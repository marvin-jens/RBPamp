import numpy as np
import scipy
import logging
import time
import sys
import os
from collections import defaultdict
from scipy.optimize import minimize, brentq, minimize_scalar

import cska.ska_kmers as cyska

from cska.caching import CachedBase, cached, pickled
from cska.affinity import Kd_to_kcal, kcal_to_Kd, AffinityDistribution
from cska.scheduler import ParamUpdateScheduler
#from cska.psam import PSAMState
#from cska import timed
#from cska.optimize import ModelOptimization
from cska.comparison import ReferenceComparison
from cska.spa import SPAState, SPAPartition, SPAModel

                 
class ModelOptimization(object):
    def __init__(self, k, rbns_analysis, known_params = [], rbp_conc=[40.], out_path="./", n_subsample=0, sub_replace=False, aff0=1e-6, aff_min=1e-12, aff_max=1000, param_file=None, seq_only=False, tm_refresh=.02, sched_params = {}, beta_interval = .01, scale_interval=100000000000000, reporter=None, mdl_params=[],t0=0, kmer_opt_global=False): # scale_interval=.02
        self.k = k
        self.nA = 4**k

        # RBNS input sample to iterate on
        self.rbns_analysis = rbns_analysis        
        self.all_reads = rbns_analysis.reads
        self.reads = self.all_reads[0] # try and phase out! TODO needs cleanup
        self.input_reads = self.all_reads[0]
        self.storages = self.rbns_analysis.acc_storages
        self.openen = self.storages[0].get_discretized(k)
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)

        self.n_params = self.nA + self.n_conc

        self.beta_interval = int(beta_interval * self.nA)
        self.scale_interval = int(scale_interval * self.nA)
        self.strategy_window = int(.1 * self.nA) # number of steps average last error to
        # decide if we want to switch from local to global optimization
        self.last_beta = 0
        self.last_scale = 0
        
        self.t = t0
        self.tm_refresh = max(10, int(tm_refresh * self.nA))
        self.last_tm_refresh = 0

        self.logger = logging.getLogger('opt.ModelOptimization')
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.opt_path = os.path.join(self.out_path, 'opt/{0}mers'.format(self.k))
        if not os.path.exists(self.opt_path):
            os.makedirs(self.opt_path)

        self.reporter=reporter
        # observations to fit to
        self.R_obs, self.R_err = self.rbns_analysis.R_value_matrix(k)
        
        # bounds for the affinity parameters
        self.aff0 = aff0
        self.aff_min = aff_min
        self.aff_max = aff_max

        # the model to be trained
        self.mdl = SPAModel(self.input_reads, self.openen, k, self.rbp_conc, n_subsample=n_subsample, sub_replace=sub_replace, seq_only=seq_only)
        
        # monitor progress
        self.errors = [] #self.global_error(self.current.R), ]
        self.correlations = []
        self.rel_improvements = []
        self.logger.error("kmer_opt_global={0}".format(kmer_opt_global))
        self.param_local_fit = not kmer_opt_global
        
        # set initial state of the model
        self.previous = None
        self.current = None
        if param_file:
            #self.update(self.mdl.load_params(param_file), np.Inf, "resuming from {0}".format(param_file))
            self.logger.info("resuming from {0}".format(param_file))
            self.current = self.mdl.evaluate(self.mdl.load_params(param_file), keep = True)
            
            ### TESTING: compare observed and predicted affinity distributions
            #import matplotlib.pyplot as pp
            #aff_in = AffinityDistribution(self.mdl, self.all_reads[0], self.storages[0].get_discretized(self.k))
            #bg = aff_in.get_affinity_distribution()
            #print "input", bg
            #for i, conc in enumerate(self.rbp_conc):
                #beta = self.mdl.params[self.mdl.nA+i]
                #pred = aff_in.predict_affinity_distribution(conc, beta=beta, bg=bg[0])
                #print "predicted", pred
                #aff = AffinityDistribution(self.mdl, self.all_reads[i+1], self.storages[i+1].get_discretized(self.k))
                #obs = aff.get_affinity_distribution()
                #print "observed", obs
                #x = (aff_in.bins[1:] + aff_in.bins[:-1])/2.
                #pp.plot(x, pred[0], '-', label='pred {0}nM'.format(conc) )
                #pp.plot(x, obs[0], '.', label='obs {0}nM'.format(conc) )
                #pp.show()
                
            #pp.xlabel('predicted')
            #pp.ylabel('observed')
            #pp.legend()
            #pp.show()
            #1/0
            
        elif len(mdl_params):
            # start with given parameterization
            params = np.array(mdl_params, dtype=np.float32)
            assert len(params) == self.nA + len(rbp_conc)
            self.current = self.mdl.evaluate(params, keep = True)
        else:
            # start from scratch
            params = np.zeros(self.nA + len(rbp_conc), dtype=np.float32)
            params[0:self.nA] += aff0
            betas = self.estimate_background()
            params[-len(betas):] = betas
            self.current = self.mdl.evaluate(params, keep = True)
            #self.update(params, np.Inf, "initialize")
        
        self.errors.append(self.global_error(self.current.R))
        
        # in case we know some parameters, use this as a reference
        if len(known_params):
            self.known_params = known_params
        else:
            self.known_params = np.ones(self.current.params.shape, dtype=np.float32) * np.nan

        # lastly, initialize the parameter update scheduler
        self.sched = ParamUpdateScheduler(self, **sched_params)

    def estimate_background(self, q=1.):
        l = self.all_reads[0].L
        betas = np.nanpercentile(self.R_obs, q, axis=1) / (l - self.k + 1)
        self.logger.info("estimated background={betas} from {q} percentile of R-value distribution".format(**locals()) )
        return betas
        
    def correlation(self, R_new=[]):
        if not len(R_new):
            R_new = self.current.R
        return np.array([np.corrcoef(np.log(Ro), np.log(Rp))[0][1] for Ro, Rp in zip(self.R_obs, R_new)])
    
    def linearity_err(self, R_new=[]):
        if not len(R_new):
            R_new = self.current.R

        lin = []
        for i in range(self.n_conc):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(R_new[i,:], self.R_obs[i,:])
            lin.append( np.fabs(1 - slope) + np.fabs(intercept) )

        return np.array(lin)
 
    def kmer_errors(self, R_new):
        return R_new - self.R_obs
           
    def kmer_error_conc(self, kmer_index, R_new, conc_i):
        return R_new[kmer_index] - self.R_obs[conc_i, kmer_index]
        
    def global_error(self, R_new):
        """used"""
        #return (((self.R_obs - R_new)**2)*self.R_obs).sum()
        #return (self.kmer_errors(R_new)**2).sum()
        lin_err = self.linearity_err(R_new)
        #print lin_err
        return ((self.kmer_errors(R_new)**2).sum(axis=1) * (1 + lin_err) ).mean()
    
    def global_error_conc(self, R_new, conc_i):
        return (self.kmer_errors(R_new)[conc_i,:]**2).mean()
    
    def local_errors(self, R_new):
        """return squared kmer R-value delta, averaged across concentrations"""
        return ((self.kmer_errors(R_new))**2).mean(axis=0)
    
    def converged(self, last=20, tol=.001):
        if len(self.rel_improvements) < last:
            return False
        
        tol *= 4**(- (self.k-5) ) # expect slower convergence for higher k, bc each kmer alone will have less to explain
        recent = np.array(self.errors[-last:])
        mean = recent.mean()
        #avg = recent.mean()
        dev = recent.std() / mean 
        self.logger.debug("mean error over past {0} iteration steps={1}, relative change={2}".format(last, mean, dev) )
        if dev < tol:
            self.logger.info("convergence with mean improvement of {0}".format(dev) )
            return True
        
        return False

    def step_tm_refresh(self):
        R_before = self.current.R
        self.current = self.mdl.evaluate(self.current.params, tm_update=True, keep = True)
        R_after = self.current.R
        round_err = np.fabs(R_before - R_after).sum()
        self.logger.debug('re-freshed thermodynamic model: rounding errors={0}'.format(round_err))
        self.last_tm_refresh = self.t

    def update(self, new_state, err, name, tick=True):
        from copy import copy
        self.previous = copy(self.current)
        
        self.current.params[self.nA:]
        if self.errors:
            better = (self.errors[-1] - err)* 100./self.errors[-1]
        else:
            better = err
            
        self.current = new_state
        self.mdl.state = new_state
        self.mdl.params = new_state.params

        # subsamples should remain stable throughout one iteration step!
        if self.t - self.last_tm_refresh >= self.tm_refresh or (better > 25.):
            self.step_tm_refresh()

        self.current.params[self.nA:]

        #self.mdl.new_subsample()
        self.errors.append(self.global_error(self.current.R))
        
        self.logger.info("status after '{0}' step at t={1}, improvement was {2:.2e}%".format(name, self.t, better))
        corr = self.correlation()
        self.correlations.append(corr)
        self.logger.debug("correlations: {0}".format(corr) )
        self.logger.debug("most recent errors: {0}".format( self.errors[-5:] ))
        
        if self.previous:
            update = self.current.params - self.previous.params
            #print "{0} step at t={1}".format(name, self.t)

        if tick:
            self.t += 1
            if self.reporter:
                self.reporter.tick(self.t)

        if self.t > 1:
            self.rel_improvements.append(better)

        return better
        
    def step_param(self, show_sweep = False):
        self.logger.info("===parameter optimization===")
        param_i = self.sched.find_worst_param()
               
        if self.param_local_fit and len(self.rel_improvements) >= self.strategy_window:
            avg_improvement = np.mean(self.rel_improvements[-self.strategy_window:])
            self.logger.info("performing local fits, past rel. improvements average={0}".format(avg_improvement) )
            
            if avg_improvement < 0:
                self.logger.info("switching to global optimization at step {0}".format(self.t))
                self.param_local_fit = False
            
        best, err, new_state = self.optimize_single_param(param_i, local = self.param_local_fit)

        update = new_state.params - self.current.params
        name = self.mdl.param_name[param_i]
        imp = self.update(new_state, err, "single parameter optimization {name} -> {best:.3e} (err={err:.3e})".format(**locals()))
        # notify the scheduler of the param change and its consequences
        self.sched.param_changed(param_i, self.t, imp)

        if not imp and show_sweep:
            self.sweep_param(param_i, best)

        return imp, new_state

    def step_betas(self, ground_state=None, update=True):
        for param_i in range(self.nA, self.n_params):
            best, err, new_state = self.optimize_single_param(param_i, ground_state=ground_state, local=False)
            
            if update:
                better = self.update(new_state, err, "beta{0} parameter optimization".format(param_i - self.nA), tick=False)
            else:
                better = np.nan

            ground_state = new_state
            
        return better, new_state
    
    def step_scale(self, min_scale=.01):
        from cska.ska_kmers import SPA_partition_function, weighted_kmer_counts
        import time
        t0 = time.time()
        params = np.array(self.current.params)

        scales = []
        errors = []
        def to_optimize(scale):
            # scale the partition function and only update pi (weighted kmer-counts)
            t0 = time.time()
            scaled = self.current * scale
            t1 = time.time()
            #better, new_state = self.step_betas(ground_state = state, update=False)
            t2 = time.time()
            err = self.global_error(scaled.R)
            t3 = time.time()
            t_scale = 1000*(t1-t0)
            t_beta = 1000*(t2-t1)
            t_err = 1000*(t3-t2)
            #print "global error={err} at scale={scale} after beta fit. t_scale={t_scale:.2f}ms t_beta={t_beta:.2f}ms t_err={t_err:.2f}ms".format(**locals())
            
            errors.append(err)
            return err

        res = minimize_scalar(to_optimize, bounds = (min_scale, 1), method='Bounded')
        dt = time.time() - t0
        self.logger.info("global affinity re-scaling: success={res.success} scale={res.x} took {dt:.2f}s".format(**locals()) )

        scaled = self.current * res.x
        better = self.update(scaled, self.global_error(scaled.R), "affinity re-scaling")
        self.step_tm_refresh()
        better, new_state = self.step_betas(update=True)
        better = self.update(new_state, self.global_error(new_state.R), "betas re-scaling after affinity rescaling")
        return better, new_state
            
    def optimize_single_param(self, param_i, ground_state=None, local=True, accept_increase=False):
        if ground_state == None:
            ground_state = self.current
        
        x0 = ground_state.params[param_i]
        if local:
            err0 = self.local_errors(ground_state.R)[param_i]
        else:
            err0 = self.global_error(ground_state.R)

        params = np.array(ground_state.params)

        if param_i < self.nA:
            # evaluate thermodynamic model, but only on the subset of sequences containing the kmer
            tm_update = True
            opt = SPAPartition(self.mdl, param_i)
        else:
            # do not evaluate the thermodynamic model, only re-compute R-values (for beta optimization)
            tm_update = False
            opt = self.mdl
            
        def to_optimize(aff):
            params[param_i] = aff #* .0001
            state = opt.evaluate(params, tm_update=tm_update, ground_state = ground_state)
            
            if local:
                err = ((state.R[:,param_i] - self.R_obs[:,param_i])**2).sum()
            else:
                err = self.global_error(state.R)

            #print params[param_i], err
            return err

        def minimize_logspaced(func, bounds = [], n_samples = 10, **kwargs):
            """
            first evaluate at log-spaced sampling points along parameter range
            then select at most 3 orders of magnitude around the lowest observed value
            for Brent optimization. Requires pos. valued bounds!
            """
            
            bmin = bounds.min()
            bmax = bounds.max()
            
            lmin = np.log10(bmin)
            lmax = np.log10(bmax)
            
            sample_x = 10**np.linspace(lmin, lmax, n_samples)
            samples = np.array([to_optimize(x) for x in sample_x])
                
            #print "logspaced sample", zip(sample_x, samples)
            i = samples.argmin()
            li = max(0, i -1)
            ri = min(n_samples-1, i+1)
            
            brent_min = sample_x[li]
            brent_max = sample_x[ri]
            #print "search optimum between", brent_min, brent_max
            
            res = minimize_scalar(func, bounds = np.array([brent_min, brent_max]), method='Bounded', **kwargs)
            return res
            
        t0 = time.time()
        res = minimize_logspaced(to_optimize, bounds = np.array([self.aff_min, self.aff_max]) )

        if res.fun > err0 and not accept_increase:
            # we have actually made it *worse* :(
            self.logger.warning("optimization increased error by {d_err}. Returning initial value instead!".format(d_err = res.fun - err0) )
            best = x0
            success = False
            params[param_i] = x0
            new_state = ground_state
            err = err0
        else:
            best = res.x
            success = res.success
            params[param_i] = best
            new_state = opt.evaluate(params, tm_update=tm_update)
            err = self.global_error(new_state.R)

        dt = time.time() - t0
        name = self.mdl.param_name[param_i]
        A0 = ground_state.params[param_i]
        rel_change = (best - A0) / A0

        if local:
            mode = 'LOCAL'
        else:
            mode = 'GLOBAL'
        
        self.logger.debug("{mode} optimal {name} affinity/value search success={success} A={best} (A0={A0} rel change={rel_change}) took {dt:.2f}s".format(**locals()) )

        return best, err, new_state

    def sweep_param(self, param_i, x0=1.):
        """
        debug function. Plot error as function of parameter. Will only resume
        after window with plot was closed.
        """
        import matplotlib.pyplot as pp
        pp.figure()
        #pp.title("t={0} conc={1}".format(self.t, self.rbp_conc[conc_i]))
        params = np.array(self.current.params)

        if param_i < self.nA:
            # evaluate thermodynamic model, but only on the subset of sequences containing the kmer
            tm_update = True
            opt = SPAPartition(self.mdl, param_i)
        else:
            # do not evaluate the thermodynamic model, only re-compute R-values
            tm_update = False
            opt = self.mdl
            
        scale = 10**np.arange(np.log10(self.aff_min),np.log10(self.aff_max),.1)
        err = []
        loc = []
        for s in scale:
            params[param_i] = s
            state = opt.evaluate(params, tm_update=tm_update)
            #err.append(self.global_error_conc(state.R, conc_i))
            err.append(self.global_error(state.R))
            error = ((state.R[:,param_i] - self.R_obs[:,param_i])**2).sum()
            loc.append(error)
         
        #print err
        pp.loglog(scale, err, label="global error")
        pp.loglog(scale, loc, label="local (kmer) error")
        pp.axvline(x0)
        pp.axhline(self.errors[-1])
        pp.legend()
        pp.show()
        pp.close()

