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
from cska.spa import SPAState, SPAPartition, SPAModel

                 
class ModelOptimization(object):
    def __init__(self, k, rbns_analysis, n_subsample=0, sub_replace=False, aff0=1e-6, aff_min=1e-12, aff_max=1000, param_file=None, tm_refresh=.02, reporter=None, mdl_params=[],t0=0, kmer_opt_global=False):
        self.k = k

        # RBNS input sample to iterate on
        self.rbns_analysis = rbns_analysis        
        self.out_path = self.rbns_analysis.out_path

        # self.all_reads = rbns_analysis.reads
        # self.reads = self.all_reads[0] # try and phase out! TODO needs cleanup
        self.input_reads = self.rbns_analysis.reads[0]
        self.rbp_conc = np.array(self.rbns_analysis.rbp_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)
        
        self.logger = logging.getLogger('opt.ModelOptimization')
        self.time_logger = logging.getLogger('timing.ModelOptimization')
        self.reporter = reporter

        # the model instance used for optimization
        self.mdl = SPAModel(self.input_reads, k, self.rbp_conc, n_subsample=n_subsample, sub_replace=sub_replace)
        self.nA = self.mdl.parameters.nA

        self.previous = None
        self.current = None

        # observations to fit to
        self.R_obs, self.R_err = self.rbns_analysis.R_value_matrix(k)

        # bounds for the affinity parameters
        self.aff0 = aff0
        self.aff_min = aff_min
        self.aff_max = aff_max
        
        # monitor progress
        self.errors = [] #self.global_error(self.current.R), ]
        self.correlations = []
        self.rel_improvements = []
        #self.logger.error("kmer_opt_global={0}".format(kmer_opt_global))
        self.param_local_fit = not kmer_opt_global

        self.t = t0
        self.tm_refresh = max(10, int(tm_refresh * self.mdl.parameters.nA))
        self.last_tm_refresh = 0

        # initialize model
        if param_file:
            # load parameters from file
            self.mdl.parameters.load(param_file)
        elif len(mdl_params):
            # start with given parameterization
            self.mdl.parameters.assign(mdl_params)
        else:
            # start from scratch
            self.mdl.parameters.reset(betas=self.estimate_background(), aff0=aff0)

        # evaluate the thermodynamic model
        self.current = self.mdl.evaluate(self.mdl.params, keep = True)


        self.errors.append(self.global_error(self.current.R))

        # lastly, initialize the parameter update scheduler
        # self.sched = ParamUpdateScheduler(self, **sched_params)

    def estimate_background(self, q=1.):
        l = self.input_reads.L
        betas = np.nanpercentile(self.R_obs, q, axis=1) / (l - self.mdl.k + 1)
        self.logger.info("estimated background={betas} from {q} percentile of R-value distribution".format(**locals()) )
        return betas
        
    def correlation(self, R_new=[]):
        if not len(R_new):
            R_new = self.current.R
        return np.array([np.corrcoef(np.log(Ro), np.log(Rp))[0][1] for Ro, Rp in zip(self.R_obs, R_new)])
    
    def kmer_errors(self, R_new):
        return R_new - self.R_obs
           
    def kmer_error_conc(self, kmer_index, R_new, conc_i):
        return R_new[kmer_index] - self.R_obs[conc_i, kmer_index]
        
    def global_error(self, R_new):
        """used"""
        #return (((self.R_obs - R_new)**2)*self.R_obs).sum()
        return (self.kmer_errors(R_new)**2).mean()
        #lin_err = self.linearity_err(R_new)
        #print lin_err
        #return ((self.kmer_errors(R_new)**2).sum(axis=1) * (1 + lin_err) ).mean()

    def global_errors(self, R_new):
        """used"""
        #return (((self.R_obs - R_new)**2)*self.R_obs).sum()
        by_conc = (self.kmer_errors(R_new)**2).mean(axis=1)
        total = np.array([self.global_error(R_new),])
        
        return np.concatenate( (by_conc, total) )
    
    def global_error_conc(self, R_new, conc_i):
        return (self.kmer_errors(R_new)[conc_i,:]**2).mean()
    
    def local_errors(self, R_new):
        """return squared kmer R-value delta, averaged across concentrations"""
        return ((self.kmer_errors(R_new))**2).mean(axis=0)
    
    def step_tm_refresh(self):
        t0 = time.time()
        R_before = self.current.R
        self.current = self.mdl.evaluate(self.current.params, tm_update=True, keep = True)
        R_after = self.current.R
        round_err = np.fabs(R_before - R_after).sum()
        t1 = time.time()
        self.logger.debug('re-freshed thermodynamic model: rounding errors={0}'.format(round_err))
        self.time_logger.debug('step_tm_refresh in {0:.2f}ms'.format(1000.*(t1-t0)) )
        self.last_tm_refresh = self.t

    def update(self, new_state, err, name, tick=True):
        from copy import copy
        self.previous = copy(self.current)

        if self.errors:
            better = (self.errors[-1] - err)* 100./self.errors[-1]
        else:
            better = err
            
        self.current = new_state
        self.mdl.state = new_state
        self.mdl.params = new_state.params

        # subsamples should remain stable throughout one iteration step!
        if self.t - self.last_tm_refresh >= self.tm_refresh:
            self.step_tm_refresh()

        #self.mdl.new_subsample()
        self.errors.append(self.global_error(self.current.R))
        
        corr = self.correlation()
        corr_str = ",".join(["{0:.3f}".format(c) for c in corr])
        self.logger.info("update '{name}' t={self.t} err={err:.3e} (down by {better:.2e}%) corr={corr_str}".format(**locals()))
        
        self.correlations.append(corr)
        self.logger.debug("most recent errors: {0}".format( self.errors[-5:] ))
        
        if self.previous:
            update = self.current.params - self.previous.params
            #print "{0} step at t={1}".format(name, self.t)

        if tick:
            self.t += 1
            if self.reporter:
                self.reporter.tick(self.t)

            if self.errors and np.fabs(better) > 5: # TODO: make adaptive
                self.reporter.trigger_plots(self.t, occasion=name, mode='temporal')

        if self.t > 1:
            self.rel_improvements.append(better)

        return better
        
    def step_betas(self, ground_state=None, update=True):
        t0 = time.time()
        for param_i in range(self.mdl.parameters.nA, self.mdl.parameters.n_params):
            best, err, new_state = self.optimize_single_param(param_i, ground_state=ground_state, local=False)
            
            if update:
                name = "beta{0}".format(param_i - self.mdl.parameters.nA)
                print name, self.current.params[param_i], '->', new_state.params[param_i]
                better = self.update(new_state, err, name, tick=False)
                print "betas", self.current.params[self.mdl.parameters.nA:]
            else:
                better = np.nan

            ground_state = new_state
            
        dt = time.time() - t0
        self.time_logger.debug('step_betas in {0:.2f}ms'.format(1000.*dt))
        return better, new_state, err
    
    def step_scale(self, min_scale=.01, max_scale=1.):
        from cska.ska_kmers import SPA_partition_function, weighted_kmer_counts
        t0 = time.time()
        params = np.array(self.current.params)

        err0 = self.global_error(self.current.R)
        # print ">>>err0", err0
        scales = []
        errors = []
        
        # return 0, self.current

        def to_optimize(scale):
            # scale the partition function and only update pi (weighted kmer-counts)
            t0 = time.time()
            scaled = self.current * scale
            # scaled.dump("scaled")
            t1 = time.time()
            better, new_state, err = self.step_betas(ground_state = scaled, update=False)
            t2 = time.time()
            # err = self.global_error(new_state.R)

            # print "scale, err", scale, err
            # print scaled.rbp_free, new_state.rbp_free
            # print self.mdl.parameters.aff_str(new_state.params[:self.mdl.nA])
            # new_state.dump("beta")

            t3 = time.time()
            t_scale = 1000*(t1-t0)
            t_beta = 1000*(t2-t1)
            t_err = 1000*(t3-t2)
            #print "global error={err} at scale={scale} after beta fit. t_scale={t_scale:.2f}ms t_beta={t_beta:.2f}ms t_err={t_err:.2f}ms".format(**locals())
            
            errors.append(err)
            scales.append(scale)
            return err

        # for scale in 10**np.arange(-3,3,.5):
        #     to_optimize(scale)

        res = minimize_scalar(to_optimize, bounds = (min_scale, max_scale), method='Bounded')
        dt = time.time() - t0
        
        # import matplotlib.pyplot as pp
        # pp.figure()
        # pp.loglog(scales, errors, 'x')
        # pp.axhline(err0)
        # pp.show()

        if res.fun > err0:
            success = False
            best = 1.
            scaled = self.current
        else:
            success = res.success
            best = res.x
            scaled = self.current * best

        self.logger.info("global affinity re-scaling: success={success} scale={best} took {dt:.2f}s".format(**locals()) )

        better = self.update(scaled, self.global_error(scaled.R), "affinity_scale={:.3e}".format(best))
        self.step_tm_refresh()
        better, new_state, new_err = self.step_betas(update=True)
        better = self.update(new_state, self.global_error(new_state.R), "betas_after_scale")
        dt = time.time() - t0
        self.time_logger.debug('step_scale in {0:.2f}ms'.format(1000*dt) )

        return better, new_state
            
    def optimize_single_param(self, param_i, ground_state=None, local=True, accept_increase=False):
        if ground_state == None:
            ground_state = self.current
        
        x0 = ground_state.params[param_i]
        if local:
            mode = 'LOCAL'            
            err0 = self.local_errors(ground_state.R)[param_i]
        else:
            mode = 'GLOBAL'            
            err0 = self.global_error(ground_state.R)

        params = np.array(ground_state.params)
        name = self.mdl.parameters.param_name[param_i]
        A0 = ground_state.params[param_i]

        if param_i < self.mdl.parameters.nA:
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
                err = self.local_errors(state.R)[param_i]
            else:
                err = self.global_error(state.R)

            #print params[param_i], err
            return err

        def minimize_logspaced(func, bounds = [], n_samples = 10, debug=False, **kwargs):
            """
            first evaluate at log-spaced sampling points along parameter range
            then select at most 3 orders of magnitude around the lowest observed value
            for Brent optimization. Requires pos. valued bounds!
            """
            t0 = time.time()
            bmin = bounds.min()
            bmax = bounds.max()
            
            lmin = np.log10(bmin)
            lmax = np.log10(bmax)
            
            sample_x = 10**np.linspace(lmin, lmax, n_samples)
            samples = np.array([to_optimize(x) for x in sample_x])
                
            if debug:
                print "logspaced sample", zip(sample_x, samples)

            i = samples.argmin()
            li = max(0, i -1)
            ri = min(n_samples-1, i+1)
            
            brent_min = sample_x[li]
            brent_max = sample_x[ri]
            if debug:
                print "search optimum between", brent_min, brent_max
            
            res = minimize_scalar(func, bounds = np.array([brent_min, brent_max]), method='Bounded', **kwargs)
            t1 = time.time()
            
            self.time_logger.debug("minimize_logspaced took {dt:.2f}ms".format(dt= 1000. * (t1-t0)) )

            return res
            
        t0 = time.time()
        res = minimize_logspaced(to_optimize, bounds = np.array([self.aff_min, self.aff_max]) )

        #self.sweep_param(param_i, x0=res.x)

        if res.fun > err0 and not accept_increase:
            #self.sweep_param(param_i, x0=err0)
            # we have actually made it *worse* :(
            d_err = res.fun - err0
            self.logger.debug("{mode} optimization of {name} {x0:.3e}->{res.x:.3e} increased error by {d_err:.3e}. Returning initial value instead! (err0={err0:.3e})".format(**locals()) )
            best = x0
            success = False
            params[param_i] = x0
            new_state = ground_state
            
        else:
            best = res.x
            success = res.success
            params[param_i] = best
            new_state = opt.evaluate(params, tm_update=tm_update, ground_state = ground_state)

        err = self.global_error(new_state.R)

        dt = time.time() - t0
        rel_change = (best - A0) / A0

        return best, err, new_state

    def sweep_param(self, param_i, x0=1.):
        """
        debug function. Plot error as function of parameter. Will only resume
        after window with plot was closed.
        """
        import matplotlib.pyplot as pp
        pp.figure()
        name = self.mdl.parameters.param_name[param_i]
        pp.title("parameter optimization")
        #pp.title("t={0} conc={1}".format(self.t, self.rbp_conc[conc_i]))
        params = np.array(self.current.params)

        if param_i < self.mdl.parameters.nA:
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
            glob = self.global_error(state.R)
            err.append(glob)
            
            if param_i < self.mdl.parameters.nA:
                error = self.local_errors(self.current.R)[param_i]
            else:
                error = np.NaN
            loc.append(error)
         
        #print err
        err = np.array(err)
        pp.semilogx(scale, err, 'r-', label=name)
        #pp.loglog(scale, loc, 'b-', label="local (kmer) error")
        pp.vlines(x0, err.min()*.9, err.min()/.9)
        pp.xlabel("kmer affinity [1/nM]")
        pp.ylabel("global R-value error")
        
        err0 = self.global_error(self.current.R)
        pp.hlines(err0, scale.min(), scale.max(), color='gray', label='err0', linestyle='dashed')
        #if param_i < self.nA:
            #pp.axhline(self.local_errors(self.current.R)[param_i], color='blue')
        pp.legend()
        pp.tight_layout()
        pp.show()
        pp.close()

    # def step_param(self, show_sweep = False):
    #     self.logger.info("===parameter optimization===")
    #     param_i = self.sched.find_worst_param()
               
    #     if self.param_local_fit and len(self.rel_improvements) >= self.strategy_window:
    #         avg_improvement = np.mean(self.rel_improvements[-self.strategy_window:])
    #         self.logger.info("performing local fits, past rel. improvements average={0}".format(avg_improvement) )
            
    #         if avg_improvement < 0:
    #             self.logger.info("switching to global optimization at step {0}".format(self.t))
    #             self.param_local_fit = False
            
    #     best, err, new_state = self.optimize_single_param(param_i, local = self.param_local_fit)

    #     update = new_state.params - self.current.params
    #     name = self.mdl.param_name[param_i]
    #     imp = self.update(new_state, err, "single parameter optimization {name} -> {best:.3e} (err={err:.3e})".format(**locals()))
    #     # notify the scheduler of the param change and its consequences
    #     self.sched.param_changed(param_i, self.t, imp)

    #     if not imp and show_sweep:
    #         self.sweep_param(param_i, best)

    #     return imp, new_state


    # def converged(self, last=20, tol=.001):
    #     if len(self.rel_improvements) < last:
    #         return False
        
    #     tol *= 4**(- (self.k-5) ) # expect slower convergence for higher k, bc each kmer alone will have less to explain
    #     recent = np.array(self.errors[-last:])
    #     mean = recent.mean()
    #     #avg = recent.mean()
    #     dev = recent.std() / mean 
    #     self.logger.debug("mean error over past {0} iteration steps={1}, relative change={2}".format(last, mean, dev) )
    #     if dev < tol:
    #         self.logger.info("convergence with mean improvement of {0}".format(dev) )
    #         return True
        
    #     return False

    # def linearity_err(self, R_new=[]):
    #     if not len(R_new):
    #         R_new = self.current.R

    #     lin = []
    #     for i in range(self.n_conc):
    #         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(R_new[i,:], self.R_obs[i,:])
    #         lin.append( np.fabs(1 - slope) + np.fabs(intercept) )

    #     return np.array(lin)
 
