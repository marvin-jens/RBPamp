import rbpbind as rbp
from cska.rbns_reads import RBNSReads
from cska.rbns_model import RBNSSimulator, RBNSGenerator
from cska.folding import RBNSOpenen, OpenenStorage, OpenenDiscretization

import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as pp

#pp.style.use('ggplot')
import logging
import os, sys, time
import numpy as np
import cska.cyska
import scipy.stats
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)

class Optimizer(object):
    def __init__(self, k, reads, openen_storage, protein_conc, R_obs, known_invkd=[], n_subsample=100000, sub_replace=False, aff0=1e-6, temp=22, debug=True, min_invkd=1e-12, max_invkd=1e1, n_blocked=1):
        self.debug = debug
        self.logger = logging.getLogger("Optimizer")

        self.k = k
        self.kmers = np.array([mer.upper().replace('T','U') for mer in cyska.yield_kmers(k)])
        self.reads = reads
        self.openen_storage = openen_storage
        self.openen = self.openen_storage.get_discretized(self.k)
        self.rbp_conc = np.array(protein_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)
        
        if not len(known_invkd):
            self.known_invkd = np.ones(4**k, dtype=np.float32) * np.nan
        else:
            self.known_invkd = known_invkd # for simulated data
        
        self.n_subsample = n_subsample
        self.sub_replace = sub_replace # subsamping with replacement is faster!

        self.temp = temp
        self.RT = (self.temp + 273.15) * 8.314459848/4.184E3 # RT in kcal/mol

        self.openen_lookup = self.openen.disc.x/self.RT # open energies in units of RT for each discretization level
        self.acc_lookup = np.exp(-self.openen_lookup) # accessibilities

        self.f0 = self.reads.kmer_frequencies(self.k)
        self.f0 /= self.f0.sum()

        # most important piece of input
        self.R_obs = R_obs

        # initial guess is flat affinity vector!
        self.R_old = np.ones((self.n_conc, 4**self.k), dtype=np.float32)
        self.R_new = np.copy(self.R_old)

        # updated with each optimization
        self.kmer_error_old = self.kmer_errors(self.R_old)
        self.kmer_error_new = self.kmer_errors(self.R_new)

        self.kmer_rel_error_old = self.kmer_rel_errors(self.R_old)
        self.kmer_rel_error_new = self.kmer_rel_errors(self.R_new)
        
        self.trial_invkd = np.ones(4**self.k, dtype=np.float32) * aff0
        self.prev_invkd = np.copy(self.trial_invkd)
        
        # bounds for root-finding. Can get more narrow as the search progresses
        self.kmer_invkd_bounds = np.repeat(np.array((min_invkd, max_invkd)), 4**self.k).reshape((2,4**self.k)).T
        self.min_invkd = min_invkd
        self.max_invkd = max_invkd
        
        self.kmer_updates = defaultdict(int)
        self.kmer_update_count = np.zeros(4**self.k)
        self.last_kmer_update = -1

        self.heat = np.zeros(4**self.k, dtype=float)
        self.blocked_kmers = []
        self.n_blocked = n_blocked
        self.n_blocked0 = n_blocked
        self.t = 0 # iteration step counter

        self.subsample_indices = self.new_subsample()
        
        from cska.rbns_model import CrosstalkMatrix
        self.cm = CrosstalkMatrix(self.k, self.reads)
        self.update_history = []
        
        self.global_errors = []
        self.update_projections = []
    
    def new_subsample(self):
        #return np.arange(self.n_subsample)
        #return np.random.randint(self.reads.N, size=self.n_subsample)
        return np.random.choice(self.reads.N, size=self.n_subsample, replace= self.sub_replace)

    def gradient(self, delta=0.0001, I = []):
        #TODO: get this out of enhanced eval_energy_model_on_seqs() + some analytics
        E0 = np.copy(self.trial_invkd)
        grad = []
        t0 = time.time()
        err0 = self.global_error(self.R_new)
        
        if not len(I):
            I = np.arange(4**self.k)
        self.logger.debug("computing gradient for {0} kmers".format(len(I)) )
        
        for i in I:
            E0[i] += delta
            p_bound, kmer_counts, openen_kmer_bincount_matrix, gradient = self.eval_thermodynamic_model(kmer_invkd = E0)
            E0[i] -= delta
            R_trial = (kmer_counts / kmer_counts.sum(axis=1)[:, np.newaxis] ) / self.f0
            
            err1 = self.global_error(R_trial)
            grad.append( (err1 - err0)/delta)
            
        dt = time.time() - t0
        self.logger.debug("gradient computation took {0:.3f} seconds".format(dt) )
        return np.array(grad)
        
    def eval_thermodynamic_model(self, kmer_invkd = [], indices = [], protein_conc = [], seq_only=False, do_jacobi=False):
        from cyska import eval_energy_model_on_seqs
        import time
        
        # prepare all variables
        if not len(kmer_invkd):
            kmer_invkd = self.trial_invkd

        if not len(protein_conc):
            protein_conc = self.rbp_conc
            
        if not len(indices):
            indices = self.subsample_indices

        seqm = self.reads.seqm[indices]
        oem = self.openen.oem[indices]

        if seq_only:
            acc_lookup = np.ones(self.acc_lookup.shape, dtype = np.float32)
        else:
            acc_lookup = self.acc_lookup
            
        t0 = time.time()
        # the model itself is implemented in Cython
        p_bound, kmer_count_matrix, openen_kmer_bincount_matrix, jacobi = eval_energy_model_on_seqs(
            seqm, 
            oem, 
            acc_lookup,
            kmer_invkd, 
            protein_conc, 
            self.k, 
            n_max = self.n_subsample,
            do_jacobi = do_jacobi
        )
        t1 = time.time()
        n = p_bound.shape[1]

        #if self.debug:
            #self.logger.debug("evaluated energy model on {0} sequences in {1:.2f} ms".format(n, 1000*(t1-t0)) )

        return p_bound, kmer_count_matrix, openen_kmer_bincount_matrix, jacobi
        
    def kmer_errors(self, R_new):
        return R_new - self.R_obs
        
    def kmer_error(self, kmer_index, R_new):
        return R_new[:,kmer_index] - self.R_obs[:,kmer_index]
    
    def kmer_rel_errors(self, R_new):
        return np.log2(R_new / self.R_obs)

    def kmer_rel_error(self, kmer_index, R_new):
        return np.log2(R_new[:,kmer_index] / self.R_obs[:,kmer_index])
    
    def kmer_error_conc(self, kmer_index, R_new, conc_i):
        return R_new[kmer_index] - self.R_obs[conc_i, kmer_index]
        
    def global_error(self, R_new):
        return np.mean((self.R_obs - R_new)**2)
    
    def emp_jacobi(self, delta=1e-5, I=[]):
        E0 = np.copy(self.trial_invkd)
        jac = []
        t0 = time.time()
        R0 = self.R_new
        
        if not len(I):
            I = np.arange(4**self.k)
        self.logger.debug("computing Jacobi matrix for {0} kmers".format(len(I)) )
        
        for i in I:
            #d = E0[i] * delta
            e0 = E0[i]
            E0[i] = e0 + delta
            p_bound, kmer_counts, openen_kmer_bincount_matrix, jac_na = self.eval_thermodynamic_model(kmer_invkd = E0)
            E0[i] = e0
            kmer_counts += 1
            R_trial = (kmer_counts / kmer_counts.sum(axis=1)[:, np.newaxis] ) / self.f0
            
            dR = (R_trial - R0)/delta
            jac.append(dR)
            
        dt = time.time() - t0
        self.logger.debug("empirical Jacobi computation took {0:.3f} seconds".format(dt) )
        return np.array(jac).transpose((1,2,0))
        
    def find_worst_kmer(self):
        
        """
        ideas: 
            * take into account if over-under-representation is systematic or only seen at some concentrations (contradicted at others?)
            * start preferring kmers whose error peaks at low concentrations (high affinity), then move to kmers whose error peaks at intermediate or high concentrations (lower affinity)
            * perhaps overall R-value correlation can serve as guide? (> .9 do a round of low affinity optimizations in fixed energy background?)
        """
        badness = np.fabs(self.kmer_error_new.mean(axis=0)) #+ np.fabs(self.kmer_rel_error_new).mean(axis=0)
        score  = np.sqrt(badness) #/ (self.heat + 1)
        
        if self.debug:
            print "!!!!! kmers with max residual error"
            for i in score.argsort()[::-1][:10]:
                #grad = self.gradient(I=[i,])
                print cyska.index_to_seq(i, self.k), "score=",score[i], "bad=",badness[i], "heat=",self.heat[i], self.known_invkd[i], self.trial_invkd[i], "residual", self.kmer_error_new[:,i]#, "grad", grad
                #for j, w in self.cm.get_shadow(i):
                    #s_mer = cyska.index_to_seq(j, self.k)
                    #s_err = self.kmer_error_new[:,j]
                    #s_msd = (self.kmer_error_new[:,i] - s_err/w)**2
                    #print "shadow: {s_mer}: {w} error: {s_err} sq.dev: {s_msd}".format(**locals())

        for i in score.argsort()[::-1]:
            if not i in self.blocked_kmers:
                yield i

        #return badness.argmax()
        
    def highest_ranked_unoptimized_kmers(self, n = 10):
        
        affinity = self.known_invkd
        
        print "!!!!! highest ranked unoptimized kmers !!!!"
        for i in affinity.argsort()[::-1]:
            if self.kmer_update_count[i] > 0:
                continue
                
            print cyska.index_to_seq(i, self.k), self.known_invkd[i], self.trial_invkd[i], "abs. error", self.kmer_error_new[:,i]
            n -= 1
            if not n:
                break

    def predict_R(self, trial_invkd = []):
        if not len(trial_invkd):
            trial_invkd = self.trial_invkd

        pb, kmer_counts, openen_kmer_bincount, jacobi = self.eval_thermodynamic_model(trial_invkd)
        self.jacobi = jacobi
        R_trial = (kmer_counts / kmer_counts.sum(axis=1)[:, np.newaxis] ) / self.f0
        return R_trial
        

    def predict_R_conc(self, invkd = [], conc_i=0):
        if not len(invkd):
            invkd = self.trial_invkd

        # evaluate the model at only one specific concentration
        pb, kmer_counts, openen_kmer_bincount, jacobi = self.eval_thermodynamic_model(
            kmer_invkd = invkd, 
            protein_conc = np.array([self.rbp_conc[conc_i],], dtype=np.float32)
        )
        kmer_counts = kmer_counts[0,:]
        R_pred = (kmer_counts / kmer_counts.sum()) / self.f0
        
        return R_pred
        

    def optimize_kmer(self, kmer_index):
        from scipy.optimize import minimize, brentq, minimize_scalar

        # flip R value, error records, and trial energies:
        self.R_new, self.R_old = self.R_old, self.R_new
        self.kmer_error_new, self.kmer_error_old = self.kmer_error_old, self.kmer_error_new
        self.kmer_rel_error_new, self.kmer_rel_error_old = self.kmer_rel_error_old, self.kmer_rel_error_new
        self.prev_invkd[:] = self.trial_invkd[:]
        
        def predict(kmer_invkd, conc_i): # todo: cache, so first brentq iteration is faster.
            self.trial_invkd[kmer_index] = kmer_invkd
            R_pred = self.predict_R_conc(self.trial_invkd, conc_i)
            return R_pred, self.kmer_error_conc(kmer_index, R_pred, conc_i)

        def to_optimize(kmer_invkd):
            R_trial, err = predict(kmer_invkd, conc_i)
            return err
    
        # find the lowest protein concentration with a root in the current interval
        conc_i = 0
        found = False
        min_invkd, max_invkd = self.kmer_invkd_bounds[kmer_index]
        
        #badness_order = np.fabs(self.kmer_error_old[:,kmer_index]).argsort()[::-1]
        #for conc_i in badness_order:
        for conc_i in np.arange(self.n_conc):
            left_R, left_err = predict(min_invkd, conc_i)
            right_R, right_err = predict(max_invkd, conc_i)
            if (left_err > 0) != (right_err > 0):
                found = True
                self.logger.debug("sign change for kmer {kmer_index}  R_err({min_invkd}) = {left_err} and R_err({max_invkd}) = {right_err} at conc {conc_i}".format(**locals()) )
                break
            else:
                self.logger.debug("NO SIGN CHANGE for kmer {kmer_index}  R_err({min_invkd}) = {left_err} and R_err({max_invkd}) = {right_err} at conc {conc_i}".format(**locals()) )
        
        if not found:
            raise ValueError("no sign change at any protein concentration")
        
        print "using concentration", conc_i
        # find the invkd value at which the kmer-error becomes 0
        invkd_opt, res = brentq(to_optimize, min_invkd, max_invkd, full_output=True)

        # accept the new state
        self.trial_invkd[kmer_index] = invkd_opt
        self.R_new = self.predict_R(self.trial_invkd)
        self.jacobi_new = self.jacobi # self.jacobi got updated implicitly by self.predict_R
        self.kmer_error_new = self.kmer_errors(self.R_new)
        self.kmer_rel_error_new = self.kmer_rel_errors(self.R_new)
        self.trial_invkd[kmer_index] = invkd_opt
        
        # narrow bounds to 6 orders of magnitude around current optimum.
        self.kmer_invkd_bounds[kmer_index,:] = [max(invkd_opt*1e-3, self.min_invkd), min(invkd_opt*1e3, self.max_invkd)] 
        
        return res, invkd_opt

    def predict_occ(self):
        theta = 1./ (1. + 1./ (self.rbp_conc[:,np.newaxis] / self.trial_invkd[np.newaxis,:]) )
        return theta
    
    def predict_R_from_cm(self):
        R = []
        for theta in self.predict_occ():
            R.append(self.cm.predict_R_from_occ(theta))
        return np.array(R)
        
    def step(self):
        self.t += 1
        self.logger.debug("iteration {self.t}".format(self=self) )
        
        # subsamples should remain stable throughout one iteration step!
        self.subsample_indices = self.new_subsample()

        print "iteration", self.t
        print "updated_kmers", self.kmer_updates
        self.highest_ranked_unoptimized_kmers()
        
        #self.test_gradient()
        for kmer_index in self.find_worst_kmer():
            kmer = self.kmers[kmer_index]
            ikd0 = self.trial_invkd[kmer_index]

            if self.debug:
                print "blocked kmers", self.blocked_kmers
                R_obs = self.R_obs[:, kmer_index]
                R_old = self.R_old[:, kmer_index]
                err_old = self.kmer_error_old[:, kmer_index]
                print "worst unblocked kmer is {kmer} with R_obs={R_obs} R_pred={R_old} error={err_old}".format(**locals())
                
            try:
                res, invkd = self.optimize_kmer(kmer_index)
            except ValueError:
                self.logger.warning("selected kmer {kmer} has no local error minimum for any concentration. Skipping!".format(kmer=kmer))
                self.trial_invkd[kmer_index] = ikd0
                continue
                
            if self.debug:
                R_new = self.R_new[:, kmer_index]
                err_new = self.kmer_error_new[:, kmer_index]
                known = self.known_invkd[kmer_index]
                delta = invkd - self.prev_invkd[kmer_index]
                print "optimal invkd={invkd} delta={delta} (known={known}) R_obs={R_obs} R_old={R_old} R_new={R_new} err_old={err_old} err_new={err_new}".format(**locals())

            self.blocked_kmers.append(kmer_index)
            if len(self.blocked_kmers) > self.n_blocked:
                self.blocked_kmers.pop(0)
            
            #self.n_blocked = self.n_blocked0 + self.t / 5
            self.logger.debug("at time {self.t} n_blocked = {self.n_blocked}".format(self=self) )
            self.kmer_updates[kmer] += 1
            self.kmer_update_count[kmer_index] += 1
            self.last_kmer_update = kmer_index
            self.heat *= 0.75
            self.heat[kmer_index] += 3
            break
    
        
    def update_from_jacobi(self):
        t0 = time.time()
        p_bound, pi, openen_kmer_bincount_matrix, jac_pi = self.eval_thermodynamic_model(do_jacobi=True)
        t1 = time.time()
        
        pi_sum = pi.sum(axis=1)
        R = (pi / pi_sum[:, np.newaxis] ) / self.f0
        
        #direct = jac_pi / pi_sum[:, np.newaxis, np.newaxis]
        diag = np.diagonal(jac_pi, axis1=1, axis2=2)
        #indirect = - pi[:, np.newaxis, :] / (pi_sum * pi_sum)[:, np.newaxis, np.newaxis] * diag[:,:, np.newaxis]
        #jac_R = direct + indirect

        jac_R = 1/pi[:, :, np.newaxis] * (R[:, :, np.newaxis] * jac_pi - (self.f0[np.newaxis,:]*R*R)[:, :, np.newaxis] * diag[:, np.newaxis, :] )

        from numpy.linalg import norm
        corr = self.known_invkd - self.trial_invkd
        corr /= norm(corr)
        
        updates = []
        for jac, R_obs, R_curr, conc in zip(jac_R, self.R_obs, self.R_new, self.rbp_conc):
            
            x = R_obs - R_curr
            indices = np.fabs(x).argsort()[::-1][:100]
            
            J = jac[indices,:][:, indices]
            jac_inv = np.linalg.inv(J)    
            x = R_obs - R_curr
            res = np.dot(jac_inv, x[indices])
            
            u = np.zeros(self.trial_invkd.shape, dtype=np.float32)
            u[indices] = res / norm(res)

            cos = np.dot(u, corr)
            angle = np.arccos(np.clip(cos, -1, 1) ) * 180./np.pi
            print "projection of update vector from {0} onto ideal update={1} and angle={2}".format(conc, cos, angle)
            
            updates.append(u)

        self.jacobi_new = jac_R
        t2 = time.time()
        self.logger.debug("computed Jacobi matrix in {0:.2f}ms, inverted and computed gradient vector in {1:.2f}ms, total={2:.2f}ms".format(t1-t0, t2-t1, t2-t0) )
        return np.array(updates, dtype=np.float32)

    def line_search(self, vec, smin=0., smax=1.):
        from scipy.optimize import minimize, brentq, minimize_scalar
            
        # find optimal scale for line-search
        def to_optimize(scale):
            err = self.global_error(self.predict_R(np.clip(self.trial_invkd + scale * vec, 1e-9, 1e3) ))
            #print "ls {0} -> err={1}".format(scale, err)
            return err
            
        t0 = time.time()
        res = minimize_scalar(to_optimize, bounds = [smin, smax], method='Bounded')
        print "line search optimization result", res.success, res.x, res.fun
        self.logger.debug("optimal line search success={0} scale={1} took {2:.2f}ms".format(res.success, res.x, time.time() - t0) )

        return vec*res.x, res.fun

    def combination_search(self, vecs, smin=0., smax=1.):
        from scipy.optimize import minimize, brentq, minimize_scalar
            
        # find optimal scale for line-search
        def to_optimize(coeff):
            coeff = np.array(coeff, dtype=np.float32)
            err = self.global_error(self.predict_R(np.clip(self.trial_invkd + np.dot(coeff, vecs), 1e-9, 1e3) ))
            print "cs {0} -> err={1}".format(coeff, err)
            return err
            
        t0 = time.time()
        
        n = len(vecs)
        bounds = np.ones((n,2), dtype=np.float32) * np.array([smin, smax], dtype=np.float32)[np.newaxis,:]
        c0 = np.ones(n, dtype=np.float32) * .25
        res = minimize(to_optimize, c0, bounds = bounds, method='L-BFGS-B')
        print "combination search optimization result", res.success, res.x, res.fun
        self.logger.debug("combination search success={0} coeff={1} took {2:.2f}ms".format(res.success, res.x, time.time() - t0) )

        return np.dot(np.array(res.x, dtype=np.float32), vecs), res.fun

        
    def annotate_update_vector(self, vec, n=10):
        print "-----------update vector------------"
        for i in np.fabs(vec).argsort()[::-1][:n]:
            print "    ", cyska.index_to_seq(i, self.k), self.trial_invkd[i], "->", self.trial_invkd[i] + vec[i], "known=", self.known_invkd[i]
        

    def step_gradient(self):
        self.t += 1

        self.logger.debug("gradient descent iteration {self.t}".format(self=self) )
        
        # subsamples should remain stable throughout one iteration step!
        self.subsample_indices = self.new_subsample()

        print "gradient descent at iteration", self.t
        #self.highest_ranked_unoptimized_kmers()

        self.R_old[:] = self.R_new[:]
        self.kmer_error_old[:] = self.kmer_error_new[:]
        self.prev_invkd[:] = self.trial_invkd[:]

        updates = self.update_from_jacobi()

        ls_results = []
        for u, rbp_conc in zip(updates, self.rbp_conc):
            ls_results.append(self.line_search(u))
            #self.annotate_update_vector(ls_results[-1][0])
    
        scaled_updates, errors = np.array(ls_results).T
        i = errors.argmin()
        print "best concentration is ",self.rbp_conc[i], "with global error", errors[i]
        update = scaled_updates[i]
    
        #update, err = self.combination_search(updates)
        #print "best linear combination update vector reaches global error", err
        self.annotate_update_vector(update)
        
        new_invkd = np.clip(self.trial_invkd + update, 1e-9, 1e3)

        residual = self.known_invkd - self.trial_invkd

        R_opt = self.predict_R(new_invkd)
        #for i in np.fabs(update).argsort()[::-1][:20]:
            
            #if np.fabs(residual[i]) > np.fabs(residual[i] - update[i]):
                #status = 'GOOD'
            #else:
                #status = 'BAD'
                
            #print cyska.index_to_seq(i, 5), status, update[i], self.trial_invkd[i], self.known_invkd[i], self.R_obs[:,i], self.R_new[:,i], R_opt[:,i]

        print "finding optimal scaling of all affinities"
        # find optimal global scale
        def to_optimize(scale):
            return self.global_error(self.predict_R(np.clip(new_invkd * scale, 1e-9, 1e3) ))

        ##t0 = time.time()
        ##res = minimize_scalar(to_optimize, bounds = [1e-3, 1], method='Bounded')
        ##print "scale optimization result", res.success, res.x
        ##self.logger.debug("optimal global scale search success={0} scale={1} took {2:.2f}ms".format(res.success, res.x, time.time() - t0) )
        
        ##new_invkd *= res.x

        #print "update vector elements that should actually matter"
        #for i in np.fabs(self.known_invkd - self.trial_invkd).argsort()[::-1][:20]:
            #print cyska.index_to_seq(i, 5), update[i], self.R_obs[:,i], self.R_old[:,i], self.known_invkd[i], self.trial_invkd[i]
        
        
        self.trial_invkd = new_invkd
        self.R_new = R_opt
        #self.jacobi_new = self.jacobi # self.jacobi got updated implicitly by self.predict_R
        self.kmer_error_new = self.kmer_errors(self.R_new)
        self.kmer_rel_error_new = self.kmer_rel_errors(self.R_new)

    

    def ripple_down(self):
        """
        Apply changes made to affinity proportionally to all lower-affinity kmers that have already been fitted at least once.
        """
        j = self.last_kmer_update
        rel_change = self.trial_invkd[j] / self.prev_invkd[j]
            
        ordering = self.trial_invkd.argsort()[::-1]
        ranks = np.arange(4**self.k)[ordering]
        top_rank = ranks[j] + 1
        for i in ordering[top_rank:]:
            if not self.kmer_update_count[i]:
                continue
            self.trial_invkd[i] *= rel_change
        

    def converged(self):
        # this is a stub. Make depend on np.fabs(self.kmer_error_new - self.kmer_error_old)
        if self.t > 100:
            return True


#class OptReporting(object):
    #def __init__(self, opt, path='./'):
        #self.opt = opt
        #self.path = path
        #from matplotlib.backends.backend_pdf import PdfPages
        #self.sweep_pdf = PdfPages(os.path.join(self.path,'local_fits.pdf') )
        #self.descent_pdf = PdfPages(os.path.join(self.path,'gradient_descent.pdf') )
        #self.R_pdf = PdfPages(os.path.join(self.path,'R_value_fit.pdf') )
        #self.invkd_pdf = PdfPages(os.path.join(self.path,'invkd_fit.pdf') )
        #self.err_pdf = PdfPages(os.path.join(self.path,'err_fit.pdf') )

    #def close(self):
        #self.sweep_pdf.close()
        #self.descent_pdf.close()
        #self.R_pdf.close()
        #self.invkd_pdf.close()
        #self.err_pdf.close()

    #def plot_gradient_descent(self, n_top=100):
        #I = self.opt.R_obs.argsort()[::-1][:n_top]
        
        #x = np.arange(len(I))
        #pp.figure()
        #t = self.opt.t
        #pp.title("gradient-descent at step {0}".format(t))
        
        #plot = pp.semilogy
        #plot(x, self.opt.known_invkd[I], 'k', label='known affinities')
        #plot(x, self.opt.prev_invkd[I], '.b', label='prev. delta')
        #plot(x, self.opt.trial_invkd[I], '.r', label='last delta')
        #pp.xlim(-1, len(x))
        #self.descent_pdf.savefig()
        #pp.savefig(os.path.join(self.path, "descent_t{0}.pdf".format(t)) )

        
    #def plot_jacobi(self):
        
        #for conc, jac in zip(self.opt.rbp_conc, self.opt.jacobi_new):
            #pp.figure()
            #pp.title("Jacobi matrix @{1}nM at t={0}".format(self.opt.t, conc))
            #pp.imshow(np.arcsinh(jac), cmap='hot')
            #pp.colorbar(label='arcsinh(jacobi matrix)')
            #pp.savefig('jacobi_{1}nM_t{0}.pdf'.format(self.opt.t, conc))

        #pp.show()
        #pp.close()
        
    #def plot_sweep(self, kmer_index=None, min_invkd = 1e-12, max_invkd = 1e2, steps=100):
  
        #if kmer_index == None:
            #kmer_index = self.opt.last_kmer_update

        #invkd = np.copy(opt.trial_invkd)
        #x = np.exp(np.linspace(np.log(min_invkd), np.log(max_invkd), steps))
        #errors = []
        #for ikd in x:
            #invkd[kmer_index] = ikd
            #R_trial = self.opt.predict_R(invkd)
            #err = R_trial[:,kmer_index] - self.opt.R_obs[:,kmer_index]
            #errors.append(err)
            
        #errors = np.array(errors).T
        
        #kmer = self.opt.kmers[kmer_index]
        #pp.figure()
        #pp.title("kmer-fit for {0} at step {1}".format(kmer, self.opt.t) )
        
        #for P, err in zip(self.opt.rbp_conc, errors):
            #pp.semilogx(x, err, label="P={0}nM".format(P))

        #pp.axvline(self.opt.known_invkd[kmer_index], color='r', label="correct value")
        #pp.axvline(self.opt.trial_invkd[kmer_index], color='k', label="fitted root")
        #pp.axhline(0, color='k')
        #if opt.kmer_updates[kmer] > 1:
            #pp.axvline(self.opt.prev_invkd[kmer_index], color='gray', label="previous value")
                
        #pp.xlabel(r"$\frac{1}{K_d}$ [nM]")
        #pp.ylabel(r"expected R - observed R")
        #pp.legend(loc='upper left')
        #pp.tight_layout()
        #self.sweep_pdf.savefig()
        #pp.savefig(os.path.join(self.path, "sweep_{0}_t{1}.pdf".format(kmer, self.opt.t)) )
        ##pp.show()
        #pp.close()

    #def plot_R_value_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        ##to_mark_i = [cyska.seq_to_index(x) for x in to_mark]
        
        #kmer_i = self.opt.last_kmer_update
        #kmer = self.opt.kmers[kmer_i]
        
        
        #pp.figure()
        #pp.title('R-value fit after step {0}'.format(self.opt.t) )
        #R_a = self.opt.R_obs
        #R_b = self.opt.R_new
        #for i,rbp_conc in enumerate(self.opt.rbp_conc):
            #corr = np.corrcoef(np.log(R_a[i]), np.log(R_b[i]))[0][1]
            #patches = pp.loglog(R_a[i], R_b[i], 'o', markeredgecolor='none', markersize=5, alpha=.75, label="P={0:.2f}nM (R={1:.3f})".format(rbp_conc, corr) )
            
        #pp.loglog(R_a[:, kmer_i], R_b[:, kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        #m = min(R_a.min(), R_b.min())
        #M = max(R_a.max(), R_b.max())
        #pp.plot([m,M],[m,M], '--k', zorder=np.inf)
        #pp.xlabel(r'{0} [R-value]'.format("observed/simulated") )
        #pp.ylabel(r'{0} [R-value]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
        #pp.legend(loc='upper left')
        #pp.tight_layout()
        #self.R_pdf.savefig()
        #pp.savefig(os.path.join(self.path, "predicted_vs_obs_R_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        #pp.close()
        
    #def plot_invkd_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        ##to_mark_i = [cyska.seq_to_index(x) for x in to_mark]
        
        #kmer_i = self.opt.last_kmer_update
        #kmer = self.opt.kmers[kmer_i]
        
        
        #pp.figure()
        #pp.title('affinity agreement after step {0}'.format(self.opt.t) )
        #A_a = self.opt.known_invkd
        #A_b = self.opt.trial_invkd
        #x = A_a
        #y = A_b
        
        #max_error_conc = np.fabs(self.opt.kmer_error_new).argmax(axis=0)
        #for i,rbp_conc in enumerate(self.opt.rbp_conc):
            #ind = max_error_conc == i
            ##print ind.shape, ind, x[ind]
            #patches = pp.loglog(x[ind], y[ind], 'o', markeredgecolor='none', markersize=5, label="max error at P={0:.2f}nM".format(rbp_conc) )

        #corr = np.corrcoef(np.log(A_a), np.log(A_b))[0][1]
        #pp.loglog(x[kmer_i], self.opt.prev_invkd[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='gray', label="previous values" )
        #pp.loglog(A_a[kmer_i], A_b[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        #m = min(A_a.min(), A_b.min())
        #M = max(A_a.max(), A_b.max())
        #pp.plot([m,M],[m,M], '--k', zorder=np.inf)
        #pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
        #pp.ylabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
        #pp.legend(loc='lower right')
        ##pp.xlim(1e-1,1e2)
        ##pp.ylim(1e-1,1e2)
        #pp.tight_layout()
        #self.invkd_pdf.savefig()
        #pp.savefig(os.path.join(self.path, "predicted_vs_obs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        #pp.close()
        
    #def plot_errors(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        #to_mark_i = [cyska.seq_to_index(x) for x in to_mark]

        #kmer_i = self.opt.last_kmer_update
        #kmer = self.opt.kmers[kmer_i]
        
        #pp.figure()
        #pp.title('residual errors after step {0}'.format(self.opt.t) )

        #abs_err = np.arcsinh(self.opt.kmer_error_new)
        ##abs_err = np.where(abs_err > 0, np.arcsinh(abs_err), -np.arcsinh(-abs_err) )
        
        #for i,rbp_conc in enumerate(self.opt.rbp_conc):
            #patches = pp.semilogx(self.opt.known_invkd, abs_err[i,:], 'o', markeredgecolor='none', markersize=3, alpha=.75, label="P={0}nM".format(rbp_conc) )
            
        #mark_x = self.opt.known_invkd[to_mark_i]
        #mark_y = abs_err[:,to_mark_i].max(axis=0)
        #for mer, index, x, y in zip(to_mark, to_mark_i, mark_x, mark_y):
            #pp.text(x*2, y, mer)

        ##pp.plot(mark_x, mark_y, 'o', markersize=10, markeredgecolor = 'black' , markerfacecolor='none')

        #pp.semilogx(np.repeat(self.opt.known_invkd[kmer_i], len(self.opt.rbp_conc)), abs_err[:, kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        
        #pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
        #pp.ylabel(r'arcsinh(residual error) [a.u.]')
        #pp.legend(loc='upper left')
        ##pp.xlim(1e-1,1e2)
        ##pp.ylim(1e-1,1e2)
        #pp.tight_layout()
        #self.err_pdf.savefig()
        #pp.savefig(os.path.join(self.path, "err_vs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        #pp.close()


def sim_rbp_ordered(k=5, seed=47110815, mode='ordered', protein_conc=[5., 40.]):
    r = reads[0]
    o = storages[0].get_discretized(k)
    print "done loading"
    sim = RBNSSimulator(r, o, k)
    gen = RBNSGenerator(k,l=40, seed=seed, mode=mode)
    #gen.store_invKd('rbns_gen_sorted.txt')

    print "computing single protein partition function model on input reads using simulated energies"
    p_bound_matrix, kmer_count_matrix, openen_kmer_bincount_matrix = sim.expected_kmer_counts(gen.kmer_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True)

    f0 = r.kmer_frequencies(k)
    f0 /= f0.sum()

    freq_matrix = np.array(kmer_count_matrix, dtype=float)
    freq_matrix /= freq_matrix.sum(axis=1)[:, np.newaxis]

    R = freq_matrix / f0[np.newaxis,:]
    
    return gen.kmer_invkd, R, f0

def sim_rbp_singleton(k=5, seed=47110815, protein_conc=[5., 40.]):
    r = reads[0]
    o = storages[0].get_discretized(k)
    import cska.rbns_model
    print "done loading"
    sim = RBNSSimulator(r, o, k)
    aff = cska.rbns_model.AffinityDistribution.singleton()
    print "best binder is", aff.invkd.argmax()
    print "computing single protein partition function model on input reads using simulated energies"
    p_bound_matrix, kmer_count_matrix, openen_kmer_bincount_matrix = sim.expected_kmer_counts(aff.invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True)

    f0 = r.kmer_frequencies(k)
    f0 /= f0.sum()

    freq_matrix = np.array(kmer_count_matrix, dtype=float)
    freq_matrix /= freq_matrix.sum(axis=1)[:, np.newaxis]

    R = freq_matrix / f0[np.newaxis,:]
    
    return aff.invkd, R, f0


#k = int(sys.argv[1])
temp = 22.
RT = (temp + 273.15) * 8.314459848/4.184E3 # RT in kcal/mol

sources = [
    ('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads',0),
    #('/scratch/data/RBNS/RBFOX2/RBFOX2_121.reads',121),
    #('/scratch/data/RBNS/RBFOX2/RBFOX2_365.reads',365),
]

#reads = [RBNSReads(src, rbp_name='RBFOX2', rbp_conc=P, pseudo_count = 10, n_max=1000000) for src,P in sources]
reads = [RBNSReads(src, rbp_name='RBFOX2', rbp_conc=P, pseudo_count = 10) for src,P in sources]
storages = [OpenenStorage(r, '/scratch/data/RBNS/RBFOX2/ska_RBFOX2/openen/', disc_mode='gamma') for r in reads]

#protein_conc = [40., ]

k = 5
#protein_conc = [.01, 40., 160., 3300.]

#sim_invkd, R_obs, f0 = sim_rbp_ordered(k=k, protein_conc=protein_conc, mode='ordered')
#sim_invkd, R_obs, f0 = sim_rbp_singleton(k=k, protein_conc=protein_conc)

print "creating optimizer"
#opt = Optimizer(k, reads[0], storages[0], protein_conc, R_obs, known_invkd=sim_invkd, n_subsample=1000000, sub_replace=False, aff0=1e-6, temp=22, debug=True, min_invkd=1e-12, max_invkd=1e1, n_blocked=0)
#rep = OptReporting(opt, 'opt_plots_rnd')

def load_table(src):
    kmers = []
    values = []
    errors = []
    
    for line in file(src):
        parts = np.array(line.split('\t'))
        if line.startswith('#'):
            continue
        n = len(parts)
        val_i = np.arange(1,n,2)
        err_i = np.arange(2,n,2)
        
        kmers.append(parts[0])
        values.append(parts[val_i])
        errors.append(parts[err_i])
        
    indices = np.array([cyska.seq_to_index(mer) for mer in kmers])
    k = len(kmers[0])
    
    l = len(values[0])
    
    values = np.array(values)
    errors = np.array(errors)

    print values.shape
    print errors.shape
    
    vals = np.zeros( (4**k, l), dtype=np.float32)
    errs = np.zeros( (4**k, l), dtype=np.float32)
    
    
    vals[indices, :] = values[:,:]
    errs[indices, :] = errors[:,:]
    
    return vals.T, errs.T


protein_conc = [40., 121., 365., 1100.]
R_obs, R_err = load_table('/scratch/data/RBNS/RBFOX2/ska_RBFOX2/RBFOX2.R_value.{0}mer.tsv'.format(k))
#protein_conc = [5., 20., 80., 320., 1300.,]
#R_obs, R_err = load_table('/scratch/data/RBNS/HNRNPA0/ska_HNRNPA0/HNRNPA0.R_value.{0}mer.tsv'.format(k))


from cska.rbns_model import ModelOptimization
openen = storages[0].get_discretized(k)
#opt = ModelOptimization(reads[0], openen, k, R_obs, R_err=[], rbp_conc=protein_conc, n_subsample=100000, param_file='params_t276.tsv') # known_params = np.concatenate((sim_invkd,[0,0,0,0]))
if len(sys.argv) > 1:
    param_file = sys.argv[1]
else:
    param_file = None
opt = ModelOptimization(reads[0], openen, k, R_obs, R_err=[], rbp_conc=protein_conc, n_subsample=0, seq_only=False, sub_replace=True,param_file=param_file) # known_params = np.concatenate((sim_invkd,[0,0,0,0]))

#opt.step_beta()

from cska.rbns_reports import OptReporting
#rep = OptReporting(opt, 'plots_HNRNPA0')
#rep = OptReporting(opt, 'plots_RBFOX2_seq_only')
rep = OptReporting(opt, 'plots_RBFOX2')
#rep.plot_R_value_agreement()
#sys.exit(1)


try:
    opt.optimize(reporter = rep)
except KeyboardInterrupt:
    pass

print "converged/interrupted after {0} steps.".format(opt.t)
opt.print_summary()
import matplotlib.pyplot as pp
#pp.figure()
#params = sim_invkd
#pp.loglog(params, params, 'x', label='reference')
#pp.loglog(params, opt.current.params, '.', label='fit')

pp.figure()
pp.semilogy(opt.errors)
pp.ylabel('global optimization error')

pp.show()

sys.exit(0)


print R_obs.shape
f0 = reads[0].kmer_frequencies(k)
f0 /= f0.sum()

print "creating optimizer"
opt = Optimizer(k, reads[0], storages[0], protein_conc, R_obs, n_subsample=100000, sub_replace=False, aff0=1e-6, temp=22, debug=True, min_invkd=1e-12, max_invkd=1e1, n_blocked=0)
rep = OptReporting(opt, 'opt_plots_RBFOX2')

#opt.test_jacobi()
#rep.plot_jacobi()
#sys.exit(0)
#opt.step()
#rep.plot_R_value_agreement()
#rep.plot_invkd_agreement()
#rep.plot_errors()
#rep.plot_gradient_descent()

try:
    while not opt.converged():
        print "optimization step"
        if opt.t % 2:
            opt.step_gradient()
        else:
            opt.step()
            
        #opt.ripple_down()
        print "="*40
        print "rendering plots"
        #rep.plot_sweep()
        
        #rep.plot_jacobi()

        rep.plot_R_value_agreement()
        #rep.plot_invkd_agreement()
        #rep.plot_errors()
        #rep.plot_gradient_descent()
        print "done rendering"
except KeyboardInterrupt:
    pass

rep.close()
sys.exit(0)





