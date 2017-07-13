import rbpbind as rbp
from cska.rbns_reads import RBNSReads
from cska.rbns_model import RBNSSimulator, RBNSGenerator
from cska.folding import RBNSOpenen, OpenenStorage, OpenenDiscretization

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pp

#pp.style.use('ggplot')
import logging
import os, sys, time
import numpy as np
import cska.ska_kmers
import scipy.stats
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)

class Optimizer(object):
    def __init__(self, k, reads, openen_storage, protein_conc, R_obs, known_invkd=[], n_subsample=100000, sub_replace=False, aff0=1e-6, temp=22, debug=True, min_invkd=1e-12, max_invkd=1e1, n_blocked=1):
        self.debug = debug
        self.logger = logging.getLogger("Optimizer")

        self.k = k
        self.kmers = np.array([mer.upper().replace('T','U') for mer in cska.ska_kmers.yield_kmers(k)])
        self.reads = reads
        self.openen_storage = openen_storage
        self.openen = self.openen_storage.get_discretized(self.k)
        self.protein_conc = np.array(protein_conc, dtype=np.float32)
        self.n_conc = len(self.protein_conc)
        
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
        from cska.ska_kmers import eval_energy_model_on_seqs
        import time
        
        # prepare all variables
        if not len(kmer_invkd):
            kmer_invkd = self.trial_invkd

        if not len(protein_conc):
            protein_conc = self.protein_conc
            
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

        if self.debug:
            self.logger.debug("evaluated energy model on {0} sequences in {1:.2f} ms".format(n, 1000*(t1-t0)) )

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
        
    def test_jacobi(self):
        p_bound, pi, openen_kmer_bincount_matrix, jac_pi = self.eval_thermodynamic_model(do_jacobi=True)

        #jac_pi = jac_pi.transpose( (0,2,1) )
        pi_sum = pi.sum(axis=1)
        R = (pi / pi_sum[:, np.newaxis] ) / self.f0

        print "current prediction R-values with invkd (best binder)", self.trial_invkd[590]
        print R[:,590]
        #print "R", R.shape, R
        #print "pi_sum", pi_sum.shape, pi_sum
        #print "jac_pi", jac_pi.shape, jac_pi[:,590,590]
        
        direct = jac_pi / pi_sum[:, np.newaxis, np.newaxis]
        #print "direct slice", direct.shape, direct[:,590,590]
        diag = np.diagonal(jac_pi, axis1=1, axis2=2)
        #print "diagonal", diag.shape, diag
        indirect = - pi[:, np.newaxis, :] / (pi_sum * pi_sum)[:, np.newaxis, np.newaxis] * diag[:,:, np.newaxis]
        #print "indirect slice", indirect.shape, indirect[:,590,590]
        jac_R = direct + indirect

        emp_jac = self.emp_jacobi()
        print "predicted R-value changes when tuning only GCATG"
        x = np.zeros(1024)
        x[590] = 1.
        
        for jac in emp_jac: #jac_R:
            print "========="
            res = np.dot(x, jac)
            
            for i in res.argsort()[::-1][:20]:
                print cska.ska_kmers.index_to_seq(i, 5), res[i]
            
            print "kmers with highest impact on GCATG R-value"
            for i in np.fabs(jac[:,590]).argsort()[::-1][:20]:
                print cska.ska_kmers.index_to_seq(i, 5), jac[:,590][i]
            
        jac_inv = np.linalg.inv(jac)    
        
        print "computed update vector for desired R-value change of +1 for GCATG"
        x = np.zeros(1024)
        x[590] = 1.
        
        for jac in jac_R:
            print "========="
            res = np.dot(jac_inv, x.T)
            
            for i in np.fabs(res).argsort()[::-1][:20]:
                print cska.ska_kmers.index_to_seq(i, 5), res[i]
            



        #emp_jac = self.emp_jacobi(I=np.array([590,]))
        #emp_jac = self.emp_jacobi()

        #print jac_R.min(), jac_R.argmin(), jac_R.max(), jac_R.argmax()
        #print "analytical slice through U-rich stuff for changing UUUUU"
        #print jac_R[:,1023,-20:]
        #print "analytical slice through U-rich stuff for changing UUUUG"
        #print jac_R[:,1022,-20:]

        #print "analytical slice through U-rich stuff for changing AAAAA"
        #print jac_R[:,0,-20:]
        #pp.figure()
        #pp.plot(jac_R[1,590,:], emp_jac[1,590,:],'.')
        #pp.savefig("emp_jac_t{0}.pdf".format(self.t) )
        #pp.close()
        #print "empirical slice"
        #print emp_jac
        
        #print "computing update vector by inverse Jacobi matrix"
        #for jac, R_obs, R_curr, conc in zip(jac_pi, self.R_obs, R, self.protein_conc):
            #jac_inv = np.linalg.inv(jac)
            #update = np.dot(R_obs - R_curr, jac_inv)
            ##print update
            #print "rbp_conc", conc
            
            #print "this should be all zeros"
            #disc = R_obs - (R_curr  +  np.dot(update, jac))
            #print disc
            #print disc.min(), disc.max()
            
            #print "update vector strongest magnitude elements"
            #mag = np.fabs(update)
            #for i in mag.argsort()[::-1][:20]:
                #print cska.ska_kmers.index_to_seq(i, 5), update[i], R_obs[i], R_curr[i], self.known_invkd[i], self.trial_invkd[i]

            #print "update vector elements that should actually matter"
            #for i in np.fabs(self.known_invkd - self.trial_invkd).argsort()[::-1][:20]:
                #print cska.ska_kmers.index_to_seq(i, 5), update[i], R_obs[i], R_curr[i], self.known_invkd[i], self.trial_invkd[i]
            
            

            #update = update / mag.max() * .1
            
            
            #pp.figure()
            #from scipy.stats import spearmanr
            #rho = spearmanr(self.known_invkd - self.trial_invkd, update)
            #pp.plot(np.arcsinh(self.known_invkd - self.trial_invkd), np.arcsinh(update), 'o', label="spearmanr = {0}".format(rho))
            #pp.savefig('update_{0}nM_t{1}.pdf'.format(conc, self.t))
            #pp.close()
        ##pp.show()
        self.jacobi_new = jac_R

    def find_worst_kmer(self):
        
        """
        ideas: 
            * take into account if over-under-representation is systematic or only seen at some concentrations (contradicted at others?)
            * start preferring kmers whose error peaks at low concentrations (high affinity), then move to kmers whose error peaks at intermediate or high concentrations (lower affinity)
            * perhaps overall R-value correlation can serve as guide? (> .9 do a round of low affinity optimizations in fixed energy background?)
        """
        badness = np.fabs(self.kmer_error_new.mean(axis=0)) #+ np.fabs(self.kmer_rel_error_new).mean(axis=0)
        score  = np.sqrt(badness) / (self.heat + 1)
        
        if self.debug:
            print "!!!!! kmers with max residual error"
            for i in score.argsort()[::-1][:10]:
                #grad = self.gradient(I=[i,])
                print cska.ska_kmers.index_to_seq(i, self.k), "score=",score[i], "bad=",badness[i], "heat=",self.heat[i], self.known_invkd[i], self.trial_invkd[i], "residual", self.kmer_error_new[:,i]#, "grad", grad
                #for j, w in self.cm.get_shadow(i):
                    #s_mer = cska.ska_kmers.index_to_seq(j, self.k)
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
                
            print cska.ska_kmers.index_to_seq(i, self.k), self.known_invkd[i], self.trial_invkd[i], "abs. error", self.kmer_error_new[:,i]
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
            protein_conc = np.array([self.protein_conc[conc_i],], dtype=np.float32)
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
        theta = 1./ (1. + 1./ (self.protein_conc[:,np.newaxis] / self.trial_invkd[np.newaxis,:]) )
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
        if self.t > 50:
            return True


class OptReporting(object):
    def __init__(self, opt, path='./'):
        self.opt = opt
        self.path = path
        from matplotlib.backends.backend_pdf import PdfPages
        self.sweep_pdf = PdfPages(os.path.join(self.path,'local_fits.pdf') )
        self.R_pdf = PdfPages(os.path.join(self.path,'R_value_fit.pdf') )
        self.invkd_pdf = PdfPages(os.path.join(self.path,'invkd_fit.pdf') )
        self.err_pdf = PdfPages(os.path.join(self.path,'err_fit.pdf') )

    def close(self):
        self.sweep_pdf.close()
        self.R_pdf.close()
        self.invkd_pdf.close()
        self.err_pdf.close()
        
    def plot_jacobi(self):
        
        for conc, jac in zip(self.opt.protein_conc, self.opt.jacobi_new):
            pp.figure()
            pp.title("Jacobi matrix @{1}nM at t={0}".format(self.opt.t, conc))
            pp.imshow(np.arcsinh(jac), cmap='hot')
            pp.colorbar(label='arcsinh(jacobi matrix)')
            pp.savefig('jacobi_{1}nM_t{0}.pdf'.format(self.opt.t, conc))

        pp.show()
        pp.close()
        
    def plot_sweep(self, kmer_index=None, min_invkd = 1e-12, max_invkd = 1e2, steps=100):
  
        if kmer_index == None:
            kmer_index = self.opt.last_kmer_update

        invkd = np.copy(opt.trial_invkd)
        x = np.exp(np.linspace(np.log(min_invkd), np.log(max_invkd), steps))
        errors = []
        for ikd in x:
            invkd[kmer_index] = ikd
            R_trial = self.opt.predict_R(invkd)
            err = R_trial[:,kmer_index] - self.opt.R_obs[:,kmer_index]
            errors.append(err)
            
        errors = np.array(errors).T
        
        kmer = self.opt.kmers[kmer_index]
        pp.figure()
        pp.title("kmer-fit for {0} at step {1}".format(kmer, self.opt.t) )
        
        for P, err in zip(self.opt.protein_conc, errors):
            pp.semilogx(x, err, label="P={0}nM".format(P))

        pp.axvline(self.opt.known_invkd[kmer_index], color='r', label="correct value")
        pp.axvline(self.opt.trial_invkd[kmer_index], color='k', label="fitted root")
        pp.axhline(0, color='k')
        if opt.kmer_updates[kmer] > 1:
            pp.axvline(self.opt.prev_invkd[kmer_index], color='gray', label="previous value")
                
        pp.xlabel(r"$\frac{1}{K_d}$ [nM]")
        pp.ylabel(r"expected R - observed R")
        pp.legend(loc='upper left')
        pp.tight_layout()
        self.sweep_pdf.savefig()
        pp.savefig(os.path.join(self.path, "sweep_{0}_t{1}.pdf".format(kmer, self.opt.t)) )
        #pp.show()
        pp.close()

    def plot_R_value_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]
        
        kmer_i = self.opt.last_kmer_update
        kmer = self.opt.kmers[kmer_i]
        
        
        pp.figure()
        pp.title('R-value fit after step {0}'.format(self.opt.t) )
        R_a = self.opt.R_obs
        R_b = self.opt.R_new
        for i,rbp_conc in enumerate(self.opt.protein_conc):
            corr = np.corrcoef(np.log(R_a[i]), np.log(R_b[i]))[0][1]
            patches = pp.loglog(R_a[i], R_b[i], 'o', markeredgecolor='none', markersize=5, alpha=.75, label="P={0:.2f}nM (R={1:.3f})".format(rbp_conc, corr) )
            
        pp.loglog(R_a[:, kmer_i], R_b[:, kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        m = min(R_a.min(), R_b.min())
        M = max(R_a.max(), R_b.max())
        pp.plot([m,M],[m,M], '--k', zorder=np.inf)
        pp.xlabel(r'{0} [R-value]'.format("observed/simulated") )
        pp.ylabel(r'{0} [R-value]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
        pp.legend(loc='upper left')
        pp.tight_layout()
        self.R_pdf.savefig()
        pp.savefig(os.path.join(self.path, "predicted_vs_obs_R_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        pp.close()
        
    def plot_invkd_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]
        
        kmer_i = self.opt.last_kmer_update
        kmer = self.opt.kmers[kmer_i]
        
        
        pp.figure()
        pp.title('affinity agreement after step {0}'.format(self.opt.t) )
        A_a = self.opt.known_invkd
        A_b = self.opt.trial_invkd
        x = A_a
        y = A_b
        
        max_error_conc = np.fabs(self.opt.kmer_error_new).argmax(axis=0)
        for i,rbp_conc in enumerate(self.opt.protein_conc):
            ind = max_error_conc == i
            #print ind.shape, ind, x[ind]
            patches = pp.loglog(x[ind], y[ind], 'o', markeredgecolor='none', markersize=5, label="max error at P={0:.2f}nM".format(rbp_conc) )

        corr = np.corrcoef(np.log(A_a), np.log(A_b))[0][1]
        pp.loglog(x[kmer_i], self.opt.prev_invkd[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='gray', label="previous values" )
        pp.loglog(A_a[kmer_i], A_b[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        m = min(A_a.min(), A_b.min())
        M = max(A_a.max(), A_b.max())
        pp.plot([m,M],[m,M], '--k', zorder=np.inf)
        pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
        pp.ylabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
        pp.legend(loc='lower right')
        #pp.xlim(1e-1,1e2)
        #pp.ylim(1e-1,1e2)
        pp.tight_layout()
        self.invkd_pdf.savefig()
        pp.savefig(os.path.join(self.path, "predicted_vs_obs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        pp.close()
        
    def plot_errors(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]

        kmer_i = self.opt.last_kmer_update
        kmer = self.opt.kmers[kmer_i]
        
        pp.figure()
        pp.title('residual errors after step {0}'.format(self.opt.t) )

        abs_err = np.arcsinh(self.opt.kmer_error_new)
        #abs_err = np.where(abs_err > 0, np.arcsinh(abs_err), -np.arcsinh(-abs_err) )
        
        for i,rbp_conc in enumerate(self.opt.protein_conc):
            patches = pp.semilogx(self.opt.known_invkd, abs_err[i,:], 'o', markeredgecolor='none', markersize=3, alpha=.75, label="P={0}nM".format(rbp_conc) )
            
        mark_x = self.opt.known_invkd[to_mark_i]
        mark_y = abs_err[:,to_mark_i].max(axis=0)
        for mer, index, x, y in zip(to_mark, to_mark_i, mark_x, mark_y):
            pp.text(x*2, y, mer)

        #pp.plot(mark_x, mark_y, 'o', markersize=10, markeredgecolor = 'black' , markerfacecolor='none')

        pp.semilogx(np.repeat(self.opt.known_invkd[kmer_i], len(self.opt.protein_conc)), abs_err[:, kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        
        pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
        pp.ylabel(r'arcsinh(residual error) [a.u.]')
        pp.legend(loc='upper left')
        #pp.xlim(1e-1,1e2)
        #pp.ylim(1e-1,1e2)
        pp.tight_layout()
        self.err_pdf.savefig()
        pp.savefig(os.path.join(self.path, "err_vs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        pp.close()


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

reads = [RBNSReads(src, rbp_name='RBFOX2', rbp_conc=P, pseudo_count = 10, n_max=2000000) for src,P in sources]
storages = [OpenenStorage(r, '/scratch/data/RBNS/RBFOX2/ska_RBFOX2/openen/', disc_mode='gamma') for r in reads]

#protein_conc = [40., ]

k = 5
protein_conc = [.01, 40., 160., 3300.]

#sim_invkd, R_obs, f0 = sim_rbp_ordered(k=k, protein_conc=protein_conc, mode='ordered')
sim_invkd, R_obs, f0 = sim_rbp_singleton(k=k, protein_conc=protein_conc)

print "creating optimizer"
opt = Optimizer(k, reads[0], storages[0], protein_conc, R_obs, known_invkd=sim_invkd, n_subsample=10000, sub_replace=False, aff0=1e-6, temp=22, debug=True, min_invkd=1e-12, max_invkd=1e1, n_blocked=1)
rep = OptReporting(opt, 'opt_plots_rnd')
opt.test_jacobi()
#rep.plot_jacobi()
sys.exit(0)
try:
    while not opt.converged():
        print "optimization step"
        opt.step()
        #opt.ripple_down()
        print "="*40
        print "rendering plots"
        opt.test_jacobi()
        #rep.plot_sweep()
        #jac = opt.jacobi_new[1]
        #print "JAC"
        #print jac.min(), jac.max(), jac.argmax(), jac.argmin()

        #print "connected to AAAAA"
        #for i in jac[0,:].argsort()[::-1][:10]:
            #print cska.ska_kmers.index_to_seq(i, 5), i, jac[0,i]

        #print "connected to UUUUG"
        #for i in jac[1022,:].argsort()[::-1][:10]:
            #print cska.ska_kmers.index_to_seq(i, 5), i, jac[1022,i]
        
        #print "connected to UUUUU"
        #for i in jac[1023,:].argsort()[::-1][:10]:
            #print cska.ska_kmers.index_to_seq(i, 5), i, jac[1023,i]
        
        #rep.plot_jacobi()

        rep.plot_R_value_agreement()
        rep.plot_invkd_agreement()
        rep.plot_errors()
        print "done rendering"
except KeyboardInterrupt:
    pass

rep.close()
sys.exit(0)










#R_order = R.min(axis=0).argsort()[::-1]


#def convergence_analysis_scatter(n_samples = [1000, 10000, 100000, 1000000]):
    #print "convergence analysis"
    #pp.figure(figsize=(6,4))
    #pp.title("subsampling vs. accuracy")
    #m = kmer_count_matrix[1].min()
    #M = kmer_count_matrix[1].max()
    #for n_sample in n_samples:
        ## randomly select n_max reads
        ##n = np.random.permutation(np.arange(sim.reads.N))[:n_sample]
        #n = np.random.randint(sim.reads.N, size=n_sample)
        #pb, sample_kmer_counts, sample_openen_bincounts = sim.expected_kmer_counts(gen.kmer_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, indices=n, n_max=n_sample)

        #scale = r.N / float(n_sample)
        #pp.loglog(kmer_count_matrix[1], sample_kmer_counts[1] * scale, 'o', label=str(n_sample))
        #m = min(m, sample_kmer_counts[1].min())
        #M = max(M, sample_kmer_counts[1].max())
        
    #print m, M
    #m += 1000
    #pp.plot([m, M], [m, M], 'k--', label=None)
    #pp.legend(loc='lower right')
    #pp.xlabel("kmer-freq predicted on subsample")
    #pp.ylabel("kmer-freq predicted on all reads".format(r.N))
    #pp.tight_layout()
    #pp.savefig('convergence_scatter.pdf')

#def convergence_analysis_CV(n_samples = [1000, 10000, 100000, 1000000], n_rep=10):
    #print "CV analysis"
    #from scipy.stats import variation
    #pp.figure(figsize=(6,4))
    #pp.title("CV of pulldown kmer-abundance estimates")
    #m = kmer_count_matrix[1].min()
    #M = kmer_count_matrix[1].max()
    #for n_sample in n_samples:
        #scale = r.N / float(n_sample)
        #scale = 1
        #reps = []
        #for i in range(n_rep):
            ## randomly select n_max reads
            ##n = np.random.permutation(np.arange(sim.reads.N))[:n_sample]
            #n = np.random.randint(sim.reads.N, size=n_sample)
            #pb, sample_kmer_counts, sample_openen_bincounts = sim.expected_kmer_counts(gen.kmer_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, indices=n, n_max=n_sample)
            #reps.append(sample_kmer_counts[1] * scale)
        
        #reps = np.array(reps)
        
        #CV = variation(reps,axis=0)
        #print "CV of highest affinity kmer", n_sample, CV[-1], reps[:,1023]
        #y, x, patches = pp.hist(CV, bins=100, label=str(n_sample), histtype='stepfilled')
        ##print x
    
    ##m += 1
    ##pp.plot([m, M], [m, M], 'k--', label=None)
    #pp.gca().set_xscale('log')
    #pp.legend(loc='upper right')
    #pp.xlabel("Coefficient of Variation")
    #pp.ylabel("frequency")
    #pp.tight_layout()
    #pp.savefig('convergence_CV.pdf')
    #pp.show()
    
##convergence_analysis_scatter()
##convergence_analysis_CV()


#def kmer_error(expected, observed, i):
    #"""This is signed and should have a root"""
    #err = np.mean(expected[:,i] - observed[:,i])
    ##err = expected[:,i] - observed[:,i]
    ##err = np.mean((np.log(expected) - np.log(observed))**2)
    ##err = np.median( np.log(expected[:,i]) - np.log(observed[:,i]))
    #return err

#def model_error(expected, observed):
    #"""This is unsigned and should have (at least) one minimum"""
    ##err = np.mean((np.log(expected) - np.log(observed))**2)
    ##err = np.mean( np.log2( (expected + 1e6) / (observed + 1e6) )**2)
    #err = np.mean( ( (observed - expected) **2) )
    #return err

    
#def select_reads_with_kmer(kmer, n_max=0):
    #presence = r.kmer_presence(kmer)
    #subset = r.seqm[presence > 0]
    
    #if not n_max:
        #n_max = len(subset)

    #return subset[:n_max], float(n_max)/r.N

#def predict_kmer_counts(trial_invkd, seqm = None):
    #print "predicting kmer counts on all reads"
    #pb_trial, kmer_count_trial, openen_kmer_bincount_trial = sim.expected_kmer_counts(trial_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, seqm=seqm)
    #print "done"
    #return kmer_count_trial

#def optimize(trial_invkd, i, local=True, min_invkd = 1e-12, max_invkd = 1e2, n_max=100000):
    
    ## operate on a local copy!
    #trial_invkd = np.copy(trial_invkd)
    #scale = sim.reads.N/ n_max

    #conc_i = 0
    
    #def predict(invkd, conc_i=None):
        
        #trial_invkd[i] = invkd
        #if conc_i == None:
            #pb_trial, kmer_count_trial, openen_kmer_bincount_trial = sim.expected_kmer_counts(trial_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, n_max=n_max)
            #R_trial = (kmer_count_trial / kmer_count_trial.sum(axis=1)[:, np.newaxis] ) / f0
            ##R_trial = (kmer_count_trial / kmer_count_trial.sum()) / f0
            #return R_trial, kmer_error(R_trial, R_obs, i)
        #else:
            ## select only one protein concentration to simulate!
            #pb_trial, kmer_count_trial, openen_kmer_bincount_trial = sim.expected_kmer_counts(trial_invkd, [protein_conc[conc_i]], _do_not_unpickle=True, _do_not_pickle=True, n_max=n_max)
            ##print "shape", kmer_count_trial.shape
            #kmer_count_trial = kmer_count_trial[0,:]
            ##R_trial = (kmer_count_trial / kmer_count_trial.sum(axis=1)[:, np.newaxis] ) / f0
            #R_trial = (kmer_count_trial / kmer_count_trial.sum()) / f0
            #return R_trial, R_trial[i] - R_obs[conc_i, i]
    
    #def to_optimize_local(invkd):
        ##invkd = args[0]
        
        #R_trial, err = predict(invkd, conc_i)
        ##err = kmer_error(R_trial, R_obs, i)
        ##print invkd, gen.kmer_energies[i], err
        ##print "invkd", invkd,  "err=", err # "expected counts", trial_counts[:,i],
        #return err

    #def to_optimize_global(invkd):
        ##invkd = np.log(args)
        
        #R_trial, err = predict(invkd)
        #err = model_error(R_trial, R_obs)
        ##print invkd, gen.kmer_energies[i], err
        #print "invkd", invkd,  "err=", err # "expected counts", trial_counts[:,i],
        #return err
    
    #from scipy.optimize import minimize, brentq, minimize_scalar
    
    #old_R = predict(trial_invkd[i])[0]
    #if local:
        #to_optimize = to_optimize_local
        
        ## find the lowest protein concentration with a root
        #found = False
        #for conc_i in range(len(protein_conc)):
            #left_R, left_err = predict(min_invkd, conc_i)
            #right_R, right_err = predict(max_invkd, conc_i)
            #print i, conc_i, min_invkd, max_invkd, left_R, right_R, left_err, right_err
            #if (left_err > 0) != (right_err > 0):
                #found = True
                #break
        
        
        #if not found:
            #print "no sign change! can't find root. reporting global minimum instead"
            ##plot_sweep_debug(i)
            #res = minimize_scalar(to_optimize_local, bounds=(min_invkd, max_invkd), method='bounded' )
            ##res = minimize_scalar(to_optimize_global, bracket=(min_invkd, max_invkd), method='brent' )
            #print res
            #return optimize(trial_invkd, i, local=False, n_max=n_max)
            ##invkd_opt = trial_invkd[i]
            ##return False, invkd_opt, to_optimize(invkd_opt)
        
        ##print "bounds", left, right
        
        #invkd_opt, res = brentq(to_optimize, min_invkd, max_invkd, full_output=True)
        ##print invkd_opt, res
        #err = to_optimize(invkd_opt)
        ##print "error", err
        #return res.converged, invkd_opt, err, predict(invkd_opt)[0], old_R
    #else:
        #to_optimize = to_optimize_global
        ##res = minimize(to_optimize_global, (1.), bounds=[(np.exp(-30.),np.exp(30.))], method='TNC')
        #res = minimize_scalar(to_optimize_global, bounds=(min_invkd, max_invkd), method='bounded' )
        ##res = minimize_scalar(to_optimize_global, bracket=(-30.,30. ), method='brent' )
        #print res
        #return res.success, res.x, res.fun, predict(res.x), old_R


#def gradient(I, trial_invkd, delta=0.0001, n_max=1000):
    #E0 = np.copy(trial_invkd)
    #grad = []
    
    ## randomly select n_max reads
    #n = np.random.permutation(np.arange(sim.reads.N))[:n_max]
    
    #scale = sim.reads.N / float(n_max)
    
    #for i in I:
        
        #pb_trial, kmer_count0, openen_kmer_bincount_trial = sim.expected_kmer_counts(E0, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, indices=n)
        #err0 = model_error(scale*kmer_count0, kmer_count_matrix)
        
        #E0[i] += delta
        #pb_trial, kmer_count1, openen_kmer_bincount_trial = sim.expected_kmer_counts(E0, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, indices=n)
        #err1 = model_error(scale*kmer_count1, kmer_count_matrix)
        #grad.append( (err1 - err0)/delta)
        #print i, grad[-1]
        
    #return np.array(grad)

#def local_optima(I, trial_invkd, n_max=1000):
    #opt = np.copy(trial_invkd)
    #for i in I:
        #plot_sweep_debug(i)
        #kmer = cska.ska_kmers.index_to_seq(i, k)
        #success, ikd_opt, err, new_expect = optimize(trial_invkd, i, local=True, n_max=n_max)
        #print "{3} real invkd={0:.2e}, current_invkd={1:.2e}, fitted_invkd={2:.2e}".format(corr_invkd[i], trial_invkd[i], ikd_opt, kmer)
        #if success:
            #print "accept"
            #opt[i] = ikd_opt

    #return opt
    
    
#def plot_simulation(protein_conc, kmer_invkd, R, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU'], mdl='simulation'):
    #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]

    #pp.figure(figsize=(6,4))
    #for rbp_conc, rv in zip(protein_conc[:], R[:]):
        #x = 1./kmer_invkd
        #y = rv
        #patches = pp.loglog(x, y, '.', markeredgecolor='none', markersize=5, alpha=.75, label="P={0} nM".format(rbp_conc))[0]
        #c = patches.get_color()
        #mark_x = x[to_mark_i]
        #mark_y = y[to_mark_i]
        
        #pp.loglog(mark_x, mark_y, 'o', markeredgecolor = c, markerfacecolor='none')
        
        #if rbp_conc == protein_conc[0]:
            #for kmer, index, x, y in zip(to_mark, to_mark_i, mark_x, mark_y):
                #pp.text(x*2, y, kmer)
            
    #pp.xlabel(r"simulated 5-mer $K_d$ [nM]")
    #pp.ylabel(r"predicted 5-mer R-value")
    #pp.legend(loc='lower left')
    #pp.tight_layout()
    #pp.savefig("{0}.pdf".format(mdl))
    #pp.close()
    
#def plot_model_R_agreement(protein_conc, R_a, R_b, mdl_a="thermodynamic simulation", mdl_b="linear independent k-mer model", to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
    #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]
    #pp.figure()
    
    #for i,rbp_conc in enumerate(protein_conc):
        #corr = np.corrcoef(np.log(R_a[i]), np.log(R_b[i]))[0][1]
        ##corr = np.corrcoef(R_a[i], R_b[i])[0][1]
        #patches = pp.loglog(R_a[i], R_b[i], 'o', markeredgecolor='none', markersize=5, alpha=.75, label="P={0}nM (R={1:.2f})".format(rbp_conc, corr) )
        #c = patches[0].get_color()
        #mark_x = R_a[i][to_mark_i]
        #mark_y = R_b[i][to_mark_i]

        #if rbp_conc == protein_conc[0]:
            #for kmer, index, x, y in zip(to_mark, to_mark_i, mark_x, mark_y):
                #pp.text(x*1.2, y, kmer)

    #m = min(R_a.min(), R_b.min())
    #M = max(R_a.max(), R_b.max())
    #pp.plot([m,M],[m,M], '--k', zorder=np.inf)
    #pp.xlabel(r'{0} [R-value]'.format(mdl_a) )
    #pp.ylabel(r'{0} [R-value]'.format(mdl_b) )
    #pp.legend(loc='upper left')
    #pp.tight_layout()
    #pp.savefig("model_{0}_vs_{1}_R.pdf".format(mdl_a, mdl_b).replace(' ','_').replace('(','').replace(')','') )
    #pp.close()

#kmers = np.array(list(cska.ska_kmers.yield_kmers(5)))
#corr_invkd = gen.kmer_invkd
#trial_invkd = np.ones(corr_invkd.shape, dtype=np.float32)*1e-6

#plot_model_R_agreement(protein_conc, R_obs, np.ones(R_obs.shape), mdl_a="using correct affinities", mdl_b="using uniform affinities")

#print "kmers with actual high affinity"
#I_real = gen.kmer_energies.argsort()[:20]

#for i in I_real:
    #print cska.ska_kmers.index_to_seq(i, k), corr_invkd[i], gen.kmer_energies[i], R[:,i], freq_matrix[:,i] * 4**k

#I = R_order[:50]



#i = I[0]
#n_max=100000
#kmer = cska.ska_kmers.index_to_seq(i, k)

#success, ikd_opt, err, new_R, old_R = optimize(trial_invkd, i, local=True, n_max=n_max)

#print "output of optimize", success, err, new_R.shape, old_R.shape, R_obs.shape

#trial_invkd[i] = ikd_opt

#print "most enriched kmer is {0} with optimal inv_kd {1} ({2})".format( kmer, ikd_opt, corr_invkd[i] )
#print success, err, new_R[:,i], old_R[:,i], R_obs[:,i]
#plot_sweep_debug(i,0)
#plot_model_R_agreement(protein_conc, R_obs, new_R, mdl_a="using correct affinities", mdl_b="using trial affinities (updated {0} in step 0)".format(kmer) )

##plot_R_value_agreement(new_R)

#blocked = [i,]
#for t in range(1,100):
    #mean_residual = np.mean(R_obs - new_R, axis=0)

    #print "kmers with max residual error after correcting for", kmer
    #for i in mean_residual.argsort()[::-1][:10]:
        #print cska.ska_kmers.index_to_seq(i, k), corr_invkd[i], trial_invkd[i], "residual", mean_residual[i], new_R[:,i], R_obs[:,i]
        
    ##print "real affinity kmers"
    ##for i in I_real:
        ##print cska.ska_kmers.index_to_seq(i, k), corr_invkd[i], "residual", mean_residual[i], new_R[:,i], R_obs[:,i]

    ## find next, different kmer to optimize
    #for i in mean_residual.argsort()[::-1]:
        #if not i in blocked:
            #break

    #if len(blocked) > 1:
        #blocked.pop(0)

    #blocked.append(i)
    
    #kmer = cska.ska_kmers.index_to_seq(i,k)
    #print ">> next kmer to update", kmer
    #plot_sweep_debug(i,t, previous_value = trial_invkd[i])

    #success, ikd_opt, err, new_R, old_R = optimize(trial_invkd, i, local=True, n_max=n_max)

    #print ">> optimal affinity", ikd_opt, "(correct affinity {0})".format(corr_invkd[i])
    #trial_invkd[i] = ikd_opt
    #plot_model_R_agreement(protein_conc, R_obs, new_R, mdl_a="using correct affinities", mdl_b="using trial affinities (updated {0} in step {1})".format(kmer, t) )

#pp.figure()
#pp.plot(corr_invkd, trial_invkd, 'o')
#pp.xlabel(r"correct $\frac{1}{K_d}$")
#pp.ylabel(r"fitted $\frac{1}{K_d}$")
#pp.savefig('opt_result.pdf')
#sys.exit(0)



#weights = R[:,I].mean(axis=0)
#weights /= (weights.max())




#print "computing local optima"
#trial_invkd = local_optima(I, trial_invkd, seqm=None)
#trial_invkd[I] *= weights

        
#new_expect = predict_kmer_counts(trial_invkd)
#mdl_err =  model_error(new_expect, kmer_count_matrix)

#for t in range(200):
    #print "computing gradient around current guess", t
    #grad = np.zeros(trial_invkd.shape)
    #grad[I] = gradient(I, trial_invkd, seqm=None, delta=.0001)
    #for i in I:
        #kmer = cska.ska_kmers.index_to_seq(i, k)
        #print "{3} real invkd={0:.2e}, current_invkd={1:.2e}, gradient={2:.2e}".format(corr_invkd[i], trial_invkd[i], grad[i], kmer)
    
    #plot_congruence(gen.kmer_energies, trial_invkd, trial_invkd - grad)
    #trial_invkd -= grad

#sys.exit(0)



#for t in range(100):
    #dG_updates = []
    #trial_invkd_before = np.copy(trial_invkd)
    #expect_before = np.copy(new_expect)

    #print "cycle {0}, optimizing kmers {1} with weights {2}".format(t, I, weights)

    #new_weights = []
    ## select top-enriched kmers
    #for j,(i, w) in enumerate(zip(I, weights)):
        #kmer = cska.ska_kmers.index_to_seq(i, k)
        ##seqm, f = select_reads_with_kmer(kmer, n_max=1000)
        #seqm = None
        ##print "selected kmer '{0}' with R-values '{1}'".format(kmer, R[:,i])
        
        ##plot_sweep_debug(i)

        #success, dG_opt, err, new_expect = optimize(trial_invkd, i, local=True, seqm=seqm )
        #new_err =  model_error(new_expect, kmer_count_matrix)
        #rel_change = (new_err - mdl_err)/mdl_err

        #if rel_change > 0:
            #print "predicted increase in error! flipping sign"
            #dG_opt *= -1
            ##new_weights.append(1.)
        ##else:
        #new_weights.append( np.fabs(rel_change))

        #print "{4} (w={5}) real energy={0:.2f}, current_energy={1:.2f}, fitted energy={2:.2f}, reduction in model_error={3:.2e} %%".format(gen.kmer_energies[i], trial_invkd[i], dG_opt, 100*rel_change, kmer, w)
        #dG_updates.append(dG_opt)
        ##pp.figure()
        ##pp.plot(kmer_count_matrix[:,i], new_expect[:,i], 'o')
        ##pp.show()
        
        ##trial_invkd[i] = dG_opt
        ##plot_congruence(kmer_count_matrix, new_expect)
        ##trial_invkd[i] = 0

    
    #dG_updates = np.array(dG_updates)
    #trial_invkd[I] = trial_invkd[I] * (1-weights) + dG_updates * weights
    #new_expect = predict_kmer_counts(trial_invkd)
    #mdl_err =  model_error(new_expect, kmer_count_matrix)
    
    ##plot_congruence(kmer_count_matrix, new_expect_before, new_expect, "kmer counts")
    #plot_congruence(gen.kmer_energies, trial_invkd_before, trial_invkd, "kmer energies, t={0}".format(t))

    #weights = np.array(new_weights)
    #weights /= (2*weights.max())

#sys.exit(0)

        




#def most_divergent_kmer_index(expected, observed, blocked = set()):

    #delta = np.mean( R  * np.log2((observed + 1e-9) / (expected + 1e-9))**2, axis=0)
    ##delta = np.mean((expected - observed), axis=0)

    #candidates = []

    #for worst in delta.argsort()[::-1]:
    ##for worst in R_order:
        #if R[:,worst].min() < 1.5:
            ## skip non-enriched kmers
            #continue

        #if not worst in blocked:
            #print "{0} d={1} R={7} trial_dG={5} real_dG={6} expect={2} observed={3} blocked={4}".format(kmers[worst], delta[worst], expected[:,worst], observed[:,worst], worst in blocked, trial_invkd[worst], gen.kmer_energies[worst], R[:,worst])
            #candidates.append(worst)
        #else:
            #print "BLOCKED", kmers[worst]
            
        #if len(candidates) >= 10:
            #break

    #return candidates

    ##for worst in delta.argsort()[::-1]:
        ##if not worst in blocked:
            ##print "selected", worst, kmers[worst], delta[worst], "real energy", gen.kmer_energies[worst]
            ##return worst
        ##worst = delta.argmax()
    ##
    ##return worst


    
#blocked = []
#failed = []

#expected_counts = predict_kmer_counts(trial_invkd)
#mdl_err = model_error(expected_counts, kmer_count_matrix)


#for r in range(100):
    #print "optimization step", r
    #if r:
        #plot_congruence()
        
    #if len(blocked) > 10:
        #blocked.pop(0)

    #if len(failed) > 10:
        #failed = []
        ## we ran into a lot of failure to optimize. Let's tweak the parameters we have modified so far.
        #print "RE-OPTIMIZATION RUN"
        #candidates = np.random.permutation((trial_invkd != 0).nonzero()[0])
        #local = False
    #else:
        #candidates = most_divergent_kmer_index(expected_counts, kmer_count_matrix, set(blocked) | set(failed))
        #local = True

    #if failed:
        #failed.pop(0)

    #for i in candidates:
        #print "selected", i, kmers[i], R[:,i], trial_invkd[i], "actual energy", gen.kmer_energies[i]
        ##print "kmer counts", kmer_count_matrix[:,i]

        #if i == 375:
            #plot_sweep_debug(i)

        #dG0 = trial_invkd[i]
        #success, dG_opt, err = optimize(trial_invkd, i, local=local)
        
        #if not success:
                #print "did not converge! trying global optimization"
            ###plot_sweep_debug(i)
            ##success, dG_opt, err = optimize(trial_invkd, i, local=False)
            ##if not success:
                #print "also failed. skipping"
                #trial_invkd[i] = dG0/2.
                #failed.append(i)
                #continue

        #trial_invkd[i] = dG_opt
        #new_expect = predict_kmer_counts(trial_invkd)
        #new_err =  model_error(new_expect, kmer_count_matrix)
        
        #rel_change = (new_err - mdl_err)/mdl_err
        
        #if new_err > (1.01) * mdl_err:
            #print "??>> dG0", dG0, "dG_opt", dG_opt, "(real one =", gen.kmer_energies[i],")"
            #print "??>> model error woukd chang by {0:.2f}%% residual error={1}".format( 100.* rel_change, new_err)
            #print "kmer optimization did not decrease model error! skipping"
            ##plot_sweep_debug(i)
            #trial_invkd[i] = dG0
            #failed.append(i)
            ##break
            #continue
        
        ## keep kmer optimization changes
        #blocked.append(i)
        #expected_counts = new_expect
        #mdl_err = new_err
        #print "!!>> dG0", dG0, "dG_opt", dG_opt, "(real one =", gen.kmer_energies[i],")"
        ##print ">> reference counts", kmer_count_matrix[:,i]
        ##print ">> expected counts", expected_counts[:,i]
        #print "!!>> model error changed by {0:.2f}%% residual error={1}".format( 100.* rel_change, new_err)
        

##print "kmer counts", kmer_count_trial[:,i]


