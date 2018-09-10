import logging, os, sys
import numpy as np
import time
import cska.gradient
import cska.cyska as cyska
from cska.pwm import PSAM
from cska.sc import SelfConsistency
from cska import vector_stats

class PartFuncModelState(object):
    def __init__(self, mdl, params, beta_fixed=True, rbp_free = None, **kwargs):
        t0 = time.time()
        self.mdl = mdl
        self.params = params.copy()
        self.rbp_conc = mdl.rbp_conc
        self.threshold = mdl.Z_thresh # TODO: cleanup! experimental
        # self.threshold = 1e-6 # TODO: cleanup! experimental
    
        # self.A, self.I = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=mdl.aff0)
        t1 = time.time()
        # print "number of relevant kmers", len(self.I)
        # assert (sorted(self.I) == (self.A > mdl.aff0).nonzero()[0] ).all()
        # TODO: make protein concentration self-consistent

        # evaluate single protein partition function of PSAM over reads
        self.Z1 = cyska.PSAM_partition_function(
            self.mdl._seqm, 
            self.mdl._acc, 
            np.array(params.psam_matrix, dtype=np.float32),# * params.A0,
            openen_ofs=self.mdl.openen.ofs - self.mdl.k_mdl + 1 + self.mdl.acc_ofs
        )
        self.Z1_read, self.Z1_read_max = cyska.clipped_sum_and_max(self.Z1, clip=1E6)
        # vector_stats(self.Z1_read)
        #self.Z1_read_max is used for thresholding

        # self-consistent free RBP concentrations
        if rbp_free is None:
            rbp_free = self.mdl.SPA_free_protein(self.Z1_read, Z_scale=params.A0)
        
        self._update_rbp_free(rbp_free)
        # print "rbp_free", self.rbp_free
        # pull-down weights for each read, in each sample
        # print "psi"
        # vector_stats(self.psi)
        # print "PSI min", self.psi.min(axis=1)
        # print "PartFuncModelState (betas=", params.betas,")"
        # print "Z1_read_max", self.Z1_read_max
        # print "psi", self.psi.shape, self.psi.min(axis=1), self.psi.max(axis=1), self.psi.mean(axis=1)
        # print "PSI max", self.psi.max(axis=1)
        # print "PSI mean", self.psi.mean(axis=1)
        # self.Q = self.psi.sum(axis=1)
        if not beta_fixed:
            betas = self.mdl.optimal_betas(self.psi, params.betas)
        else:
            betas = self.params.betas
            
        self._update_betas(betas)

        self.mdl.n_fev += 1
        self.mdl.t_fev += time.time() - t1
        self.mdl.t_aff += 0

    def _update_rbp_free(self, rbp_free):
        self.rbp_free = rbp_free
        self.psi = cyska.p_bound(self.Z1_read, self.rbp_free*self.params.A0)
        self.q = self.mdl.PD_kmer_weights(self.psi)
        self.Q = self.q.sum(axis=1)

    def _update_betas(self, betas):
        self.params.betas[:] = betas
        # print self.mdl.F0.dtype, self.params.betas.dtype
        self.w = np.array(self.q + self.mdl.F0[np.newaxis,:] * self.params.betas[:,np.newaxis], dtype=np.float32)
        # print self.w.dtype
        self.W = self.w.sum(axis=1)
        # print self.W.dtype
        # print "W", self.W
        # print "Q/W", self.Q/self.W
        self.R = self.w / self.mdl.f0[np.newaxis,:] / self.W[:,np.newaxis]
        # gcaug = cyska.seq_to_index('ugcaugu')
        # print "R(uGCAUGu)", self.R[:,gcaug]
        # print "R0(uGCAUGu)", self.mdl.R0[:,gcaug]

        self.R_errors = np.array(self.R, dtype=np.float64) - self.mdl.R0
        self.error = (self.R_errors**2).mean()
        
    @property
    def beta_estimators(self):
        est = (self.mdl.Rf0 * self.Q[:,np.newaxis] - self.q) / self.mdl.beta_denom
        return est

    @property
    def A0_estimators(self):
        psi0 = self.psi / self.params.A0
        q0 = self.mdl.PD_kmer_weights(psi0)
        Q0 = q0.sum(axis=1)
        est = self.params.betas[:,np.newaxis] * self.mdl.beta_denom / (Q0[:,np.newaxis] * self.mdl.Rf0 - q0)
        return est


    @property
    def correlations(self):
        from scipy.stats import pearsonr
        p_values = []
        R_values = []

        lR = np.log2(self.R)
        # vector_stats(lR)
        for r, r0 in zip(lR, self.mdl.lR0):
            pears_R, p_val = pearsonr(r,r0)
            p_values.append(p_val)
            R_values.append(pears_R)

        return np.array(R_values), np.array(p_values)

    def kmer_affinity_weights(self, cutoff=.01):
        ## EXPERIMENTAL: restrict to current highest affinity kmers
        aff = self.params.as_PSAM().affinities

        I = aff.argsort()[::-1]
        kmer_weights = np.zeros(self.mdl.nA, dtype=np.float32)

        INDMAX = int(4**self.mdl.k) - 1
        for i in I:
            ratio = aff[i] / self.params.A0
            if ratio < cutoff:
                break
            nmer = cyska.index_to_seq(i, self.mdl.k_mdl)
            # print nmer, ratio
            d = self.mdl.k_mdl - self.mdl.k
            assert d >= 0
            for x in range(d+1):
                j = (i >> 2*x) & INDMAX
                kmer_weights[j] += ratio
                # print cyska.index_to_seq(j, self.k), kmer_weights[j]

        return kmer_weights

    @property
    def grad(self):
        t0 = time.time()
        self.mdl.n_grad += 1
        # self.mdl.set_mask( self.Z1_read > self.mdl.Z_thresh * self.Z1_read_max)
        # _grad = cska.gradient.emp_grad(self, eps=1e-4)
        # self.mdl.set_mask()
        # print "state.psi", self.psi.shape
        # print "state.Q", self.Q.shape
        # print "state.rbp_free", self.rbp_free
        # print "state.w", self.w.shape

        self.E_weights = np.ones(self.mdl.nA, dtype=np.float32) + 10*self.kmer_affinity_weights(cutoff=.01)
        self.E_weights /= self.E_weights.mean()
        _grad = cyska.PSAM_partition_function_gradient(self)
        # _grad.A0 *= 0
        # _grad.betas *= 0
        # _grad.psam_vec[:] = 0 # HACK to test beta value convergence

        # from cska.gradient import emp_grad
        # _grad = emp_grad(self)

        # print "skipped reads below Z1_threshold", self.skipped

        # print "parallel"
        # _grad = cyska.PSAM_partition_function_gradient_parallel(self)
        # # _grad.A0 *= 4*_grad.k
        # _grad.betas = np.where(self.params.betas > 0, self.params.n_samples * _grad.betas, 0)
        # _grad.betas = np.where(self.params.betas > 0, _grad.betas, 0)
        self.mdl.t_grad += time.time() - t0
        return _grad


    def archive(self):
        from copy import copy
        arc = copy(self)
        arc.Z1 = None
        arc.Z1_read = None
        arc.psi  = None
        arc.Q = None
        arc.w = None
        arc.R_errors = None
        return arc


    def __str__(self):
        buf = [str(self.params)]
        buf.append("error = {}".format(self.error))
        buf.append("rbp_free = {}".format(self.rbp_free))
        buf.append("correlation = {}".format(self.correlations))

        return "\n".join(buf)


class PartFuncModel(object):
    """
    Evaluate the partition function on samples of RBNS reads to approximate the expected R-values.
    Importantly, k_monitor can be < params.n, i.e. the model can be more complex than the kmer frequencies
    being used to estimate agreement with the experiment.
    """
    def __init__(self, reads, params0, R0, rbp_conc=[], aff0=1e-6, Z_thresh=0, **kwargs):
        self.logger = logging.getLogger('opt.PartFuncModel')
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.reads = reads
        self.params = params0
        self.k_mdl = self.params.k
        self.acc_k = self.params.acc_k
        self.acc_ofs = self.params.acc_ofs
        self.Z_thresh = Z_thresh

        self.n_samples, self.nA = R0.shape
        assert self.n_samples == params0.n_samples
        self.k = int(np.log(self.nA) / np.log(4)) # nA = 4**k
        # print "partfuncmodel: k_mdl, k_fit", self.k_mdl, self.k
        self.F0 = np.array(reads.kmer_counts(self.k), dtype=np.uint32) # actual counts
        # print type(self.F0), self.F0.dtype
        self.f0 = np.array(self.F0 + reads.pseudo_count, dtype=np.float32)
        self.f0 /= self.f0.sum() # relative frequencies
        self.set_R0(R0)

        self.aff0 = aff0
        self.opt = None

        self.im = self.reads.get_index_matrix(self.k)
        self.seqm = self.reads.get_padded_seqm(self.k_mdl) #2bit coded read sequences, including flanking adapter overlap
        # self.im_mdl = self.reads.get_index_matrix(self.k_mdl) # k_mdl-mer indices from the reads
        self.openen = self.reads.acc_storage.get_raw(self.acc_k) # corresponding accessibilities
        self.acc = self.openen.acc

        # in case a mask is set, this can be a subset
        self.indices = []
        self._seqm = self.seqm
        self._acc = self.acc
        self._im = self.im
        self.N = np.float32(len(self._seqm))

        self.n_fev = 0
        self.t_fev = 0
        self.n_grad = 0
        self.t_grad = 0
        self.t_aff = 0

    def set_R0(self, R0):
        self.R0 = np.array(R0, dtype=np.float32)
        mu = self.R0.mean(axis=1)
        sig = self.R0.std(axis=1)
        z = (self.R0 - mu[:,np.newaxis]) / sig[:,np.newaxis]
        z_cut = 3.5
        I_top = (z > z_cut).sum(axis=0).nonzero()[0]
        print I_top
        self.top_Ri = I_top
        self.lR0 = np.log2(self.R0)
        self.Rf0 = self.R0 * self.f0[np.newaxis]
        self.beta_denom = self.F0[np.newaxis,:] * (1 - self.R0)

    def tune(self, state, A0_plot="", debug=True, maxiter=5):
        params = state.params
        A00 = params.A0
        # state2 = self.predict(params)
        # print "TUNE STATE"
        # print state
        # print state.Z1_read

        from scipy.optimize import minimize_scalar
        from scipy.stats import sem
        from cska.gradient import minimize_logspaced
        sc = SelfConsistency(state.Z1_read, self.reads.rna_conc, bins=1000)
        kmer_weights = state.kmer_affinity_weights(cutoff=.1)
        e0 = state.error
        # mask = np.fabs(state.mdl.R0 - 1) > .001
        # mask = np.logical_or(state.mdl.R0 > 1.1, state.mdl.R0 < .9)
        mask = state.mdl.R0 > 1.1
        print "estimators used", mask.sum() 

        a0s = []
        # beta0s = []
        R_err = {}
        A0_est = {}
        A0_sem = {}
        R_corr = {}

        def err(A0):
            state.params.A0 = A0

            rbp_free = np.array([sc.free_rbp(total, Z_scale=A0) for total in self.rbp_conc], dtype= np.float32)
            state._update_rbp_free(rbp_free)

            betas = self.optimal_betas(state.psi, state.params.betas)
            state._update_betas(betas)

            a0s.append(A0)
            # beta0s.append(betas[0])

            if debug:
                top = kmer_weights.argmax()
                topmer = cyska.index_to_seq(top, self.k)
                print "R({0})={1} [R0={2}]".format(topmer, state.R[:,top], self.R0[:,top])

            est = state.A0_estimators
            
            err = sem(est[mask], axis=None)
            if debug:
                print A0, "->", err, state.error
            
            A0_sem[A0] = err
            R_err[A0] = state.error
            R_corr[A0] = np.array(state.correlations).max()
            A0_est[A0] = est

            return state.error #err + state.error * self.nA

        # res = minimize_scalar(err, bounds=(1e-3, 100), method='Bounded')
        res = minimize_logspaced(err, bounds=np.array((1e-3, 1000)), n_samples=7, nested=2, options=dict(maxiter=maxiter))

        if debug:
            print res
    
        if 5e-3 < res.x < 500:
            state.params.A0 = res.x
        else:
            self.logger.warning("fit would push A0 to boundaries. Letting drift through gradient-only instead.")
            state.params.A0 = A00
        
        state = state.mdl.predict(state.params, beta_fixed=False)
        # HACK: attach to state object
        a0 = sorted(a0s)
        rerr = np.array([R_err[a] for a in a0])
        rcorr = np.array([R_corr[a] for a in a0])
        asem = np.array([A0_sem[a] for a in a0])
        
        self.logger.info("spectrum of mean squared error {} {} {}".format(rerr.max(), rerr.min(), rerr.max() / rerr.min()) )
        from gradient import Tracked
        state._A0_data = Tracked(a0=a0, rerr=rerr, asem=asem, rcorr=rcorr)
        return state

    def set_mask(self, indices=[]):
        if not len(indices):
            # unset mask
            self._seqm = self.seqm
            self._acc = self.acc
            self._im = self.im
            self.logger.debug("set_mask() unset")
        else:
            self._seqm = self.seqm[indices]
            self._acc = self.acc[indices]
            self._im = self.im[indices]
            frac = float(len(self._seqm)) / len(self.seqm)
            self.logger.debug("set_mask() to {0:.2f}% of reads".format(100. * frac))

        self.N = np.float32(len(self._seqm))

    def SPA_free_protein(self, Z1, Z_scale=1.):
        sc = SelfConsistency(Z1, self.reads.rna_conc, bins=1000)
        rbp_free = [sc.free_rbp(total, Z_scale=Z_scale) for total in self.rbp_conc]
        self._last_sc = sc # keep for debugging or re-use (if Z1 is unaltered)
        return np.array(rbp_free, dtype= np.float32)

    def PD_kmer_weights(self, psi):
        w = np.zeros( (self.n_samples, self.nA), dtype=np.float32) 
        for j in range(self.n_samples):
            w[j] = cyska.weighted_kmer_counts(self._im, psi[j], self.k) + 1e-20

        return w

    def predict(self, params, aff0=1e-6, debug=False, tune=False, **kwargs):
        t0 = time.time()
        state = PartFuncModelState(self, params, **kwargs)
        self.logger.debug("predict() took {0:.2f} ms".format(1000. * (time.time() - t0)))
        if tune:
            state = self.tune(state)

        return state

    def estimate_betas(self, state):
        est = state.beta_estimators
        est_b = np.median(est[:,self.top_Ri],axis=1)

        return est_b

    def optimal_betas(self, state_psi, opt_betas, n=5, q_top=10):
        from cska.gradient import minimize_logspaced
        from scipy.optimize import minimize_scalar
        
        # print "initial guess", opt_betas
        # print "state_psi", state_psi

        t0 = time.time()
        top_i = self.R0.max(axis=0).argmax()
        top_mer = cyska.index_to_seq(top_i, self.k)

        for i in range(self.n_samples):
            # R0_quants = np.percentile(self.R0[i], np.linspace(0,q_top,n))
            # print "R0 quantiles", R0_quants
            psi = state_psi[i]
            w0 = cyska.weighted_kmer_counts(self._im, psi, self.k)
            # w0 = state.q[i]
            # TODO: should be w0 = state.q
            # if i == 0:
                # print "B0", b0[:10]


            def to_optimize(beta):
                # b = cyska.weighted_kmer_counts(self.im, psi + beta, self.k)
                w = w0 + self.F0 * beta
                W = w.sum()
                R = w / self.f0 / W

                # R_quants = np.percentile(R, np.linspace(0,q_top,n))
                # print "beta", beta, "R quants", R_quants
                R_errors = np.array(R - self.R0[i], dtype=np.float32)
                # R_errors = R_quants - R0_quants
                # print "R-errors min/max/mean", R_errors.min(), R_errors.max(), R_errors.mean()
                # print "most over-predicted", cyska.index_to_seq(R_errors.argmax(), self.k)
                # print "most under-predicted", cyska.index_to_seq(R_errors.argmin(), self.k)
                error = (R_errors**2).mean()
                # print beta, "->", error, "R({})".format(top_mer), R[top_i], self.R0[i,top_i]
                return error
            
            res = minimize_logspaced(to_optimize, bounds=np.array([1e-7, 10]), n_samples=7, debug=False)
            # res = minimize_scalar(to_optimize, opt_betas[i], bounds=np.array([1e-7, 10]), method='Bounded')
            # print "beta",i, res
            if res.success:
                opt_betas[i] = res.x

        # print "final values", opt_betas
        # self.logger.debug("optimal_betas took {0:.2f} ms".format(1000. * (time.time() - t0)))
        return np.array(opt_betas, dtype=np.float32)

    # def optimal_A0(self, state, debug=False, plot=""):
    #     from scipy.optimize import minimize_scalar
    #     from cska.gradient import minimize_logspaced
    #     params = state.params.copy()
    #     sc = SelfConsistency(state.Z1_read, self.reads.rna_conc, bins=1000)

    #     perc = np.linspace(0,100,50)
    #     R0_quant = np.percentile(self.R0, perc, axis=1)
    #     kmer_weights = state.kmer_affinity_weights(cutoff=.1)
    #     e0 = state.error
    #     def err(A0):
    #         # s = np.exp(x)

    #         rbp_free = np.array([sc.free_rbp(total, Z_scale=A0) for total in self.rbp_conc], dtype= np.float32)
    #         psi = cyska.p_bound(state.Z1_read, rbp_free*A0)
    #         # print "PSI min", self.psi.min(axis=1)
    #         # print "PSI max", self.psi.max(axis=1)
    #         # print "PSI mean", self.psi.mean(axis=1)
    #         params.A0 = A0
    #         betas = self.optimal_betas(psi, params.betas)
    #         params.betas[:] = betas
    #         q = self.PD_kmer_weights(psi)
    #         Q = q.sum(axis=1)
    #         w = q + self.F0[np.newaxis,:] * betas[:,np.newaxis]

    #         W = w.sum(axis=1)
    #         # print "Q", self.Q
    #         R = w / self.f0[np.newaxis,:] / W[:,np.newaxis]
    #         if debug:
    #             top = kmer_weights.argmax()
    #             topmer = cyska.index_to_seq(top, self.k)
    #             print "R({0})={1} [R0={2}]".format(topmer, R[:,top], self.R0[:,top])

    #         # R_quant = np.percentile(R, perc, axis=1)
    #         # !!!TODO!!! restrict to kmers that are currently covered by motif!!!
    #         R_errors = np.array(R - self.R0, dtype=np.float32) * kmer_weights[np.newaxis,:]
    #         # R_errors = R_quant - R0_quant
    #         error = (R_errors**2).mean()
    #         if debug:
    #             print "A0={0:.3e} -> err={1:.3e} betas={2} rbp_free={3} Q={4}".format(A0, error, betas, rbp_free, Q), psi.shape
    #         return error - e0


    #     # res = minimize_scalar(err, 1., bounds=np.array([1e-3, 10]), method='Bounded')
    #     res = minimize_logspaced(err, bounds=np.array([1e-3, 10000]), debug=False, nested=2, n_samples=5, options=dict(maxiter=5), plot=plot)
    #     # print "debug"
    #     # err(1.)
    #     # print res
    #     return res.x

    # def fit_A0_and_betas(self, state, plot=""):
    #     from scipy.optimize import minimize_scalar
    #     # from cska.gradient import minimize_logspaced

    #     params = state.params.copy()
    #     sc = SelfConsistency(state.Z1_read, self.reads.rna_conc, bins=1000)

    #     e0 = state.error

    #     I = self.top_Ri
    #     Rf0 = self.Rf0[:,I]
    #     bd = self.beta_denom[:,I]

    #     # psi0 = state.psi / state.params.A0
    #     # q0 = self.PD_kmer_weights(psi0)
    #     # Q0 = q0.sum(axis=1)

    #     sampled = {}
    #     def compute(A0):
    #         rbp_free = np.array([sc.free_rbp(total, Z_scale=A0) for total in self.rbp_conc], dtype= np.float32)
    #         psi = cyska.p_bound(state.Z1_read, rbp_free*A0)
    #         psi0 = np.array(state.Z1_read[np.newaxis,:] * (rbp_free * A0)[:,np.newaxis], dtype=np.float32)
    #         q0 = self.PD_kmer_weights(psi0)
    #         Q0 = q0.sum(axis=1)

    #         q = self.PD_kmer_weights(psi)
    #         Q = q.sum(axis=1)

    #         # print "Q0", Q0
    #         # print "Q", Q
    #         beta_est = (Rf0 * Q[:,np.newaxis] - q[:,I]) / bd
    #         betas = np.mean(beta_est, axis=1)
    #         betas = params.betas
    #         A0_est = betas[:,np.newaxis] * bd / (Q0[:,np.newaxis] * Rf0 - q0[:,I])
    #         return betas, beta_est, rbp_free, A0_est

    #     def to_optimize(A0):
    #         betas, beta_est, rbp_free, A0_est = compute(A0)

    #         err_beta = np.median((beta_est - betas[:,np.newaxis])**2)
    #         err_A0 = np.median((A0_est - A0)**2)
    #         err = err_beta + err_A0
    #         print A0, "->", rbp_free, "->", betas, "->", err_beta, err_A0, err
    #         sampled[A0] = (err_beta, err_A0, err)
    #         return err

    #     res = minimize_scalar(to_optimize, state.params.A0, bounds=np.array([1e-6, 10000]), method='Bounded')
    #     betas, beta_est, rbp_free, A0_est = compute(res.x)

    #     params.A0 = res.x
    #     params.betas[:] = betas
        
    #     if plot:
    #         import matplotlib.pyplot as plt
    #         plt.figure()
    #         plt.title("fit_A0_and_betas")
    #         x = sorted(sampled.keys())
    #         y_beta, y_A0, y_total = np.array([sampled[i] for i in x]).T
    #         plt.loglog(x,y_total)
    #         plt.loglog(x,y_total, 'xr')
    #         plt.loglog(x,y_beta,label="beta_err")
    #         plt.loglog(x,y_A0,label="A0_err")
    #         plt.legend(loc='lower right')
    #         plt.show()
    #         plt.savefig(plot)
    #         plt.close()

    #     return self.predict(params, beta_fixed=True)


            

    def quantile_fit(self, state, n=20):
        """True
        Optimize A0 and betas jointly to match the distribution of R-values, not considering the exact
        identity of the kmers. The idea is to "straighten the Banana" which is sometimes the shape of the
        kmer scatter plots if betas+A0 are stuck in sub-optimal values.
        """
        # observed R-value quantiles
        Rq0 = np.array([np.percentile(self.R0[i], np.linspace(0,100,n)) for i in range(self.n_samples)])

        # track most enriched kmer
        top_i = self.R0.max(axis=0).argmax()
        top_mer = cyska.index_to_seq(top_i, self.k)

        sc = SelfConsistency(state.Z1_read, self.reads.rna_conc, bins=1000)

        def to_optimize(args):
            args = np.array(args, dtype=np.float32)
            A0, betas = args[0], args[1:]
            # rbp_free = self.SPA_free_protein(state.Z1_read, Z_scale=A0)
            rbp_free = np.array([sc.free_rbp(total, Z_scale=A0) for total in self.rbp_conc], dtype= np.float32)

            # print "rbp_free", rbp_free
            # pull-down weights for each read, in each sample
            psi = cyska.p_bound(state.Z1_read, rbp_free*A0)

            b = self.PD_kmer_weights(psi)
            # print "b min", self.b.min(axis=1)
            # print "b max", self.b.max(axis=1)
            # print "b mean", self.b.mean(axis=1)

            Q = b.sum(axis=1)
            # print "Q", self.Q
            R = b / self.f0[np.newaxis,:] / Q[:,np.newaxis]

            # self.R_errors = np.array(self.R - self.mdl.R0, dtype=np.float32)
            # self.error = (self.R_errors**2).mean()
       
            Rq = np.array([np.percentile(r, np.linspace(0,100,n)) for r in R])
            Rq_errors = Rq - Rq0
            # for beta, rq in zip(betas, Rq_errors):
            #     print "beta", beta, "quantile errors", rq
            
            # print "R-errors min/max/mean", R_errors.min(), R_errors.max(), R_errors.mean()
            # print "most over-predicted", cyska.index_to_seq(R_errors.argmax(), self.k)
            # print "most under-predicted", cyska.index_to_seq(R_errors.argmin(), self.k)
            error = (Rq_errors**2).mean()
            # print args, "->", error, "R({})".format(top_mer), R[:,top_i], self.R0[:,top_i]
            return error

        from scipy.optimize import minimize
        args0 = np.array([state.params.A0,] + list(state.params.betas))
        print "args0", args0
        bounds = np.array([(1e-4,1e3),] + [(1e-9, 10)] * self.n_samples)
        print "bounds", bounds
        res = minimize(to_optimize, args0, bounds=bounds, method='L-BFGS-B')
        print res
        return res.x

    @property
    def affinities(self):
        A, I = cyska.params_from_pwm(self.params.psam_matrix, A0=self.params.A0, aff0=self.aff0)
        return A




# # # old code from gradient.py

# class KmerModelState(object):
#     def __init__(self, mdl, params):
#         self.mdl = mdl
#         self.params = params
        
#         # print "psam params vector", psam
#         cyZ = cyska.PSAM_partition_function(mdl.sub_padded, mdl.sub_acc, params.psam_matrix, openen_ofs=mdl.acc_ofs)
        
#         cyZ *= params.A0
#         Zj = cyZ.sum(axis=1)
#         psi = P*Zj/ (P*Zj+1)

#         # print "PSI", psi, psi.min(), psi.mean(), psi.max()
#         self.cyZ = cyZ
#         self.Zj = Zj
#         # avoid zeros in Zj by all means!
#         np.clip(self.Zj, 1e-9, None, out=self.Zj)
#         self.psi = psi
#         self.pi = cyska.weighted_kmer_counts(mdl.sub_im, psi + params.beta, mdl.k_monitor)

#         self.R = self.pi / self.pi.sum() / mdl.f0
#         self.error = self.mdl.error(self.R) 
#         self.mdl.n_fev += 1

#     @property
#     def grad(self):
#         N = len(cyZ)
#         pi, d_pi = cyska.PSAM_kmer_gradient(self.mdl.sub_padded, self.cyZ, self.Zj, self.psi, self.mdl.sub_im, self.params.psam_vec, self.mdl.k_monitor)
#         pi += N * self.f0 * params.beta 

#         # print R.shape, d_pi.shape
#         dR_dM = (self.R/self.pi)[:,np.newaxis] * (d_pi - (self.f0 * self.R)[:,np.newaxis] * d_pi.sum(axis=0)[np.newaxis,:])
#         dR_dbeta =  self.f0 / self.pi * (self.R - self.R**2)

#         dR = np.hstack( (dR_dM, dR_dbeta[:,np.newaxis]) )

#         # TODO: refactor the R^2 part of gradient into optimizer?
#         _grad = 2 * ((1. * (self.R - self.mdl.opt.R0))[:,np.newaxis] * dR).sum(axis=0)
#         param_grad = self.params.copy().set_vector(_grad)
#         self.mdl.n_grad += 1
#         return param_grad

# class PartFuncModel(object):
#     def __init__(self, reads, params, R0, k_monitor=5, subsample=1.):
#         self.logger = logging.getLogger('PartFuncModel')
#         self.reads = reads
#         self.opt = None
#         self.params = params
#         self.R0 = R0
#         self.openen = self.reads.acc_storage.get_raw(self.params.k)
#         self.acc_ofs = self.reads.l5 - params.k + 1
#         print "acc_ofs", self.acc_ofs
#         self.acc = self.openen.acc
#         print self.acc.shape
#         self.n = params.k
#         adap5 = cyska.seq_to_bits(reads.adap5)
#         adap3 = cyska.seq_to_bits(reads.adap3)
#         self.padded = cyska.seqm_pad_adapters(reads.seqm, adap5, adap3, self.n)

#         self.k_monitor = k_monitor
#         self.im = reads.get_index_matrix(k_monitor)

#         f0 = reads.kmer_frequencies(k_monitor)
#         self.f0 = f0 / f0.sum()

#         self.subsample = subsample
#         self.new_subsample()
    
#         self.n_fev = 0
#         self.n_grad = 0

#     def new_subsample(self):
#         self.logger.info('new subsample')
#         n = int(self.subsample * self.reads.N)
#         if n == self.reads.N:
#             self.sub_indices = np.arange(n)
#         else:
#             self.sub_indices = np.random.permutation(self.reads.N)[:n]

#         self.sub_padded = self.padded[self.sub_indices]
#         self.sub_im = self.im[self.sub_indices]
#         self.sub_acc = self.acc[self.sub_indices]
        
#     def predict(self, params):
#         self.n_fev += 1
#         return KmerModelState(self, params)

#     def error(self, R):
#         return ((self.R0 - R)**2).mean()

