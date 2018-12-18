import logging
import numpy as np
import time
import cska.cyska as cyska
from cska.sc import SelfConsistency


class PartFuncModelState(object):
    def __init__(self, mdl, params, beta_fixed=True, rbp_free=None, **kwargs):
        self.mdl = mdl
        self.params = params.copy()
        self.rbp_conc = mdl.rbp_conc
        self.threshold = mdl.Z_thresh   # TODO: cleanup! experimental
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
            np.array(params.psam_matrix, dtype=np.float32),  # * params.A0,
            openen_ofs=self.mdl.openen.ofs - self.mdl.k_mdl + 1 + self.mdl.acc_shift
        )
        self.Z1_read, self.Z1_read_max = cyska.clipped_sum_and_max(self.Z1, clip=1E6)
        # vector_stats(self.Z1_read)
        # self.Z1_read_max is used for thresholding

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
        self.w = np.array(self.q + self.mdl.F0[np.newaxis, :] * self.params.betas[:, np.newaxis], dtype=np.float32)
        # print self.w.dtype
        self.W = self.w.sum(axis=1)
        # print self.W.dtype
        # print "W", self.W
        # print "Q/W", self.Q/self.W
        self.R = self.w / self.mdl.f0[np.newaxis, :] / self.W[:, np.newaxis]
        # gcaug = cyska.seq_to_index('ugcaugu')
        # print "R(uGCAUGu)", self.R[:,gcaug]
        # print "R0(uGCAUGu)", self.mdl.R0[:,gcaug]

        self.R_errors = np.array(self.R, dtype=np.float64) - self.mdl.R0
        self.error = (self.R_errors**2).mean()

    @property
    def beta_estimators(self):
        est = (self.mdl.Rf0 * self.Q[:, np.newaxis] - self.q) / self.mdl.beta_denom
        return est

    @property
    def A0_estimators(self):
        psi0 = self.psi / self.params.A0
        q0 = self.mdl.PD_kmer_weights(psi0)
        Q0 = q0.sum(axis=1)
        est = self.params.betas[:, np.newaxis] * self.mdl.beta_denom / (Q0[:, np.newaxis] * self.mdl.Rf0 - q0)
        return est


    @property
    def correlations(self):
        from scipy.stats import pearsonr
        p_values = []
        R_values = []

        lR = np.log2(self.R)
        # vector_stats(lR)
        for r, r0 in zip(lR, self.mdl.lR0):
            pears_R, p_val = pearsonr(r, r0)
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

        self.E_weights = np.ones(self.mdl.nA, dtype=np.float32) + 10 * self.kmer_affinity_weights(cutoff=.01)
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
    Evaluate the partition function on samples of RBNS reads to
    approximate the expected R-values.
    Importantly, k_monitor can be < params.n, i.e. the model can
    be more complex than the kmer frequencies
    being used to estimate agreement with the experiment.
    """
    def __init__(self, reads, params0, R0, rbp_conc=[], aff0=1e-6, Z_thresh=0, **kwargs):
        self.logger = logging.getLogger('model.PartFuncModel')
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.reads = reads
        self.params = params0
        self.k_mdl = self.params.k
        self.acc_k = self.params.acc_k
        self.acc_shift = self.params.acc_shift
        self.acc_scale = self.params.acc_scale

        self.Z_thresh = Z_thresh

        self.n_samples, self.nA = R0.shape
        assert self.n_samples == params0.n_samples
        self.k = int(np.log(self.nA) / np.log(4)) # nA = 4**k
        # print "partfuncmodel: k_mdl, k_fit", self.k_mdl, self.k
        self.F0 = np.array(reads.kmer_counts(self.k), dtype=np.uint32)  # actual counts
        # print type(self.F0), self.F0.dtype
        self.f0 = np.array(self.F0 + reads.pseudo_count, dtype=np.float32)
        self.f0 /= self.f0.sum() # relative frequencies
        self.set_R0(R0)

        self.aff0 = aff0
        self.opt = None

        self.im = self.reads.get_index_matrix(self.k)
        self.seqm = self.reads.get_padded_seqm(self.k_mdl)  #2bit coded read sequences, including flanking adapter overlap
        # self.im_mdl = self.reads.get_index_matrix(self.k_mdl) # k_mdl-mer indices from the reads
        self.openen = self.reads.acc_storage.get_raw(self.acc_k)  # corresponding accessibilities
        self.acc = self.openen.acc
        if self.params.acc_scale != 1.:
            self.logger.debug("scaling accessibilities by {}".format(self.params.acc_scale))
            self.acc = np.array(self.acc, dtype=np.float32)  # make a scaled *copy*
            cyska.pow_scale(self.acc, self.params.acc_scale)

        assert np.isfinite(self.acc).all()

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
        z = (self.R0 - mu[:, np.newaxis]) / sig[:, np.newaxis]
        z_cut = 3.5
        I_top = (z > z_cut).sum(axis=0).nonzero()[0]
        # print I_top
        self.top_Ri = I_top
        self.lR0 = np.log2(self.R0)
        self.Rf0 = self.R0 * self.f0[np.newaxis]
        self.beta_denom = self.F0[np.newaxis, :] * (1 - self.R0)

    def tune(self, state, debug=False, maxiter=5):
        params = state.params
        A00 = params.A0
        from cska.gradient import minimize_logspaced

        sc = SelfConsistency(state.Z1_read, self.reads.rna_conc, bins=1000)
        kmer_weights = state.kmer_affinity_weights(cutoff=.1)

        a0s = []
        R_err = {}
        R_corr = {}

        def err(A0):
            state.params.A0 = A0

            rbp_free = sc.free_rbp_vector(self.rbp_conc, Z_scale=A0)
            state._update_rbp_free(rbp_free)

            betas = self.optimal_betas(state.psi, state.params.betas)
            state._update_betas(betas)

            a0s.append(A0)

            if debug:
                top = kmer_weights.argmax()
                topmer = cyska.index_to_seq(top, self.k)
                print "R({0})={1} [R0={2}]".format(topmer, state.R[:, top], self.R0[:, top])
                print A0, "->", err, state.error
            
            R_err[A0] = state.error
            R_corr[A0] = np.array(state.correlations).max()

            return state.error

        res = minimize_logspaced(err, bounds=np.array((1e-3, 1000)), n_samples=7, nested=2, options=dict(maxiter=maxiter))

        if debug:
            print res
    
        if 5e-3 < res.x < 500:
            state.params.A0 = res.x
        else:
            self.logger.debug("fit would push A0 to boundaries. Letting drift through gradient-only instead.")
            state.params.A0 = A00
        
        state = state.mdl.predict(state.params, beta_fixed=False)
        # HACK: attach to state object
        a0 = sorted(a0s)
        rerr = np.array([R_err[a] for a in a0])
        rcorr = np.array([R_corr[a] for a in a0])
        # asem = np.array([A0_sem[a] for a in a0])
        
        if debug:
            self.logger.debug("spectrum of mean squared error {} {} {}".format(rerr.max(), rerr.min(), rerr.max() / rerr.min()))

        from gradient import Tracked
        state._A0_data = Tracked(a0=a0, rerr=rerr, rcorr=rcorr)  # asem=asem, 
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
        rbp_free = sc.free_rbp_vector(self.rbp_conc, Z_scale=Z_scale)
        self._last_sc = sc  # keep for debugging or re-use (if Z1 is unaltered)
        return np.array(rbp_free, dtype=np.float32)

    def PD_kmer_weights(self, psi):
        w = np.zeros( (self.n_samples, self.nA), dtype=np.float32) 
        for j in range(self.n_samples):
            w[j] = cyska.weighted_kmer_counts(self._im, psi[j], self.k) + 1e-20

        return w

    def predict(self, params, aff0=1e-6, debug=False, tune=False, **kwargs):
        t0 = time.time()
        state = PartFuncModelState(self, params, **kwargs)
        self.logger.debug("predict(acc_k={} acc_shift={}) took {:.2f} ms".format(self.acc_k, self.acc_shift, 1000. * (time.time() - t0)))
        if tune:
            state = self.tune(state)

        return state

    def estimate_betas(self, state):
        est = state.beta_estimators
        est_b = np.median(est[:, self.top_Ri], axis=1)

        return est_b

    def optimal_betas(self, state_psi, opt_betas, n=5, q_top=10):
        from cska.gradient import minimize_logspaced
        
        # print "initial guess", opt_betas
        # print "state_psi", state_psi

        # t0 = time.time()
        # top_i = self.R0.max(axis=0).argmax()
        # top_mer = cyska.index_to_seq(top_i, self.k)

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
  
    @property
    def affinities(self):
        A, I = cyska.params_from_pwm(self.params.psam_matrix, A0=self.params.A0, aff0=self.aff0)
        return A


