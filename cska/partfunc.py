import logging, os, sys
import numpy as np
import time
import cska.gradient
import cska.cyska as cyska
from cska.pwm import PSAM
from cska.sc import SelfConsistency

class PartFuncModelState(object):
    def __init__(self, mdl, params):
        t0 = time.time()
        self.mdl = mdl
        self.params = params
        self.rbp_conc = mdl.rbp_conc
    
        # self.A, self.I = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=mdl.aff0)
        t1 = time.time()
        # print "number of relevant kmers", len(self.I)
        # assert (sorted(self.I) == (self.A > mdl.aff0).nonzero()[0] ).all()
        # TODO: make protein concentration self-consistent

        # evaluate single protein partition function of PSAM over reads
        self.Z1 = cyska.PSAM_partition_function(
            self.mdl.seqm, 
            self.mdl.openen.acc, 
            np.array(params.psam_matrix, dtype=np.float32) * params.A0,
            openen_ofs=self.mdl.openen.ofs - self.mdl.k_mdl + 1
        )
        self.Z1_read = self.Z1.sum(axis=1)
        # self-consistent free RBP concentrations
        self.rbp_free = self.mdl.SPA_free_protein(self.Z1_read)
        # print "rbp_free", self.rbp_free
        # pull-down weights for each read, in each sample
        self.psi, w = cyska.p_bound(self.Z1_read, self.rbp_free, params.betas)
        # print "PSI min", self.psi.min(axis=1)
        # print "PSI max", self.psi.max(axis=1)
        # print "PSI mean", self.psi.mean(axis=1)

        self.b = self.mdl.PD_kmer_weights(w)
        # print "b min", self.b.min(axis=1)
        # print "b max", self.b.max(axis=1)
        # print "b mean", self.b.mean(axis=1)

        self.Q = self.b.sum(axis=1)
        # print "Q", self.Q
        self.R = self.b / self.mdl.f0[np.newaxis,:] / self.Q[:,np.newaxis]
        # gcaug = cyska.seq_to_index('ugcaugu')
        # print "R(uGCAUGu)", self.R[:,gcaug]
        # print "R0(uGCAUGu)", self.mdl.R0[:,gcaug]

        self.R_errors = np.array(self.R - self.mdl.R0, dtype=np.float32)
        self.error = (self.R_errors**2).mean()

        self.mdl.n_fev += 1
        self.mdl.t_fev += time.time() - t1
        self.mdl.t_aff += 0
        
    @property
    def grad(self):
        t0 = time.time()
        self.mdl.n_grad += 1
        # _grad = cska.gradient.emp_grad(self, eps=1e-4)
        # print "state.psi", self.psi.shape
        # print "state.Q", self.Q.shape
        # print "state.rbp_free", self.rbp_free
        # print "state.b", self.b.shape
        _grad = cyska.PSAM_partition_function_gradient(self)
        # print "parallel"
        # _grad = cyska.PSAM_partition_function_gradient_parallel(self)
        # # _grad.A0 *= 4*_grad.k
        # _grad.A0 *= 0
        # _grad.psam_vec[:] = 0 # HACK to test beta value convergence
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
        arc.b = None
        arc.R_errors = None
        return arc
        

class PartFuncModel(object):
    """
    Evaluate the partition function on samples of RBNS reads to approximate the expected R-values.
    Importantly, k_monitor can be < params.n, i.e. the model can be more complex than the kmer frequencies
    being used to estimate agreement with the experiment.
    """
    def __init__(self, reads, params0, R0, rbp_conc=[], aff0=1e-6, **kwargs):
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.reads = reads
        self.params = params0
        self.k_mdl = self.params.k

        self.R0 = R0
        self.n_samples, self.nA = R0.shape
        assert self.n_samples == params0.n_samples
        self.k = int(np.log(self.nA) / np.log(4)) # nA = 4**k
        print "partfuncmodel: k_mdl, k_fit", self.k_mdl, self.k
        f0 = reads.kmer_frequencies(self.k)
        self.f0 = f0 / f0.sum()

        self.aff0 = aff0
        self.opt = None

        self.im = self.reads.get_index_matrix(self.k)
        self.seqm = self.reads.get_padded_seqm(self.k_mdl) #2bit coded read sequences, including flanking adapter overlap
        # self.im_mdl = self.reads.get_index_matrix(self.k_mdl) # k_mdl-mer indices from the reads
        self.openen = self.reads.acc_storage.get_raw(self.k_mdl) # corresponding accessibilities

        self.n_fev = 0
        self.t_fev = 0
        self.n_grad = 0
        self.t_grad = 0
        self.t_aff = 0


    def SPA_free_protein(self, Z1):
        sc = SelfConsistency(Z1, self.reads.rna_conc, bins=10000)
        rbp_free = [sc.free_rbp(total) for total in self.rbp_conc]
        return np.array(rbp_free, dtype= np.float32)

    def PD_kmer_weights(self, psi):
        w = np.zeros( (self.n_samples, self.nA), dtype=np.float32)
        for i in range(self.n_samples):
            w[i] = cyska.weighted_kmer_counts(self.im, psi[i], self.k)

        return w

    def predict(self, params, aff0=1e-6, debug=False):
        state = PartFuncModelState(self, params)
        return state

    def estimate_betas(self, state, q=5):
        R_ns = np.percentile(self.R0, q, axis=1)
        beta = R_ns / (1 - R_ns)
        return state.Q * beta / self.reads.N

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

