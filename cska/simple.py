import logging, os, sys
import numpy as np
import time
import cska.gradient
import cska.ska_kmers as cyska
from cska.pwm import PSAM

class SimpleModelState(object):
    def __init__(self, mdl, params):
        t0 = time.time()
        self.mdl = mdl
        self.params = params
        self.rbp_conc = mdl.rbp_conc
    
        self.A, self.I = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=mdl.aff0)
        t1 = time.time()
        print "number of relevant kmers", len(self.I)
        # assert (sorted(self.I) == (self.A > mdl.aff0).nonzero()[0] ).all()
        # TODO: make protein concentration self-consistent
        cyska.PSAM_mean_field_eval(self) # This call populates self.error, self.occ, self.pi etc...

        self.mdl.n_fev += 1
        self.mdl.t_fev += time.time() - t1
        self.mdl.t_aff += t1 - t0
        
    @property
    def grad(self):
        t0 = time.time()
        self.mdl.n_grad += 1
        _grad = self.params.copy()
        t1 = time.time() - t0
        print "dt params.copy=", t1
        _grad.data[:] = cyska.PSAM_mean_field_gradient(self)
        self.mdl.t_grad += time.time() - t0
        return _grad
    

class SimpleModel(object):
    """
    No crosstalk matrix, no background parameter.
    """
    def __init__(self, reads, params0, R0, rbp_conc=[], aff0=1e-6, **kwargs):
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.reads = reads
        self.params = params0
        self.R0 = R0
        self.aff0 = aff0
        self.opt = None
        k = self.params.k
        self.k = self.params.k
        self.xm = CrosstalkMatrix(k, reads)
        
        f0 = reads.kmer_frequencies(k)
        self.f0 = f0 / f0.sum()
        
        self.n_fev = 0
        self.t_fev = 0
        self.n_grad = 0
        self.t_grad = 0
        self.t_aff = 0

    def predict(self, params, aff0=1e-6, debug=False):
        state = MeanFieldModelState(self, params)
        return state

    def estimate_betas(self, state, q=5):
        R_ns = np.percentile(self.R0, q, axis=1)
        # print "R_ns,j", R_ns
        beta = R_ns / (1 - R_ns)
        # print "beta?", beta
        # print "sum_pi with current beta estimate", state.sum_pi
        return state.sum_pi * beta

    @property
    def affinities(self):
        state = self.predict(self.params)
        return state.A
