import cska.gradient
from cska.crosstalk_matrix import CrosstalkMatrix
import logging, os, sys
import cska.cyska as cyska
from cska.pwm import PSAM
import numpy as np
import time

def debug_kmer_vector(vec, top=10, ref=None, header='values'):
    k = int(np.log(len(vec))/ np.log(4))
    
    I = vec.argsort()
    print "highest {0}".format(header)
    for i in I[::-1][:top]:
        out = [cyska.index_to_seq(i, k), str(vec[i])]
        if not ref is None:
            out.append(str(ref[i]))
        print "\t".join(out)
        

class MeanFieldModelState(object):
    def __init__(self, mdl, params):
        t0 = time.time()
        self.mdl = mdl
        self.params = params
        self.rbp_conc = mdl.rbp_conc
    
        self.A, self.I = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=0.1*mdl.aff0)
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

    def archive(self):
        self.A = None
        self.I = None
        return self
    

class MeanFieldModel(object):
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








class InvMeanFieldModelState(object):
    def __init__(self, mdl, params):
        t0 = time.time()
        self.mdl = mdl
        self.params = params
        self.rbp_conc = mdl.rbp_conc
    
        self.A, self.I = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=mdl.aff0)
        t1 = time.time()
        #print "number of relevant kmers", len(self.I)
        # assert (sorted(self.I) == (self.A > mdl.aff0).nonzero()[0] ).all()
        # TODO: make protein concentration self-consistent
        #cyska.PSAM_mean_field_eval(self) # This call populates self.error, self.occ, self.pi etc...
        Z = self.rbp_conc[:,np.newaxis] * self.A[np.newaxis,:]
        self.occ = Z/(Z+1)
        self.Q = params.betas + (self.occ * self.mdl.xm.wrm[np.newaxis,:]).sum(axis=1)
        self.sum_pi = (self.occ * self.mdl.f0).sum(axis=1)
        # print "sum_pi", self.sum_pi

        self._pd = self.occ + params.betas[:,np.newaxis] * self.mdl.icsum[np.newaxis,:]
        self.right = 1/self.Q[:,np.newaxis]*self._pd
        self.left = self.mdl.iR0
        # print "params", params
        # gcaug = cyska.seq_to_index('gcaug')
        # print left[:,gcaug]
        # print right[:,gcaug]
        # print zip(left[:,gcaug], self.occ[:,gcaug]/total), total
        self._E = self.left - self.right
        self._err = (self.mdl.w*((self._E)**2)).sum(axis=1)
        self.error = self._err.sum()

        # self._err = 2*(1/total[:,np.newaxis] * (self.occ + params.betas[:,np.newaxis] * self.mdl.iI[np.newaxis,:]) - self.mdl.iR0)
        
        self.mdl.n_fev += 1
        self.mdl.t_fev += time.time() - t1
        self.mdl.t_aff += t1 - t0
        
        # for j in range(len(self.mdl.rbp_conc[:1])):
        #     print "sample",j, self.error, self._err
        #     iR = left[j]
        #     pred = right[j]
        #     I = iR.argsort()[::-1]
        #     for i in I[:10]:
        #         print iR[i], cyska.index_to_seq(i, params.k), 'iI', pred[i]
        

    @property
    def R(self):
        Mocc = np.array([np.dot(self.mdl.xm.M, o) for o in self.occ], dtype=np.float32)
        self.pi = self.mdl.f0[np.newaxis,:] * (Mocc  + self.params.betas[:,np.newaxis])
        self.sum_pi = self.pi.sum(axis=1)
        return self.pi / self.sum_pi[:,np.newaxis] / self.mdl.f0[np.newaxis,:]
        
        
    @property
    def grad(self):
        import cska.gradient
        # for now
        self.mdl.n_grad += 1
        t0 = time.time()
        g = cska.gradient.emp_grad(self, eps=1e-4)
        self.mdl.t_grad += time.time() - t0
        
        #self.mdl.n_grad += 1
        #_grad = self.params.copy()
        
        docc = (self.occ - self.occ*self.occ)
        sml = (self.mdl.xm.wrm[np.newaxis,:] * docc).sum(axis=1)
        pre = self._pd / (self.Q**2)[:,np.newaxis]
        dA = pre * sml[:,np.newaxis] - 1/self.Q[:,np.newaxis] * docc
        dA *= 2 * self._E * self.mdl.w
        

        # print 'docc', docc.shape, docc.min(), docc.max()
        # print 'sml', sml.shape, 'w', sml.min(), sml.max()
        # Q = self.Q
        # print 'Q', Q.min(), Q.max()
        # print '1/Q^2', (1/Q**2).min(), (1/Q**2).max()
        # # g.A0 = dA.sum() / self.params.A0
        g.psam_vec = (self.mdl.docc_mask[np.newaxis,:,:] * dA[:,:,np.newaxis]).sum(axis=(1,0))
        # #print "dt params.copy=", t1
        # #_grad.data[:] = cyska.PSAM_mean_field_gradient(self)
        # #self.mdl.t_grad += time.time() - t0
        # #return _grad
        
        # db = -2 * (self.mdl.icsum[np.newaxis,:] / self.Q[:,np.newaxis] - 1/(self.Q**2)[:,np.newaxis] * self._pd).mean(axis=1)

        db = pre - self.mdl.icsum[np.newaxis,:] / self.Q[:,np.newaxis]
        db *= 2 * self._E * self.mdl.w
        g.betas = db.sum(axis=1)

        return g

class InvMeanFieldModel(object):
    def __init__(self, reads, params0, R0, rbp_conc=[], aff0=1e-6, **kwargs):
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.reads = reads
        self.params = params0
        self.R0 = R0
        self.w = R0 / R0.sum(axis=1)[:,np.newaxis] # R-value weight
        self.w = 1. / R0.size # uniform weights
        self.aff0 = aff0
        self.opt = None
        k = self.params.k
        self.k = k
        self.xm = CrosstalkMatrix(k, reads)
        x = np.diag(self.xm.M_inv).argmax()
        print cyska.index_to_seq(x, k), 'diag max'
        x = np.diag(self.xm.M_inv).argmin()
        print cyska.index_to_seq(x, k), 'diag min'
        print self.xm.M_inv.min(), self.xm.M.max(), np.median(self.xm.M)

        self.iR0 = np.array([np.dot(self.xm.M_inv.T, r) for r in self.R0], dtype=np.float32)

        rR0 = np.array([np.dot(self.xm.M.T, r) for r in self.iR0], dtype=np.float32)
        print "allclose?", np.allclose(rR0,R0)
        print np.fabs(rR0 - R0).max(),rR0.max()

        self.iI = np.array(np.dot(self.xm.M_inv.T, np.ones(R0.shape[1], dtype=np.float32)), dtype=np.float32)
        self.icsum = self.xm.M_inv.T.sum(axis=1)
        print "ICSUM", self.icsum.min(), self.icsum.max(), self.icsum.mean()
        docc = np.zeros( (4**k, 4*k+1), dtype=np.float32)
        for i in range(4**k):
            for d in range(k):
                n = (i >> 2*(k-1-d)) & 3
                x = (d << 2) + n + 1
                # print k, d, n, x
                docc[i,x] = 1
            docc[i,0] = 1
        self.docc_mask = docc
                
        #gcaug = cyska.seq_to_index('gcaug')
        #print 'iR0', self.iR0.shape
        #for ir in self.iR0:
            #print ir.min(), ir.mean(), ir.max(), cyska.index_to_seq(ir.argmax(), 5)
            

        #print 'iI', self.iR0.shape
        #print self.iI.min(), self.iI.mean(), self.iI.max(), cyska.index_to_seq(self.iI.argmax(), 5)

            
        #print 'iR0(GCAUG)'
        #for ir in self.iR0[:,gcaug]:
            #print ir, self.iI[gcaug]
            
        f0 = reads.kmer_frequencies(k)
        self.f0 = f0 / f0.sum()
        
        self.n_fev = 0
        self.t_fev = 0
        self.n_grad = 0
        self.t_grad = 0
        self.t_aff = 0

    def predict(self, params, aff0=1e-6, debug=False):
        state = InvMeanFieldModelState(self, params)
        return state

    def estimate_betas(self, state, q=5):
        R_ns = np.percentile(self.R0, q, axis=1)
        # print "R_ns,j", R_ns
        beta = R_ns / (1 - R_ns)
        print "beta?", beta
        print "sum_pi with current beta estimate", state.Q
        return state.Q * beta

    @property
    def affinities(self):
        state = self.predict(self.params)
        return state.A



        

        
if __name__ == "__main__":
    
    from cska.reads import RBNSReads
    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt')
    xm = CrosstalkMatrix(6, reads)
    occ = np.zeros(4**6)
    occ[cyska.seq_to_index('UGCAUG')] = .5
    occ[cyska.seq_to_index('AGCAUG')] = .3
    occ[cyska.seq_to_index('CGCAUG')] = .25
    occ[cyska.seq_to_index('GGCAUG')] = .05
    occ[cyska.seq_to_index('UGCACG')] = .1
    t0 = time.time()
    r = xm.r_values_from_occupancies(occ, .01)
    print time.time() - t0
    debug_kmer_vector(r)