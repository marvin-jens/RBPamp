import cska.gradient
from cska.crosstalk_matrix import CrosstalkMatrix
import logging, os, sys
import cska.ska_kmers as cyska
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
        

class MeanFieldDescent(cska.gradient.GradientDescent):

    def __init__(self, *argc, **kwargs):
        rbp_conc = kwargs.pop('rbp_conc', [])
        kwargs['subsample'] = 0
        super(MeanFieldDescent, self).__init__(*argc, **kwargs)
        self.rbp_conc = np.array(rbp_conc)
        k = self.params.k
        self.xm = CrosstalkMatrix(k, argc[0])
        
        docc = np.zeros( (4**k, self.params.n), dtype=np.float32)
        for i in range(4**k):
            for d in range(k):
                n = (i >> 2*(k-1-d)) & 3
                docc[i,d << 2 + n + 1] = 1

        self.docc_mask = docc

    def predict_R(self, params, grad=False, aff0=1e-6):
        A = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=aff0)
        P = self.rbp_conc
        Z = P[:,np.newaxis]*A[np.newaxis,:]
        occ = Z / (Z + 1)
        debug_kmer_vector(occ[0], header="occ @5nM")
        
        r_values = self.R
        t0 = time.time()
        omegas = (occ * kappa[np.newaxis,:]).sum(axis=1)
        # TODO: make this self-consistent

        # adding baseline
        
        pi = np.array([self.xm.kappa * (np.dot(self.M, o ) + beta ) for o, beta in zip(occ, params.betas)])
        sum_pi = pi.sum(axis=1)
        
        R_predict = pi / sum_pi[np.newaxis,:] / kappa[np.newaxis,:]
        r_predict = np.array([self.xm.r_values_from_occupancies(o, beta) for o, beta in zip(occ, params.betas)])
        
        print np.allclose(R_predict, r_predict)
        print time.time() - t0, "seconds r_predict"

        dR = None
        if grad:
            Nk = 4**self.params.k
            docc = [ (o - o**2)[:,np.newaxis] * 1./A[np.newaxis,:] * self.docc_mask for o in occ]
            
            dpi = [self.xm.kappa * np.tensordot(self.xm.M, do, axes=([1,],[1,])) for do in docc]
            dR = np.array([1./p[:,np.newaxis] * (R[:,np.newaxis] * dp -  (R**2)[:,np.newaxis] * dp.sum(axis=0)[np.newaxis,:]) for p, dp, R in zip(pi, dpi, R_predict)])
            
        return r_predict, dR
    

        
class MeanFieldAnalysis(object):
    def __init__(self, rbns, pwm, k):
        self.rbns = rbns
        self.k = k
        self.R, self.R_err = rbns.R_value_matrix(k)
        
        params = cska.gradient.ModelParametrization(k, len(rbns.reads) - 1, psam=pwm.psam, A0=pwm.A0)
        params.betas[:] = .02
        
        self.descent = MeanFieldDescent(rbns.reads[0], params, self.R, k_monitor=k)
        R, dR = self.descent.predict_R(params, grad=True)
        
        #self.pwm = pwm
        #self.pwm.A0 = 1./1.6
        #print self.pwm
        
        #A = self.pwm.kmer_affinity_table()
        #debug_kmer_vector(A, header="affinities")
        
        #P = np.array(self.rbns.rbp_conc)
        #Z = P[:,np.newaxis]*A[np.newaxis,:]
        #occ = Z / (Z + 1)
        #debug_kmer_vector(occ[0], header="occ @5nM")
        
        #r_values = self.R
        #t0 = time.time()
        #r_predict = np.array([self.xm.r_values_from_occupancies(o, .02) for o in occ])
        #print time.time() - t0, "s"
        
        debug_kmer_vector(self.R[2], ref=R[2])
        #print occ.shape

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