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
        print "rbp_conc", rbp_conc, self.params.n, len(rbp_conc)
        k = self.params.k
        self.xm = CrosstalkMatrix(k, argc[0])
        
        docc = np.zeros( (4**k, self.params.n), dtype=np.float32)
        for i in range(4**k):
            for d in range(k):
                n = (i >> 2*(k-1-d)) & 3
                x = (d << 2) + n + 1
                # print k, d, n, x
                docc[i,x] = 1

        print "docc_mask(GCAUG)"
        print  docc[cyska.seq_to_index('GCAUG')][1:21].reshape(5,4)
        self.docc_mask = docc

    def predict_R(self, params, grad=False, aff0=1e-6, debug=False):
        A = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=aff0)
        P = self.rbp_conc
        Z = P[:,np.newaxis]*A[np.newaxis,:]
        print Z.shape, P.shape, A.shape
        occ = Z / (Z + 1)
        debug_kmer_vector(occ[0], header="occ @5nM")
        
        r_values = self.R0
        t0 = time.time()
        omegas = (occ * self.f0[np.newaxis,:]).sum(axis=1)
        # TODO: make this self-consistent

        # adding baseline
        
        pi = np.array([self.f0 * (np.dot(self.xm.M, o ) + beta ) for o, beta in zip(occ, params.betas)])
        sum_pi = pi.sum(axis=1)
        
        R_predict = pi / sum_pi[:,np.newaxis] / self.f0[np.newaxis,:]

        r_predict = np.array([self.xm.r_values_from_occupancies(o, beta) for o, beta in zip(occ, params.betas)])
        
        print np.allclose(R_predict, r_predict)
        print time.time() - t0, "seconds r_predict"

        dR = None
        if grad:
            Nk = 4**self.params.k
            docc = np.array([ (o - o**2)[:,np.newaxis] * A[:,np.newaxis] * self.docc_mask / params.as_vector() for o in occ])
            print "docc minmax", docc.min(), docc.max()
            i = docc.max(axis=2).max(axis=0).argmax()
            print "docc maximum at", cyska.index_to_seq(i,5)
            for do in docc[:,i,:]:
                print do[1:21].reshape(5,4)

            # gcaug = cyska.seq_to_index('GCAUG')
            # print "docc (GCAUG) @5nM"
            # for do in docc[:1,gcaug][:,1:21].reshape(1,5,4):
            #     print do

            # print "SHADOW"
            # for i, s in self.xm.get_shadow(gcaug):
            #     print cyska.index_to_seq(i, 5), s, docc[0,i]

            # print "HULL"
            # for i, s in self.xm.get_hull(gcaug):
            #     print cyska.index_to_seq(i, 5), s, docc[0,i]
                
            # dpi = np.array([self.f0[:,np.newaxis] * np.tensordot(self.xm.M, do, axes=([0,],[0,])) for do in docc])
            dpi = np.array([self.f0[:,np.newaxis] * np.dot(self.xm.M, do) for do in docc])
            # print "dpi", dpi.shape
            # print "dpi minmax", dpi.min(), dpi.max(), np.median(dpi)
            # print "dpi (GCAUG)"
            # for do in dpi[:,cyska.seq_to_index('GCAUG')][:,1:21].reshape(5,5,4):
            #     print do

            # sys.exit(0)

            dR = np.array([1./p[:,np.newaxis] * (R[:,np.newaxis] * dp -  (R**2)[:,np.newaxis] * dp.sum(axis=0)[np.newaxis,:]) for p, dp, R in zip(pi, dpi, R_predict)])
            # print "dR", dR.shape
            # print "dR(GCAUG)", dR[:,cyska.seq_to_index('GCAUG')]
            # print "dR minmax", dR.min(), dR.max(), np.median(dR)
            

        return r_predict, dR
    
    def grad_from_R(self, R1, dR, debug=False):
        grad = 2 * ((1. * (R1 - self.R0))[:,:,np.newaxis] * dR).sum(axis=(1,0))
        param_grad = self.params.copy().set_vector(grad)
        return param_grad #.unity_bounded()

        
class MeanFieldAnalysis(object):
    def __init__(self, rbns, pwm, k):
        self.rbns = rbns
        self.k = k
        self.R, self.R_err = rbns.R_value_matrix(k)
        
        params = cska.gradient.ModelParametrization(k, len(rbns.reads) - 1, psam=pwm.psam, A0=pwm.A0)
        params.betas[:] = .035
        
        self.descent = MeanFieldDescent(rbns.reads[0], params, self.R, k_monitor=k, rbp_conc = rbns.rbp_conc)
        self.descent.set_reference(self.R)
        R, dR = self.descent.predict_R(params, grad=True)
        print self.descent.grad_from_R(R, dR).unity()
        # print self.descent.ana_grad(params)
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