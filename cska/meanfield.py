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
        

class MeanFieldModelState(object):
    def __init__(self, mdl, params):
        self.mdl = mdl
        self.params = params
        self.rbp_conc = mdl.rbp_conc

        self.A = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=mdl.aff0)
        # TODO: make protein concentration self-consistent
        Z = self.rbp_conc[:,np.newaxis]*self.A[np.newaxis,:]
        self.occ = Z / (Z + 1)
        # debug_kmer_vector(occ[0], header="occ @5nM")

        # adding baseline
        self.pi = np.array([self.mdl.f0 * (np.dot(self.mdl.xm.M, o ) + beta ) for o, beta in zip(self.occ, self.params.betas)])
        self.sum_pi = self.pi.sum(axis=1)
        self.R = self.pi / self.sum_pi[:,np.newaxis] / self.mdl.f0[np.newaxis,:]
        self.error = self.mdl.opt.error(self.R)

        self.mdl.n_fev += 1
        
    @property
    def grad(self):
        self.mdl.n_grad += 1
        _grad = self.params.copy()
        _grad.data[:] = cyska.PSAM_mean_field_gradient(self)
        return _grad
    

class MeanFieldModel(object):
    def __init__(self, reads, params0, rbp_conc=[], aff0=1e-6, **kwargs):
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.reads = reads
        self.params = params0
        self.aff0 = aff0
        self.opt = None
        # print "rbp_conc", rbp_conc, self.params.n, len(rbp_conc)
        k = self.params.k
        self.xm = CrosstalkMatrix(k, reads)
        
        f0 = reads.kmer_frequencies(k)
        self.f0 = f0 / f0.sum()
        
        self.n_fev = 0
        self.n_grad = 0
        #docc = np.zeros( (4**k, self.params.n), dtype=np.float32)
        #for i in range(4**k):
            #for d in range(k):
                #n = (i >> 2*(k-1-d)) & 3
                #x = (d << 2) + n + 1
                ## print k, d, n, x
                #docc[i,x] = 1

        #self.docc_mask = docc

    def predict(self, params, aff0=1e-6, debug=False):
        state = MeanFieldModelState(self, params)
        return state

        
        
    #def predict_R(self, params, grad=False, aff0=1e-6, debug=False):
        #A = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=aff0)
        #P = self.rbp_conc
        #Z = P[:,np.newaxis]*A[np.newaxis,:]
        ## print Z.shape, P.shape, A.shape
        #occ = Z / (Z + 1)
        ## debug_kmer_vector(occ[0], header="occ @5nM")
        
        #r_values = self.R0
        #t0 = time.time()
        #omegas = (occ * self.f0[np.newaxis,:]).sum(axis=1)
        ## TODO: make this self-consistent

        ## adding baseline
        
        #pi = np.array([self.f0 * (np.dot(self.xm.M, o ) + beta ) for o, beta in zip(occ, params.betas)])
        #sum_pi = pi.sum(axis=1)
        
        #R_predict = pi / sum_pi[:,np.newaxis] / self.f0[np.newaxis,:]

        #r_predict = np.array([self.xm.r_values_from_occupancies(o, beta) for o, beta in zip(occ, params.betas)])
        
        ## print np.allclose(R_predict, r_predict)
        ## print time.time() - t0, "seconds r_predict"

        #dR = None
        #if grad:
            #Nk = 4**self.params.k
            #docc = np.array([ (o - o**2)[:,np.newaxis] * A[:,np.newaxis] * self.docc_mask / params.as_vector() for o in occ])
            ## print "docc minmax", docc.min(), docc.max()
            ## i = docc.max(axis=2).max(axis=0).argmax()
            ## print "docc maximum at", cyska.index_to_seq(i,5)
            ## for do in docc[:,i,:]:
            ##     print do[1:21].reshape(5,4)

            #gcaug = cyska.seq_to_index('GCAUG')
            ## print "docc (GCAUG) @5nM"
            ## for do in docc[:1,gcaug][:,1:21].reshape(1,5,4):
            ##     print do

            ## print "SHADOW"
            ## for i, s in self.xm.get_shadow(gcaug):
            ##     print cyska.index_to_seq(i, 5), s, docc[0,i]

            ## print "HULL"
            ## for i, s in self.xm.get_hull(gcaug):
            ##     print cyska.index_to_seq(i, 5), s, docc[0,i]
                
            ## dpi = np.array([self.f0[:,np.newaxis] * np.tensordot(self.xm.M, do, axes=([0,],[0,])) for do in docc])
            #dpi = np.array([self.f0[:,np.newaxis] * np.dot(self.xm.M, do) for do in docc])
            ## print "dpi", dpi.shape
            ## print "dpi minmax", dpi.min(), dpi.max(), np.median(dpi)
            ## print "dpi (GCAUG)"
            ## for do in dpi[:,cyska.seq_to_index('GCAUG')][:,1:21].reshape(5,5,4):
            ##     print do

            ## sys.exit(0)

            #dR = np.array([1./p[:,np.newaxis] * (R[:,np.newaxis] * dp -  (R**2)[:,np.newaxis] * dp.sum(axis=0)[np.newaxis,:]) for p, dp, R in zip(pi, dpi, R_predict)])
            ## print "dR", dR.shape
            ## print "dR(GCAUG)", dR[:,cyska.seq_to_index('GCAUG')]
            ## print "dR minmax", dR.min(), dR.max(), np.median(dR)
            #dR_dbeta = r_predict / pi - self.f0[np.newaxis,:] * r_predict**2 / pi
            ## print "dR_dbeta", dR_dbeta.shape, dR_dbeta.min(), dR_dbeta.max()
            #for i in range(len(dR_dbeta)):
                #dR[i,:,self.params.betas_start+i] = dR_dbeta[i,:]
            ## print "dR_dbeta(GCAUG)", dR[:,gcaug,params.betas_start:]

        #return r_predict, dR
    
    #def grad_from_R(self, R1, dR, debug=False):
        #grad = 2 * ((1. * (R1 - self.R0))[:,:,np.newaxis] * dR).sum(axis=(1,0))
        #param_grad = self.params.copy().set_vector(grad)
        #return param_grad #.unity_bounded()

        
class MeanFieldAnalysis(object):
    def __init__(self, rbns, pwm, k):
        self.rbns = rbns
        self.k = k
        self.R, self.R_err = rbns.R_value_matrix(k)
        
        params = cska.gradient.ModelParametrization(k, len(rbns.reads) - 1, psam=pwm.psam, A0=.21)
        params.betas[:] = [.013,.023,.062,.18,.19]
        params.psam_matrix[3,1] = .1
        
        model = MeanFieldModel(rbns.reads[0], params, rbp_conc = rbns.rbp_conc)
        self.descent = cska.gradient.GradientDescent(model, params, self.R)
        state = model.predict(params)
        
        import matplotlib.pyplot as pp
        pp.figure()
        pp.loglog(self.R[0],state.R[0],'x')
        pp.loglog(self.R[1],state.R[1],'x')
        pp.loglog(self.R[2],state.R[2],'x')
        pp.loglog(self.R[3],state.R[3],'x')
        pp.savefig('unoptimized.pdf')
        pp.close()
        # pp.show()

        debug_kmer_vector(self.R[0], ref=state.R[0])
        debug_kmer_vector(self.R[1], ref=state.R[1])
        debug_kmer_vector(self.R[2], ref=state.R[2])
        debug_kmer_vector(self.R[3], ref=state.R[3])
        debug_kmer_vector(self.R[4], ref=state.R[4])
        print "error", state.error
        print "PARAMS"
        print params
        print "ANALYTICAL"
        print state.grad
        print "EMPIRICAL"
        print cska.gradient.emp_grad(state)
        self.descent.optimize(maxiter=100)
        print "OPTIMIZATION RESULTS"
        print self.descent.params

        state = model.predict(self.descent.params)
        import matplotlib.pyplot as pp
        pp.figure()
        pp.loglog(self.R[0],state.R[0],'x')
        pp.loglog(self.R[1],state.R[1],'x')
        pp.loglog(self.R[2],state.R[2],'x')
        pp.loglog(self.R[3],state.R[3],'x')
        pp.savefig('optimized.pdf')
        pp.close()


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