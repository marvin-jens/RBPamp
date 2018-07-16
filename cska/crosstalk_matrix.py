import numpy as np
import logging
import scipy
import logging
import time
import sys
import os
from collections import defaultdict
#from scipy.optimize import minimize, brentq, minimize_scalar
import cska.cyska as cyska
from cska.caching import CachedBase, cached, pickled
#from cska.rbns_reads import RBNSReads


class CrosstalkMatrix(CachedBase):
    def __init__(self, k, input_reads):
        CachedBase.__init__(self)

        self.k = k
        self.l = input_reads.L # oligo size
        self.input_reads = input_reads
        self.kappa = self.input_reads.kmer_frequencies(k) / 4**k
        
        self.logger = logging.getLogger("CrosstalkMatrix")
        

    @property
    def cache_key(self):
        return "crosstalkmatrix_k={0}_{1}".format(self.k, self.input_reads.cache_key)

    @property
    @cached
    @pickled
    def M(self):
        k = self.k
        N = 4**k
        
        M = np.zeros((N,N), dtype=np.float32)
        
        # kmer overlap extension frequencies
        kfreqs = {}
        for x in range(1,k):
            kfreqs[x] = self.input_reads.kmer_frequencies(x) / 4**x
        
        t0 = time.time()        
        for i in np.arange(N):
            M[i,i] = 1
            for x in range(1,k):
                weights, shifts = cyska.weighted_kmer_shifts(i, k, self.l, x, kfreqs[x]) 
                for s,f in zip(shifts, weights):
                    M[i,s] += f

        t1 = time.time()
        self.logger.debug("computed crosstalk matrix for k={0} in {1:.3f}s".format(self.k, t1-t0) )
        return M

    @property
    @cached
    @pickled
    def wrm(self):
        """return abundance weighted row-mean"""
        return (self.kappa * self.M).sum(axis=0) / self.kappa.sum()
    
    @property
    @cached
    @pickled
    def M_inv(self):
        M = self.M
        t1 = time.time()
        M_inv = np.linalg.inv(M)
        t2 = time.time()
        self.logger.debug("inverted crosstalk matrix for k={0} in {1:.3f}s".format(self.k, t2-t1) )
        
        return M_inv


    @property
    def sparseness(self):
        M = self.M
        nzero = (M == 0).nonzero()[0].size
        return nzero / float(M.size)


    def get_shadow(self, kmer_index):
        row = (self.M - np.identity(4**self.k))[kmer_index]
        shadow = [(i, row[i]) for i in row.argsort()[::-1] if row[i] > 0]
        return shadow

    def get_inv_shadow(self, kmer_index):
        row = (self.M_inv.T - np.identity(4**self.k))[kmer_index]
        shadow = [(i, row[i]) for i in row.argsort()[::-1] if row[i] > 0]
        return shadow
    
    def matrix_plot(self, fname='crosstalk_matrix.pdf'):
        M = self.M
        M_inv = self.M_inv

        import matplotlib.pyplot as pp
        pp.figure(figsize=(8,4))
        pp.subplot(121)
        pp.imshow(np.log10(M), cmap=pp.get_cmap('plasma'))
        pp.colorbar(label=r'$\log_{10}(M)$',fraction=0.046, pad=0.04)

        pp.subplot(122)
        pp.imshow(np.log10(M_inv), cmap=pp.get_cmap('plasma'))
        pp.colorbar(label=r'$\log_{10}(M^{-1})$',fraction=0.046, pad=0.04)
        pp.tight_layout()
        
        self.logger.info("storing figure in '{0}'".format(fname) )
        pp.savefig(fname)

    def r_values_from_occupancies(self, occ, bg=None):
        k = self.k
        kappa = self.kappa
        omega = (occ * kappa).sum()

        # adding baseline
        l = self.l - k + 1
        if bg == None:
            bg = omega*(l-2.*(k-1))
        
        pi = kappa * (np.dot(self.M, occ ) + bg )
        sum_pi = pi.sum()
        
        r = pi / sum_pi / kappa
        return r
    

    def o_values(self, r_values, sum_pi=1., beta=1.):
        k = self.k
        
        M = self.M
        M_inv = self.M_inv
        
        r_inv = np.dot(M_inv, r_values)
        bg_vec = np.dot(M_inv,np.ones(4**k))
        
        occ = r_inv * sum_pi - bg_vec * beta
        
        return occ
        
        
    def fit_occupancies(self, r_values):
        k = self.k
        
        M = self.M
        M_inv = self.M_inv
        
        r_inv = np.dot(M_inv, r_values)
        bg_vec = np.dot(M_inv,np.ones(4**k))
        
        kappa = self.input_reads.kmer_frequencies(k)
        kappa /= kappa.sum()

        def r2occ(sum_pi, beta):
            occ = r_inv * sum_pi - bg_vec * beta
            return occ
        
        def occ2r(occ, beta):
            omega = (occ * kappa).sum()

            # adding baseline
            #l = self.l - k + 1
            #print M.shape, occ.shape
            pi = kappa * (np.dot(M, occ ) + beta )
            sum_pi = pi.sum()
            
            r = pi / sum_pi / kappa
            return r, sum_pi
            
        def err(r_predict, r_obs):
            return np.mean((r_predict - r_obs)**2)
            #return np.mean(np.log2(np.fabs(r_predict/r_obs))**2)
        
        def score(params):
            sum_pi, beta = params
            
            occ = r2occ(sum_pi, beta)
            print "occ top5", occ[-5:]
            r_predict, sum_pi_pred = occ2r(occ, beta)
            print "sum_pi", sum_pi, "sum_pi_pred", sum_pi_pred
            print "r_predict top5", r_predict[-5:]
            print "r_value   top5", r_values[-5:]
            
            e = err(r_predict, r_values) # super close to zero
            print "error r_predict - r_values", e
            e += (occ[occ < 0].sum())**2
            e += (occ[occ > 1].sum())**2 
            e += np.log(sum_pi/sum_pi_pred)**2

            print "score({0},{1}) -> err={2}".format(sum_pi, beta, e) 
            print occ[-20:]
            return e 
        
        from scipy.optimize import minimize
        x0 = (0.01, 0.0001)
        bounds = np.array([
            [1e-6, .9],
            [1e-6, .9]
        ])
        res = minimize(score, x0, method='SLSQP', bounds = bounds)
        print res
        sum_pi, beta = res.x
        print score(res.x), "<- optimized score"
        
        
        opt_occ = r2occ(sum_pi, beta)
        #r_pred = occ2r()
        return opt_occ, beta


    def linear_fit(self, rbp_conc, r_matrix, indices, temp=22):
        k = self.k
        n = len(rbp_conc)

        # all the expensive matrix multiplications are 
        # done *once*
        
        M = self.M
        M_inv = self.M_inv
        r_inv = np.array([np.dot(M_inv, r) for r in r_matrix])
        bg_vec = np.dot(M_inv,np.ones(4**k))
        
        I = indices
        
        kmers = np.array(list(cyska.yield_kmers(k)))[I]
        
        r_inv = r_inv[:,I]
        bg_vec = bg_vec[I]
        

        def infer_lkd(sum_pi, bg):
            occ = r_inv * sum_pi[:,np.newaxis] - bg_vec[np.newaxis,:] * bg[:,np.newaxis]
            
            kd = rbp_conc[:,np.newaxis] * (1/occ - 1)
            return np.log(np.where(kd > 0, kd, 1e8))
        
        
        def objective_function(k_est):
            #k_mean = k_est.mean(axis=0)
            k_mean = np.median(k_est, axis=0)
            
            #err_weight = 1./k_mean
            #err_weight /= err_weight.sum()
            
            S = ((k_est - k_mean[np.newaxis,:])**2).mean()
            
            #n = len(k_est)
            #S = (err_weight[np.newaxis,:] * np.log(k_est / k_mean[np.newaxis,:])**2).sum() / n
            #S = (np.log(k_est / k_mean[np.newaxis,:])**2).sum() / n
            return S


        def score(x):
            sum_pi = x[:n]
            bg = x[n:]
            
            lkd = infer_lkd(sum_pi, bg)
            
            #err = RBNSKmerModel.objective_function(kd)
            err = objective_function(lkd)
            
            print "score(",sum_pi,bg,")-> err=",err
            return err

        
        def best_fit():
            from scipy.optimize import minimize, basinhopping
            x0 = np.ones(2*n, dtype=float)*0.01
            bounds = np.ones((2*n,2), dtype=float) * 0.1
            bounds[:,0] = 1e-3
            #min_dict = dict(bounds = bounds, method='SLSQP')
            #res = basinhopping(score, x0, minimizer_kwargs=min_dict)
            
            res = minimize(score, x0, method='SLSQP', bounds = bounds)#, options=dict(eps=1e-3))
            print res
            x = res.x
            print score(x), "<- optimized score"
            sum_pi = x[:n]
            bg = x[n:]
            
            lkd = infer_lkd(sum_pi, bg)
            
            return np.exp(lkd), sum_pi, bg
        

        kd, sum_pi, bg = best_fit()
        dG = Kd_to_kcal(kd, temp=temp).mean(axis=0)
        
        return kmers, kd, dG

if __name__ == "__main__":
    from cska.reads import RBNSReads
    import cska.cyska as cyska
    import logging
    logging.basicConfig(level=logging.DEBUG)
    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', rbp_name='RBFOX3', storage_kw=dict(T=4))
    pd = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_80.txt', rbp_name='RBFOX3', storage_kw=dict(T=4))

    
    def kmer_increase_weights(reads, k):
        im1 = reads.get_index_matrix(k)
        im2 = reads.get_index_matrix(k+1)
        #print im1
        #print im2
        #import cska.cyska as cyska
        M = cyska.kmer_crosstalk_matrix(im1, im2, k, k+1)
        return M
    k = 5
    mer = 'UGCAUG'
    ind = cyska.seq_to_index(mer)
    
    M = kmer_increase_weights(reads, k)
    P = kmer_increase_weights(pd, k)
    contrib = P[:,ind]
    #print contrib.sum()
    #I = contrib.nonzero()[0]
    I = contrib.argsort()[::-1]
    

    Z1 = M[:,ind].max()
    Z2 = P[:,ind].max()

    MAX = M.max(axis=0)
    W = M / MAX[np.newaxis,:] # normalize weights to maximal contribution of 1
    print (M[:,ind] / Z1).sum()
    print (P[:,ind] / Z2).sum()
    for i in I[:20]:
        x = M[i,ind] / Z1
        y = P[i,ind] / Z2
        print cyska.index_to_seq(i,5), x, y, y/x 
    
    from cska.spa import SPAModel
    mdl = SPAModel(reads, 5, [5,20,80,320,1300], T=4, n_subsample=0)
    state = mdl.evaluate(params = mdl.load_params('/scratch/data/RBNS/RBFOX3/5mer_affinities.tsv'), keep=True)
    aff = state.params[:mdl.nA]
    #M = CrosstalkMatrix(k, reads)
    #P = CrosstalkMatrix(k, pd)
    aff2 = np.dot(aff, W)

    #print "shadow"
    #for i, w in P.get_shadow(ind)[:10]:
        #print cyska.index_to_seq(i,k), w
    
    #print "inv. shadow"
    #for i, w in M.get_inv_shadow(ind)[:10]:
        #print cyska.index_to_seq(i,k), w
    
    
    
    #print "R-value of crosstalk"
    #in_slice = M.M[ind]
    #pd_slice = P.M[ind]
    #print in_slice
    #print pd_slice
    #R = np.where(in_slice > 0, pd_slice / in_slice, 0)
    #R = pd_slice - in_slice
    #print R
    #for i in R.argsort()[:10]:
        ##w = 
        #print cyska.index_to_seq(i,k), R[i], in_slice[i], pd_slice[i]
