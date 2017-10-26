import numpy as np
import scipy
import logging
import time
import sys
import os
from collections import defaultdict
from scipy.optimize import minimize, brentq, minimize_scalar

import cska.ska_kmers 

from cska.rbns_reads import RBNSReads

from cska.caching import CachedBase, cached, pickled
#<<<<<<< Updated upstream
#=======
#from cska.affinity import Kd_to_kcal, kcal_to_Kd, AffinityDistribution
#from cska.crosstalk_matrix import CrosstalkMatrix
#from cska.simulator import RBNSGenerator, RBNSSimulator                             
#from cska.kmersoup import RBNSKmerModel
#from cska.scheduler import ParamUpdateScheduler
#from cska.psam import PSAMState
#from cska import timed
#from cska.optimize import ModelOptimization
#>>>>>>> Stashed changes

def Kd_to_kcal(K,temp=22):
    RT = (temp + 273.15) * 8.314459848# RT in Joules/mol
    kcal = 4.184E3 # kcal in Joules
    # Kd in nM to E in kcal/mol
    return np.log(K/1E9)*RT/kcal


def kcal_to_Kd(E,temp=22):
    RT = (temp + 273.15) * 8.314459848# RT in Joules/mol
    kcal = 4.184E3 # kcal in Joules
    #print "1./RT",kcal/RT
    # kcal/mol to Kd in nM
    return np.exp(E*kcal/RT)*1e9

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
                weights, shifts = cska.ska_kmers.weighted_kmer_shifts(i, k, self.l, x, kfreqs[x]) 
                for s,f in zip(shifts, weights):
                    M[i,s] += f

        t1 = time.time()
        self.logger.debug("computed crosstalk matrix for k={0} in {1:.3f}s".format(self.k, t1-t0) )
        return M

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
        
        kappa = self.input_reads.kmer_frequencies(k) / 4**k

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
        return opt_occ


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
        
        kmers = np.array(list(cska.ska_kmers.yield_kmers(k)))[I]
        
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

class AffinityDistribution(object):
    def __init__(self, invkd, T=22):
        self.RT = (T + 273.15) * 8.314459848 / 4.184E3 # RT in kcal/mol
        self.invkd = invkd
        
    @classmethod
    def singleton(cls, best = 'GCATG', best_kd=1., bg_kd=1e6, T = 22):
        k = len(best)
        invkd = np.ones(4**k, dtype=np.float32) / bg_kd

        best_i = cska.ska_kmers.seq_to_index(best)
        invkd[best_i] = 1./best_kd
        
        return cls(invkd, T=T)
    
    @classmethod
    def doublet(cls, a = 'GCATG', b='GCACG', a_kd=1., b_kd=20., bg_kd=1e6, T=22):
        k = len(best)
        invkd = np.ones(4**k, dtype=np.float32) / bg_kd

        invkd[cska.ska_kmers.seq_to_index(a)] = 1./a_kd
        invkd[cska.ska_kmers.seq_to_index(b)] = 1./b_kd
        
        return cls(invkd, T=T)
        
class RBNSGenerator(CachedBase):
    def __init__(self, k, l=20, min_E=-11., seed=None, temp=22, mode='ordered', **kwargs):
        
        CachedBase.__init__(self)

        self.k = k
        self.l = l
        self.min_E = min_E
        self.temp = temp
        self.RT = (temp + 273.15) * 8.314459848 / 4.184E3 # RT in kcal/mol

        if seed: 
            np.random.seed(seed)
            cska.ska_kmers.rand_seed(seed)

        raw_energies = RBNSGenerator.energy_distribution(min_E=min_E, N=4**k, **kwargs)
        if mode == 'ordered':
            self.kmer_energies = np.array(sorted(raw_energies)[::-1], dtype=np.float32) / self.RT
        else:
            self.kmer_energies = np.array(raw_energies, dtype=np.float32) / self.RT
            
        self.kmer_invkd = np.exp(-self.kmer_energies)*1e-9
        self.logger = logging.getLogger("RBNSGenerator")
        self.input_reads = None

    def energy_plot(self, store="kmer_energies.pdf"):
        import matplotlib.pyplot as pp
        pp.figure()
        y, bins = np.histogram(self.kmer_energies*self.RT,bins=50)
        print bins.shape, y.shape
        pp.semilogx(kcal_to_Kd(bins[:-1]), y, linestyle='steps', linewidth=2.)
        pp.xlabel(r'$K_d$ [nM]')
        #pp.xlabel(r'$\Delta G$ [kcal/mol]')
        pp.ylabel('frequency')
        pp.savefig(store)

        ##pp.figure()
        #pp.hist(kcal_to_Kd(self.kmer_energies*self.RT),bins=100)
        #pp.xlabel(r'$K_d$ [nM]')
        #pp.ylabel('frequency')
        

    def store_invKd(self, fname):
        """
        write flat file with kmer 1./Kd as neede by RBPbind
        """
            
        invKd = 1./(np.exp(self.kmer_energies)*1e9)
        with file(fname,'w') as f:
            f.write(" Motifs invKd\n===================\n")
            for kmer, ikd in zip(cska.ska_kmers.yield_kmers(self.k), invKd):
                f.write("{0} {1}\n".format(kmer, ikd))
        

    def __str__(self):
        top_kmers = []
        
        for i in self.kmer_energies.argsort()[:10]:
            kmer = cska.ska_kmers.index_to_seq(i, self.k)
            E = self.kmer_energies[i]* self.RT
            top_kmers.append( "{0}\t{1}\t{2}".format(kmer, E, kcal_to_Kd(E*self.RT, self.temp) ) )
            
        return "\n".join(top_kmers)

    @staticmethod
    def energy_distribution(mu=10., sigma=.2, min_E = -11., N=1024, temp=22.):
        E = - np.random.lognormal(10., sigma, N) 
        E -= E.mean()
        E *= min_E / E.min()
        return E

    @staticmethod
    def Kd_distribution(**kwargs):
        return kcal_to_Kd(RBNSGenerator.energy_distribution(**kwargs))

    @property
    def Kd(self):
        return np.exp(self.kmer_energies)*1e9
        
    def assign_experimental_input(self, real_input, pseudo_count=0):
        from cska.rbns_reads import RBNSReads
        reads = RBNSReads(real_input, pseudo_count=pseudo_count)
        nt_freq = reads.kmer_frequencies(1) / 4.
        di_freq = reads.kmer_frequencies(2).reshape(4,4) / 16.
        di_freq /= di_freq.sum(axis=1)[:, np.newaxis] # normalize rows to one
        
        self.input_reads = reads
        self.input_nt_freq = nt_freq
        self.input_di_freq = di_freq
        
        return nt_freq, di_freq
        
    def generate_input_reads(self, N=20000000, store=""):
        if self.input_reads:
            self.logger.debug("generating random sequence matrix, mimicking '{0}'".format(self.input_reads.fname))
            t0 = time.time()
            seqm = cska.ska_kmers.generate_random_sequence_matrix_dinuc(self.l, N, self.input_nt_freq, self.input_di_freq)
            dt = time.time() - t0
            
            rps = N / dt
            self.logger.debug("took {0:.3f} seconds. {1:.1f} reads per second".format(dt, rps) )

        else:
            self.logger.debug("generating random sequence matrix")

            t0 = time.time()
            seqm = cska.ska_kmers.generate_random_sequence_matrix(self.l, N)
            dt = time.time() - t0
            
            rps = N / dt
            self.logger.debug("took {0:.3f} seconds. {1:.1f} reads per second".format(dt, rps) )

        if store:
            cska.ska_kmers.write_seqm(seqm, file(store, 'w') )

        return seqm
   
    def generate_bound_reads(self, N=20000000, P=320., p_ns=0.01, real_input="", store=""):
        self.logger.debug("simulating bound sequence matrix, mimicking input from '{0}'".format(self.input_reads.fname))
        
        t0 = time.time()
        seqm, bound_fraction = cska.ska_kmers.simulate_rbns_reads(self.l, N, self.k, self.input_nt_freq, self.input_di_freq, self.kmer_energies, P, p_ns)
        # Hacky wacky just to debug and troubleshoot!
        #seqm, self.input_kmer_counts, self.pd_kmer_weights, self.pd_kmer_counts, self.bound_fraction, self.Z_full = cska.ska_kmers.simulate_rbns_reads(self.l, N, self.k, self.input_nt_freq, self.input_di_freq, self.kmer_energies, P, p_ns)
        
        dt = time.time() - t0
        
        print "bound fraction", bound_fraction #, "non-specific", n_ns
        rps = N / dt
        self.logger.debug("took {0:.3f} seconds. {1:.1f} reads per second".format(dt, rps) )
        
        if store:
            cska.ska_kmers.write_seqm(seqm, file(store, 'w') )

        reads = RBNSReads(store, seqm=seqm, rbp_conc=P, n_subsamples=0, pseudo_count=0)
        # simulation results should be temporary or we run into trouble!
        reads._do_not_unpickle = True
        reads._do_not_pickle = True
        return reads
        #return seqm

    @cached
    def boltzmann_weights(self, P=320.):
        # chemical potential in units of RT. 
        # $\mu = \log(c/c_0)$ c_0=1M, P is in nM, hence 1e-9
        mu = np.log(P*1e-9) 
        
        w = np.exp(- self.kmer_energies + mu)
        return w
        
    def predict_occupancies(self, P=320., store=""):
        Kd = np.exp(self.kmer_energies)*1e9
        occ = P / (P+Kd)
        
        if store:
            with file(store,'w') as f:
                for kmer, o in zip(cska.ska_kmers.yield_kmers(self.k), occ):
                    f.write("{0}\t{1}\n".format(kmer, o) )
            
        return occ
    
    def predict_r_values(self, P=320., store="", limit=True):
        occ = self.predict_occupancies(P)
        r, pi_sum, beta = self.predict_r_values_from_occ(occ, limit=limit)
        
        if store:
            with file(store,'w') as f:
                for kmer, o in zip(cska.ska_kmers.yield_kmers(self.k), r):
                    f.write("{0}\t{1}\n".format(kmer, o) )

        return r, pi_sum, beta
    
    def predict_r_values_from_occ(self, occ, store="", limit=True):
        k = self.k
        kappa = self.input_reads.kmer_frequencies(k) / 4**k
        omega = (occ * kappa).sum()
        #print "omega", omega
        l = self.l - self.k + 1
        beta = omega*(l-2.*(self.k) + 1)
        
        M, M_inv = self.crosstalk_matrix_and_inverse()
        
        
        vec = (np.dot(M, occ) + beta)
        print "lin approx max expected binding", vec.max()
        if limit:
            #vec = np.where(vec > 1, 1, vec)  # hard cap
            #vec = vec / (vec + 2.) # sigmoidal
            
            vec = np.arcsinh(vec*1.5 ) # log-like for high values
            print "lin approx max binding regularized", vec.max()
            
        pi = kappa * vec
            
        r = pi / pi.sum() / kappa
        return r, pi.sum(), beta
    

    def binding_constants_from_R_vec(self, r, k, P=320., limit=True):
        # WORK IN PROGRESS!
        print "correct Kd", self.Kd[-5:]

        M, M_inv = self.crosstalk_matrix_and_inverse()
        k = self.k
        kappa = self.input_reads.kmer_frequencies(k) / 4**k

        
        r_inv = np.dot(M_inv, r)
        ofs = np.dot(M_inv, np.ones(r.shape))

        def get_occ(x):
            a,b = x
            return a*(r_inv + b)
        
        def err(x):
            r_pred = self.predict_r_values_from_occ(get_occ(x))
            E = ((r - r_pred)**2).mean()
            print x, E
            return E
        
        from scipy.optimize import minimize
        res = minimize(err, (.01,.01))
        print res.success, res.x
        
        inv_occ = get_occ(res.x)
        
        #import matplotlib.pyplot as pp
        #pp.figure()
        #pp.plot(self.predict_occupancies(P=P), inv_occ, 'ob')
        #pp.show()
        
    
    @cached
    def crosstalk_matrix_and_inverse(self, store=""):
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
                weights, shifts = cska.ska_kmers.weighted_kmer_shifts(i, k, self.l, x, kfreqs[x]) 
                for s,f in zip(shifts, weights):
                    M[i,s] += f

        t1 = time.time()
        M_inv = np.linalg.inv(M)
        t2 = time.time()
        self.logger.debug("computed crosstalk matrix for k={0} in {1:.3f}s, inverted in {2:.3f}s".format(self.k, t1-t0, t2-t1) )
        return M, M_inv
    
    
    def linear_fit(self, rbp_conc, r_matrix, n_top=20):

        k = self.k
        n = len(rbp_conc)

        def correct_parameters():
            corr_sum_pi = []
            corr_bg = []
            for P in rbp_conc:
                occ = self.predict_occupancies(P)
                kappa = self.input_reads.kmer_frequencies(k) / 4**k
                omega = (occ * kappa).sum()

                # adding baseline
                l = self.l - k + 1
                bg = omega*(l-2.*(k-1)) #* kappa 
                corr_bg.append(bg)

                # predict amount in pulldown
                pi = kappa * (np.dot(M, occ ) + bg )
                sum_pi = pi.sum()
                corr_sum_pi.append(sum_pi)
        
            return np.array(corr_sum_pi), np.array(corr_bg)
        
        # all the expensive matrix multiplications are 
        # done *once*
        
        M, M_inv = self.crosstalk_matrix_and_inverse()
        r_inv = np.array([np.dot(M_inv, r) for r in r_matrix])
        bg_vec = np.dot(M_inv,np.ones(4**k))
        
        #occ_recover = r_inv * sum_pi - bg_vec * bg 
        
        
        I = np.arange(4**k - n_top, 4**k)
        
        kmers = np.array(list(cska.ska_kmers.yield_kmers(k)))[I]
        
        r_inv = r_inv[:,I]
        bg_vec = bg_vec[I]
        

        def infer_lkd(sum_pi, bg):
            occ = r_inv * sum_pi[:,np.newaxis] - bg_vec[np.newaxis,:] * bg[:,np.newaxis]
            
            kd = rbp_conc[:,np.newaxis] * (1/occ - 1)
            return np.log(np.where(kd > 0, kd, 1e8))
        
        
        sp0, bg0 = correct_parameters()
        print "correct parameters are", sp0, bg0
        #kd = infer_kd(sp0, bg0)


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
            x0 = np.ones(2*n, dtype=float)*0.02
            bounds = np.ones((2*n,2), dtype=float) * 0.1
            bounds[:,0] = 1e-3
            min_dict = dict(bounds = bounds, method='L-BFGS-B')
            x_best = np.concatenate( (sp0, bg0) )
            
            print score(x_best), "<- best score"
            res = minimize(score, x_best, method='SLSQP', bounds = bounds)#, options=dict(eps=1e-3))
            #res = basinhopping(score, x0, minimizer_kwargs=min_dict)
            print res
            x = res.x
            print score(x), "<- optimized score"
            sum_pi = x[:n]
            bg = x[n:]
            
            lkd = infer_lkd(sum_pi, bg)
            
            return lkd
        
        kd = np.exp(best_fit())
            
        #import matplotlib.pyplot as pp
        #pp.figure()
        #pp.loglog(occ, occ_recover,'xr')
        #pp.show()
        print "correct parameters are", sp0, bg0
        print kd
        dG = self.RT * np.log(kd/1e9).mean(axis=0)
        
        return kmers, kd, dG

class RBNSSimulator(CachedBase):
    def __init__(self, reads, openen, k, temp=22.):
        CachedBase.__init__(self)
        
        self.logger = logging.getLogger("RBNSSimulator")
        self.temp = temp
        self.RT = (temp + 273.15) * 8.314459848/4.184E3
        self.reads = reads
        if openen.disc:
            self.openen = openen
        else:
            # we need to discretize first
            self.openen = openen.discretize()
            self.openen.store()

        self.openen_lookup = self.openen.disc.x/self.RT # open energies in units of RT for each discretization level
        self.acc_lookup = np.exp(-self.openen_lookup)
        
        self.k = k
        self.L = reads.L

    @property
    def cache_key(self):
        return "RBNSSimulator {self.reads.cache_key} {self.openen.cache_key} {self.k}".format(self=self)
    
    #@pickled
    #def expected_kmer_counts(self, kmer_energies, protein_conc, n_max=0, E_ns = 0, seqm=None):
        #from cska.ska_kmers import eval_energy_model_on_seqs
        #if seqm == None:
            #seqm = self.reads.seqm
        #p_bound, kmer_count_matrix, openen_kmer_bincount_matrix = eval_energy_model_on_seqs(seqm, self.openen.oem, self.openen_lookup, kmer_energies, np.array(protein_conc, dtype=np.float32), self.k, E_ns = E_ns, n_max=n_max)
        #return p_bound, kmer_count_matrix, openen_kmer_bincount_matrix

    @pickled
    def expected_kmer_counts(self, kmer_invkd, protein_conc, n_max=0, E_ns = 0, indices=None, seq_only=False):
        from cska.ska_kmers import eval_energy_model_on_seqs
        if indices == None:
            seqm = self.reads.seqm
            oem = self.openen.oem
        else:
            seqm = self.reads.seqm[indices]
            oem = self.openen.oem[indices]

        if seq_only:
            acc_lookup = np.ones(self.acc_lookup.shape, dtype = np.float32)
        else:
            acc_lookup = self.acc_lookup
            
        t0 = time.time()
        p_bound, kmer_count_matrix, openen_kmer_bincount_matrix, jacobi = eval_energy_model_on_seqs(seqm, oem, acc_lookup, kmer_invkd, np.array(protein_conc, dtype=np.float32), self.k, E_ns = E_ns, n_max=n_max)
        t1 = time.time()
        if n_max:
            n = n_max
        else:
            n = len(seqm)
        self.logger.debug("evaluated energy model on {0} sequences in {1:.2f} ms".format(n, 1000*(t1-t0)) )
        return p_bound, kmer_count_matrix, openen_kmer_bincount_matrix
                             
class KmerSoupModel(object):
    def __init__(self, k=5, min_E = -11., temp=22., seed=None):
        self.k=5
        self.min_E = min_E
        if seed: np.random.seed(seed)
        self.Kd = np.array(sorted(Kd_distribution(min_E = self.min_E, N=4**k, temp=temp)))
        
    def occ(self, P):
        return P[:,np.newaxis] / (P[:,np.newaxis] + self.Kd[np.newaxis,:])
    
    def specific_ratio(self, P, beta):
        occ = self.occ(P)
        omega = occ.mean(axis=1)
        
        return omega / beta
        
    def enrichment(self, P, beta):
        occ = self.occ(P)
        omega = occ.mean(axis=1)
        
        return beta / (beta + omega[:,np.newaxis]) + occ / (beta + omega[:,np.newaxis])
    
    def estimate_k(self, P, beta, r, r_ns):
        return P / (beta * (r/r_ns - 1)) - P

class RBNSKmerModel(object):
    def __init__(self, kmers, r_matrix, rbp_conc, rna_conc=1000., temp=22., unfolding_energies=None, tracked_kmers=[], out_path= './', rbp_name="RBP"):
        """
        kmers and r_matrix should be sorted from most to least strongly bound kmer
        """
        self.k = len(kmers[0])
        #self.kmers = cska.ska_kmers.yield_kmers(k)
        self.kmers = kmers
        self.r = r_matrix
        self.n_conc, self.n_kmers = r_matrix.shape
        self.rbp_name = rbp_name
        self.rbp_conc = rbp_conc
        self.rna_conc = rna_conc
        self.temp = temp
        self.unfolding = unfolding_energies
        self.tracked_kmers = {}
        self.track_kmers(tracked_kmers)
        self.out_path = out_path

        self.logger = logging.getLogger("RBNSKmerModel({self.rbp_name},{self.k}mers) ".format(self=self) )
        self.logger.debug("rbp_conc={self.rbp_conc}, rna_conc={self.rna_conc}, temp={self.temp}".format(self=self))

    @classmethod
    def from_file(cls, fname, col_start=None, col_end=None):
        rs = []
        rbp_conc = []
        kmers = []
        ranks = []
        
        with file(fname) as f:
            I = f.__iter__()
            head = I.next().rstrip().split('\t')
            rbp_conc = [float(c.replace('nM','')) for c in head[1:]]
            
            for line in I:
                parts = line.rstrip().split('\t')
                rs.append( [float(c) for c in parts[1:]] )
                kmers.append(parts[0])
                ranks.append(int(float(parts[1])))

        rs = np.array(rs)
        if col_start and col_end:
            rs = rs[:, col_start:col_end]
            rbp_conc = rbp_conc[col_start:col_end]
            
        most_enriched = np.percentile(rs, axis=0, q=99.9)
        best_sample = most_enriched.argmax()
        I = rs[:,best_sample].argsort()[::-1]
        return cls(np.array(kmers)[I], rs[I].T, np.array(rbp_conc))

    @classmethod
    def from_analysis(cls, a, k, **kwargs):
        rbp_conc = a.rbp_conc
        rna_conc = a.reads[0].rna_conc # assume constant over different experiments
        kmers = np.array(list(cska.ska_kmers.yield_kmers(k)))

        r_values, r_errors = a.SKA_weight_matrix(k) # a.pure_F_ratio_matrix(k) # a.SKA_weight_matrix(k)
        #order = a.get_optimal_kmer_ranking(k)
       
            
        mdl = cls(
            kmers, 
            r_values, 
            np.array(rbp_conc), 
            rna_conc=rna_conc, 
            rbp_name=a.rbp_name, 
            **kwargs
        )

        if a.known_kd:
            mdl.load_known(a.known_kd)
        
        return mdl
        

    def load_known(self, path):
        self.logger.info("loading known Kd values from '{0}'".format(path) )

        ref_kds = {}
        def is_valid(kmer):
            nucs = set(['A','C','G','T'])

            for n in kmer:
                if not n in nucs:
                    return False
            return True
            

        for line in file(path):
            if line.startswith('#'): 
                continue

            kmer, kd, err = line.split()[:3]
            kmer = kmer.upper().replace('U','T')
            
            if not is_valid(kmer):
                self.logger.warning("non-standard nucleotides in kmer '{0}', skipping.".format(kmer) )
                continue
            
            ref_kds[kmer] = float(kd) #float(err) )
        
        self.ref_kds = ref_kds
        self.track_kmers(ref_kds.keys())
        return ref_kds
        
    def track_kmers(self, kmers):
        for mer in kmers:
            self.tracked_kmers[mer] = cska.ska_kmers.seq_to_index(mer)

            
    def estimate_r_nonspecific(self, q=10):
        rns = np.percentile(self.r, q, axis=1)
        self.logger.debug("estimated r_ns='{0}'".format(rns) )
        return rns


    def select_kmers(self, n_top=10):
        most_enriched = np.percentile(self.r.T, axis=0, q=99.9)
        best_sample = most_enriched.argmax()
        
        selected = self.tracked_kmers.values()
        I = self.r[best_sample,:].argsort()[::-1]
        j = 0
        while len(selected) < n_top:
            i = I[j]
            if not i in selected:
                selected.append(i)
            j += 1

        I = np.array(selected)

        kmers = self.kmers[I]
        r = self.r[:,I]

        self.logger.debug("select_kmers: {0}".format(kmers))
        self.logger.debug("select_peak_r_values: {0}".format(r.max(axis=0)) )
        return kmers, r


    @staticmethod
    def objective_function(k_est):
        k_mean = k_est.mean(axis=0)
        
        
        err_weight = 1./k_mean
        err_weight /= err_weight.sum()
        
        #S = (np.log(k_est / k_mean[np.newaxis,:])**2).sum()
        
        n = len(k_est)
        S = (err_weight[np.newaxis,:] * np.log(k_est / k_mean[np.newaxis,:])**2).sum() / n
        #S = (np.log(k_est / k_mean[np.newaxis,:])**2).sum() / n
        return S

        #n = len(k_est)
        #err = np.var(np.log(k_est), axis=0)
        #err_weight = np.median(1./k_est, axis=0)
        #err_weight /= err_weight.sum()

        ##print self.rbp_conc
        ##print kmers[0], err[0], "opt", k_est[:,0]
        #return (err * err_weight[np.newaxis,:]).sum() / n
        
        

    def fit_naive(self, **kwargs):
        self.logger.info("fitting naive kmer energy model")

        kmers, r = self.select_kmers(**kwargs)
        n = self.n_conc
        
        def estimate_k(omegas):
            occ_matrix = r * omegas[:,np.newaxis]
            k_est = self.rbp_conc[:,np.newaxis] * (1/occ_matrix - 1)
            return np.array(k_est)

        def score(omegas):
            return self.objective_function( estimate_k(omegas) )
            
        #print "ratios", (1./(r/rns[:,np.newaxis] - 1)).min(axis=1)
        omegas_init = np.ones(n, dtype=float)*0.02
        bounds = np.ones((n,2), dtype=float)
        bounds[:,0] = 1e-6
        bounds[:,1] = 1./(r.max(axis=1) + 1e-3   ) 
        
        #print "BOUNDS", bounds
        from scipy.optimize import minimize
        res = minimize(score, omegas_init, method='L-BFGS-B', bounds = bounds)
        
        omegas = res.x
        k_est = estimate_k(omegas)
        err = self.objective_function( k_est )

        if not res.success:
            self.logger.warning("optimization did not converge! binding constant estimates are not possible")
        else:
            self.logger.info("fit converged. minimal value of objective function={0}.".format(err) )

        return kmers, res.x, k_est


    def fit_background(self, q=5, **kwargs):
        self.logger.info("fitting kmer energy model with background")
        
        kmers, r = self.select_kmers(**kwargs)
        rns = self.estimate_r_nonspecific(q=q)
        n = self.n_conc

        def estimate_k(betas):
            occ_matrix = betas[:,np.newaxis]*(r/rns[:,np.newaxis] - 1)
            k_est = self.rbp_conc[:,np.newaxis] * (1/occ_matrix - 1)
            return np.array(k_est)

        def score(omegas):
            return self.objective_function( estimate_k(omegas) )
            
        #print "ratios", (1./(r/rns[:,np.newaxis] - 1)).min(axis=1)
        betas_init = np.ones(n, dtype=float)*0.02
        bounds = np.ones((n,2), dtype=float)
        bounds[:,0] = 1e-6
        rr = (r/rns[:,np.newaxis] - 1)
        bounds[:,1] = 1./(rr.max(axis=1) + 1e-3 ) 
        
        #print "BOUNDS", bounds
        from scipy.optimize import minimize
        res = minimize(score, betas_init, method='L-BFGS-B', bounds = bounds)
        
        betas = res.x
        k_est = estimate_k(betas)
        err = self.objective_function( k_est )
        
        if not res.success:
            self.logger.warning("optimization did not converge! binding constant estimates are not possible")
        else:
            self.logger.info("fit converged. minimal value of objective function={0}.".format(err) )

        return kmers, res.x, k_est

        
    def fit_full(self, q=50, **kwargs):
        self.logger.info("fitting kmer energy model with background and secondary structure unfolding energy")
        
        kmers, r = self.select_kmers(**kwargs)
        rns = self.estimate_r_nonspecific(q=q)
        n = self.n_conc
        
        r_eff = (r - rns[:,np.newaxis]) / (1 - rns[:,np.newaxis])

        def estimate_k(omegas):
            k_est = []
            occ_matrix = r_eff * omegas[:,np.newaxis]
            
            for j,occ_row in enumerate(occ_matrix):
                k_row = []
                for i, occ in enumerate(occ_row):
                    k_bare = self.unfolding.k_bare_from_occ(kmers[i], self.rbp_conc[j], occ, temp=self.temp)
                    k_row.append(k_bare)
                k_est.append(k_row)
                
            return np.array(k_est)

        def score(omegas):
            return self.objective_function( estimate_k(omegas) )
            
        bounds = np.ones((n,2), dtype=float)
        bounds[:,0] = 1e-5
        bounds[:,1] = (1./r_eff).min(axis=1)*(1- 1e-2)
        #omegas_init = np.ones(n, dtype=float)*0.02
        omegas_init = bounds[:,1] / 2.
        
        #print "BOUNDS", bounds
        from scipy.optimize import minimize
        res = minimize(score, omegas_init, method='L-BFGS-B', bounds = bounds)
        
        omegas = res.x
        occ = r_eff * omegas[:,np.newaxis]
        
        k_est = estimate_k(omegas)
        err = score( omegas )
        
        if not res.success:
            self.logger.warning("optimization did not converge! binding constant estimates are invalid")
        else:
            self.logger.info("fit converged. minimal value of objective function={0}.".format(err) )
            
        return kmers, res.x, k_est

    def test_known_kds(self, q=10):
        kmers, r = self.select_kmers(n_top=0) # select only tracked/known kmers
        rns = self.estimate_r_nonspecific(q=q)
        n = self.n_conc
        print "rns", rns
        print "r", r
        r_eff = (r - rns[:,np.newaxis]) / (1 - rns[:,np.newaxis])
        print "r_eff", r_eff
        k_bare = [self.ref_kds[mer] for mer in kmers]

        omegas = []
        for i in range(n):
            P = self.rbp_conc[i]
            occ_fold = np.array([self.unfolding.occ(mer, P, self.ref_kds[mer]) for mer in kmers])
            om = occ_fold / r_eff[i]
            omegas.append(om)
        
        print "omegas", omegas
        
        

    def test_known(self, kmers, kds, q=1):
        # TODO: update!!
        import matplotlib.pyplot as pp
        import cska.ska_kmers
        I = np.array([cska.ska_kmers.seq_to_index(mer) for mer in kmers])
        kds = np.array(kds)
        print kds, kmers
        r = self.r[:,I]
        rns = self.estimate_r_nonspecific(q=q)
        
        L = len(self.rbp_conc)
        N = len(I)
        
        pp.figure()
        occ = np.zeros( (L,N), dtype=float)
        for i, P in enumerate(self.rbp_conc):
            beta_estimates = []
            for j, k_bare in enumerate(kds):
                kmer = kmers[j]
                o = oc.occ(kmer, P, k_bare, disable=True)
                #print o
                occ[i,j] = o
                r_ratio = r[i,j]/rns[i] - 1
                #print r_ratio
                beta_est = o / r_ratio
                beta_estimates.append(beta_est)
                print "{P}nM {kmer} K_bare={k_bare} nM, occ={o:.4f} r/rns -1 = {r_ratio:.4f} beta_est={beta_est:.3f}".format(**locals())
                
                pp.text(P*1.1, beta_est, kmer)
        
            #pp.loglog(occ[i,:], r[i,:]/rns[:] - 1, 'o', label="{P}nM".format(**locals()) )
            x = np.ones(len(beta_estimates)) * P
            pp.loglog(x, beta_estimates,  'ob', label="{P}nM".format(**locals()) )
        pp.xlabel(r'$P$ [nM]')
        pp.ylabel(r'$\beta$')
        pp.title('no structure')
        pp.ylim(1e-3,1e3)
        pp.savefig('test_nodU.pdf')
        
        pp.close()
        pp.figure()

        for i, P in enumerate(self.rbp_conc):
            beta_estimates = []
            for j, k_bare in enumerate(kds):
                kmer = kmers[j]
                o = oc.occ(kmer, P, k_bare, disable=False)
                #print o
                occ[i,j] = o
                r_ratio = r[i,j]/rns[i] - 1
                #print r_ratio
                beta_est = o / r_ratio
                beta_estimates.append(beta_est)
                #print "{P}nM {kmer} K_bare={k_bare} nM, occ={o:.4f} r/rns -1 = {r_ratio:.4f} beta_est={beta_est:.3f}".format(**locals())
                
                pp.text(P*1.1, beta_est, kmer)
        
            #pp.loglog(occ[i,:], r[i,:]/rns[:] - 1, 'o', label="{P}nM".format(**locals()) )
            x = np.ones(len(beta_estimates)) * P
            pp.loglog(x, beta_estimates,  'or', label="{P}nM".format(**locals()) )


        #pp.legend(loc='lower left')
        pp.xlabel(r'$P$ [nM]')
        pp.ylabel(r'$\beta$')
        pp.title('with dU correction')
        pp.ylim(1e-3,1e3)
        pp.savefig('test_dU.pdf')


    def plot_k(self, kmers, k_est, title='naive model'):
        import matplotlib.pyplot as pp
        import brewer2mpl
        bcolors = brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors
        pp.figure()
        pp.title(title)
        plot = pp.loglog
        for i, (mer, k_est) in enumerate(zip(kmers, k_est.T)):
            plot(self.rbp_conc, k_est, '-', color = bcolors[i % 12], marker='o', label=mer)
            
        pp.xlabel(r'$P_j$ [nM]')
        pp.ylabel(r'$\hat{k_i}$ [nM]')
        pp.legend()

        fname = '{0}.pdf'.format(title.replace(' ','_').replace(':',''))
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        path = os.path.join(self.out_path, fname)
        
        self.logger.info("plot_k: rendering PDF '{0}'".format(path) )
        pp.savefig(path)
        
class SPAState(object):
    def __init__(self, mdl, params, Z1, p_bound, pi_kmer, rbp_free, openen_bin_counts = [], jacobi = []):
        self.mdl = mdl

        self.params = params
        self.A = params[:self.mdl.nA] # affinities
        self.betas = params[self.mdl.nA:] # background coefficients
        self.Z1 = Z1
        self.p_bound = p_bound
        self.pi_kmer = pi_kmer
        self.rbp_free = rbp_free
        self.openen_bin_counts = openen_bin_counts
        self.jacobi = jacobi

        self.N = self.mdl.n_subsample
        if not self.N:
            self.N = self.mdl.reads.N
            
        #print "BETAS", self.betas
        self.pd_freq = self.pi_kmer + self.betas[:, np.newaxis] * self.mdl.f0 * self.N
        self.pd_sum = self.pd_freq.sum(axis=1)
        self.R = (self.pd_freq/ self.pd_sum[:, np.newaxis] ) / self.mdl.f0[np.newaxis,:]
        
    @property
    def dR_dA_matrices(self):
        R = self.R
        pi = self.pi_kmer
        jac_pi = self.jacobi
        
        diag = np.diagonal(jac_pi, axis1=1, axis2=2)
        return 1/pi[:, :, np.newaxis] * (R[:, :, np.newaxis] * jac_pi - (self.mdl.f0[np.newaxis,:]*R*R)[:, :, np.newaxis] * diag[:, np.newaxis, :] )

    @property
    def dR_dbeta_vectors(self):
        #return self.N * self.mdl.f0[np.newaxis, :] / self.pd_sum[:, np.newaxis] * (1 - self.R)
        return self.N * self.mdl.f0[np.newaxis, :] / self.pd_sum[:, np.newaxis] * (1 - self.R)
    
    @property
    def jacobi_matrices(self):
        jac = np.concatenate( (self.dR_dA_matrices, self.dR_dbeta_vector[:,:,np.newaxis]), axis=2)
        return jac
    
    def sum_square_gradients(self, R_obs):
        nA = len(self.A)
        grad = np.zeros((self.mdl.n_conc, len(self.params)), dtype=np.float32)

        grad_A = - 2 * ((self.R - R_obs)[:,:,np.newaxis] * self.dR_dA_matrices).sum(axis=1)
        grad_betas = - 2 * ((self.R - R_obs)[:,:] * self.dR_dbeta_vectors).sum(axis=1)
        grad[:,:nA] = grad_A[:,:]
        grad[:,nA:] = np.diag(grad_betas)
        
        return grad

    def sum_square_gradient(self, R_obs):
        return self.sum_square_gradients(R_obs).sum(axis=0)
    
    def sum_square_emp_gradients(self, R_obs, delta = 1e-5, f= 1e-3 ):
        R_obs = np.array(R_obs, dtype=float) # higher precision?
        grad = np.zeros((self.mdl.n_conc, len(self.params)), dtype=np.float32)
        params = np.array(self.params)
        err0 = ((R_obs - self.R)**2).sum(axis=1)
        for i in np.arange(len(params)):
            p0 = params[i]
            dp = - max(p0*f, delta)
            params[i] += dp
            state = self.mdl.evaluate(params, do_jacobi=False, keep=False)
            derr = err0 - ((R_obs - state.R)**2).sum(axis=1)
            #print dp, derr
            grad[:,i] = derr / dp
            params[i] = p0

        return grad
        
class ReferenceComparison(object):
    
    def __init__(self, opt, ref_file):
        self.opt = opt
        self.sequences = []
        self.kmer_sets = []
        self.names = []
        self.seqs = []
        self.affinities = []
        self.affinity_errs = []
        self.uniq_kmers = set()
        
        self.logger = logging.getLogger("ReferenceComparison")
        import cska.ska_kmers
        for line in file(ref_file):
            if line.startswith("#"):
                continue

            if not line.strip():
                continue
            
            parts = line.rstrip().split('\t')
            if len(parts) < 5:
                continue

            rbp, name, seq, kd, kd_err = parts[:5]
            if not rbp == self.opt.reads.rbp_name:
                continue
            
            kmers = self.split_kmers(seq)
            if not kmers:
                # can not predict affinity for sequence with non-canonical bases
                continue

            self.uniq_kmers |= set(kmers)
            self.seqs.append(seq)
            self.kmer_sets.append(np.array([cska.ska_kmers.seq_to_index(mer) for mer in kmers]))
            
            self.logger.debug("{seq} {kmers}".format(**locals()) )

            a = 1./float(kd)
            self.affinities.append(a)
            self.affinity_errs.append(a**2 * float(kd_err))
            self.names.append(name)
            
        self.observed_affinities = np.array(self.affinities)
        self.observed_affinity_errors = np.array(self.affinity_errs)
        
        self.logger.info("read {0} reference affinities".format(len(self.seqs)) )

    def split_kmers(self, seq):
        kmers = []
        k = self.opt.k
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer.count('A') + kmer.count('C') + kmer.count('G') + kmer.count('U') < k:
                # discovered non-canonical nucleotide
                continue
            kmers.append(kmer)
            
        return kmers

    @property
    def expected_affinities(self):
        a = []
        for s in self.kmer_sets:
            a.append(self.opt.current.params[s].sum())

        return np.array(a)
            
            
        
            
            
        
        
class SPAModel(object):
    def __init__(self, reads, openen, k, protein_conc, T=22, sub_replace=True, seq_only=False, out_path="./", n_subsample=100000, params = None):
        self.k = k
        self.nA = 4**k
        self.rbp_conc = np.array(protein_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)

        self.param_name = np.array(list(cska.ska_kmers.yield_kmers(self.k)) + ["beta{0}".format(i) for i in range(self.n_conc)])
        self.param_index = {}
        for i, name in enumerate(self.param_name):
            self.param_index[name] = i

        self.kmers = self.param_name

        self.T = T
        self.RT = (self.T + 273.15) * 8.314459848/4.184E3 # RT in kcal/mol
        self.logger = logging.getLogger('SPAModel')

        self.out_path = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
      
        # reads from RBNS random RNA pool and corresonding kmer-frequencies
        self.reads = reads
        self.seq_only = seq_only
        self.f0 = self.reads.kmer_frequencies(self.k)
        self.f0 /= self.f0.sum()

        # secondary structure accessibility
        self.openen = openen
        self.openen_lookup = self.openen.disc.x/self.RT # open energies in units of RT for each discretization level
        self.acc_lookup = np.exp( - self.openen_lookup) # accessibilities
        
        # subsampling related stuff
        self.sub_replace = sub_replace
        self.n_subsample = n_subsample
        
        # all subsample fields are set by new_subsample
        self.subsample_indices = []
        self.subsample_index_matrix = []
        self.subsample_oem = []
        self.new_subsample()
        # useful to normalize the kmer_pi's into quasi-occupancies
        self.n_kmers_in_sample = self.reads.N * (self.reads.L - self.k + 1) 
        
        # current state of the model
        self.state = None
        self.params = params
        
    def new_subsample(self):
        from cska.ska_kmers import fast_randint
        t0 = time.time()

        if not self.n_subsample:
            if len(self.subsample_indices):
                # de-activated subsampling and we already have everything in place!
                return
            
            indices = np.arange(self.reads.N)
        else:
            self.logger.debug('subsampling {self.n_subsample} out of {self.reads.N} sequences. replacement={self.sub_replace}'.format(self=self) )
            if self.sub_replace:
                indices = fast_randint(self.n_subsample, self.reads.N)
            else:
                indices = np.random.choice(self.reads.N, size=self.n_subsample, replace= self.sub_replace)
            t1 = time.time()
            self.logger.debug('generating random subsample indices took {0:.2f} ms'.format(1000* (t1-t0)) )
            self.n_kmers_in_sample = self.n_subsample * (self.reads.L - self.k + 1)

        self.subsample_indices = indices

        seqm = self.reads.seqm[indices]
        t2 = time.time()
        self.subsample_index_matrix = cska.ska_kmers.seq_matrix_to_index_matrix(seqm, self.k)
        t3 = time.time()
        self.logger.debug('converting subsample to index-matrix took {0:.2f} ms'.format(1000* (t3-t2)) )

        self.subsample_oem = self.openen.oem[indices]
        self.logger.debug("entire new_subsample() run took {0:.2f} ms".format(1000*(time.time() - t0)) )
        
    def self_consistent_free_rbp(self, Z1, rbp_total):
        from scipy.optimize import minimize_scalar
        rna_conc = self.reads.rna_conc
        N = len(Z1)
        
        t0 = time.time()
        def to_optimize(p_free):
            Z = p_free * Z1
            p = Z / (Z + 1.)
            
            rbp_bound = (p * rna_conc).sum() / N
            
            return ((rbp_total - rbp_bound) - p_free)**2
            
        res = minimize_scalar(to_optimize, bounds = (0, rbp_total), method='Bounded')
        t1 = time.time()
        perc = res.x / rbp_total
        self.logger.debug("self consistency: total={0:.1f} free={1:.1f} ({2:.2f}%) in {3:.2f} ms".format(rbp_total, res.x, perc, 1000*(t1-t0)) )
        
        return res.x
        
        
    def _spa_partition_function(self, im, oem, acc_lookup, kmer_invkd):
        from cska.ska_kmers import SPA_partition_function
        import time

        #t0 = time.time()
        # the model itself is implemented in Cython
        Z1 = SPA_partition_function(im, oem, acc_lookup, kmer_invkd, self.k, n_max = self.n_subsample, openen_ofs = self.openen.ofs)
        return Z1
    
    def _spa_free_protein(self, Z1, rbp_conc):
        rbp_free = [self.self_consistent_free_rbp(Z1, total) for total in rbp_conc]
        return np.array(rbp_free, dtype= np.float32)

    def _spa_p_rna_bound(self, Z1, rbp_free):
        n = len(Z1)
        n_conc = len(rbp_free)
        p_bound = np.zeros( (n_conc, n), dtype=np.float32)
        
        for i in range(n_conc):
            Z = rbp_free[i] * Z1
            p_bound[i] = Z / (Z + 1)
        
        return p_bound
        

    def _spa_kmer_pi(self, p_bound, im):
        from cska.ska_kmers import SPA_partition_function, weighted_kmer_counts
        n_conc, n = p_bound.shape
        
        pi = np.zeros( (n_conc, self.nA), dtype=np.float32)
        for i in range(n_conc):
            pi[i] = weighted_kmer_counts(im, p_bound[i], self.k)

        return pi

    def _eval_tm(self, im, oem, acc_lookup, kmer_invkd, rbp_conc):
        """ LEGACY: WILL BE REMOVED"""
        self.logger.debug("_eval_tm called!")
        Z1 = self._spa_partition_function(im, oem, acc_lookup, kmer_invkd)
        p_bound = self._spa_p_rna_bound(Z1, rbp_conc)
        pi = self._spa_kmer_pi(p_bound, im)

        return Z1, p_bound, pi

        
    def evaluate(self, params, keep=False, indices = [], rbp_conc = [], seq_only=None, do_jacobi=False, tm_update=True, ground_state=None):
        """
        Evaluate the thermodynamic model (single protein approximation) on a sub-sample of reads. Return an SPAState instance
        """
        # prepare all variables
        kmer_invkd = params[:self.nA]
           
        if not len(indices):
            indices = self.subsample_indices
            im = self.subsample_index_matrix
            oem = self.subsample_oem
        else:
            seqm = self.reads.seqm[indices]
            im = cska.ska_kmers.seq_matrix_to_index_matrix(seqm, self.k)
            oem = self.openen.oem[indices]


        if seq_only == None:
            seq_only = self.seq_only # use SPAModel instance setting

        if seq_only:
            acc_lookup = np.ones(self.acc_lookup.shape, dtype = np.float32)
        else:
            acc_lookup = self.acc_lookup

        if not len(rbp_conc):
            rbp_conc = self.rbp_conc
            
        n_conc = len(rbp_conc)
        if ground_state == None:
            ground_state = self.state

        if tm_update:
            Z1 = self._spa_partition_function(im, oem, acc_lookup, kmer_invkd)
            rbp_free = self._spa_free_protein(Z1, rbp_conc)
                
            p_bound = self._spa_p_rna_bound(Z1, rbp_free)
            pi_kmer = self._spa_kmer_pi(p_bound, im)

            state = SPAState(self, params, Z1, p_bound, pi_kmer, rbp_free)

        else:
            # skip thermodynamic model. 
            # Useful when changed parameter is not affinity (i.e. betas)
            # copy all thermodynamic model results from previous state.

            state = SPAState(self, params, ground_state.Z1, ground_state.p_bound, ground_state.pi_kmer, ground_state.rbp_free, ground_state.openen_bin_counts, ground_state.jacobi)

        if keep:
            self.params = params
            self.state = state

        return state
    
    def store_params(self, fname, params=[]):
        self.logger.info("storing current parameters in '{0}'".format(fname))
        with file(fname, 'w') as f:
            for i in xrange(len(self.params)):
                f.write('{0}\t{1}\n'.format(self.param_name[i], self.params[i]))

    def extrapolation(self, k, fname=""):
        """
        Using current affinities, extrapolate expected affinties for k > self.k
        """
        assert k > self.k
        kmers = list(cska.ska_kmers.yield_kmers(k))
        seqm = cska.ska_kmers.read_raw_seqs_chunked(kmers, chunklines=len(kmers))
        index_matrix = cska.ska_kmers.seq_matrix_to_index_matrix(seqm, self.k)
        affinities = self.params[index_matrix].sum(axis=1)
        
        params = np.concatenate((affinities, self.params[self.nA:]))
        mdl = SPAModel(self.reads, self.openen, k, self.rbp_conc, T= self.T, out_path =self.out_path, params = params)
        if fname:
            mdl.store_params(fname)
        
        return mdl
        
        
        
    def load_params(self, fname):
        params = []
        self.logger.info("reading parameters from '{0}'".format(fname))
        with file(fname, 'r') as f:
            for line in f:
                params.append(float(line.split('\t')[1]))

        #self.evaluate(np.array(params, dtype=np.float32), keep=True)
        return np.array(params, dtype=np.float32)


class SPAPartition(object):
    """
    breaks up the data into reads that *contain* kmer_i and those that do not. Allows very fast single k-mer optimization.
    """

    def __init__(self, mdl, kmer_i):
        self.mdl = mdl
        #self.params = np.array(mdl.params)
        self.kmer_i = kmer_i
        
        from cska.ska_kmers import index_matrix_rows_with_kmer
        kmer_hits = index_matrix_rows_with_kmer(self.mdl.subsample_index_matrix, self.mdl.k, kmer_i)
        self.kmer_indices = kmer_hits
        self.im_kmer = self.mdl.subsample_index_matrix[kmer_hits]
        self.oem_kmer = self.mdl.subsample_oem[kmer_hits]
        
        pi = self.mdl.state.pi_kmer
        rbp_free = self.mdl.state.rbp_free
        self.Z1 = self.mdl.state.Z1
        self.p_bound = self.mdl.state.p_bound
        self.rbp_free = self.mdl.state.rbp_free
        
        kmer_Z1 = self.mdl._spa_partition_function(self.im_kmer, self.oem_kmer, self.mdl.acc_lookup, self.mdl.params[:self.mdl.nA])
        kmer_p_bound = self.mdl._spa_p_rna_bound(kmer_Z1, rbp_free)
        kmer_pi = self.mdl._spa_kmer_pi(kmer_p_bound, self.im_kmer)

        self.other_pi = pi - kmer_pi
        
        self.mdl.logger.debug('SPAPartition of {0} sequences'.format(len(self.im_kmer)) )
        
    def evaluate(self, params, **kwargs):
        # re-evaluate the model *only* on the sequences with the kmer whose affinity is changed

        # update relevant partition functions
        kmer_Z1 = self.mdl._spa_partition_function(self.im_kmer, self.oem_kmer, self.mdl.acc_lookup, params[:self.mdl.nA])
        self.Z1[self.kmer_indices] = kmer_Z1

        # update free protein concentrations (probably not necessary)
        #rbp_free = self.mdl._spa_free_protein(self.Z1, self.mdl.rbp_conc)
        rbp_free = self.rbp_free

        # and re-compute expected pulldown kmer abundances
        kmer_p_bound = self.mdl._spa_p_rna_bound(kmer_Z1, rbp_free)
        kmer_pi = self.mdl._spa_kmer_pi(kmer_p_bound, self.im_kmer)

        # pi is pi from the non kmer-containing + pi from the kmer-containing subset of sequences
        pi = self.other_pi + kmer_pi
        # p_bound is not needed for R value computation. 
        # So we keep the unchanged value. Incorrect but convenient and not used anyway.
        self.Z1[self.kmer_indices] = kmer_Z1

        return SPAState(self.mdl, params, self.Z1, self.mdl.state.p_bound, pi, rbp_free)
        
        
class ParamUpdateScheduler(object):
    #def __init__(self, opt, n_blocked=5, ttl=10, n_max=100, n_avg=5, beta_burn_in=True):
    def __init__(self, opt, n_blocked=5, ttl=5, n_max=100, n_avg=5, beta_burn_in=True, monitor_params=[]):
        # iterative kmer selection 
        self.opt = opt
        self.logger = logging.getLogger("ParamUpdateScheduler")
        self.n_blocked = n_blocked
        self.n_max = n_max
        self.n_avg = n_avg
        self.ttl = ttl
        self.N = len(self.opt.current.params)
        self.monitor_params = [self.opt.mdl.param_index[p] for p in monitor_params]
        if not self.monitor_params:
            # top 10 R-value k-mers
            self.monitor_params = list(self.opt.R_obs.max(axis=0).argsort()[::-1][:10])
            # betas
            self.monitor_params += range(self.opt.nA, self.N)
        
        if beta_burn_in:
            self.blocked_params = range(self.opt.nA, len(self.opt.current.params))
        else:
            self.blocked_params = []

        self.param_last_improvement = {}
        self.param_last_error = {}
        self.param_last_updated = defaultdict(int)
        self.last_param_update = None
        self.R_max = self.opt.R_obs.max(axis=1)
        
    def update(self, pick):
        """
        param selection data structure maintenance
        """

        # param blocked list
        self.blocked_params.append(pick)
        while len(self.blocked_params) > self.n_blocked:
            self.blocked_params.pop(0)

        for i,rel in self.param_last_improvement.items():
            if self.opt.t - self.param_last_updated[i] > self.ttl:
                del self.param_last_improvement[i]

    def param_changed(self, param_i, t, better):
        self.last_param_update = param_i
        self.param_last_updated[param_i] = t
        self.param_last_improvement[param_i] = better
        
    @property
    def kmer_residuals(self):
        ## max mismatch between predicted and observed R-values
        return (self.opt.kmer_errors(self.opt.current.R)**2).sum(axis=0)

    @property
    def beta_residuals(self):
        residual_beta = []
        for i in range(self.opt.n_conc):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(self.opt.current.R[i,:], self.opt.R_obs[i,:])
            #print "LINREGRESS", self.rbp_conc[i], slope, intercept, r_value, p_value, std_err
            self.logger.debug("beta{i} slope={slope:.3e} intercept={intercept:.3e} r_value={r_value:.3e} p_value={p_value:.3e} std_err={std_err:.3e}".format(**locals()) )
            #residual_beta.append(0)
            residual_beta.append( self.R_max[i] * (np.fabs(1 - slope) + np.fabs(intercept) ) ) 

        self.logger.debug("beta residuals: {0}".format(residual_beta) )
        return np.array(residual_beta)
  
    @property
    def susceptibility(self):
        # high affinity kmers are more susceptible to changes, unless we look at under-estimation
        error = self.opt.kmer_errors(self.opt.current.R)
        affinities = self.opt.current.params[:self.opt.nA]
        a = affinities / affinities.max()
        suscept = (error > 0).all(axis=0) * a + (error < 0).sum(axis=0)
        suscept = np.concatenate( (suscept, np.ones(self.opt.n_conc)) )
        #caccc = cska.ska_kmers.seq_to_index('CACCC')
        #print "suscept of GCATG", suscept[590]
        #print "suscept of CACCC", suscept[caccc], a[caccc]

        return suscept

    @property
    def expectation(self):
        ## expected improvement upon parameter optimization
        if len(self.opt.rel_improvements):
            # default = average over past improvements + 10% benefit
            exp0 = np.array(self.opt.rel_improvements[-self.n_avg:]).mean()
        else:
            # fallback
            exp0 = 1
            
        exp0 = max(.01, exp0)

        expect = np.ones(self.N) * exp0
        for i in self.param_last_improvement.keys():
            expect[i] = self.param_last_improvement[i]

        return expect

    @property
    def time_passed(self):
        ## time since last update
        dt = np.ones(self.N, dtype=int) * (self.opt.t + 1)
        for i in self.param_last_updated.keys():
            dt[i] -= self.param_last_updated[i]
        
        dt[self.opt.nA:] *= 10 # make beta updates 10 times more often
        return dt
    
    def debug_monitor(self):
        self.logger.debug(">>>>> monitored parameters <<<<<")
        self.logger.debug("kmer\tblocked\tscore\tresidual\tdt\texpect\tcurrent\tknown\tdR")

        for i in self.monitor_params:
            self.logger.debug(self.debug_str_from_param(i))

    def debug_str_from_param(self, i):
        if i in self.blocked_params:
            state = 'B'
        else:
            state = ' '

        if i < self.opt.nA:
            R_str = ",".join(["{0:.2f}".format(x) for x in self.opt.current.R[:,i] - self.opt.R_obs[:,i]])
        else:
            R_str = 'n/a'
        
        return "{0:10s} {1}\t{2:.2e}\t{3:.2e}\t{4}\t{5:.2e}\t{6:.3e}\t{7:.3e}\t{8:.3e}\t{9}".format( 
            self.opt.mdl.param_name[i], 
            state, 
            self._score[i], 
            self._residual[i], 
            self._dt[i], 
            self._expect[i], 
            self._suscept[i],
            self.opt.current.params[i], 
            self.opt.known_params[i], 
            R_str 
        )
        
    def find_worst_param(self):
        """
        ideas: 
            * take into account if over-under-representation is systematic or only seen at some concentrations (contradicted at others?)
            * start preferring kmers whose error peaks at low concentrations (high affinity), then move to kmers whose error peaks at intermediate or high concentrations (lower affinity)
            * perhaps overall R-value correlation can serve as guide? (> .9 do a round of low affinity optimizations in fixed energy background?)
        """

        self._residual = np.concatenate( (self.kmer_residuals, 0*self.beta_residuals) ) # HACK: de-activate beta updates from inside same framework
        self._dt = self.time_passed
        self._expect = self.expectation
        self._suscept = self.susceptibility
        #score = residual * dt * expect * self.susceptibility
        self._score = self._residual * self._dt * self._expect * self._suscept
        #self.logger.debug("beta expect {0}".format(expect[self.opt.nA:]))
        #self.logger.debug("beta scores {0}".format(score[self.opt.nA:]))
        
        self.debug_monitor()
        
        self.logger.debug(">>>>> candidate search <<<<<")
        self.logger.debug("kmer\tblocked\tscore\tresidual\tdt\texpect\tsuscept\tcurrent\tknown\tdR")
        
        cand = []
        for j,i in enumerate(self._score.argsort()[::-1]):
            if not i in self.blocked_params:
                cand.append(i)
            if j < 20:
                self.logger.debug(self.debug_str_from_param(i))
            if j > 20 and cand:
                break

        pick = cand[0]
        self.update(pick)
        self.logger.debug("selected {0} {1}".format(pick, self.opt.mdl.param_name[pick]) )
        return pick
        
        
                 
class ModelOptimization(object):
    def __init__(self, reads, storage, k, R_obs, R_err=[], known_params = [], rbp_conc=[40.], out_path="./", n_subsample=0, sub_replace=False, aff0=1e-6, aff_min=1e-12, aff_max=1000, param_file=None, seq_only=False, tm_refresh=10, sched_params = {}, beta_interval = .01, scale_interval=100000000.): # scale_interval=.02
        self.k = k
        self.nA = 4**k

        # RBNS input sample to iterate on
        self.reads = reads
        self.openen = storage.get_discretized(k)
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)

        self.n_params = self.nA + self.n_conc

        self.beta_interval = int(beta_interval * self.nA)
        self.scale_interval = int(scale_interval * self.nA)
        self.last_beta = 0
        self.last_scale = 0
        
        self.kmers = np.array(list(cska.ska_kmers.yield_kmers(self.k)))
        self.t = 0
        self.tm_refresh = tm_refresh
        self.last_tm_refresh = 0

        self.logger = logging.getLogger('ModelOptimization')
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)


        # observations to fit to
        self.R_obs = R_obs
        self.R_err = R_err
        
        # bounds for the affinity parameters
        self.aff0 = aff0
        self.aff_min = aff_min
        self.aff_max = aff_max

        # the model to be trained
        self.mdl = SPAModel(self.reads, self.openen, k, self.rbp_conc, n_subsample=n_subsample, sub_replace=sub_replace, seq_only=seq_only)
        
        # monitor progress
        self.errors = [] #self.global_error(self.current.R), ]
        self.rel_improvements = []
        
        # set initial state of the model
        self.previous = None
        self.current = None
        if param_file:
            #self.update(self.mdl.load_params(param_file), np.Inf, "resuming from {0}".format(param_file))
            self.logger.info("resuming from {0}".format(param_file))
            self.current = self.mdl.evaluate(self.mdl.load_params(param_file), keep = True)
        else:
            params = np.zeros(self.nA + len(rbp_conc), dtype=np.float32)
            params[0:self.nA] += aff0
            betas = self.estimate_background()
            params[-len(betas):] = betas
            self.current = self.mdl.evaluate(params, keep = True)
            #self.update(params, np.Inf, "initialize")
        
        self.errors.append(self.global_error(self.current.R))
        
        # in case we know some parameters, use this as a reference
        if len(known_params):
            self.known_params = known_params
        else:
            self.known_params = np.ones(self.current.params.shape, dtype=np.float32) * np.nan

        # lastly, initialize the parameter update scheduler
        self.sched = ParamUpdateScheduler(self, **sched_params)

    def estimate_background(self, q=1.):
        return np.nanpercentile(self.R_obs, q, axis=1)
        
    def correlation(self, R_new=[]):
        if not len(R_new):
            R_new = self.current.R
        return np.array([np.corrcoef(np.log(Ro), np.log(Rp))[0][1] for Ro, Rp in zip(self.R_obs, R_new)])
    
    def linearity_err(self, R_new=[]):
        if not len(R_new):
            R_new = self.current.R

        lin = []
        for i in range(self.n_conc):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(R_new[i,:], self.R_obs[i,:])
            lin.append( np.fabs(1 - slope) + np.fabs(intercept) )

        return np.array(lin)
 
    def kmer_errors(self, R_new):
        return R_new - self.R_obs
        
    def kmer_error_conc(self, kmer_index, R_new, conc_i):
        return R_new[kmer_index] - self.R_obs[conc_i, kmer_index]
        
    def global_error(self, R_new):
        """used"""
        #return (((self.R_obs - R_new)**2)*self.R_obs).sum()
        #return (self.kmer_errors(R_new)**2).sum()
        lin_err = self.linearity_err(R_new)
        #print lin_err
        return ((self.kmer_errors(R_new)**2).sum(axis=1) * (1 + lin_err) ).sum()
    
    def global_error_conc(self, R_new, conc_i):
        return (self.kmer_errors(R_new)[conc_i,:]**2).sum()
    
    def sweep_param(self, param_i, x0=1.):
        import matplotlib.pyplot as pp
        pp.figure()
        #pp.title("t={0} conc={1}".format(self.t, self.rbp_conc[conc_i]))
        params = np.array(self.current.params)

        if param_i < self.nA:
            # evaluate thermodynamic model, but only on the subset of sequences containing the kmer
            tm_update = True
            opt = SPAPartition(self.mdl, param_i)
        else:
            # do not evaluate the thermodynamic model, only re-compute R-values
            tm_update = False
            opt = self.mdl
            
        scale = 10**np.arange(-9,3,.1)
        err = []
        for s in scale:
            params[param_i] = s
            state = opt.evaluate(params, tm_update=tm_update)
            #err.append(self.global_error_conc(state.R, conc_i))
            err.append(self.global_error(state.R))
         
        #print err
        pp.loglog(scale, err)
        pp.axvline(x0)
        pp.axhline(self.errors[-1])
        pp.show()
        pp.close()

    def print_update_vector(self, vec, n=-1):
 
        print "-----------update vector------------"
        for i in np.fabs(vec).argsort()[::-1][:n]:
        #for i in vec.nonzero()[0]:
            if vec[i] == 0:
                break
            print "    ", self.mdl.param_name[i], self.current.params[i], "+", vec[i], "known=", self.known_params[i]
        
    def print_summary(self):
        print "-----------optimization results------------"
        delta = self.current.params - self.previous.params
        R_err = self.current.R - self.R_obs
        for i in range(len(self.current.params)):
            if self.current.params[i] > self.known_params[i]*1.001:
                status = "OVER"
            elif self.current.params[i] < self.known_params[i]*0.999:
                status = "UNDER"
            else:
                status = "sweet"

            if i < self.nA:
                R_dev = str(R_err[:,i])
            else:
                R_dev = 'N/A'

            out =  ["    ", self.mdl.param_name[i], str(self.current.params[i]), "known=", str(self.known_params[i]), status, "last grad=", str(delta[i]), R_dev]
            print "\t".join(out)
    
    def debug_output_parameters(self, n=20):
        
        self.logger.debug(">>> current highest affinities/beta values <<<")
        for i in self.current.params.argsort()[::-1][:n]:
            if i in self.blocked_params:
                state = 'B'
            else:
                state = ' '
            
            if i < self.nA:
                R_str = ",".join(["{0:.2f}".format(x) for x in self.current.R[:,i] - self.R_obs[:,i]])
            else:
                R_str = 'n/a'
            
            self.logger.debug("{0} {1}\t{2:.3e}\t{3:.3e}\t{4}".format( self.mdl.param_name[i], state, self.current.params[i], self.known_params[i], R_str ) )
      

    def converged(self, last=10, tol=1e-5):
        if len(self.rel_improvements) < last:
            return False
        
        tol *= 4**(- (self.k-5) ) # expect slower convergence for higher k, bc each kmer alone will have less to explain
        recent = np.array(self.rel_improvements[-last:])
        #avg = recent.mean()
        dev = np.mean(recent)
        self.logger.debug("mean relative improvement over past {0} iteration steps={1}".format(last, dev) )
        if dev < tol:
            self.logger.info("convergence with mean improvement of {0}".format(dev) )
            return True
        
        return False
    
    def optimize(self, reporter = None, max_iter=None, snapshots=False, **kwargs):
       
        if reporter:
            reporter.tick(self.t)

        if max_iter == None:
            # roughly, expect ~10% of kmers to have relevant affinity. 
            # So this gives the chance of updating each ~10 times.
            max_iter = self.nA 
            
        imp, new_state = self.step_betas()
        #imp, new_state = self.step_scale(reporter=reporter)
        #sys.exit(1)       
        while not self.converged() and self.t < max_iter:
            if reporter:
                reporter.tick(self.t)

            p_bound = (new_state.p_bound * self.mdl.reads.rna_conc).sum(axis=1) / new_state.N
            self.logger.info("total amount of bound protein={0}".format(p_bound) )

            #imp, new_state = self.step_scale()
            imp, new_state = self.step_param()
            #imp_b = self.step_gradient()
            #imp_b = self.step_beta()
            
            #if imp_a < 1: #and imp_b < 1:
                #self.logger.warning( "Got stuck. Optimizing background parameters")
                #imp = self.step_beta()
                #self.logger.info("improvements were betas: {0:.4f}%".format(imp))

            if snapshots:
                self.mdl.store_params(os.path.join(self.out_path, 'params_t{0}.tsv'.format(self.t)) )

            if self.t - self.last_scale > self.scale_interval:
                self.step_scale()
                #self.step_betas()
                self.last_scale = self.t
                self.last_beta = self.t

            if self.t - self.last_beta > self.beta_interval:
                self.step_betas()
                self.last_beta = self.t


        if reporter:
            reporter.plot_R_value_agreement()              

    def step_param(self, show_sweep = False):
        self.logger.info("===parameter optimization===")
        param_i = self.sched.find_worst_param()
        
        #self.sweep_param(param_i)
        
        best, err, new_state = self.optimize_single_param(param_i)

        update = new_state.params - self.current.params
        name = self.mdl.param_name[param_i]
        imp = self.update(new_state, err, "single parameter optimization {name} -> {best:.3e} (err={err:.3e})".format(**locals()))
        # notify the scheduler of the param change and its consequences
        self.sched.param_changed(param_i, self.t, imp)

        if not imp and show_sweep:
            self.sweep_param(param_i, best)

        return imp, new_state

    def step_betas(self, ground_state=None):
        for param_i in range(self.nA, self.n_params):
            best, err, new_state = self.optimize_single_param(param_i, ground_state=ground_state)
            better = self.update(new_state, err, "beta{0} parameter optimization".format(param_i - self.nA), tick=False)

        return better, new_state
    
    def step_scale(self, reporter=None):
        from cska.ska_kmers import SPA_partition_function, weighted_kmer_counts
        import time
        t0 = time.time()
        params = np.array(self.current.params)
        Z1 = self.current.Z1
        im = self.mdl.subsample_index_matrix
        n = len(Z1)
        p_bound = np.zeros( (self.n_conc, n), dtype=np.float32)
        pi = np.zeros( (self.n_conc, self.nA), dtype=np.float32)
       
        scales = []
        errors = []
        
        def to_optimize(scale):
            # scale the partition function and only update pi (weighted kmer-counts)
            scales.append(scale)
            params[:self.nA] = self.current.params[:self.nA] * scale
            for i in range(self.n_conc):
                Z_scaled = Z1 * scale
                Z = self.rbp_conc[i] * Z_scaled
                p_bound[i] = Z / (Z + 1)
                pi[i] = weighted_kmer_counts(im, p_bound[i], self.k)
            
            
            # construct a new state object from that, by-passing evaluate
            # TODO: integrate into evaluate by better re-factor.
            state = SPAState(self.mdl, params, Z_scaled, p_bound, pi)
            err = self.global_error(state.R)
            #print "global error={0} at scale={1} before beta fit".format(err, scale)
            
            # find optimal betas at each step
            for param_i in range(self.nA, self.n_params):
                best, err, new_state = self.optimize_single_param(param_i, ground_state=state)
                state.params[param_i] = best
                
            err = self.global_error(new_state.R)
            #print "global error={0} at scale={1} after beta fit".format(err, scale)
            
            errors.append(err)
            return err

        res = minimize_scalar(to_optimize, bounds = (.1,10), method='Bounded')
        dt = time.time() - t0
        self.logger.info("global affinity re-scaling: success={res.success} scale={res.x} took {dt:.2f}s".format(**locals()) )

        if reporter:
            reporter.plot_sweep(param="scale", errors = errors, x = scales)

        params[:self.nA] = self.current.params[:self.nA] * res.x
        new_state = self.mdl.evaluate(params, tm_update=True)

        better = self.update(new_state, self.global_error(new_state.R), "affinity re-scaling")
        if better > 0:
            # un-block all affinities to allow unbiased optimization.
            self.sched.n_blocked = []
        return better, new_state
            
    def optimize_single_param(self, param_i, ground_state=None):
        if ground_state == None:
            ground_state = self.current
        
        params = np.array(ground_state.params)

        if param_i < self.nA:
            # evaluate thermodynamic model, but only on the subset of sequences containing the kmer
            tm_update = True
            opt = SPAPartition(self.mdl, param_i)
        else:
            # do not evaluate the thermodynamic model, only re-compute R-values
            tm_update = False
            opt = self.mdl
            
        def to_optimize(aff):
            params[param_i] = aff * .0001
            state = opt.evaluate(params, tm_update=tm_update, ground_state = ground_state)
            err = self.global_error(state.R)
            return err

        t0 = time.time()
        s_mid = (self.aff_max + self.aff_min)/2.
        res_a = minimize_scalar(to_optimize, bounds = 10000 * np.array([self.aff_min, s_mid]), method='Bounded')
        res_b = minimize_scalar(to_optimize, bounds = 10000. * np.array([s_mid, self.aff_max]), method='Bounded')
        if res_a.fun < res_b.fun:
            res = res_a
        else:
            res = res_b

        best = res.x * 0.0001
        dt = time.time() - t0
        name = self.mdl.param_name[param_i]
        A0 = ground_state.params[param_i]
        rel_change = (best - A0) / A0
        self.logger.debug("optimal {name} affinity/value search success={res.success} A={best} (A0={A0} rel change={rel_change}) took {dt:.2f}s".format(**locals()) )

        params[param_i] = best
        new_state = opt.evaluate(params, tm_update=tm_update)
        
        return best, res.fun, new_state


    def update(self, new_state, err, name, tick=True):
        from copy import copy
        self.previous = copy(self.current)
        
        if self.errors:
            better = (self.errors[-1] - err)* 100./self.errors[-1]
            if better <= 0:
                self.logger.warning("unable to lower error in {1} step at t={0}".format(self.t, name))
                # reject the changes!
                #params = np.array(self.current.params)
                better = 0
                new_state = self.current
        else:
            better = err
            
        self.current = new_state
        self.mdl.state = new_state
        self.mdl.params = new_state.params
        
        # subsamples should remain stable throughout one iteration step!
        if self.t - self.last_tm_refresh > self.tm_refresh:
            R_before = self.current.R
            self.current = self.mdl.evaluate(self.current.params, keep = True)#, do_jacobi = True) # dont use gradient for now!
            R_after = self.current.R
            
            round_err = np.fabs(R_before - R_after).sum()
            self.logger.debug('re-freshed thermodynamic model: rounding errors={0}'.format(round_err))
            self.last_tm_refresh = self.t
            
        #self.mdl.new_subsample()
            

        self.errors.append(self.global_error(self.current.R))
        
        self.logger.info("status after '{0}' step at t={1}, improvement was {2:.2e}%".format(name, self.t, better))
        self.logger.info("correlations: {0}".format(self.correlation()) )
        self.logger.info("most recent errors: {0}".format( self.errors[-5:] ))
        
        if self.previous:
            update = self.current.params - self.previous.params
            #print "{0} step at t={1}".format(name, self.t)
            #self.print_update_vector(update)

        if tick:
            self.t += 1
        if self.t > 1:
            self.rel_improvements.append(better)
        return better
       
      
if __name__ == "__main__":
    import matplotlib
    #matplotlib.use('pdf')
    import matplotlib.pyplot as pp
    logging.basicConfig(level=logging.DEBUG)

    from cska.rbns_reads import RBNSReads
    from cska.folding import RBNSOpenen, OpenenStorage
    import cska.folding
    reads = RBNSReads('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads', n_max=10000000)
    storage = OpenenStorage(reads, '/scratch/data/RBNS/RBFOX2/ska_RBFOX2/openen/', disc_mode='gamma')
    openen = storage.get_discretized(5)
    #openen._do_not_unpickle = True
    disc = openen.disc

    sim = RBNSSimulator(reads, openen, 5)    
    
    print "joint frequencies"
    kmer_openen = openen.kmer_openen_counts()

    gen = RBNSGenerator(5,l=40, seed=47110815)
    gen.store_invKd("/home/mjens/git/RBPbind/server/proteinfiles/RBNSGenerator_rev.txt")
    #sys.exit(1)
    
    kmer_energies = np.random.permutation(gen.kmer_energies) #- 1.5 # non-specific binding
    kmer_energies = gen.kmer_energies #- 1.5 # non-specific binding


    print kmer_energies
    print "simulating binding"
    rbp_conc = [.5,1.,10.,120.,360.]
    #sim._do_not_unpickle=True
    counts, openen_bincounts = sim.expected_kmer_counts(kmer_energies, rbp_conc, E_ns=-2/gen.RT)
    print counts.shape, openen_bincounts.shape
    print ">>>openen-bins"
    
    best_i = kmer_energies.argmin()
    med_i = (kmer_energies == np.median(kmer_energies)).argmax()

    print "median kmer energy", kmer_energies[med_i]*gen.RT, "kcal/mol"
    colors = ['k','r','g','y','c']
    styles = ['x','*','^','.','o']
    
    pp.figure(figsize=(10,8) )
    for I,color in zip([best_i, med_i], colors):
        ref = kmer_openen[I]
        x, ref_y = disc.get_hist_xy(ref)
        
        for i,style in zip(range(len(openen_bincounts)), styles):
            bc = openen_bincounts[i,I,:]
            x, y = disc.get_hist_xy(bc)
            
            lratio = np.log(y, ref_y)
            seq = cska.ska_kmers.index_to_seq(I, 5)
            pp.plot(disc.x, lratio, color+style, label="{1} P={0}".format(rbp_conc[i], seq))

        
    #pp.show()
    pp.xlabel(r"$\Delta U$ [kcal/mol]")
    pp.ylabel(r"$\log(\frac{pd}{input})$")
    pp.legend()
    pp.savefig('simulated.pdf')
    
    
    freqs = counts * (4**5)/counts.sum(axis=1)[:,np.newaxis]
    print freqs[:,-10:]
    
    R = freqs/reads.kmer_frequencies(5)
    print "R-values", R.min(), R.max()
    print R[:,-10:]
    
    pp.show()    
    
    
    
        
    #test_fastrand()
    sys.exit(0)
    
    
    ### test partition function for overlap
    gen = RBNSGenerator(5,l=40, seed=47110815)
    #gen.assign_experimental_input("/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads")
    gen.assign_experimental_input("bla.reads")
    
    #P = 3200.
    #P = 320.
    #P = 50.
    P = 10.
    
    B = np.exp(- gen.kmer_energies + np.log(P*1e-9) )
    print "Boltzmann weights for top sites", B[-5:]
    
    k = 5
    N = 4**k
        
    # kmer overlap extension frequencies
    kfreqs = {}
    for x in range(1,k+1):
        kfreqs[x] = gen.input_reads.kmer_frequencies(x) / 4**x
        
    # test using simulated reads
    from cska.rbns_analysis import RBNSComparison
    #reads = gen.generate_input_reads(N=1000000, store="6mer@0nM.reads")
    reads = gen.generate_bound_reads(P=P, p_ns=0.00, N=10000, store="7mer@{0}nM.reads".format(P))
    #print gen.Z_full
    print ">>> most abundant kmer-frequencies in pulldown simulation", reads.kmer_frequencies(k)[-5:]
    comp = RBNSComparison(gen.input_reads, reads)
    R, R_err = comp.R_values(k, _do_not_unpickle=True)
    print "simulation R-values", R[-5:]

    def P_bound_given_kmer_nn(x):
        P = 0
        fx = kfreqs[k][x]
        
        for j in range(4):
            fj = kfreqs[1][j]
            
            # overlapping kmers
            ij = (x << 2 | j) & (N-1)
            #print "kmer i={0:b} ij={1:b}".format(i,ij)
            ji = j << ((k-1)*2) | x >> 2
            #print "kmer i={0:b} ij={1:b} ji={2:b}".format(i,ij, ji)
            
            Zij = B[x] + B[ij]
            Zji = B[x] + B[ji]
                            
            Pij = Zij / (Zij + 1)
            Pji = Zji / (Zji + 1)
            
            P += fj * (Pij + Pji)

        return fx * P/2.


    def P_bound_given_kmer_nnn(x):
        P = 0
        fx = kfreqs[k][x]
        
        for l in range(4):
            fl = kfreqs[1][l]
            
            for r in range(4):
                fr = kfreqs[1][r]
                
                xr = (x << 2 | l) & (N-1)
                lx = l << ((k-1)*2) | x >> 2
                
                Zlxr = B[lx] + B[x] + B[xr]
                                
                Plxr = Zlxr / (Zlxr + 1)
                
                P += fl * fr * Plxr

        for l in range(16):
            fl = kfreqs[2][l]
            l1 = l & 3
            
            # overlapping kmers
            lx = l << ((k-2)*2) | x >> 4
            l1x = l1 << ((k-1)*2) | x >> 2
            
            Zlx = B[lx] + B[l1x] + B[x]
            Plx = Zlx/ (Zlx + 1)
            
            P += fl * Plx

        for r in range(16):
            fr = kfreqs[2][r]
            r1 = r >> 2
            
            # overlapping kmers
            xr = (x << 4 | r) & (N-1)
            xr1 = (x << 2 | r1) & (N-1)
            
            Zxr = B[x] + B[xr1] + B[xr]
            Pxr = Zxr/ (Zxr + 1)
            
            P += fr * Pxr

        return fx * P/3.


    
    def P_kmer_in_read():
        Z = np.zeros(N)
        
        for i in range(N):
            fi = kfreqs[k][i]
            
            #Z[i] += fi
            
            for j in range(4):
                fj = kfreqs[1][j]
                
                # overlapping kmers
                ij = (i << 2 | j) & (N-1)
                
                ji = j << ((k-1)*2) | i >> 2
                
                Z[ij] += fj * fi
                Z[ji] += fj * fi

        return Z / Z.sum()
        
       
    kmer_occ = B / (B + 1)
    naive_freq = kmer_occ * kfreqs[k]
    naive_r = naive_freq / naive_freq.sum() / kfreqs[k]
    print "naive r",naive_r[-5:]
    
    lin_approx_r, pisum, beta = gen.predict_r_values(P, limit=False)
    print "lin matrix approx.", lin_approx_r[-5:]
    print "factors pisum", pisum, "beta", beta
    #p_in = kfreqs[k] * 2#P_kmer_in_read()
    #p_in = P_kmer_in_read()
    
    M, M_inv = gen.crosstalk_matrix_and_inverse()
    # inverting linear approximation and comparing to real occupancies
    occ_approx = np.dot(M_inv, lin_approx_r * pisum - beta)
    
    pp.figure()
    plot = pp.plot
    plot(kmer_occ, np.dot(M_inv, R * pisum - beta), 'or')
    plot(kmer_occ, np.dot(M_inv, R * 1.5*pisum - beta), 'oy')
    plot(kmer_occ, np.dot(M_inv, lin_approx_r * pisum - beta), 'ok' ,label="pos. control")
    
    pp.xlabel('kmer occ')
    pp.ylabel('lin approx occ')
    
    pp.show()
    
    nn_freq = np.array([P_bound_given_kmer_nn(x) for x in np.arange(N)])
    nn_r = nn_freq / nn_freq.sum() / kfreqs[k]
    print "nn r",nn_r[-5:]

    nnn_freq = np.array([P_bound_given_kmer_nnn(x) for x in np.arange(N)])
    nnn_r = nnn_freq / nnn_freq.sum() / kfreqs[k]
    print "nnn r",nnn_r[-5:]

    
    pp.figure()
    M = max(R.max(), nn_r.max(), nnn_r.max(), lin_approx_r.max(), naive_r.max() )
    M = max(R.max(), lin_approx_r.max())
    
    plot = pp.loglog
    plot = pp.plot
    
    plot(np.array([1.,M]),np.array([1., M]), '--', color='gray')

    #plot(R, naive_r, 'ok', alpha=.2, label="naive kmer soup R={0:.2f}".format(np.corrcoef(np.log(R), np.log(naive_r))[0][1]) )
    #plot(R, nn_r, 'oy', alpha=.5, label="nearest neighbor avg. R={0:.2f}".format(np.corrcoef(np.log(R), np.log(nn_r))[0][1]) )
    #plot(R, nnn_r, 'or', alpha=.5, label="next-nearest neighbor avg. R={0:.2f}".format(np.corrcoef(np.log(R), np.log(nnn_r))[0][1]) )
    plot(R, lin_approx_r, 'og', label="linear matrix R={0:.2f}".format(np.corrcoef(np.log(R), np.log(lin_approx_r))[0][1]) )
    pp.xlim(1,M)
    pp.ylim(1,M)
    pp.legend(loc='upper left')
    
    gen.binding_constants_from_R_vec(R, k, P)
    #pp.show()
        #t0 = time.time()        
        #for i in np.arange(N):
            #M[i,i] = 1
            #for x in range(1,k):
                #weights, shifts = cska.ska_kmers.weighted_kmer_shifts(i, k, self.l, x, kfreqs[x]) 
                #for s,f in zip(shifts, weights):
                    #M[i,s] += f

        #t1 = time.time()
        #M_inv = np.linalg.inv(M)
        #t2 = time.time()
        #self.logger.debug("computed crosstalk matrix for k={0} in {1:.3f}s, inverted in {2:.3f}s".format(self.k, t1-t0, t2-t1) )
        #return M, M_inv

    
    #sys.exit(0)
    
    
    
    #gen = RBNSGenerator(5,l=40, seed=47110815)
    #pp.figure()
    
    #colors = ['k','b','y','g','r']
    #for P,c in zip([1., 10., 100., 500.], colors):
        ##Q = PartitionFunction("GAATGGAGTTTTTTGTTTTC",P, gen.kmer_energies, 5)
        #Q = PartitionFunction("TTTTT",P, gen.kmer_energies, 5)
        #Pb_1 = Q.P_bound_single()
        #Pb_ex = Q.P_bound_indep_exact()
        ##print Q.P_bound_indep()
        #Pb_corr = Q.P_bound_rec()
        
        #print "single protein unit binding",Pb_1
        #P_bound_single = 1 - np.product( 1 - Pb_1)
        #P_bound_indep = 1 - np.product( 1 - Pb_ex)
        #P_bound_corr = 1 - np.product( 1- Pb_corr)
        #print "P", P, "indep",P_bound_indep, "corr",P_bound_corr, "ratio", P_bound_indep/P_bound_corr
        ##print Pb_ex
        #pp.semilogy(Pb_1, '^-', color=c, label="P={0}".format(P))
        #pp.semilogy(Pb_ex, 'x--', color=c, label="P={0}".format(P))
        #pp.semilogy(Pb_corr, 'o-', color=c, label=None)
    
    #pp.legend(loc='lower right')
    #pp.show()
    #import sys
    #sys.exit(0)

    gen.assign_experimental_input("/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads")
    print gen
    #gen.energy_plot()
    #gen.generate_input_reads(store="input.reads", N=100)
    # test different partition functions
    
    


    
    #gen.generate_bound_reads(store="bound_320.reads", P=320., p_ns=0.00, N=20000000)
    #occ = gen.predict_occupancies(P=320., store="occ_320.tsv")
    
    
    r_obs = []
    mers = []
    for line in file('ska_sim/RBP.R_value.5mer.tsv'):
        if line.startswith('#'): 
            continue
        parts = line.rstrip().split('\t')
        r_obs.append([float(parts[i]) for i in [1,3,5,7,9]])
        mers.append(parts[0])
            
    mers = np.array(mers)
    r_obs = np.array(r_obs).T[:,mers.argsort()]
    
    
    rbp_conc = np.array([1., 40, 80, 160, 320])
    #rbp_conc = np.array([1.,5.])
    rbp_conc = np.array([1., 80, 320])
    
    pp.figure()
    r_matrix = []
    occ_matrix = []
    for i,P in enumerate(rbp_conc):
        r = gen.predict_r_values(P=P, store="r_{0}.tsv".format(P))
        occ = gen.predict_occupancies(P=P, store="occ_{0}.tsv".format(P))
        r_matrix.append(r)
        occ_matrix.append(occ)
        #gen.generate_bound_reads(store="bound_{0}.reads".format(P), P=P, p_ns=0.00, N=1000000)
        
        pp.loglog(occ, r, 'o', alpha=.5, label="P={0:.0f}nM".format(P))
        
    pp.legend(loc='upper left')
    pp.xlabel('predicted occupancy')
    pp.ylabel('predicted R-value')
    pp.savefig("R_pred_vs_occ.pdf")
    pp.close()
    
    pp.figure()
    rmin = 1.
    rmax = 1.
    for P, ro, r in zip(rbp_conc, r_obs, r_matrix):
        pp.loglog(r, ro, 'o', alpha=.5, label="P={0:.0f}nM".format(P))

        rmin = min(r.min(), ro.min(), rmin)
        rmax = max(r.max(), ro.max(), rmax)
    
    rmin /= 2
    rmax *= 2
    pp.plot([rmin, rmax], [rmin, rmax], '--k')
    #print "minmax", rmin, rmax
    pp.legend(loc='upper left')
    pp.xlabel('predicted R-value')
    pp.ylabel('observed R-value')


    M, M_inv = gen.crosstalk_matrix_and_inverse()

    def inverse(r):
        r_inv = np.dot(M_inv, r) 
        ofs = np.dot(M_inv, np.ones(r.shape))

        delta = r_inv - ofs
        corr = max(- delta.min(), 0) # no negative terms allowed
        
        beta = corr / ofs
        print (r_inv + beta*ofs).min()
        
        pre_scaled = r_inv + beta * ofs
        scale = 1./pre_scaled.max()
        
        beta = beta * scale
        inv = pre_scaled * scale
        print "inv minmax", inv.min(), inv.max()
        return inv
        
    pp.figure()
    for P, ro, occ, r in zip(rbp_conc, r_obs, occ_matrix, r_matrix):
        Z = 1
        beta = 1
        ro_inv = inverse(ro)
        r_inv = inverse(r)

        print r_inv[-10:]
        pp.loglog(r_inv, ro_inv, 'o', alpha=.5, label="P={0:.0f}nM".format(P))

    pp.legend(loc='upper left')
    #pp.xlabel('predicted occupancy')
    pp.xlabel(r'$M^{-1} \cdot r_{pred}$')
    pp.ylabel(r'$M^{-1} \cdot r_{obs}$')
    
    #pp.show()
        
        
    kmers, k_est, dG = gen.linear_fit(rbp_conc, np.array(r_matrix), n_top=20)
    

    import matplotlib.pyplot as pp
    ##pp.figure()
    ##pp.loglog(r, rm, 'ok')
    
    #pp.figure(figsize=(8,4))
    #pp.subplot(121)
    #pp.imshow(np.log10(M), cmap=pp.get_cmap('plasma'))
    #pp.colorbar(label=r'$\log_{10}(M)$',fraction=0.046, pad=0.04)

    #pp.subplot(122)
    #pp.imshow(np.log10(M_inv), cmap=pp.get_cmap('plasma'))
    #pp.colorbar(label=r'$\log_{10}(M^{-1})$',fraction=0.046, pad=0.04)
    #pp.tight_layout()
    #pp.savefig('crosstalk_matrix.pdf')
        
    pp.figure()
    pp.title("inferred binding energies")
    print gen.kmer_energies[-20:].shape, dG.shape
    pp.plot(gen.kmer_energies[-20:]*gen.RT, dG, 'ob')
    pp.xlabel("simulated energies [kcal/mol]")
    pp.ylabel("inferred energies [kcal/mol]")

    pp.show()
    #pp.savefig('predict.pdf')
    
