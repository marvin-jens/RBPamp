import numpy as np
import logging
import time
import sys
from collections import defaultdict
from scipy.optimize import minimize, brentq, minimize_scalar

import cska.ska_kmers 

from cska.rbns_reads import RBNSReads

from cska.caching import CachedBase, cached, pickled

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
    
        #occ = self.predict_occupancies(P)
        #kappa = self.input_reads.kmer_frequencies(k) / 4**k
        #omega = (occ * kappa).sum()

        ## adding baseline
        #l = self.l - k + 1
        #bg = omega*(l-2.*(k-1)) #* kappa 
        
        ##pi = np.dot(M, occ ) * kappa + base

        #pi = kappa * (np.dot(M, occ ) + bg )
        
        #sum_pi = pi.sum()
        
        #print "background terms", bg, sum_pi
        #r = pi / sum_pi / kappa
        ##import matplotlib.pyplot as pp
        ##pp.figure()
        ##pp.loglog(pi / kappa, r,'ok')
        ##pp.show()
        #I = r.argsort()[::-1]
        #for i in I[:10]:
            #print i, cska.ska_kmers.index_to_seq(i, k), pi[i], r[i]

        #if store:
            #with file(store,'w') as f:
                #for kmer, o in zip(cska.ska_kmers.yield_kmers(self.k), r):
                    #f.write("{0}\t{1}\n".format(kmer, o) )
        
        ## all the expensive matrix multiplications are 
        ## done *once*
        #M_inv = np.linalg.inv(M)
        #r_inv = np.dot(M_inv, r)
        #bg_vec = np.dot(M_inv,np.ones(4**k)) 

    
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
    def __init__(self, mdl, params, p_bound, pi_kmer, openen_bin_counts = [], jacobi = []):
        self.mdl = mdl

        self.params = params
        self.A = params[:self.mdl.nA] # affinities
        self.betas = params[self.mdl.nA:] # background coefficients
        self.p_bound = p_bound
        self.pi_kmer = pi_kmer
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
        
        
        
        
class SPAModel(object):
    def __init__(self, reads, openen, k, protein_conc, T=22, sub_replace=True, seq_only=False, n_subsample=100000):
        self.k = k
        self.nA = 4**k
        self.kmers = np.array(list(cska.ska_kmers.yield_kmers(self.k)))

        self.rbp_conc = np.array(protein_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)

        self.T = T
        self.RT = (self.T + 273.15) * 8.314459848/4.184E3 # RT in kcal/mol
        self.logger = logging.getLogger('SPAModel')

        # secondary structure accessibility
        self.openen = openen
        self.openen_lookup = self.openen.disc.x/self.RT # open energies in units of RT for each discretization level
        self.acc_lookup = np.exp( - self.openen_lookup) # accessibilities

        # reads from RBNS random RNA pool and corresonding kmer-frequencies
        self.reads = reads
        self.seq_only = seq_only
        self.f0 = self.reads.kmer_frequencies(self.k)
        self.f0 /= self.f0.sum()
        
        # subsampling related stuff
        self.sub_replace = sub_replace
        self.n_subsample = n_subsample
        
        # all subsample fields are set by new_subsample
        self.subsample_indices = []
        self.subsample_index_matrix = []
        self.subsample_oem = []
        self.new_subsample()

        # current state of the model
        self.state = None
        self.params = None
        
    def new_subsample(self):
        from cska.ska_kmers import fast_randint
        t0 = time.time()

        if not self.n_subsample:
            indices = np.arange(self.reads.N)
        else:
            self.logger.debug('subsampling {self.n_subsample} out of {self.reads.N} sequences. replacement={self.sub_replace}'.format(self=self) )
            if self.sub_replace:
                indices = fast_randint(self.n_subsample, self.reads.N)
            else:
                indices = np.random.choice(self.reads.N, size=self.n_subsample, replace= self.sub_replace)
            t1 = time.time()
            self.logger.debug('generating random subsample indices took {0:.2f} ms'.format(1000* (t1-t0)) )

        self.subsample_indices = indices

        seqm = self.reads.seqm[indices]
        t2 = time.time()
        self.subsample_index_matrix = cska.ska_kmers.seq_matrix_to_index_matrix(seqm, self.k)
        t3 = time.time()
        self.logger.debug('converting subsample to index-matrix took {0:.2f} ms'.format(1000* (t2-t3)) )

        self.subsample_oem = self.openen.oem[indices]
        self.logger.debug("entire new_subsample() run took {0:.2f} ms".format(1000*(time.time() - t0)) )
       
    def evaluate(self, params, keep=False, indices = [], sub_indices = [], rbp_conc = [], seq_only=None, do_jacobi=False, tm_update=True):
        """
        Evaluate the thermodynamic model (single protein approximation) on a sub-sample of reads. Return an SPAState instance
        """
        from cska.ska_kmers import eval_energy_model_on_index_matrix, SPA_partition_function, weighted_kmer_counts
        import time
        
        # prepare all variables
        kmer_invkd = params[:self.nA]
           
        if not len(indices):
            indices = self.subsample_indices
            im = self.subsample_index_matrix
            oem = self.subsample_oem
            # TODO: for fractional runs, needs further planning in building "merged states"?
            if len(sub_indices):
                im = im[sub_indices]
                oem = oem[sub_indices]
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
            
        if tm_update:
            t0 = time.time()
            # the model itself is implemented in Cython
            
            #p_bound, pi_kmer, openen_bin_counts, jacobi = eval_energy_model_on_index_matrix(
                #im, 
                #oem, 
                #acc_lookup,
                #kmer_invkd, 
                #rbp_conc, 
                #self.k, 
                #n_max = self.n_subsample,
                #do_jacobi = do_jacobi
            #)
            
            Z1 = SPA_partition_function(im, oem, acc_lookup, kmer_invkd, self.k, n_max = self.n_subsample)
            n = len(Z1)

            p_bound = np.zeros( (self.n_conc, n), dtype=np.float32)
            pi_kmer = np.zeros( (self.n_conc, self.nA), dtype=np.float32)

            for i in range(self.n_conc):
                Z = self.rbp_conc[i] * Z1
                p_bound[i] = Z / (Z + 1)
                pi_kmer[i] = weighted_kmer_counts(im, p_bound[i], self.k)

            t1 = time.time()

            self.logger.debug("evaluated energy model on {0} sequences in {1:.2f} ms".format(n, 1000*(t1-t0)) )
            state = SPAState(self, params, p_bound, pi_kmer)
        else:
            # skip thermodynamic model. 
            # Useful when changed parameter is not affinity (i.e. betas)
            # copy all thermodynamic model results from previous state.
            state = SPAState(self, params, self.state.p_bound, self.state.pi_kmer, self.state.openen_bin_counts, self.state.jacobi)

        if keep:
            self.params = params
            self.state = state

        return state
    
    def param_name(self, i):
        if i < self.nA:
            return self.kmers[i]
        else:
            return "beta{0}".format(i - self.nA)

    def store_params(self, fname):
        self.logger.info("storing current parameters to '{0}'".format(fname))
        with file(fname, 'w') as f:
            for i in xrange(len(self.params)):
                f.write('{0}\t{1}\n'.format(self.param_name(i), self.params[i]))

    def load_params(self, fname):
        params = []
        self.logger.info("reading parameters from '{0}'".format(fname))
        with file(fname, 'r') as f:
            for line in f:
                params.append(float(line.split('\t')[1]))

        #self.evaluate(np.array(params, dtype=np.float32), keep=True)
        return np.array(params, dtype=np.float32)

#class Stepper(object):
    #def __init__(self, opt, mdl):
        #self.opt = opt
        #self.mdl = mdl
 
    
class ModelOptimization(object):
    def __init__(self, reads, openen, k, R_obs, R_err=[], known_params = [], rbp_conc=[40.], n_subsample=0, sub_replace=False, aff0=1e-6, aff_min=1e-12, aff_max=1000, param_file=None, seq_only=False):
        self.k = k
        self.kmers = np.array(list(cska.ska_kmers.yield_kmers(self.k)))
        self.t = 0
        self.logger = logging.getLogger('ModelOptimization')

        # RBNS input sample to iterate on
        self.reads = reads
        self.openen = openen
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)

        # observations to fit to
        self.R_obs = R_obs
        self.R_err = R_err
        
        # bounds for the affinity parameters
        self.aff0 = aff0
        self.aff_min = aff_min
        self.aff_max = aff_max

        # the model to be trained
        self.mdl = SPAModel(reads, openen, k, self.rbp_conc, n_subsample=n_subsample, sub_replace=sub_replace, seq_only=seq_only)
        
        # monitor progress
        self.errors = [] #self.global_error(self.current.R), ]
        self.rel_improvements = []
        
        # initialize parameters: flat affinity vector and background from lowest percentile
        self.nA = 4**k
        
        
        # set initial state of the model
        self.current = None
        if param_file:
            self.update(self.mdl.load_params(param_file), np.Inf, "resuming from {0}".format(param_file))
        else:
            params = np.zeros(self.nA + len(rbp_conc), dtype=np.float32)
            params[0:self.nA] += aff0
            betas = self.estimate_background()
            params[-len(betas):] = betas
            self.update(params, np.Inf, "initialize")
        
        

        # in case we know some parameters, use this as a reference
        if len(known_params):
            self.known_params = known_params
        else:
            self.known_params = np.ones(self.current.params.shape, dtype=np.float32) * np.nan

        # iterative kmer selection 
        self.blocked_kmers = []
        self.kmer_last_improvement = {}
        self.kmer_last_updated = defaultdict(int)
        self.last_kmer_update = None

    def estimate_background(self, q=1.):
        return np.nanpercentile(self.R_obs, q, axis=1)
        
    def correlation(self):
        return np.array([np.corrcoef(np.log(Ro), np.log(Rp))[0][1] for Ro, Rp in zip(self.R_obs, self.current.R)])
    
    def kmer_errors(self, R_new):
        return R_new - self.R_obs
        
    def kmer_error_conc(self, kmer_index, R_new, conc_i):
        return R_new[kmer_index] - self.R_obs[conc_i, kmer_index]
        
    def global_error(self, R_new):
        """used"""
        #return (((self.R_obs - R_new)**2)*self.R_obs).sum()
        return (self.kmer_errors(R_new)**2).sum()
    
    def global_error_conc(self, R_new, conc_i):
        return (self.kmer_errors(R_new)[conc_i,:]**2).sum()

    def find_worst_kmer(self, n_max=100, n_blocked=5, n_avg=5, ttl=10):
        
        """
        ideas: 
            * take into account if over-under-representation is systematic or only seen at some concentrations (contradicted at others?)
            * start preferring kmers whose error peaks at low concentrations (high affinity), then move to kmers whose error peaks at intermediate or high concentrations (lower affinity)
            * perhaps overall R-value correlation can serve as guide? (> .9 do a round of low affinity optimizations in fixed energy background?)
        """
        badness = np.fabs(self.kmer_errors(self.current.R)).mean(axis=0)
        
        #score  = np.sqrt(badness) #/ (self.heat + 1)
        self.logger.debug("kmer\tscore\tbadness\tdt\texpect\t")
        cand = []
        for j,i in enumerate(badness.argsort()[::-1][:n_max]):          
            if i in self.blocked_kmers:
                state = 'B'
            else:
                state = ' '
            
            if i in self.kmer_last_improvement:
                rel = self.kmer_last_improvement[i]
            elif len(self.rel_improvements):
                rel = np.array(self.rel_improvements[-n_avg:]).mean()
            else:
                rel = 1
                           
            dt = self.t - self.kmer_last_updated[i]
            
            score = badness[i] * dt * rel
            if j < 20:
                self.logger.debug("{0} {1}\t{2:.2f}\t{3:.2f}\t{4}\t{5:.2f}\t{6}\t{7}\t{8}".format( self.kmers[i], state, score, badness[i], dt, rel, self.current.params[i], self.known_params[i], self.current.R[:,i] - self.R_obs[:,i] ) )
            if state != 'B':
                cand.append( (score, i) )

        score, pick = sorted(cand, reverse=True)[0]
        
        # kmer blocked list
        self.blocked_kmers.append(pick)
        while len(self.blocked_kmers) > n_blocked:
            self.blocked_kmers.pop(0)

        # kmer selection data structure maintenance
        for i,rel in self.kmer_last_improvement.items():
            if self.t - self.kmer_last_updated[i] > ttl:
                del self.kmer_last_improvement[i]

            #if rel < 1:
                ## bring low expectations back to 1. in 10 iterations.
                #self.kmer_rel_improvement[i] += .1
            #else:
                ## bring high expectations exponentially down, back to 1.
                #self.kmer_rel_improvement[i] *= .5
            
        self.logger.debug("selected {0}".format(self.kmers[pick]) )
        return pick

    def optimize_single_param(self, param_i, tm_update=True):
        params = np.array(self.current.params)

        def to_optimize(aff):
            params[param_i] = aff
            state = self.mdl.evaluate(params, do_jacobi=False, keep=False, tm_update=tm_update)
            err = self.global_error(state.R)
            #err = (self.param_errors(state.R)[:,param_i]**2).mean()
            #print aff, "->", err, state.R[:, param_i], self.R_obs[:, param_i]
            return err
            
        t0 = time.time()
        res = minimize_scalar(to_optimize, bounds = [self.aff_min, self.aff_max], method='Bounded')
        #print "line search optimization result", res.success, res.x, res.fun
        dt = time.time() - t0
        name = self.mdl.param_name(param_i)
        A0 = self.current.params[param_i]
        rel_change = (res.x - A0) / A0
        self.logger.debug("optimal {name} affinity/value search success={res.success} A={res.x} (A0={A0} rel change={rel_change}) took {dt:.2f}s".format(**locals()) )
        return res.x, res.fun

    def sweep_ls(self, vec, x0=1.):
        import matplotlib.pyplot as pp
        pp.figure()
        #pp.title("t={0} conc={1}".format(self.t, self.rbp_conc[conc_i]))
        scale = np.arange(-1,1,.02)
        err = []
        for s in scale:
            params = np.clip(self.current.params + s * vec, 1e-9, 1e3)
            state = self.mdl.evaluate(params, do_jacobi=False, keep=False)
            #err.append(self.global_error_conc(state.R, conc_i))
            err.append(self.global_error(state.R))
         
        #print err
        pp.semilogy(scale, err)
        pp.axvline(x0)
        pp.axhline(self.errors[-1])
        pp.show()
        pp.close()
    
    def sweep_param(self, i, x0=1.):
        import matplotlib.pyplot as pp
        pp.figure()
        #pp.title("t={0} conc={1}".format(self.t, self.rbp_conc[conc_i]))
        params = np.array(self.current.params)
        scale = 10**np.arange(-6,3,.1)
        err = []
        for s in scale:
            params[i] = s
            state = self.mdl.evaluate(params, do_jacobi=False, keep=False)
            #err.append(self.global_error_conc(state.R, conc_i))
            err.append(self.global_error(state.R))
         
        #print err
        pp.loglog(scale, err)
        pp.axvline(x0)
        pp.axhline(self.errors[-1])
        pp.show()
        pp.close()

        
    def line_search(self, vec, smin=0., smax=10.):
        #err0 = self.global_error_conc(self.current.R, conc_i)
        err0 = self.errors[-1]
        #rbp_conc = np.array([self.rbp_conc[conc_i],])
        
        def to_optimize(scale):
            params = np.clip(self.current.params + scale * vec, 1e-9, 1e3)
            #state = self.mdl.evaluate(params, rbp_conc=rbp_conc, do_jacobi=False, keep=False)
            #err = self.global_error_conc(state.R, conc_i)
            state = self.mdl.evaluate(params, do_jacobi=False, keep=False)
            err = self.global_error(state.R)

            return err
            
        t0 = time.time()
        smid = (smin+smax)/2.
        res0 = minimize_scalar(to_optimize, bounds = [smin, smid], method='Bounded')
        res1 = minimize_scalar(to_optimize, bounds = [smid, smax], method='Bounded')
        if res0.fun < res1.fun:
            res = res0
        else:
            res = res1
        
        if err0 < res.fun:
            self.logger.warning("WARNING: line_search optimum *raises* global error!!!")
            #opt_x = 0
            #opt_err = err0
            opt_succ = False
        else:
            opt_succ = True

        opt_x = res.x
        opt_err = res.fun
        self.sweep_ls(vec, x0=opt_x)
        self.logger.debug("optimal line search success={0} scale={1} took {2:.2f}ms".format(opt_succ, opt_x, time.time() - t0) )
        return opt_x, opt_err

    #def optimal_beta(self, conc_i, smin=0, smax=10.):
        #params = np.array(self.current.params)
        #rbp_conc = np.array([self.rbp_conc[conc_i],])

        #def to_optimize(beta):
            #params[self.nA+conc_i] = beta
            #state = self.mdl.evaluate(params, do_jacobi=False, rbp_conc=rbp_conc, keep=False)
            #err = ((self.R_obs[conc_i] - state.R[0])**2).sum()
            
            #return err
            
        #t0 = time.time()
        #res = minimize_scalar(to_optimize, bounds = [smin, smax], method='Bounded')
        ##print "line search optimization result", res.success, res.x, res.fun
        #self.logger.debug("optimal beta search for {3}nM success={0} beta={1} took {2:.2f}ms".format(res.success, res.x, time.time() - t0, self.rbp_conc[conc_i] ) )
        #return res.x
        
    def print_update_vector(self, vec, n=-1):
 
        print "-----------update vector------------"
        for i in np.fabs(vec).argsort()[::-1][:n]:
        #for i in vec.nonzero()[0]:
            if vec[i] == 0:
                break
            print "    ", self.mdl.param_name(i), self.current.params[i], "+", vec[i], "known=", self.known_params[i]
        
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

            out =  ["    ", self.mdl.param_name(i), str(self.current.params[i]), "known=", str(self.known_params[i]), status, "last grad=", str(delta[i]), R_dev]
            print "\t".join(out)
        
            
    #def update_from_jacobi(self):

        #from numpy.linalg import norm
        #corr = self.known_params - self.current.params
        #corr /= norm(corr)

        #t0 = time.time()
        #updates = []
        #for grad, R_obs, R_curr, conc in zip(self.current.sum_square_gradients(self.R_obs), self.R_obs, self.current.R, self.rbp_conc):
            ##x = R_obs - R_curr
            #u = grad / norm(grad)
            ##cos = np.dot(u, corr)
            ##angle = np.arccos(np.clip(cos, -1, 1) ) * 180./np.pi
            ###print "projection of update vector from {0} onto ideal update={1} and angle={2}".format(conc, cos, angle)
            #updates.append(u)

        #t1 = time.time()
        ##self.logger.debug("inverted Jacobi matrices and computed gradient vectors in {0:.2f}ms".format(t1-t0) )
        #return np.array(updates, dtype=np.float32)

    def converged(self, last=10, tol=1e-5):
        if len(self.rel_improvements) < last:
            return False
        
        recent = np.array(self.rel_improvements[-last:])
        #avg = recent.mean()
        dev = np.mean(recent)
        self.logger.debug("mean relative improvement over past {0} iteration steps={1}".format(last, dev) )
        if dev < tol:
            self.logger.info("convergence with mean improvement of {0}".format(dev) )
            return True
        
        return False
    
    def update(self, params, err, name):
        self.previous = self.current
        
        if self.errors:
            better = (self.errors[-1] - err)* 100./self.errors[-1]
            if better <= 0:
                self.logger.warning("unable to lower error in {1} step at t={0}".format(self.t, name))
                # reject the changes!
                params = np.array(self.current.params)
                better = 0
        else:
            better = err

        # subsamples should remain stable throughout one iteration step!
        self.mdl.new_subsample()
        self.current = self.mdl.evaluate(params, keep = True)#, do_jacobi = True) # dont use gradient for now!
        self.errors.append(self.global_error(self.current.R))
        
        self.logger.info("status after '{0}' step at t={1}, improvement was {2:.2f}%".format(name, self.t, better))
        self.logger.info("correlations: {0}".format(self.correlation()) )
        self.logger.info("most recent errors: {0}".format( self.errors[-5:] ))
        
        if self.previous:
            update = self.current.params - self.previous.params
            print "{0} step at t={1}".format(name, self.t)
            self.print_update_vector(update)

        self.t += 1
        if self.t > 1:
            self.rel_improvements.append(better)
        return better
       
    def optimize(self, reporter = None, max_iter=1000, report_interval=5, **kwargs):
        if reporter:
            reporter.plot_R_value_agreement()
        last_report = self.t
        
        while not self.converged() and self.t < max_iter:
            imp_a = self.step_kmer()
            #imp_b = self.step_gradient()
            #imp_b = self.step_beta()
            
            #self.logger.info("improvements were kmer_step: {0:.4f}% gradient: {1:.4f}%".format(imp_a, imp_b))
            #self.logger.info("improvements were kmer_step: {0:.4f}% beta: {1:.4f}%".format(imp_a, imp_b))
            
            if imp_a < 1: #and imp_b < 1:
                self.logger.warning( "Got stuck. Optimizing background parameters")
                imp = self.step_beta()
                self.logger.info("improvements were betas: {0:.4f}%".format(imp))

            self.mdl.store_params('params_t{0}.tsv'.format(self.t) )
            if reporter and self.t > last_report + report_interval:
                reporter.plot_R_value_agreement()
                last_report = self.t

        if reporter:
            reporter.plot_R_value_agreement()
                

    def step_kmer(self, show_sweep = False):
        self.logger.info("===kmer fit step===")
        kmer_i = self.find_worst_kmer()
        best, err = self.optimize_single_param(kmer_i)
        
        if show_sweep:
            self.sweep_param(kmer_i, best)
            
        params = np.array(self.current.params)
        params[kmer_i] = best
        update = params - self.current.params
        
        self.last_kmer_update = kmer_i
        self.kmer_last_updated[kmer_i] = self.t
        imp = self.update(params, err, "k-mer optimization")
        self.kmer_last_improvement[kmer_i] =  imp
        

        return imp

        
    def step_gradient(self):
        self.logger.info("===gradient descent step===")
        
        #updates = self.update_from_jacobi()
        #ls_results = []
        #for conc_i, (u, rbp_conc) in enumerate(zip(updates, self.rbp_conc)):

            #scale, err = self.line_search(u)#, conc_i)
            #ls_results.append((u*scale, err))
            ##self.sweep_ls(u, scale, conc_i)

            #self.logger.debug("line search finds minimal error={err} for scale={scale}".format(**locals()))
            ##self.print_update_vector(ls_results[-1][0])
    
        #scaled_updates, errors = np.array(ls_results).T
        #i = errors.argmin()
        #err = errors[i]

        grad = self.current.sum_square_gradient(self.R_obs)
        grad /= np.linalg.norm(grad)
        self.print_update_vector(grad)
        
        scale, err = self.line_search(grad)
                
        #self.logger.debug("best concentration is {0} nM with global error {1}".format(self.rbp_conc[i], errors[i]) )
        #update = scaled_updates[i]
        params = np.clip(self.current.params + grad*scale, 1e-9, 1e3)

        return self.update(params, err, "gradient descent")


    def step_beta(self, smin=0, smax=10.):
        params = np.array(self.current.params)
        for beta_i in range(self.nA, self.nA+self.n_conc):
            val, err = self.optimize_single_param(beta_i, tm_update=False)
            if err < self.errors[-1]:
                params[beta_i] = val

        state = self.mdl.evaluate(params, do_jacobi=False, keep=False)
        err = self.global_error(state.R)
        
        return self.update(params, err, "beta optimization")


      
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
    
