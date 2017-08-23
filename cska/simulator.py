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
