import numpy as np
import logging
import time

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
        
        pi = kappa * (np.dot(M, occ ) + bg )
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
            return r
            
        def err(r_predict, r_obs):
            #return np.mean((r_predict - r_obs)**2)
            return np.mean(np.log2(np.fabs(r_predict/r_obs))**2)
        
        def score(params):
            sum_pi, beta = params
            
            occ = r2occ(sum_pi, beta)
            print "occ top5", occ[-5:]
            r_predict = occ2r(occ, beta)
            print "r_predict top5", r_predict[-5:]
            print "r_value   top5", r_values[-5:]
            
            e = err(r_predict, r_values)
            print "score({0},{1}) -> err={2}".format(sum_pi, beta, e) 
            print occ[-20:]
            return e + (occ[occ < 0]**2).sum() + (occ[occ > 1]**2).sum()
        
        from scipy.optimize import minimize
        x0 = (0.01, 0.01)
        bounds = np.array([
            [1e-6, .9],
            [1e-6, .9]
        ])
        res = minimize(score, x0, method='SLSQP', bounds = bounds)
        print res
        sum_pi, beta = res.x
        print score(res.x), "<- optimized score"
        
        
        opt_occ = r2occ(sum_pi, beta)
        r_pred = occ2r()
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

class RBNSGenerator(CachedBase):
    def __init__(self, k, l=20, min_E=-11., seed=None, temp=22, **kwargs):
        
        CachedBase.__init__(self)

        self.k = k
        self.l = l
        self.min_E = min_E
        self.temp = temp
        self.RT = (temp + 273.15) * 8.314459848 / 4.184E3 # RT in kcal/mol

        if seed: 
            np.random.seed(seed)
            cska.ska_kmers.rand_seed(seed)

        self.kmer_energies = np.array(sorted(RBNSGenerator.energy_distribution(min_E=min_E, N=4**k, **kwargs))[::-1], dtype=np.float32) / self.RT
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

    def assign_experimental_input(self, real_input):
        from cska.rbns_reads import RBNSReads
        reads = RBNSReads(real_input)
        nt_freq = reads.kmer_frequencies(1) / 4.
        di_freq = reads.kmer_frequencies(2).reshape(4,4) / 16.
        di_freq /= di_freq.sum(axis=1)[:, np.newaxis] # normalize rows to one
        
        self.input_reads = reads
        self.input_nt_freq = nt_freq
        self.input_di_freq = di_freq
        
        return nt_freq, di_freq
        
    def generate_input_reads(self, N=20000000, store=""):
        if self.input_reads:
            self.logger.debug("generating random sequence matrix, mimicking '{0}'".format(self.input_reads.name))
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
        self.logger.debug("simulating bound sequence matrix, mimicking input from '{0}'".format(self.input_reads.name))
        
        t0 = time.time()
        seqm, n_bound, n_ns = cska.ska_kmers.simulate_rbns_reads(self.l, N, self.k, self.input_nt_freq, self.input_di_freq, self.kmer_energies, P, p_ns)
        dt = time.time() - t0
        
        print "specific", n_bound, "non-specific", n_ns
        rps = N / dt
        self.logger.debug("took {0:.3f} seconds. {1:.1f} reads per second".format(dt, rps) )
        
        if store:
            cska.ska_kmers.write_seqm(seqm, file(store, 'w') )

        return seqm

    def predict_occupancies(self, P=320., store=""):
        Kd = np.exp(self.kmer_energies)*1e9
        occ = P / (P+Kd)
        
        if store:
            with file(store,'w') as f:
                for kmer, o in zip(cska.ska_kmers.yield_kmers(self.k), occ):
                    f.write("{0}\t{1}\n".format(kmer, o) )
            
        return occ
    
    def predict_r_values(self, P=320., store=""):
        occ = self.predict_occupancies(P)
        k = self.k
        kappa = self.input_reads.kmer_frequencies(k) / 4**k
        omega = (occ * kappa).sum()
        print "omega", omega
        pi = np.empty(occ.shape, dtype=np.float32)
        kfreqs = {}
        for x in range(1,k):
            kfreqs[x] = self.input_reads.kmer_frequencies(x) / 4**x
        
        I = occ.argsort()[::-1]
        for i in I:
            kmer = cska.ska_kmers.index_to_seq(i, k)
            p = occ[i] * kappa[i]
            #print "direct", kmer, occ[i], kappa[i], "pi_naked", p
            
            for x in range(1,k):
                weights, shifts = cska.ska_kmers.weighted_kmer_shifts(i, k, self.l, x, kfreqs[x]) 
                for s,f in zip(shifts, weights):
                    #print s, f, type(s), k 
                    ks = cska.ska_kmers.index_to_seq(int(s), self.k)
                    #print "shifted", ks, "weight", f, "abundance", kappa[s]
                    p += occ[s] * f * kappa[i]

                #print "pi_with_shift_{0}".format(x), p
            #print "pi_with_all_shifts", p
            pi[i] = p

        # adding baseline
        l = self.l - k + 1
        pi += omega*(l-2.*(k-1)) * kappa 
        
        r = pi / pi.sum() / kappa
        I = r.argsort()[::-1]
        for i in I[:10]:
            print i, cska.ska_kmers.index_to_seq(i, k), pi[i], r[i]

        if store:
            with file(store,'w') as f:
                for kmer, o in zip(cska.ska_kmers.yield_kmers(self.k), r):
                    f.write("{0}\t{1}\n".format(kmer, o) )
        
        return r

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
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    gen = RBNSGenerator(5,l=40, seed=47110815)
    gen.assign_experimental_input("/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads")
    print gen
    gen.energy_plot()

    #gen.energy_plot()
    #gen.generate_input_reads(store="input.reads")

    #gen.generate_bound_reads(store="bound_320.reads", P=320., p_ns=0.00, N=20000000)
    #occ = gen.predict_occupancies(P=320., store="occ_320.tsv")
    
    rbp_conc = np.array([40, 80, 160, 320])
    rbp_conc = np.array([1.,5.])
    r_matrix = []
    for P in rbp_conc:
        r = gen.predict_r_values(P=P, store="r_{0}.tsv".format(P))
        occ = gen.predict_occupancies(P=P, store="occ_{0}.tsv".format(P))
        r_matrix.append(r)
        #gen.generate_bound_reads(store="bound_{0}.reads".format(P), P=P, p_ns=0.00, N=20000000)
        
    kmers, k_est, dG = gen.linear_fit(rbp_conc, np.array(r_matrix), n_top=20)
    
    M, M_inv = gen.crosstalk_matrix_and_inverse()
    nzero = (M == 0).nonzero()[0].size
    print "sparseness of M", nzero / float(M.size)

    import matplotlib.pyplot as pp
    #pp.figure()
    #pp.loglog(r, rm, 'ok')
    
    pp.figure(figsize=(8,4))
    pp.subplot(121)
    pp.imshow(np.log10(M), cmap=pp.get_cmap('plasma'))
    pp.colorbar(label=r'$\log_{10}(M)$',fraction=0.046, pad=0.04)

    pp.subplot(122)
    pp.imshow(np.log10(M_inv), cmap=pp.get_cmap('plasma'))
    pp.colorbar(label=r'$\log_{10}(M^{-1})$',fraction=0.046, pad=0.04)
    pp.tight_layout()
    pp.savefig('crosstalk_matrix.pdf')
        
    pp.figure()
    pp.title("inferred binding energies")
    print gen.kmer_energies[-20:].shape, dG.shape
    pp.plot(gen.kmer_energies[-20:]*gen.RT, dG, 'ob')
    pp.xlabel("simulated energies [kcal/mol]")
    pp.ylabel("inferred energies [kcal/mol]")

    pp.show()
    #pp.savefig('predict.pdf')
    
