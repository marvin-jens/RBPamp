import numpy as np
import scipy
import logging
import time
import sys
import os
from collections import defaultdict
from scipy.optimize import minimize, brentq, minimize_scalar
import cska.cyska as cyska
from cska.rbns_reads import RBNSReads
from cska.caching import CachedBase, cached, pickled

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
        #self.kmers = cyska.yield_kmers(k)
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
        kmers = np.array(list(cyska.yield_kmers(k)))

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
            self.tracked_kmers[mer] = cyska.seq_to_index(mer)

            
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
        import cyska
        I = np.array([cyska.seq_to_index(mer) for mer in kmers])
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
