import numpy as np
import scipy
import logging
import time
import sys
import os
from collections import defaultdict
from scipy.optimize import minimize, brentq, minimize_scalar

import cska.ska_kmers 
import cska.ska_kmers as cyska

from cska.rbns_reads import RBNSReads

from cska.caching import CachedBase, cached, pickled
from cska.affinity import Kd_to_kcal, kcal_to_Kd, AffinityDistribution
from cska.scheduler import ParamUpdateScheduler
#from cska.psam import PSAMState
#from cska import timed
#from cska.optimize import ModelOptimization

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

                 
class ModelOptimization(object):
    def __init__(self, k, rbns_analysis, known_params = [], rbp_conc=[40.], out_path="./", n_subsample=0, sub_replace=False, aff0=1e-6, aff_min=1e-12, aff_max=1000, param_file=None, seq_only=False, tm_refresh=.01, sched_params = {}, beta_interval = .01, scale_interval=100000000000000, reporter=None, mdl_params=[],t0=0): # scale_interval=.02
        self.k = k
        self.nA = 4**k

        # RBNS input sample to iterate on
        self.rbns_analysis = rbns_analysis        
        self.all_reads = rbns_analysis.reads
        self.reads = self.all_reads[0] # try and phase out! TODO needs cleanup
        self.input_reads = self.all_reads[0]
        self.storages = self.rbns_analysis.acc_storages
        self.openen = self.storages[0].get_discretized(k)
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)

        self.n_params = self.nA + self.n_conc

        self.beta_interval = int(beta_interval * self.nA)
        self.scale_interval = int(scale_interval * self.nA)
        self.strategy_window = int(.1 * self.nA) # number of steps average last error to
        # decide if we want to switch from local to global optimization
        self.last_beta = 0
        self.last_scale = 0
        
        self.kmers = np.array(list(cska.ska_kmers.yield_kmers(self.k)))
        self.t = t0
        self.tm_refresh = int(tm_refresh * self.nA)
        self.last_tm_refresh = 0

        self.logger = logging.getLogger('ModelOptimization')
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.opt_path = os.path.join(self.out_path, 'opt/{0}mers'.format(self.k))
        if not os.path.exists(self.opt_path):
            os.makedirs(self.opt_path)

        self.reporter=reporter
        # observations to fit to
        self.R_obs, self.R_err = self.rbns_analysis.R_value_matrix(k)
        
        # bounds for the affinity parameters
        self.aff0 = aff0
        self.aff_min = aff_min
        self.aff_max = aff_max

        # the model to be trained
        self.mdl = SPAModel(self.reads, self.openen, k, self.rbp_conc, n_subsample=n_subsample, sub_replace=sub_replace, seq_only=seq_only)
        
        # monitor progress
        self.errors = [] #self.global_error(self.current.R), ]
        self.correlations = []
        self.rel_improvements = []
        self.param_local_fit = True
        
        # set initial state of the model
        self.previous = None
        self.current = None
        if param_file:
            #self.update(self.mdl.load_params(param_file), np.Inf, "resuming from {0}".format(param_file))
            self.logger.info("resuming from {0}".format(param_file))
            self.current = self.mdl.evaluate(self.mdl.load_params(param_file), keep = True)
            
            ## TESTING: compare observed and predicted affinity distributions
            import matplotlib.pyplot as pp
            aff_in = AffinityDistribution(self.mdl, self.all_reads[0], self.storages[0].get_discretized(self.k))
            bg = aff_in.get_affinity_distribution()
            print "input", bg
            for i, conc in enumerate(self.rbp_conc):
                beta = self.mdl.params[self.mdl.nA+i]
                pred = aff_in.predict_affinity_distribution(conc, beta=beta, bg=bg[0])
                print "predicted", pred
                aff = AffinityDistribution(self.mdl, self.all_reads[i+1], self.storages[i+1].get_discretized(self.k))
                obs = aff.get_affinity_distribution()
                print "observed", obs
                x = (aff_in.bins[1:] + aff_in.bins[:-1])/2.
                pp.plot(x, pred[0], '-', label='pred {0}nM'.format(conc) )
                pp.plot(x, obs[0], '.', label='obs {0}nM'.format(conc) )
                pp.show()
                
            pp.xlabel('predicted')
            pp.ylabel('observed')
            pp.legend()
            pp.show()
            1/0
            
        elif len(mdl_params):
            # start with given parameterization
            params = np.array(mdl_params, dtype=np.float32)
            assert len(params) == self.nA + len(rbp_conc)
            self.current = self.mdl.evaluate(params, keep = True)
        else:
            # start from scratch
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
        l = self.all_reads[0].L
        betas = np.nanpercentile(self.R_obs, q, axis=1) / (l - self.k + 1)
        self.logger.info("estimated background={betas} from {q} percentile of R-value distribution".format(**locals()) )
        return betas
        
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
        return ((self.kmer_errors(R_new)**2).sum(axis=1) * (1 + lin_err) ).mean()
    
    def global_error_conc(self, R_new, conc_i):
        return (self.kmer_errors(R_new)[conc_i,:]**2).mean()
    
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
            
        scale = 10**np.arange(np.log10(self.aff_min),np.log10(self.aff_max),.1)
        err = []
        loc = []
        for s in scale:
            params[param_i] = s
            state = opt.evaluate(params, tm_update=tm_update)
            #err.append(self.global_error_conc(state.R, conc_i))
            err.append(self.global_error(state.R))
            error = ((state.R[:,param_i] - self.R_obs[:,param_i])**2).sum()
            loc.append(error)
         
        #print err
        pp.loglog(scale, err)
        pp.loglog(scale, loc)
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
      

    def converged(self, last=20, tol=.001):
        if len(self.rel_improvements) < last:
            return False
        
        tol *= 4**(- (self.k-5) ) # expect slower convergence for higher k, bc each kmer alone will have less to explain
        recent = np.array(self.errors[-last:])
        mean = recent.mean()
        #avg = recent.mean()
        dev = recent.std() / mean 
        self.logger.debug("mean error over past {0} iteration steps={1}, relative change={2}".format(last, mean, dev) )
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
                self.step_betas()
                self.last_scale = self.t
                self.last_beta = self.t

            if self.t - self.last_beta > self.beta_interval:
                self.step_betas()
                self.last_beta = self.t


        if reporter:
            reporter.plot_R_value_agreement()              
            reporter.close()

    def pwm_fit(self, max_iter=None, snapshots=False, k_max=7, start_kmer="", **kwargs):
        from cska.pwm import PSAM
        if self.reporter:
            self.reporter.tick(self.t)
                
        # initialize the model by fitting the worst kmer and then background estimates once.
        if start_kmer:
            param_i = cyska.seq_to_index(start_kmer)
            name = start_kmer
        else:
            param_i = self.sched.find_worst_param()
            name = self.mdl.param_name[param_i]

        best, err, new_state = self.optimize_single_param(param_i, local = True)
        imp = self.update(new_state, err, "initial fit of {name} -> {best:.3e} (err={err:.3e})".format(**locals()))
        self.step_betas()

        kmer_pwm_map = {}
        
        def optimize_pwm_set(seed_i):
            # begin actual optimization 
            
            # first, update the error scores!
            self.sched.find_worst_param() 

            # optimize kmer and all of its 1-mismatch relatives, ordered by their scheduler scores
            kmer_indices = self.sched.pwm_set(seed_i, self.k)
            for i in kmer_indices:
                name = self.mdl.param_name[i]
            
                best, err, new_state = self.optimize_single_param(i, local = True)
                imp = self.update(new_state, err, "local kmer optimization {name} -> {best:.3e} (err={err:.3e})".format(**locals()))
            
            kmer_aff = [new_state.params[i] for i in kmer_indices]
            kmers = [self.mdl.param_name[i] for i in kmer_indices]

            pwm = PSAM.from_kmer_variants(kmers, np.array(kmer_aff))
            for mer in kmers:
                kmer_pwm_map[mer] = pwm
            
            # output/storage of current results
            print pwm
            logo_path = os.path.join(
                self.opt_path, 
                'pwm_seed_{pwm.kmer_seed}_t={self.t}.pdf'.format(**locals()) 
            )
            logo_title = 'A0={pwm.A0}'.format(**locals())
            pwm.save_logo(logo_path, title=logo_title)

            if self.reporter:
                self.reporter.plot_R_value_agreement()
       
            self.logger.info("correlations: {0}".format(self.correlation()) )
            return pwm
        
        def is_shifted(kmer, s_max=2):
            k = len(kmer)
            for x in range(1,s_max+1):
                for pad in list(cyska.yield_kmers(x)):
                    rshifted = pad + kmer[:k-x]
                    lshifted = kmer[x:]+pad
                    #print "l", x, kmer, lshifted
                    #print "r", x, kmer, rshifted
                    if lshifted in kmer_pwm_map:
                        return -x, kmer_pwm_map[lshifted]
                    elif rshifted in kmer_pwm_map:
                        return x, kmer_pwm_map[rshifted]

            return 0, None

        # initially, param_i points to the kmer with highest R-value
        while not self.converged():
            
            self.logger.error("optimzing PWM set seeded by {0}".format(name) )
            #for t in range(int(.75*self.k)):
            for t in range(2):
                # optimize a PWM set multiple times
                optimize_pwm_set(param_i)
                self.step_betas()
                if self.reporter:
                    self.reporter.plot_R_value_agreement()
        
            param_i = self.sched.find_worst_param()
            name = self.mdl.param_name[param_i]
            
            # debug output
            ranked = self.sched._residual.argsort()[::-1]
            print self.sched._residual[ranked]
            
            self.sched.debug_monitor(param_list = ranked[:10], title="candidate search")

            
            # see if this kmer is included in an existing PWM set
            if name in kmer_pwm_map:
                # gather the set of related kmers belonging to this PWM set
                seed = kmer_pwm_map[name].kmer_seed
                param_i = cyska.seq_to_index(seed)
                # optimization touches kmers ordered by scheduler. So the original param_i *will* come first!
                self.logger.error("next up is {1} ->selecting PWM set seeded by {0}".format(name, seed) )
            else:
                shift, pwm = is_shifted(name)
                self.logger.error("shift = {0} pwm = {1}".format(shift,pwm))
                if abs(shift) > 1:
                    # this kmer is shifted too far. Ignore for now by marking as "updated"
                    self.sched.update(param_i)
                    continue
                elif shift == None:
                    # this has no similarity to previously fitted kmers. Start a new PWM:
                    pass
                elif abs(shift) == 1:
                    # need to switch to k+1 model!
                    if self.k+1 > k_max:
                        self.logger.warning("reached k_max, ending optimization")
                        break
                    
                    if shift < 0:
                        newmer = name[:-shift] + pwm.kmer_seed
                    else:
                        newmer = pwm.kmer_seed + name[-shift:]

                    self.logger.info("switching to k+1 = {0} at t={1} for k+1 mer {2}".format(self.k+1, self.t, newmer) )
                    new_param = self.params_for_k_extension(self.k)
                    #print "NEWPARAM", len(new_param), 4**(self.k+1)
                    new_opt = ModelOptimization(self.k+1, self.rbns_analysis,
                        rbp_conc=self.rbp_conc, 
                        out_path=self.out_path, 
                        mdl_params = new_param,
                        t0 = self.t
                    )
                    new_opt.errors = self.errors
                    new_opt.correlations = self.correlations
                    new_opt.rel_improvements = self.rel_improvements
                    
                    #new_opt.t = self.t
                    new_opt.reporter = self.reporter
                    self.reporter.opt = new_opt
                    return new_opt.pwm_fit(max_iter=max_iter, snapshots=snapshots, start_kmer = newmer, **kwargs)

        self.mdl.store_params(os.path.join(self.opt_path, '{self.k}mer_affinities.tsv'.format(self=self)))
        
        #if self.reporter:
            #self.reporter.close()
        # notify the scheduler of the param change and its consequences
        #self.sched.param_changed(param_i, self.t, imp)

    def params_for_k_extension(self, k):
        """
        generate kmer parameters for k+1 by expanding an existing table for k
        """
        new = np.zeros(4**(k+1) + len(self.rbp_conc), dtype=np.float32)
        params = self.mdl.params
        
        # copy over beta values
        new[-self.n_conc:] = params[-self.n_conc:]
        
        # copy each kmer value we presently have, to the 8 k+1 
        # values (4 left-padded, 4 right-padded) in the new array
        nts = np.arange(4)

        for i in np.arange(self.nA):
            for nt in nts:
                left = i | (nt << (k*2))
                right = (i << 2) | nt
                new[left] = params[i]
                new[right] = params[i]
        
        return new
        
    def step_param(self, show_sweep = False):
        self.logger.info("===parameter optimization===")
        param_i = self.sched.find_worst_param()
        
        #self.sweep_param(param_i)
               
        if self.param_local_fit and len(self.rel_improvements) >= self.strategy_window:
            avg_improvement = np.mean(self.rel_improvements[-self.strategy_window:])
            self.logger.info("performing local fits, past rel. improvements average={0}".format(avg_improvement) )
            
            if avg_improvement < 0:
                self.logger.info("switching to global optimization at step {0}".format(self.t))
                self.param_local_fit = False
            
        best, err, new_state = self.optimize_single_param(param_i, local = self.param_local_fit)

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
            best, err, new_state = self.optimize_single_param(param_i, ground_state=ground_state, local=False)
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
                best, err, new_state = self.optimize_single_param(param_i, ground_state=state, local=False)
                state.params[param_i] = best
                
            err = self.global_error(new_state.R)
            #print "global error={0} at scale={1} after beta fit".format(err, scale)
            
            errors.append(err)
            return err

        res = minimize_scalar(to_optimize, bounds = (.1,1), method='Bounded')
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
            
    def optimize_single_param(self, param_i, ground_state=None, local=True):
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
            params[param_i] = aff #* .0001
            state = opt.evaluate(params, tm_update=tm_update, ground_state = ground_state)
            
            if local:
                err = ((state.R[:,param_i] - self.R_obs[:,param_i])**2).sum()
            else:
                err = self.global_error(state.R)

            #print params[param_i], err
            return err

        def minimize_logspaced(func, bounds = [], n_samples = 10, **kwargs):
            """
            first evaluate at log-spaced sampling points along parameter range
            then select at most 3 orders of magnitude around the lowest observed value
            for Brent optimization. Requires pos. valued bounds!
            """
            
            bmin = bounds.min()
            bmax = bounds.max()
            
            lmin = np.log10(bmin)
            lmax = np.log10(bmax)
            
            sample_x = 10**np.linspace(lmin, lmax, n_samples)
            samples = np.array([to_optimize(x) for x in sample_x])
                
            #print "logspaced sample", zip(sample_x, samples)
            i = samples.argmin()
            li = max(0, i -1)
            ri = min(n_samples-1, i+1)
            
            brent_min = sample_x[li]
            brent_max = sample_x[ri]
            #print "search optimum between", brent_min, brent_max
            
            res = minimize_scalar(func, bounds = np.array([brent_min, brent_max]), method='Bounded', **kwargs)
            return res
            
        t0 = time.time()
        #s_mid = (self.aff_max + self.aff_min)/2.
        #bounds_a = 10000 * np.array([self.aff_min, 1.5*s_mid])
        #bounds_b = 10000. * np.array([.75*s_mid, self.aff_max])
        #print "first fit bounds", bounds_a
        #res_a = 
        #print "second fit", bounds_b
        #res_b = minimize_scalar(to_optimize, bounds = bounds_b, method='Bounded')
        #if res_a.fun < res_b.fun:
            #res = res_a
        #else:
            #res = res_b
        res = minimize_logspaced(to_optimize, bounds = np.array([self.aff_min, self.aff_max]) )

        best = res.x #* 0.0001
        dt = time.time() - t0
        name = self.mdl.param_name[param_i]
        A0 = ground_state.params[param_i]
        rel_change = (best - A0) / A0

        if local:
            mode = 'LOCAL'
        else:
            mode = 'GLOBAL'
        
        self.logger.debug("{mode} optimal {name} affinity/value search success={res.success} A={best} (A0={A0} rel change={rel_change}) took {dt:.2f}s".format(**locals()) )

        params[param_i] = best
        new_state = opt.evaluate(params, tm_update=tm_update)
        
        if best > .1 * self.aff_max and param_i < self.nA:
            print self.kmers[param_i]
            print "local", local
            print "res", res
            #print "res_b", res_b
            print "error at minimum", res.fun
            print "optimal affinity", best

            self.sweep_param(param_i, x0=best)

        err = self.global_error(new_state.R)
        return best, err, new_state


    def update(self, new_state, err, name, tick=True):
        from copy import copy
        self.previous = copy(self.current)
        
        if self.errors:
            better = (self.errors[-1] - err)* 100./self.errors[-1]
            #if better <= 0:
                #self.logger.warning("unable to lower error in {1} step at t={0}".format(self.t, name))
                ## reject the changes!
                ##params = np.array(self.current.params)
                #better = 0
                #new_state = self.current
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
        corr = self.correlation()
        self.correlations.append(corr)
        self.logger.debug("correlations: {0}".format(corr) )
        self.logger.debug("most recent errors: {0}".format( self.errors[-5:] ))
        
        if self.previous:
            update = self.current.params - self.previous.params
            #print "{0} step at t={1}".format(name, self.t)
            #self.print_update_vector(update)

        if tick:
            self.t += 1
            if self.reporter:
                self.reporter.tick(self.t)

        if self.t > 1:
            self.rel_improvements.append(better)
        return better
       
