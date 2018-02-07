__license__ = "MIT"
__version__ = "0.9.8"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import os
import sys
import logging
import numpy as np
from scipy.optimize import minimize_scalar
import cska.ska_kmers as cyska
import time
from cska.caching import cached, pickled, CachedBase
from cska.sc import SelfConsistency
#from cska import timed

"""
Single Protein Approximation (SPA) thermodynamic model for an RNA bind'n'seq (RBNS) experiment.
This module contains class wrappers around the lower-level Cython code and encapsulates the implementation of the thermodynamic model itself. Input to the model are essentially an efficient binary representation of (millions of) sequences randomly sampled from the RBNS pool, and the corresponding kmer accessibilities throughout those sequences. These constitute the sequence matrix (seqm) and accessibility (originally open-energy) matrix (oem) variables.
As a first step, the model computes the one-protein partition function for interaction of one RBP with each sequence (sum over all possible binding sites in a sequence). From this and the rbp_concentration follows the probability that the sequnce is bound by at least one molecule (p_bound), from which in turn follows the expected kmer composition of the pulldown.

Further perks are support for a non-specific contribution from background binding (beta parameters) and determining the self-consistent concentration of actually *free* RBP.
"""

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

    def __mul__(self, scale):
        """
        multiply all affinities by 'scale'. For scale ~1 rounding errors
        should be small, so we do not need to recompute the partition 
        function, just rescale that as well.
        """
        
        # parameter scale <-> part. function scale
        params = np.array(self.params)
        params[:self.mdl.nA] = self.params[:self.mdl.nA] * scale
        Z_scaled = self.Z1 * scale
        
        # update dependent values
        rbp_free = self.mdl._spa_free_protein(Z_scaled, self.mdl.rbp_conc)
        p_bound = self.mdl._spa_p_rna_bound(Z_scaled, rbp_free)
        pi_kmer = self.mdl._spa_kmer_pi(p_bound, self.mdl.subsample_index_matrix)

        # construct a new state object
        state = SPAState(self.mdl, params, Z_scaled, p_bound, pi_kmer, rbp_free)
        return state

    def store_Z(self, fname):
        np.save(fname, self.Z1)

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
        self.kmer_i = kmer_i
        
        kmer_hits = cyska.index_matrix_rows_with_kmer(self.mdl.subsample_index_matrix, self.mdl.k, kmer_i)
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
        
        self.mdl.logger.debug('model.SPAPartition of {0} sequences'.format(len(self.im_kmer)) )
        
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
    """
    Single Protein Approximation (SPA) thermodynamic model of RBNS.
    """
    def __init__(self, reads, openen, k, protein_conc, T=22, sub_replace=True, seq_only=False, out_path="./", n_subsample=100000, params = None):
        self.k = k
        self.nA = 4**k
        self.rbp_conc = np.array(protein_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)

        self.param_name = np.array(list(cyska.yield_kmers(self.k)) + ["beta{0}".format(i) for i in range(self.n_conc)])
        self.param_index = {}
        for i, name in enumerate(self.param_name):
            self.param_index[name] = i

        self.kmers = self.param_name

        self.T = T
        self.RT = (self.T + 273.15) * 8.314459848/4.184E3 # RT in kcal/mol
        self.logger = logging.getLogger('model.SPAModel')

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
        
        # current state of the model
        self.state = None
        self.params = params
        
    def new_subsample(self):
        t0 = time.time()

        if not self.n_subsample:
            if len(self.subsample_indices):
                # de-activated subsampling and we already have everything in place!
                return
            
            indices = np.arange(self.reads.N)
        else:
            self.logger.debug('subsampling {self.n_subsample} out of {self.reads.N} sequences. replacement={self.sub_replace}'.format(self=self) )
            if self.sub_replace:
                indices = cyska.fast_randint(self.n_subsample, self.reads.N)
            else:
                indices = np.random.choice(self.reads.N, size=self.n_subsample, replace= self.sub_replace)
            t1 = time.time()
            self.logger.debug('generating random subsample indices took {0:.2f} ms'.format(1000* (t1-t0)) )


        self.subsample_indices = indices
        self.subsample_index_matrix = self.reads.get_index_matrix(self.k, indices=indices)
        self.subsample_oem = self.openen.oem[indices]
        self.logger.debug("entire new_subsample() run took {0:.2f} ms".format(1000*(time.time() - t0)) )
        
    def _spa_partition_function(self, im, oem, acc_lookup, kmer_invkd):
        Z1 = cyska.SPA_partition_function(im, oem, acc_lookup, kmer_invkd, self.k, n_max = self.n_subsample, openen_ofs = self.openen.ofs - self.k + 1)
        return Z1
    
    def _spa_free_protein(self, Z1, rbp_conc):
        sc = SelfConsistency(Z1, self.reads.rna_conc, bins=1000)
        rbp_free = [sc.free_rbp(total) for total in rbp_conc]
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
        n_conc, n = p_bound.shape
        
        pi = np.zeros( (n_conc, self.nA), dtype=np.float32)
        for i in range(n_conc):
            pi[i] = cyska.weighted_kmer_counts(im, p_bound[i], self.k)

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
            im = cyska.seq_matrix_to_index_matrix(seqm, self.k)
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
            f.write('# {0}mer\taffinity [1/nM]\tKd [nM]\n'.format(self.k) )
            for i in xrange(len(self.params)):
                f.write('{0}\t{1}\t{2}\n'.format(self.param_name[i], self.params[i], 1./self.params[i]))

    def extrapolation(self, k, fname=""):
        """
        Using current affinities, extrapolate expected affinties for k > self.k
        """
        assert k > self.k
        kmers = list(cyska.yield_kmers(k))
        seqm = cyska.read_raw_seqs_chunked(kmers, chunklines=len(kmers))
        index_matrix = cyska.seq_matrix_to_index_matrix(seqm, self.k)
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
                if line.startswith('#'): 
                    continue
                params.append(float(line.split('\t')[1]))

        #self.evaluate(np.array(params, dtype=np.float32), keep=True)
        return np.array(params, dtype=np.float32)

