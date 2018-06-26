__license__ = "MIT"
__version__ = "0.9.8"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import os
import sys
import logging
import numpy as np
from scipy.optimize import minimize_scalar
import cska.cyska as cyska
import time
from cska.caching import cached, pickled, CachedBase
from cska.sc import SelfConsistency
import cska
from cska.affinity import *
#from cska import timed

"""
Single Protein Approximation (SPA) thermodynamic model for an RNA bind'n'seq (RBNS) experiment.
This module contains class wrappers around the lower-level Cython code and encapsulates the implementation of the thermodynamic model itself. Input to the model are essentially an efficient binary representation of (millions of) sequences randomly sampled from the RBNS pool, and the corresponding kmer accessibilities throughout those sequences. These constitute the sequence matrix (seqm) and accessibility (originally open-energy) matrix (oem) variables.
As a first step, the model computes the one-protein partition function for interaction of one RBP with each sequence (sum over all possible binding sites in a sequence). From this and the rbp_concentration follows the probability that the sequnce is bound by at least one molecule (p_bound), from which in turn follows the expected kmer composition of the pulldown.

Further perks are support for a non-specific contribution from background binding (beta parameters) and determining the self-consistent concentration of actually *free* RBP.
"""

class SPAState(object):
    def __init__(self, mdl, params, Z1, p_bound, pi_kmer, rbp_free, jacobi = [], sc=None):
        self.mdl = mdl
        self.sc=sc
        self.params = params
        self.A = params[:self.mdl.nA] # affinities
        self.betas = params[self.mdl.nA:] # background coefficients
        self.Z1 = Z1
        self.p_bound = p_bound
        self.pi_kmer = pi_kmer
        self.rbp_free = rbp_free
        #self.openen_bin_counts = openen_bin_counts
        self.jacobi = jacobi

        self.N = self.mdl.n_subsample
        if not self.N:
            self.N = self.mdl.reads.N
            
        #print "BETAS", self.betas
        self.pd_freq = self.pi_kmer + self.betas[:, np.newaxis] * self.mdl.f0 * self.N
        self.pd_sum = self.pd_freq.sum(axis=1)
        self.R = (self.pd_freq/ self.pd_sum[:, np.newaxis] ) / self.mdl.f0[np.newaxis,:]

    def R_at_k(self, k):
        """
        Using current self.k-mer model, predict R values at another k
        """
        im = self.mdl.reads.get_index_matrix(k)
        f0 = self.mdl.reads.kmer_frequencies(k)
        f0 /= f0.sum()
        
        n_conc, n = self.p_bound.shape
        pi_kmer = np.zeros( (n_conc, 4**k), dtype=np.float32)
        for i in range(n_conc):
            pi_kmer[i] = cyska.weighted_kmer_counts(im, self.p_bound[i], k)

        pd_freq = pi_kmer + self.betas[:, np.newaxis] * f0 * self.N
        pd_sum = pd_freq.sum(axis=1)
        R = (pd_freq / pd_sum[:, np.newaxis]) / f0[np.newaxis,:]

        return R

    def __mul__(self, scale):
        """
        multiply all affinities by 'scale'. For scale ~1 rounding errors
        should be small, so we do not need to recompute the partition 
        function, just rescale that as well.
        """
        t0 = time.time()
        # parameter scale <-> part. function scale
        params = np.array(self.params)
        params[:self.mdl.nA] = self.params[:self.mdl.nA] * scale
        Z_scaled = self.Z1 * scale
        
        # update dependent values
        if not self.sc:
            #print "Making SelfConsistency object for scaling"
            self.sc = SelfConsistency(self.Z1, self.mdl.reads.rna_conc, bins=10000)
        #else:
            #print "found cached version!"
        #rbp_free = self.mdl._spa_free_protein(Z_scaled, self.mdl.rbp_conc)
        t1 = time.time()
        rbp_free = self.sc.free_rbp_vector(self.mdl.rbp_conc, Z_scale=scale)
        t11 = time.time()
        betas = self.params[self.mdl.nA:]
        p_bound, w = self.mdl._spa_p_rna_bound(Z_scaled, rbp_free, betas) # w includes beta contribution
        t12 = time.time()
        pi_kmer = self.mdl._spa_kmer_pi(p_bound, self.mdl.subsample_index_matrix)

        t2 = time.time()
        # construct a new state object
        state = SPAState(self.mdl, params, Z_scaled, p_bound, pi_kmer, rbp_free)
        t3 = time.time()
        t_setup = 1000. *(t1 - t0)
        t_comp = 1000. *(t2 - t1)
        t_create = 1000. *(t3 - t2)

        tfree = 1000. *(t11 - t1)
        tbound = 1000. *(t12 - t11)
        tpi = 1000. *(t2 - t12)
        
        self.mdl.logger.debug("mul: setup={t_setup:.2f} compute={t_comp:.2f} ({tfree:.2f}, {tbound:.2f}, {tpi:.2f}) create={t_create:.2f}".format(**locals()) )
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
        
    def dump(self, msg=""):
        I = self.params[:self.mdl.nA].argsort()[::-1]
        for i in I[:10]:
            print msg, cyska.index_to_seq(i,self.mdl.k), self.params[i], self.R[:,i]


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
        #self.oem_kmer = self.mdl.subsample_oem[kmer_hits]
        self.acc_kmer = self.mdl.subsample_acc[kmer_hits]
        
        pi = self.mdl.state.pi_kmer
        rbp_free = self.mdl.state.rbp_free
        self.Z1 = self.mdl.state.Z1
        self.p_bound = self.mdl.state.p_bound
        self.rbp_free = self.mdl.state.rbp_free
        
        betas = self.mdl.params[self.mdl.nA:]
        kmer_Z1 = self.mdl._spa_partition_function(self.im_kmer, self.acc_kmer, self.mdl.params[:self.mdl.nA])
        kmer_p_bound, w = self.mdl._spa_p_rna_bound(kmer_Z1, rbp_free, betas)
        kmer_pi = self.mdl._spa_kmer_pi(kmer_p_bound, self.im_kmer)

        self.other_pi = pi - kmer_pi
        
        self.mdl.logger.debug('model.SPAPartition of {0} sequences'.format(len(self.im_kmer)) )
        
    def evaluate(self, params, **kwargs):
        # re-evaluate the model *only* on the sequences with the kmer whose affinity is changed

        # update relevant partition functions
        kmer_Z1 = self.mdl._spa_partition_function(self.im_kmer, self.acc_kmer, params[:self.mdl.nA])
        self.Z1[self.kmer_indices] = kmer_Z1

        # update free protein concentrations (probably not necessary)
        #rbp_free = self.mdl._spa_free_protein(self.Z1, self.mdl.rbp_conc)
        rbp_free = self.rbp_free

        # and re-compute expected pulldown kmer abundances
        betas = self.mdl.params[self.mdl.nA:]
        kmer_p_bound, w  = self.mdl._spa_p_rna_bound(kmer_Z1, rbp_free, betas)
        kmer_pi = self.mdl._spa_kmer_pi(kmer_p_bound, self.im_kmer)

        # pi is pi from the non kmer-containing + pi from the kmer-containing subset of sequences
        pi = self.other_pi + kmer_pi
        # p_bound is not needed for R value computation. 
        # So we keep the unchanged value. Incorrect but convenient and not used anyway.
        self.Z1[self.kmer_indices] = kmer_Z1

        return SPAState(self.mdl, params, self.Z1, self.mdl.state.p_bound, pi, rbp_free)
        
class ParamNameProxy(object):
    def __init__(self, iface):
        self.iface = iface
    
    def __len__(self):
        return self.iface.n_params
    
    def __getitem__(self, I):
        if hasattr(I, "__len__"):
            # it's an array
            return np.array([self.get_param_name(i) for i in I])
        else:
            return self.get_param_name(I)

    def get_param_name(self, i):
        if i < self.iface.nA:
            return cyska.index_to_seq(i, self.iface.k)
        else:
            return "beta{0}".format(i - self.iface.nA)

    def __iter__(self):
        for i in xrange(self.iface.n_params):
            yield self[i]

class ParamInterface(object):
    """
    Delegate class. Used by SPAModel to handle parameter related functionality. Format checking,
    conversions, loading and storing to files, etc.
    """
    def __init__(self, mdl):
        self.mdl = mdl
        self.k = self.mdl.k
        self.nA = 4**self.mdl.k
        self.n_beta = self.mdl.n_conc
        self.n_params = self.nA + self.n_beta
        self.logger = logging.getLogger("model.ParamInterface")
        self.source = "n/a"
        self.param_name = ParamNameProxy(self)
        # initialize empty parameter vector
        if self.mdl.params == None or len(self.mdl.params) == 0:
            self.mdl.params = np.ones(self.n_params, dtype=np.float32)

    @property
    def affinities(self):
        return self.mdl.params[:self.nA]
    
    @property
    def energies(self):
        return Kd_to_kcal(1./self.affinities, temp=self.mdl.T)
    
    @property
    def betas(self):
        return self.mdl.params[self.nA:]

    def load_rbpbind(self, path):
        params = np.ones(self.n_params, dtype=np.float32) * 1e-6

        aff = []
        ind = []
        for i, line in enumerate(file(path)):
            if i < 2:
                # skip header
                continue
            seq, a = line.split()
            aff.append(a)
            ind.append(cyska.seq_to_index(seq))

        aff = np.array(aff, dtype=np.float32)
        ind = np.array(ind)
        params[ind] = aff
        m = params[:self.nA].min()
        if m < 0:
            self.mdl.logger.warning("negative affinities detected in '{path}'. Shifting entire affinity distribution!".format(path=path))
            params[:self.nA] += (1e-6 - m)

        self.mdl.params = params


    def load(self, path):
        params = []
        k = 0
        if os.path.isdir(path):
            fname = os.path.join(path, '{self.k}mer_affinities.tsv'.format(self=self))
        else:
            fname = path

        with file(fname, 'r') as f:
            for line in f:
                if line.startswith('#'): 
                    continue

                parts = line.split('\t')
                params.append(float(parts[1]))
                if not k:
                    k = len(parts[0])

        params = np.array(params, dtype=np.float32)
        assert k == self.mdl.k
        if len(params) != self.n_params:
            self.logger.warning("expected {0} but read {1} params. Trying to accomodate.".format(self.n_params, len(params)))

        if len(params) < self.n_params:
            self.mdl.params[:self.nA] = params[:self.nA]
        else:
            self.mdl.params = params[:self.n_params]

        self.logger.info("loaded model parameters from {0}".format(fname))
        self.source = fname

    def store(self, path, params=[], suffix=""):
        fname = cska.ensure_path(os.path.join(path, '{self.k}mer_affinities{suffix}.tsv'.format(self=self, suffix=suffix)))
        if not len(params):
            params = self.mdl.params

        self.logger.info("storing model parameters in '{0}'".format(fname))
        with file(fname, 'w') as f:
            f.write('# {0}mer\taffinity [1/nM]\tKd [nM]\n'.format(self.k) )
            for i in xrange(len(params)):
                f.write('{0}\t{1}\t{2}\n'.format(self.param_name[i], params[i], 1./params[i]))
        
    def assign(self, params):
        assert len(params) == self.n_params
        self.mdl.params = np.array(params, dtype=np.float32)
        self.source = "assigned"
        
    def reset(self, betas=[], aff0=1e-7):
        assert len(betas == self.n_beta)
        params = np.zeros(self.n_params, dtype=np.float32)
        params[:self.nA] += aff0
        params[self.nA:] = betas
        self.assign(params)
        self.source = "scratch"

    def resume(self, path, k=0, **kwargs):
        # TODO: also store temperature in file!
        if not k:
            k = self.k
        mdl = SPAModel(self.mdl.reads, k, self.mdl.rbp_conc, **kwargs)
        mdl.parameters.load(path)

        return mdl

    def aff_str(self, affinities):
        aff = affinities
        aff0 = 1e-6
        n = (aff > aff0).sum()
        mina = aff.min()
        maxa = aff.max()
        mink = self.param_name[aff.argmin()]
        maxk = self.param_name[aff.argmax()]
        return "model params from '{self.source}': {n} above aff0, min_aff={mina:.3e} ({mink}), max_aff={maxa:.3e} ({maxk})".format(**locals()) 

    def __str__(self):
        return self.aff_str(self.affinities)

    def params_for_next_k(self, core_param=None, waterline=1e-6, aff0=1e-7):
        k = self.k

        kmer_seqs = list(cyska.yield_kmers(k+1))
        kmers = cska.reads.RBNSReads.from_seqs( kmer_seqs )
        im = kmers.get_index_matrix(k)[:,k-1:k+1] #no "adapters" here!!!

        # prepare parameter vector for k+1 model
        new = np.zeros(4**(k+1) + self.n_beta, dtype=np.float32)
        if self.n_beta:
            new[-self.n_beta:] = self.betas

        
        T = self.mdl.T
        if core_param and False:
            self.logger.info("params_for_next_k(): using parameters from {0} to derive core energies".format(core_param))
            
            # second k-mer index right shifted gives k-1 core
            # shared between both k-mers
            core_i = im[:,1] >> 2
            dG_core = core_param.energies[core_i]
            dG = self.energies[im]
            dG_l = np.where(dG_core < 0, dG.sum(axis=1) - dG_core, dG.mean(axis=1) )
            new_aff = 1./kcal_to_Kd(dG_l, temp=T)
            for i in new_aff.argsort()[:-10:-1]:
                print cyska.index_to_seq(i,k+1), new_aff[i], dG[i], dG_core[i]


        else:
            self.logger.info("params_for_next_k(): summing Boltzmann weights")
            # add up Boltzmann weights/affinities
            new_aff = self.affinities[im].sum(axis=1)

            # # use heuristic
            # for i,(l,r) in enumerate(self.affinities[im]):
            #     a,b = max(l,r), min(l,r)
            #     kcal = Kd_to_kcal(1/a, temp=T) + 0.25 * Kd_to_kcal(1/b, temp=T)
            #     aff = 1./kcal_to_Kd(kcal, temp=T)
            #     if aff >= waterline:
            #         new[i] = aff
            #     else:
            #         new[i] = min(aff, aff0)

        # detect overflows
        if new_aff.max() > 1000.:
            io = (new_aff > 1000.).nonzero()[0]
            self.logger.warning("affinity overlow for {0} kmers".format(len(io)))
            core_i = im[:,1] >> 2
            for i in io:
                print i, cyska.index_to_seq(i, k+1), dG[i], "core", cyska.index_to_seq(core_i[i], k-1), dG_core[i]
        
            new_aff = (new_aff / new_aff.max()) * 500.

        # implement the waterline
        new[:4**(k+1)] = np.where(new_aff > waterline, new_aff, aff0)
        return new


class SPAModel(object):
    """
    Single Protein Approximation (SPA) thermodynamic model of RBNS.
    """
    def __init__(self, reads, k, protein_conc, sub_replace=True, seq_only=False, n_subsample=100000, params = None):
        self.k = k
        self.nA = 4**k
        self.rbp_conc = np.array(protein_conc, dtype=np.float32)
        self.n_conc = len(self.rbp_conc)

        self.T = reads.temp
        self.RT = (self.T + 273.15) * 8.314459848/4.184E3 # RT in kcal/mol
        self.logger = logging.getLogger('model.SPAModel')

        # self.out_path = out_path
        # if not os.path.exists(out_path):
        #     os.makedirs(out_path)
      
        # reads from RBNS random RNA pool and corresonding kmer-frequencies
        self.reads = reads
        self.seq_only = seq_only
        self.f0 = self.reads.kmer_frequencies(self.k)
        self.f0 /= self.f0.sum()

        # secondary structure accessibility
        #self.openen = self.reads.acc_storage.get_discretized(k)
        self.openen = self.reads.acc_storage.get_raw(k)
        
        # subsampling related stuff
        self.sub_replace = sub_replace
        self.n_subsample = n_subsample
        
        # all subsample fields are set by new_subsample
        self.subsample_indices = []
        self.subsample_index_matrix = []
        #self.subsample_oem = []
        self.subsample_acc = []
        self.new_subsample()
        
        # current state of the model
        self.state = None
        self.params = params

        # delegate class for convenient interface with parameter vector
        self.parameters = ParamInterface(self)


    def new_subsample(self, indices = []):
        t0 = time.time()

        if not len(indices):
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
        self.subsample_acc = self.openen.acc[indices]
        if self.seq_only:
            self.subsample_acc[:,:] = 1.

        self.logger.debug("entire new_subsample() run took {0:.2f} ms".format(1000*(time.time() - t0)) )
        
    #def _spa_partition_function(self, im, oem, acc_lookup, kmer_invkd):
        #Z1 = cyska.SPA_partition_function(im, oem, acc_lookup, kmer_invkd, self.k, n_max = self.n_subsample, openen_ofs = self.openen.ofs - self.k + 1)
        #return Z1

    def _spa_partition_function(self, im, acc, kmer_invkd):
        Z1 = cyska.SPA_partition_function_raw(im, acc, kmer_invkd, self.k, n_max = self.n_subsample, openen_ofs = self.openen.ofs - self.k + 1)
        return Z1
    
    def _spa_free_protein(self, Z1, rbp_conc):
        sc = SelfConsistency(Z1, self.reads.rna_conc, bins=10000)
        rbp_free = [sc.free_rbp(total) for total in rbp_conc]
        return np.array(rbp_free, dtype= np.float32)

    def _spa_p_rna_bound(self, Z1, rbp_free, betas):
        return cyska.p_bound(Z1, np.array(rbp_free, dtype=np.float32), np.array(betas, dtype=np.float32))
    
    def _spa_kmer_pi(self, p_bound, im):
        n_conc, n = p_bound.shape
        
        pi = np.zeros( (n_conc, self.nA), dtype=np.float32)
        for j in range(n_conc):
            pi[j] = cyska.weighted_kmer_counts(im, p_bound[j], self.k)

        return pi

        
    def evaluate(self, params, keep=False, indices = [], rbp_conc = [], seq_only=None, do_jacobi=False, tm_update=True, ground_state=None):
        """
        Evaluate the thermodynamic model (single protein approximation) on a sub-sample of reads. Return an SPAState instance
        """
        # prepare all variables
        kmer_invkd = params[:self.nA]
        if not len(indices):
            indices = self.subsample_indices
            im = self.subsample_index_matrix
            acc = self.subsample_acc
        else:
            seqm = self.reads.seqm[indices]
            im = cyska.seq_matrix_to_index_matrix(
                seqm, 
                self.k,
                adap5 = cyska.seq_to_bits(self.reads.adap5[-self.k+1:]),
                adap3 = cyska.seq_to_bits(self.reads.adap3[:self.k-1]),
            )
            #oem = self.openen.oem[indices]
            acc = self.openen.acc[indices]


        if seq_only == None:
            seq_only = self.seq_only # use SPAModel instance setting

        #if seq_only:
            #acc_lookup = np.ones(self.openen.acc_lookup.shape, dtype = np.float32)
        #else:
            #acc_lookup = self.openen.acc_lookup

        if not len(rbp_conc):
            rbp_conc = self.rbp_conc
            
        n_conc = len(rbp_conc)
        if ground_state == None:
            ground_state = self.state

        # print "tm_update=",tm_update
        if tm_update:
            #Z1 = self._spa_partition_function(im, oem, acc_lookup, kmer_invkd)
            Z1 = self._spa_partition_function(im, acc, kmer_invkd)
            rbp_free = self._spa_free_protein(Z1, rbp_conc)
            betas = params[self.nA:]
            p_bound, pi = self._spa_p_rna_bound(Z1, rbp_free, betas)
            pi_kmer = self._spa_kmer_pi(p_bound, im)

            state = SPAState(self, params, Z1, p_bound, pi_kmer, rbp_free)

        else:
            # skip thermodynamic model. 
            # Useful when changed parameter is not affinity (i.e. betas)
            # copy all thermodynamic model results from previous state.
            state = SPAState(self, params, ground_state.Z1, ground_state.p_bound, ground_state.pi_kmer, ground_state.rbp_free, ground_state.jacobi)

        if keep:
            self.params = params
            self.state = state

        return state
    