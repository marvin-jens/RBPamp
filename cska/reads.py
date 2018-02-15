__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import numpy as np
import time
import os
import logging
import cska.ska_kmers as cyska

from cska.caching import cached, pickled, CachedBase
import cska.fold

class RBNSReads(CachedBase):
    def __init__(self, fname, chunklines=2000000, n_max=0, pseudo_count=10, seqm=[], rbp_name='RBP', rbp_conc=300., rna_conc=1000., n_subsamples = 0, adap5="gggaguucuacaguccgacgauc", adap3="uggaauucucgggugucaagg", acc_storage_path='openen', storage_kw=dict(disc_mode='linear')):
        
        CachedBase.__init__(self)
        
        self.name = "{rbp_name}@{rbp_conc}nM".format(**locals())
        self.rbp_name = rbp_name
        self.rbp_conc = rbp_conc
        self.rna_conc = rna_conc
        self.adap5 = adap5
        self.adap3 = adap3
        self.l5 = len(adap5)
        self.l3 = len(adap3)
        self.fname = fname
        self.path = os.path.dirname(fname)
        self.pseudo_count = pseudo_count
        self.chunklines = chunklines
        self.n_max = n_max
        self.n_subsamples = n_subsamples
        self.logger = logging.getLogger('rbns.RBNSReads({self.rbp_name}@{self.rbp_conc}nM/RNA={self.rna_conc}nM)'.format(self=self))
        
        if len(seqm):
            self.is_subsample = True
            self.cache_preload("seqm", seqm)
            N, L = seqm.shape
            self.cache_preload("N", N)
            self.cache_preload("L", L)
        else:
            self.is_subsample = False

        # TODO: rel-path
        self.acc_storage = cska.fold.OpenenStorage(self, os.path.join(self.path, acc_storage_path), **storage_kw)

    @classmethod
    def from_seqs(cls, seqs, **kwargs):
        
        reads = cls("", **kwargs)
        reads._do_not_unpickle = True
        reads._do_not_pickle = True
        seqm = cyska.read_raw_seqs_chunked(seqs, chunklines=reads.chunklines, n_max=reads.n_max)
        N, L = seqm.shape
        reads.cache_preload("seqm", seqm)
        reads.cache_preload("N", N)
        reads.cache_preload("L", L)
        
        return reads

    @property
    def cache_key(self):
        return "{self.fname}.nmax{self.n_max}.pseudo{self.pseudo_count}".format(self=self)

    def _subsample(self, i, N):
        """
        Returns the i-th out of N (i < N) equally sized chunks of the total data. 
        Returned object is again a RBNSReads object.
        """
        chunk_n = self.N / float(N)
        start = int(np.floor(i * chunk_n))
        end = min(self.N, int(np.floor((i+1) * chunk_n)))
        l = end - start
        self.logger.debug("returning subsample of len={l} from {start}:{end}".format(**locals()) )
        
        ss = RBNSReads(
            self.fname, 
            seqm=self.seqm[start:end],
            pseudo_count=self.pseudo_count,
            rbp_name="{self.rbp_name}_subsample_{i:02d}".format(**locals()),
            rbp_conc = self.rbp_conc,
        )
        # disable caching on the subsamples, bc they are only used once
        ss._do_not_cache = True
        return ss

    @property
    @cached
    def subsamples(self):
        # TODO: do this more rigorously. Perhaps bootstrapping is better?
        self.logger.info("subsampling reads...")
        return [self._subsample(i, self.n_subsamples) for i in range(self.n_subsamples)]

    @property
    @cached
    def seqm(self):
        """
        load and keep all sequences in memory (numerically A=0,...T=3 )
        """
        self.logger.info('reading sequences from {self.fname}'.format(self=self) )

        if isinstance(self.fname, basestring):
            src = file(self.fname)
        else:
            # already file-like
            src = self.fname

        t0 = time.time()
        seqm = cyska.read_raw_seqs_chunked(src, chunklines=self.chunklines, n_max=self.n_max)
        t1 = time.time()
        N, L = seqm.shape
        self.logger.info("read {0:.3f}M sequences of length {1} in {2:.1f} seconds".format(N/1E6, L, (t1-t0) ) )

        return seqm

    @cached
    def get_index_matrix(self, k, indices=[]):
        """
        Returns N x (L+k-1) matrix with all k-mer indices in each read, including
        positions that overlap the adapter.
        """
        seqm = self.seqm
        if len(indices):
            seqm = self.seqm[indices]

        im = cyska.seq_matrix_to_index_matrix(
            seqm, 
            k, 
            adap5 = cyska.seq_to_bits(self.adap5[-k+1:]),
            adap3 = cyska.seq_to_bits(self.adap3[:k-1]),
        )
        return im
            
       
    @property
    @cached
    @pickled
    def N(self):
        N, L = self.seqm.shape
        return N

    @property
    @cached
    @pickled
    def L(self):
        N, L = self.seqm.shape
        return L

    @cached
    @pickled
    def kmer_counts(self, k):
        """
        Returns kmer counts. Keeps counts cached so that successive queries for 
        the same k are just a lookup.
        """
        self.seqm # trigger loading, so that timer is correct
        im = self.get_index_matrix(k)
        t0 = time.time()
        counts = cyska.index_matrix_kmer_counts(im, k)
        t = time.time() - t0
        self.logger.debug("counted {0}mer occurrences in {1:.3f} ms".format( k, 1000.*t ) )
        
        return counts

    @cached
    @pickled
    def kmer_counts_acc_weighted(self, k):
        """
        Returns kmer counts, weighted by accessibility
        """
        self.seqm # trigger loading, so that timer is correct
        im = self.get_index_matrix(k)
        print "IM", im.shape
        openen = self.acc_storage.get_raw(k)
        acc = openen.acc
        print "acc", acc.shape, acc.min(), acc.max()
        print "openen_ofs", openen.ofs - k + 1
        t0 = time.time()
        weighted = cyska.kmer_counts_acc_weighted(im, acc, k, openen_ofs=openen.ofs - k + 1)
        t = time.time() - t0
        openen.cache_flush()
        self.logger.debug("counted weighted {0}mer occurrences in {1:.3f} ms".format( k, 1000.*t ) )
        return weighted

    def kmer_frequencies(self,k):
        """
        Returns relative kmer frequencies, scaled such that they add up 4**k.
        This means that a uniform kmer distribution would give 1 for every kmer.
        """
        counts = self.kmer_counts(k) + self.pseudo_count
        N = counts.sum()
        freqs = np.array(counts/float(N) * (4**k), dtype=np.float32)

        return freqs

    @cached
    @pickled
    def joint_kmer_profiles(self, k_core, k_flank):
        t0 = time.time()
        counts = cyska.joint_kmer_profiles(self.seqm, k_core, k_flank)
        t = time.time() - t0
        self.logger.debug("counted joint occurrences of flanking {1}mers around core {0}mers in {2:.3f} ms".format( k_core, k_flank, 1000.*t ) )
        return counts 

    @cached
    @pickled
    def reads_with_kmers(self, k):
        t0 = time.time()
        res = cyska.count_reads_with_kmers(self.seqm, k)
        t = time.time() - t0
        self.logger.debug("counted reads with {0}mers {1:.3f} ms".format( k, 1000.*t ) )
        
        return res

    @cached
    def kmer_presence(self, kmer):
        k = len(kmer)
        kmer_index = cyska.kmer_to_index(kmer)
        
        res = cyska.seq_set_kmer_flag(self.seqm, k, kmer_index)
        
        return res
    
    @cached
    @pickled
    def fraction_of_reads_with_kmers(self, k):
        # NOTE: since multiple kmers occur in the same read, this does not sum up to 1!
        return (self.reads_with_kmers(k) + self.pseudo_count) / float(self.N + self.pseudo_count)
        
    @cached
    @pickled
    def fraction_of_reads_with_pure_kmers(self, k, candidates, out_file=None, n_sample=100000):
        """
        candidates is a kmer-indexed np.array with the (non-zero) 
        ranks/ids of candidate kmers to consider. The numbers in it are 
        arbitrary but used to flag presence/absence of *exactly one* 
        corresponding kmer, and *none of the others* with non-zero entries 
        in candidates in each read. Returns a normal kmer-indexed rel. 
        frequency array (counting "pure", as defined above, occurrences only) 
        and a n_reads sized flag array with 0 (no candidate hit), -1 (multiple 
        candidate hits), or the number assigned to the candidate kmer in your input
        if it is a "pure" occurrence.
        """
        if out_file:
            # open the file only here when the function is actually executed, 
            # to avoid starting a new file whithout the actual call performed
            # due to caching!
            out_file = file(out_file, 'w')

        t0 = time.time()
        #counts = cyska.count_pure_hits(self.seqm, candidates, out_file=out_file, n_sample=n_sample)
        counts = cyska.count_reads_with_hits(self.seqm, candidates, out_file=out_file, n_sample=n_sample, adap5=self.adap5, adap3=self.adap3)
        t = time.time() - t0
        self.logger.debug("counted reads with pure {0}mers {1:.3f} ms".format( k, 1000.*t ) )
        if out_file:
            out_file.close()
            
        #N = (flags > 0).sum() # fraction of pure reads
        fraction = (counts + self.pseudo_count ) / float(self.N + self.pseudo_count)

        return fraction

    @cached
    @pickled
    def recall(self, k, kmer_order, reorder=True):
        
        kmer_ranks = np.zeros(len(kmer_order))
        kmer_ranks[kmer_order] = np.arange(len(kmer_order))
        
        t0 = time.time()
        counts_by_kmer_rank = cyska.count_best_ranked_hits(self.seqm, np.array(kmer_ranks,dtype=np.uint32) ) 
        t = time.time() - t0
        self.logger.debug("counted reads by {0}mer-rank in {1:.3f} ms".format( k, 1000.*t ) )

        recall = (counts_by_kmer_rank + self.pseudo_count) / (float(self.N) + self.pseudo_count)
        
        if reorder:
            return recall[kmer_order]
        else:
            return recall

    @cached
    @pickled
    def kmer_profiles(self, k):
        t0 = time.time()
        profiles = cyska.kmer_profiles(self.seqm, k)
        t = time.time() - t0
        self.logger.debug("built {0}mer-profiles {1:.3f} ms".format( k, 1000.*t ) )
        
        return profiles
    
    #@cached
    #@pickled
    def expected_kmer_cooccurrence_distance_tensor(self, kmer_list):
        """
        Predict the cooccurrence frequency of kmers from kmer_list
        at each distance from their positional kmer profiles under
        the assumption of independence.
        """
        k = len(kmer_list[0])
        n = len(kmer_list)
        l = self.L - k + 1
        kmer_indices = np.array([cyska.seq_to_index(mer) for mer in kmer_list])
        kmer_lookup = np.zeros(4**k, dtype=np.uint64)
        kmer_lookup[kmer_indices] = np.arange(n) + 1
        
        tensor = np.zeros( (n, n, l) ,dtype=np.float32)
        profiles = self.kmer_profiles(k)
        
        max_index = {}
        for s in np.arange(0,k):
            max_index[s] = 4**(k-s) - 1

        def omega(x,y, s):
            "returns 1 if overlap is all matches, zero for non-overlap or mismatch"
            if s < k:
                return (y >> 2*s) == x & max_index[s]
            
            return 1
            
        for i,x in enumerate(kmer_indices):
            for j,y in enumerate(kmer_indices):
                f_x = profiles[x]/float(self.N)
                f_y = profiles[y]
                #if i == 0 and j == 0:
                    #print f_x
                for s in np.arange(1,l):
                    tensor[i,j,s] = np.array([f_x[m] * f_y[m+s] * omega(x,y,s) for m in np.arange(0, l-s)]).mean()
        
        return tensor
                
        
    @cached
    @pickled
    def kmer_cooccurrence_distance_tensor(self, kmer_list):
        k = len(kmer_list[0])
        n = len(kmer_list)
        kmer_indices = np.array([cyska.seq_to_index(mer) for mer in kmer_list])
        kmer_lookup = np.zeros(4**k, dtype=np.uint64)
        kmer_lookup[kmer_indices] = np.arange(n) + 1
        
        t0 = time.time()
        tensor = cyska.kmer_cooccurrence_distance_tensor(self.seqm, kmer_lookup, k, n)
        t = time.time() - t0
        self.logger.debug("built {0}mer-cooccurrence tensor in {1:.3f} ms".format( k, 1000.*t ) )
        
        return tensor
        
    
    @cached
    @pickled
    def kmer_flank_profiles(self, kmer, k_flank):
        """
        use kmer_filter first and then compute the average occurrences of kmers
        with k=k_flank (k_flank = 1..k_max) around the desired "central" kmer.
        """
        return cyska.kmer_flank_profiles(self.seqm, kmer, k_flank=k_flank)
    
    def __str__(self):
        return "RBNSReads('{self.fname}' N={self.N} L={self.L})".format(self=self)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    CachedBase.debug_caching=True
    test_reads = [
        "TGCAGCTGAGCTAGCGTAGCGAT",
        "AGAGGAGAGAGAGAGTCGCGCGA",
        "CGCGCGCGTCGCGATAGCGTCGA",
    ]
    reads = RBNSReads.from_seqs(test_reads)
    print reads
