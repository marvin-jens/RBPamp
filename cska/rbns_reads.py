__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import numpy as np
import time
import logging
import cska.ska_kmers

from cska.caching import cached, pickled, CachedBase


class RBNSReads(CachedBase):
    def __init__(self, fname, chunklines=2000000, n_max=0, pseudo_count=10, seqm=[], rbp_name='RBP', rbp_conc=300., rna_conc=100000., n_subsamples = 0):
        
        CachedBase.__init__(self)
        
        self.name = "{rbp_name}@{rbp_conc}nM".format(**locals())
        self.rbp_name = rbp_name
        self.rbp_conc = rbp_conc
        self.rna_conc = rna_conc
        self.fname = fname
        self.pseudo_count = pseudo_count
        self.chunklines = chunklines
        self.n_max = n_max
        self.n_subsamples = n_subsamples
        
        self.logger = logging.getLogger('RBNSReads({self.rbp_name}@{self.rbp_conc}nM/RNA={self.rna_conc}nM)'.format(self=self))
        
        if len(seqm):
            self.cache_preload("__cached_seqm", seqm)
            self.N, self.L = seqm.shape
        else:
            self.N = 0
            self.L = 0

    
    @property
    def cache_key(self):
        return "{self.name}.nmax{self.n_max}.pseudo{self.pseudo_count}".format(self=self)

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
        
        t0 = time.time()
        seqm = cska.ska_kmers.read_raw_seqs_chunked(file(self.fname), chunklines=self.chunklines, n_max=self.n_max)
        self.N, self.L = seqm.shape
        t1 = time.time()

        self.logger.info("read {0:.3f}M sequences of length {1} in {2:.1f} seconds".format(self.N/1E6, self.L, (t1-t0) ) )

        return seqm

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

    @cached
    @pickled
    def kmer_counts(self, k):
        """
        Returns kmer counts. Keeps counts cached so that successive queries for 
        the same k are just a lookup.
        """
        t0 = time.time()
        counts = cska.ska_kmers.seq_set_kmer_count(self.seqm, k)
        t1 = time.time()
        self.logger.debug("counted {0}mer occurrences in {1:.3f} ms".format( k, (t1-t0)*1000. ) )
        
        return counts

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
    def reads_with_kmers(self, k):
        return cska.ska_kmers.count_reads_with_kmers(self.seqm, k)

    def fraction_of_reads_with_kmers(self, k):
        # NOTE: since multiple kmers occur in the same read, this does not sum up to 1!
        return (self.reads_with_kmers(k) + self.pseudo_count) / float(self.N + self.pseudo_count)
        
    @cached
    def fraction_of_reads_with_pure_kmers(self, candidates):
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
        counts, flags, indices = cska.ska_kmers.count_pure_hits(self.seqm, candidates)
        
        N = (flags > 0).sum() # fraction of pure reads
        fraction = (counts + self.pseudo_count ) / float(N + self.pseudo_count)

        return fraction, flags, indices

    def write_pure_reads_fasta(self, out_file, k, candidates, n_sample=100000):
        counts, flags, indices = self.fraction_of_reads_with_pure_kmers(candidates)
        n = np.ones(4**k, dtype=np.uint32)*n_sample

        cska.ska_kmers.store_pure_reads(out_file, self.seqm, flags, indices, n)
        

    @pickled
    def kmer_cooccurrence_distance_tensor(self, kmer_list):
        k = len(kmer_list[0])
        n = len(kmer_list)
        kmer_indices = np.array([cska.ska_kmers.seq_to_index(mer) for mer in kmer_list])
        kmer_lookup = np.zeros(4**k, dtype=np.uint64)
        kmer_lookup[kmer_indices] = np.arange(n) + 1
        
        return cska.ska_kmers.kmer_cooccurrence_distance_tensor(self.seqm, kmer_lookup, k, n)
        
        
    def kmer_filter(self, kmer):
        """
        returns the subset of seqm that contains sequences with the desired kmer
        and a boolean matrix with ones at the positions of kmer occurrence
        """
        return cska.ska_kmers.kmer_filter(self.seqm, kmer)

    def recall(self, kmer_order, reorder=True):
        
        kmer_ranks = np.zeros(len(kmer_order))
        kmer_ranks[kmer_order] = np.arange(len(kmer_order))
        
        counts_by_kmer_rank = cska.ska_kmers.count_best_ranked_hits(self.seqm, np.array(kmer_ranks,dtype=np.uint32) ) 
        
        recall = (counts_by_kmer_rank + self.pseudo_count) / (float(self.N) + self.pseudo_count)
        
        if reorder:
            return recall[kmer_order]
        else:
            return recall

    @pickled
    def kmer_flank_profiles(self, kmer, k_flank):
        """
        use kmer_filter first and then compute the average occurrences of kmers
        with k=k_flank (k_flank = 1..k_max) around the desired "central" kmer.
        """
        return cska.ska_kmers.kmer_flank_profiles(self.seqm, kmer, k_flank=k_flank)
    
    def __str__(self):
        return "RBNSReads('{self.fname}' N={self.N} L={self.L})".format(self=self)
