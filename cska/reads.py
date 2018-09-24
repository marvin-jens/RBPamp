__license__ = "MIT"
__version__ = "0.9.8"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import numpy as np
import time
import os
import logging
import cska.cyska as cyska

from cska.caching import cached, pickled, CachedBase
import cska.fold

class RBNSReads(CachedBase):
    def __init__(self, fname, format='raw', chunklines=2000000, n_max=0, pseudo_count=10, seqm=[], rbp_name='RBP', rbp_conc=300., rna_conc=1000., temp=22, n_subsamples = 0, adap5="gggaguucuacaguccgacgauc", adap3="uggaauucucgggugucaagg", acc_storage_path='cska/acc', storage_kw=dict(disc_mode='linear')):
        
        CachedBase.__init__(self)
        
        self.name = "{rbp_name}@{rbp_conc}nM".format(**locals())
        self.rbp_name = rbp_name
        self.rbp_conc = rbp_conc
        self.rna_conc = rna_conc
        self.temp = temp
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
        self.time_logger = logging.getLogger('timing.rbns.RBNSReads')
        self.format = format

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

    def iter_reads(self):
        if hasattr(self.fname, "read"):
            # already file-like
            return self.fname

        f = file(self.fname,'r')
        if self.format == 'raw':
            return f

        elif self.format == 'fasta':
            import byo.io
            def readsrc():
                for fa_id, seq in byo.io.fasta_chunks(f):
                    yield seq
            return readsrc()


    @classmethod
    def from_seqs(cls, seqs, fname = "", **kwargs):
        
        reads = cls(fname, **kwargs)
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

        t0 = time.time()
        seqm = cyska.read_raw_seqs_chunked(self.iter_reads(), chunklines=self.chunklines, n_max=self.n_max)
        t1 = time.time()
        N, L = seqm.shape
        self.logger.info("read {0:.3f}M sequences of length {1}.".format(N/1E6, L) )
        self.time_logger.debug("read {0:.3f}M x {1}nt in {2:.2f}ms.".format(N/1E6, L, 1000. * (t1-t0)) )

        return seqm

    def get_padded_seqm(self, k):
        adap5 = cyska.seq_to_bits(self.adap5)
        adap3 = cyska.seq_to_bits(self.adap3)

        padded = cyska.seqm_pad_adapters(self.seqm, adap5, adap3, k)
        return padded

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
        self.logger.debug("get_index_matrix k={0}".format(k))
        return im
            
    def get_acc_matrix(self, k, indices=[], disc_mode='linear', ofs=None):
        """
        Returns N x (L+k-1) matrix with all k-mer accessibilities in each read, including
        positions that overlap the adapter.
        disc_mode is discretization mode: linear, gamma, raw
        """
        if disc_mode == 'raw':
            openen = self.acc_storage.get_raw(k)
        else:
            openen = self.acc_storage.get_discretized(k,disc_mode=disc_mode)
            
        if ofs is None:
            ofs = openen.ofs - k + 1
        else:
            ofs = openen.ofs - ofs

        if indices:
            return openen.oem[indices, ofs:], openen
        else:
            return openen.oem[:, ofs:], openen

    @pickled        
    def get_acc_profiles(self, k_seq, k_acc):
        acc, openen = self.get_acc_matrix(k_acc, ofs=k_seq-1)
        im = self.get_index_matrix(k_seq)
        self.logger.debug("kmer_openen_profiles k_seq={0} k_acc={1}".format(k_seq, k_acc))
        prof = cyska.kmer_openen_profiles(im, acc, k_seq, k_acc, 0)

        return prof

    def PSAM_partition_function(self, params):
        """
        Note: it is more efficient to request the necessary ingredients once and re-use them, as
        PartFuncModel does. But if you just want to evaluate a PSAM model once and get the scores,
        this should do the trick! Set params.acc_k=0 to disable accessibility scoring.
        """
        seqm = self.get_padded_seqm(params.k)
        w = self.L - params.k + 1 + self.l5 + self.l3
       
        acc_k = getattr(params, "acc_k", None)
        if not acc_k:
            print "acc_k=0 pretending everything is accessible"
            acc1 = np.ones( (self.N, w), dtype=np.float32)
            ofs = self.l5
        else:
            openen = self.acc_storage.get_raw(acc_k)
            acc1 = np.array(openen.acc)
            ofs = openen.ofs
            acc_scale = getattr(params, "acc_scale", 1.)
            if acc_scale != 1.:
                # print "power"
                # import time
                # t0 = time.time()
                # np.power(acc1, acc_scale)
                t1 = time.time()
                cyska.pow_scale(acc1, acc_scale)
                # t2 = time.time()
                # print "got it", t1-t0, t2-t1
    
        non_specific = getattr(params, "non_specific", 0.)
        Z1 = cyska.PSAM_partition_function(
            seqm, 
            acc1,
            np.array(params.psam_matrix, dtype=np.float32),
            openen_ofs=ofs - params.k + 1 + params.acc_shift, non_specific = non_specific
        )
        
        return Z1 # relative affinities of all motif instances everywhere

    def weighted_accessibility_profile(self, Z1, k_motif, pad=0, k_acc=1, row_w = None, **kwargs):
        """
        Use Boltzmann-weights in Z1 to weigh k-nt accesibility (p-unpaired) profiles across the motifs + <pad> nts 
        on either side.
        """
        from time import time
        # t0 = time()
        acc = self.acc_storage.get_raw(k_acc)
        # t1 = time()
        fp = cyska.acc_footprints(Z1, acc.acc, k_motif, k_acc, acc.ofs - k_motif + 1, pad=pad, row_w = row_w)
        # t2 = time()
        # print "t_get={:.2f} t_fp={:.2f}".format(1000. * (t1-t0), 1000. * (t2-t1))
        
        return fp

    @pickled
    def get_kmer_accessibility_binned(self, k):
        counts, openen = cyska.kmer_acc_counts(self, k)
        return counts, openen.acc_lookup
        
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

    def extrapolated_kmer_frequencies(self, k, level=2):
        """
        Uses mono-, di- ... up to <level>-nucleotide frequencies to extrapolate
        frequencies for arbitrary k > level.
        """
        init = self.kmer_frequencies(level-1)
        init /= init.sum()
        
        transition = self.kmer_frequencies(level).reshape( (4**(level-1), 4) )
        transition /= transition.sum(axis=1)[:,np.newaxis]
        
        #print "transition matrix", transition.shape
        #for row in transition:
            #print row
        
        #return cyska.extrapolate_kmer_freqs(k, np.log(init), np.log(transition), level)
        return cyska.extrapolate_kmer_freqs(k, init, transition, level), init, transition
    
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
        res = cyska.count_reads_with_kmers(self.get_index_matrix(k), k)
        t = time.time() - t0
        self.logger.debug("counted reads with {0}mers {1:.3f} ms".format( k, 1000.*t ) )
        
        return res

    @cached
    @pickled
    def reads_with_kmer_set(self, kmers):
        t0 = time.time()
        # construct kmer lookup table
        k = len(kmers[0])
        kmap = np.zeros(4**k, dtype=np.uint8)
        for mer in kmers:
            kmap[cyska.seq_to_index(mer)] = 1

        res = cyska.count_reads_with_kmap_hit(self.get_index_matrix(k), kmap)
        t = time.time() - t0
        self.logger.debug("counted reads with {0}mers {1:.3f} ms".format( k, 1000.*t ) )
        
        return res

    @cached
    @pickled
    def joint_kmer_freq_distance_profile(self, k):
        # warning! You don't want to use high values of k here!
        im = self.get_index_matrix(k)
        joint = np.array(cyska.joint_freq_at_distance(im, k) + self.pseudo_count, dtype=np.float32)
        joint_freqs = joint / joint.sum(axis=(1,0))[np.newaxis,np.newaxis,:]

        return joint_freqs

    def kmer_mutual_information_profile(self, k):
        joint = self.joint_kmer_freq_distance_profile(k)
        indep = self.kmer_frequencies(k)
        indep /= indep.sum()
        # print indep.sum(), joint.sum(axis=(1,0))

        return (joint * np.log2(joint / (np.outer(indep, indep)[:,:,np.newaxis]))).sum(axis=(1,0))

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
    def fraction_of_reads_with_kmer_set(self, kmers):
        # NOTE: since multiple kmers occur in the same read, this does not sum up to 1!
        return (self.reads_with_kmer_set(kmers) + self.pseudo_count) / float(self.N + self.pseudo_count)

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
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    CachedBase.debug_caching=True
    test_reads = [
        "TAATTTTTGCATGAAAAATCGAT",
        "AGAGGAGAGAGAGAGTCGCGCGA",
        "CGCGCGCGTCGCGATAGCGTCGA",
    ]
    
    reads = RBNSReads.from_seqs(test_reads)
    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', n_max=1000000)
    k = 4
    l = 3
    ext, init, transition = reads.extrapolated_kmer_frequencies(k,level=l)
    mono = reads.kmer_frequencies(1)
    base = cyska.seq_to_index('auca')
    corr = (1 + 1./(20 + l -1))
    print "corr ", corr
    for n in range(4):
        ext[base + n] *= corr / init[cyska.seq_to_index('uc')] * mono[0] # a

    print "corr ", corr
    for s in ['augg','cugg','gugg','uugg']:
        n = cyska.seq_to_index(s)
        c = corr / init[cyska.seq_to_index('ug')] * mono[2] # G
        ext[n] *= c

    ext /= ext.sum()
    
    obs = reads.kmer_frequencies(k)
    obs /= obs.sum()
    
    from scipy.stats import pearsonr
    import matplotlib.pyplot as pp
    R, p_val = pearsonr(np.log(obs), np.log(ext))
    print R, p_val
    pp.figure()
    pp.loglog(obs, ext, 'x')
    pp.xlabel("observed")
    pp.ylabel("extrapolated")
    pp.tight_layout()
    pp.savefig("kmer_extrapolation_test.pdf")
    
    lfc = np.log2(obs/ext)
    I = np.fabs(lfc).argsort()[::-1]
    for i in I[:20]:
        print cyska.index_to_seq(i,k), obs[i], ext[i], lfc[i], 2**lfc[i], corr
        
    sys.exit(1)
    

    import cska.cyska as cyska
    adap5 = cyska.seq_to_bits(reads.adap5)
    adap3 = cyska.seq_to_bits(reads.adap3)
    print adap5
    print adap3
    
    padded = cyska.seqm_pad_adapters(reads.seqm, adap5, adap3, 5)
    for r in padded:
        print r
