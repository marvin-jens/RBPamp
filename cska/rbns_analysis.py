__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import os
import numpy as np
import time
import logging
import cska.ska_kmers

from cska.caching import cached, pickled, CachedBase

        
class RBNSComparison(CachedBase):
    def __init__(self, in_reads, pd_reads, ska_runner):
        
        CachedBase.__init__(self)
        
        self.logger = logging.getLogger('RBNSComparison')
        self.pd_reads = pd_reads
        self.in_reads = in_reads
        self.ska_runner = ska_runner
        
        self.name = 'RBNS:{pd_reads.rbp_conc}nM:{in_reads.rbp_conc}nM'.format(**locals())
        
    @property
    def cache_key(self):
        return "{self.name}.{self.pd_reads.cache_key}.{self.in_reads.cache_key}".format(self=self)
        
    def _subsampled(self, func):
        """
        Adds error estimates using subsamples of the underlying RBNSReads instance
        for the pulldown sample.
        """
        
        res = func(self.pd_reads, self.in_reads)

        sampled = np.array([
            func(sample, self.in_reads)
            for sample in self.pd_reads.subsamples
        ])
        errors = sampled.std(axis=0)

        return res, errors
        
    @cached
    @pickled
    def R_values(self, k):
        def compute_R(sample, control):
            return sample.kmer_frequencies(k) / control.kmer_frequencies(k)
        
        return self._subsampled(compute_R)

    @cached
    @pickled
    def SKA_weights(self, k):
        def compute_SKA(sample, control):
            return self.ska_runner.stream_counts(k, sample, control)
        
        return self._subsampled(compute_SKA)

    @cached
    @pickled
    def F_ratios(self, k):
        def compute_F_ratio(sample, control):
            return sample.fraction_of_reads_with_kmers(k) / control.fraction_of_reads_with_kmers(k)
        
        return self._subsampled(compute_F_ratio)

    @cached
    @pickled
    def recall_ratios(self, kmer_order):
        def compute_recall_ratio(sample, control):
            return sample.recall(kmer_order, reorder=False) / control.recall(kmer_order, reorder=False)
        
        return self._subsampled(compute_recall_ratio)

    @cached
    @pickled
    def pure_F_ratios(self, candidates):
        def compute_pure_F_ratio(sample, control):
            f_pd, flags, indices = sample.fraction_of_reads_with_pure_kmers(candidates)
            f_in, flags, indices = control.fraction_of_reads_with_pure_kmers(candidates)
        
            return f_pd / f_in

        return self._subsampled(compute_pure_F_ratio)

    def ska_z_scores(self, k):
        s = self.SKA_weights(k)
        z = (s - s.mean()) / s.std()
        return z
            
    def __str__(self):
        # TODO: update
        I = self.ska_weights.argsort()[::-1]
        buf = [self.name]
        for i in I[:10]:
            buf.append("{0}\t{1:.2f}".format(cska.ska_kmers.index_to_seq(i, self.k), self.ska_weights[i] ) )
        
        return '\n'.join(buf)


class RBNSAnalysis(CachedBase):
    def __init__(self, rbp_name ='RBP', out_path='ska_results', ska_runner=None):
        
        CachedBase.__init__(self)
        
        self.reads = []
        self.rbp_name = rbp_name
        self.out_path = out_path
        self.ska_runner = ska_runner
        self.logger = logging.getLogger('RBNSAnalysis({self.rbp_name}) -> "{self.out_path}"'.format(self=self))
        
        self.rbp_conc = []
        self.comparisons = []

        self.k_range = []
        self.runs = {}
        self.AUCs = {}
        #self.pair_screens = collections.defaultdict(dict)
        
    def flush(self):
        for comp in self.comparisons:
            comp.cache_flush()

    def add_reads(self, rbns_reads):
        self.logger.info("adding {0}".format(rbns_reads.name) )
        self.reads.append(rbns_reads)
        if len(self.reads) > 1:
            self.comparisons.append(RBNSComparison(self.reads[0], rbns_reads, self.ska_runner) )
            self.rbp_conc.append(rbns_reads.rbp_conc)
    
    def _make_matrices(self, comp_attr, k):
        M = np.array([getattr(comp, comp_attr)(k) for comp in self.comparisons])
        values = M[:,0,:]
        errors = M[:,1,:]
        
        return values, errors
        
    def R_value_matrix(self, k):
        return self._make_matrices("R_values", k)
            
    def SKA_weight_matrix(self, k):
        return self._make_matrices("SKA_weights", k)

    def F_ratio_matrix(self, k):
        return self._make_matrices("F_ratios", k)
            
    def recall_ratio_matrix(self, k):
        kmer_order = self.get_optimal_kmer_ranking(k)
        return self._make_matrices("recall_ratios", kmer_order)

    def pure_F_ratio_matrix(self, k):
        order = self.get_optimal_kmer_ranking(k)
        R, R_err = self.R_value_matrix(k)
        Rm = np.median(R - R_err, axis=0)[order]
        
        i_cut = (Rm > 1).argmin()
        candidates = np.zeros(4**k, dtype=np.uint32)
        candidates[order[:i_cut]] = np.arange(i_cut) + 1
        
        for x in candidates.nonzero()[0]:
            print cska.ska_kmers.index_to_seq(x, k), candidates[x]
            
        return self._make_matrices("pure_F_ratios", candidates)

    def write_pure_reads_fasta(self, k, n_sample=100000):
        order = self.get_optimal_kmer_ranking(k)
        R, R_err = self.R_value_matrix(k)
        Rm = np.median(R - R_err, axis=0)[order]
        
        i_cut = (Rm > 1).argmin()
        candidates = np.zeros(4**k, dtype=np.uint32)
        candidates[order[:i_cut]] = np.arange(i_cut) + 1
        
        for x in candidates.nonzero()[0]:
            print cska.ska_kmers.index_to_seq(x, k), candidates[x]

        for reads in self.reads:
            fname = "pure_{k}mer_reads_{reads.name}.fa".format(**locals() )
            reads.write_pure_reads_fasta(file(os.path.join(self.out_path, fname), 'w'), k, candidates, n_sample=n_sample)
        
    @cached
    def get_optimal_kmer_ranking(self, k):
        # TODO: factor in consistently elevated scores with increasing protein concentration
        from scipy.stats.mstats import gmean
        all_ska_weights = self.SKA_weight_matrix(k)[0]

        #return gmean(all_ska_weights, axis=0).argsort()[::-1]
        return np.median(all_ska_weights, axis=0).argsort()[::-1]

    def select_significant_kmers(self,k, n_max=10):
        #TODO: do this based on some statistics, taking into 
        # account the errors of ska weights from subsampling
        
        # TODO: less ugly!
        all_f_ratios = np.array([self.runs[(k, reads.rbp_conc)].f_ratios for reads in self.reads[1:] ])

        from scipy.stats.mstats import gmean
        mf = gmean(all_f_ratios, axis=0)
        order = mf.argsort()[::-1]
        print "geometric mean f-ratios for k",k, mf[order][:n_max]
        rank_cut = min((mf[order] < 1).argmax(), n_max)

        best_sample_i = all_f_ratios[:,order[0]].argmax()
        kmers = [cska.ska_kmers.index_to_seq(i, k) for i in order[:rank_cut]]
        return kmers, order[:rank_cut], self.reads[best_sample_i+1].rbp_conc

    def store_all_results(self, k):
        order = self.get_optimal_kmer_ranking(k)
        kmers = np.array(list(cska.ska_kmers.yield_kmers(k)))
        
        for name in ["pure_F_ratio", "recall_ratio", "SKA_weight", "R_value", "F_ratio"]:
            values, errors = getattr(self, "{name}_matrix".format(name=name) )(k)

            fname = "{self.rbp_name}.{name}.{k}mer.tsv".format(**locals())
            path = os.path.join(self.out_path, fname)

            self.write_kmer_matrix(path, kmers, values.T, errors.T, order)

    def write_kmer_matrix(self, out_path, kmers, values, errors, order):
        self.logger.info("writing data matrix '{out_path}'".format(out_path=out_path) )
        
        def round_to_2(x):
            if x:
                return round(x, max(-int(np.floor(np.log10(abs(x)))), 2) ) 
            else:
                return x
        
        def round_to_err(x, x_err):
            if x_err:
                n_dig = int(np.ceil(-np.log10(x_err)))+1
                x_err = round(x_err,n_dig)

                n_dig = int(np.ceil(-np.log10(x_err)))+1
                x = round(x,n_dig)
            
            return [str(x), str(x_err)]

        def roundrobin(*iterables):
            from itertools import cycle, islice
            "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
            # Recipe credited to George Sakkis
            pending = len(iterables)
            nexts = cycle(iter(it).next for it in iterables)
            while pending:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    pending -= 1
                    nexts = cycle(islice(nexts, pending))

        header = ['# kmer'] + list(roundrobin(['{0}nM'.format(c) for c in self.rbp_conc], ['error' for c in self.rbp_conc]))
        with file(os.path.join(out_path), 'w') as of:

            of.write("\t".join(header) + '\n')
            for mer, values_row, error_row in zip(kmers[order], values[order], errors[order]):
                
                cols = [str(mer), ]
                for x, err in zip(values_row, error_row):
                    cols.extend(round_to_err(x, err))

                of.write("\t".join(cols) + '\n')
            of.close()
 
