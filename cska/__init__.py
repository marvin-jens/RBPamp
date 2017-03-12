#!/usr/bin/env python
__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import sys
import itertools
import numpy as np
import copy
import time
import os
import logging
import collections
import cska.ska_kmers
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as pp
import cPickle as pickle

def cached(func):
    """
    Decorator for class methods that keeps the results of the first call and 
    returns the cached result for subsequent calls. Works by adding a 
    "__cached_<func_name>" dictionary to the decorated method's class instance.
    """
    cache_name = "__cached_{name}".format(name=func.__name__)
    
    def cached_func(self, *argc):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, dict() )
        
        cache = getattr(self, cache_name)
        if not argc in cache:
            cache[argc] = func(self, *argc)
            
        return cache[argc]
    
    return cached_func
  
def pickled(func):
    """
    Decorator for class methods that returns an un-pickled result if it exists. 
    Otherwise, stores the result of the call in a pickle file. Requires that the 
    class has an out_path attribute and a pickle_key method that returns a distinct 
    key for all the parameters that influence the results, ensuring that the correct
    object is unpickled.
    """
    
    def pickled_func(self, *argc, **kwargs):
        inst_key = self.pickle_key()
        argc_key = "_".join([str(a) for a in argc])
        kw_key = "__".join(["{0}={1}".format(k,v) for k,v in sorted(kwargs.items()) ])
        
        path = os.path.join(self.out_path,"pkl")
        pkl_name = "{inst_key}.{func.__name__}.{argc_key}.{kw_key}.pkl".format(**locals() )
        if not os.path.exists(os.path.join(path,pkl_name)):
            res = func(self, *argc, **kwargs)
            try:
                os.makedirs(path)
            except OSError:
                # already exists
                pass
            self.logger.debug("storing pickle of '{0}'".format(pkl_name) )
            pickle.dump(res, file(os.path.join(path,pkl_name),'wb'), protocol=-1)
        else:
            self.logger.debug("un-pickling '{0}'".format(pkl_name) )
            res = pickle.load(file(os.path.join(path,pkl_name),'rb'))
        
        return res
    
    return pickled_func

class RBNSReads(object):
    out_path = './'
    
    def __init__(self, fname, chunklines=2000000, n_max=0, pseudo_count=10, seqm=[], rbp_name='RBP', rbp_conc=300., rna_conc=100000., n_subsamples = 0):
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
            self._seqm = seqm
            self.N, self.L = self._seqm.shape
        else:
            self._seqm = []
            self.N = 0
            self.L = 0

        self._subsamples = []
    
    def pickle_key(self):
        return "{self.fname}.{self.name}.nmax{self.n_max}.pseudo{self.pseudo_count}".format(self=self)

    def flush(self, items = ["kmer_counts","reads_with_kmers", "fraction_of_reads_with_pure_kmers"]):
        for item in items:
            setattr(self, "__cached_{name}".format(name=item), dict() )

    @property
    def subsamples(self):
        # TODO: do this more rigorously. Perhaps bootstrapping is better?
        if not self._subsamples:
            self.logger.info("subsampling reads...")
            self._subsamples = [self._subsample(i, self.n_subsamples) for i in range(self.n_subsamples)]

        return self._subsamples

    @property
    def seqm(self):
        if not len(self._seqm):
            self.logger.info('reading sequences from {self.fname}'.format(self=self) )
            # load and keep all sequences in memory (numerically A=0,...T=3 )
            t0 = time.time()
            self._seqm = cska.ska_kmers.read_raw_seqs_chunked(file(self.fname), chunklines=self.chunklines, n_max=self.n_max)
            self.N, self.L = self._seqm.shape
            t1 = time.time()

            self.logger.info("read {0:.3f}M sequences of length {1} in {2:.1f} seconds".format(self.N/1E6, self.L, (t1-t0) ) )

        return self._seqm

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
        
        return RBNSReads(
            self.fname, 
            seqm=self.seqm[start:end],
            pseudo_count=self.pseudo_count,
            rbp_name="{self.rbp_name}_subsample_{i:02d}".format(**locals()),
            rbp_conc = self.rbp_conc,
        )

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
        counts, flags = cska.ska_kmers.count_pure_hits(self.seqm, candidates)
        fraction = (counts + self.pseudo_count ) / float(self.N + self.pseudo_count)

        return fraction, flags

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

class SKARunner(object):
    def __init__(self, max_iterations=10, convergence=0.5, subsamples=10):
        self.logger = logging.getLogger('SKARunner')
        self.max_iterations = max_iterations
        self.convergence = convergence
        self.n_subsamples = subsamples
        
    def stream_counts(self, k, pd_reads, in_reads):
        self.logger.info("streaming {k} mers in {pd_reads.name}".format(**locals()))
        t0 = time.time()
        kmers = list(yield_kmers(k))
        #background, bg_source = self.try_load_background_freqs(k)

        pd_freqs = pd_reads.kmer_frequencies(k)
        in_freqs = in_reads.kmer_frequencies(k)
        R_values = pd_freqs / in_freqs # kmer-frequency 'R'-atios

        current_weights = copy.copy(R_values)
        
        weight_history = []
        for iteration_i in range(self.max_iterations):
            new_weights = cska.ska_kmers.seq_set_SKA(pd_reads.seqm, current_weights, in_freqs, k)
                
            weight_history.append(new_weights)
            current_weights = copy.copy(new_weights)
            
            if len(weight_history) > 1:
                delta = weight_history[-1] - weight_history[-2]
                
                change = np.fabs(delta)
                mean_change = change.mean()
                max_i = change.argmax()
                max_change = change[max_i]
                
                self.logger.debug("iteration {0}: mean_change={1} max_change={2} for '{3}' ({4} -> {5})".format(iteration_i, mean_change, max_change, kmers[max_i], weight_history[-2][max_i], weight_history[-1][max_i]))
                if max_change < self.convergence:
                    self.logger.info("reached convergence after {0} iterations".format(iteration_i))
                    break
        
        if iteration_i >= self.max_iterations:
            self.logger.warning("reached max_iterations without convergence!")

        t = time.time() - t0
        self.logger.debug("streaming of {0:.2f}M reads took {1:.2f}ms".format(pd_reads.N / 1e6,  1000. * t) )
        return current_weights

class RBNSResult(object):
    """
    Generic wrapper around all quantities that are computed by comparing two 
    RBNS samples, such as R-values, F-values, SKA-weights, etc.
    Adds transparent caching and pickling/unpickling of the values as well as
    error estimates using the subsamples of the underlying RBNSReads instance
    for the pulldown sample.
    """
    
    out_path = "./"
    
    def __init__(self, pd_reads, in_reads, name, func):
        """
        Name is arbitrary, the pd_reads and in_reads are RBNSReads instances of 
        the two samples being compared. Func is a callable that will receive
        pd_reads and in_reads as first arguments, plus *argc, **kwargs from
        the __call__ to the RBNSResults instance.
        """
        self.pd_reads = pd_reads
        self.in_reads = in_reads
        self.func = func
        self.name = name
        self.logger = logging.getLogger('RBNSResult({self.name})'.format(self=self) )
        
        ## make the @cached and @pickled work
        #self.__call__.name = name
        
    def pickle_key(self):
        return ".".join([self.name, self.pd_reads.pickle_key(), self.in_reads.pickle_key()])
    
    @cached
    @pickled
    def __call__(self, *argc, **kwargs):
        res = self.func(self.pd_reads, self.in_reads, *argc, **kwargs)
        sampled = np.array([
            self.func(sample, self.in_reads, *argc, **kwargs)
            for sample in self.pd_reads.subsamples
        ])
    
        errors = sampled.std(axis=0)
    
        return res, errors
        
    def __str__(self):
        return "{self.name} ({self.pd_reads.rbp_conc}nM / {self.in_reads.rbp_conc}nM)".format(self=self)
           
class RBNSComparison(object):
    def __init__(self, in_reads, pd_reads, ska_runner):
        self.logger = logging.getLogger('RBNSResults')
        self.pd_reads = pd_reads
        self.in_reads = in_reads
        
        self.name = 'RBNS:{pd_reads.rbp_conc}nM:{in_reads.rbp_conc}nM'.format(**locals())
        
        def compute_R(sample, control, k):
            return sample.kmer_frequencies(k) / control.kmer_frequencies(k)
        
        def compute_SKA(sample, control, k):
            return ska_runner.stream_counts(k, sample, control)
            
        def compute_F_ratio(sample, control, k):
            return sample.fraction_of_reads_with_kmers(k) / control.fraction_of_reads_with_kmers(k)
            
        self.R_values = RBNSResult(pd_reads, in_reads, "R-value", compute_R)
        self.SKA_weights = RBNSResult(pd_reads, in_reads, "SKA-weight", compute_SKA)
        self.F_ratios = RBNSResult(pd_reads, in_reads, "F-ratio", compute_F_ratio)
        self._pure_f_ratios = None
            

    def ska_z_scores(self, k):
        s = self.SKA_weights(k)
        z = (s - s.mean()) / s.std()
        return z

    def pure_f_ratios(self, k, candidates):

        f_pd, flags = self.pd_reads.fraction_of_reads_with_pure_kmers(candidates)
        f_in, flags = self.in_reads.fraction_of_reads_with_pure_kmers(candidates)
        
        f_ratios = f_pd / f_in

        f_sampled = np.array([
            sample.fraction_of_reads_with_pure_kmers(candidates) / f_in
            for sample in self.pd_reads.subsamples()
        ])
        
        f_ratios_err = f_sampled.std(axis=0)
            
        return f_ratios, f_ratios_err
            
        

    def __str__(self):
        # TODO: update
        I = self.ska_weights.argsort()[::-1]
        buf = [self.name]
        for i in I[:10]:
            buf.append("{0}\t{1:.2f}".format(cska.ska_kmers.index_to_seq(i, self.k), self.ska_weights[i] ) )
        
        return '\n'.join(buf)



class RBNSAnalysis(object):
    def __init__(self, rbp_name ='RBP', out_path='ska_results', ska_runner=None):
        self.reads = []
        self.rbp_name = rbp_name
        self.out_path = out_path
        self.ska_runner = ska_runner
        self.logger = logging.getLogger('RBNSAnalysis({self.rbp_name}) -> {self.out_path}'.format(self=self))
        
        self.rbp_conc = []
        self.comparisons = []

        self.k_range = []
        self.runs = {}
        self.AUCs = {}
        self.pair_screens = collections.defaultdict(dict)
        
    def add_reads(self, rbns_reads):
        print "added", rbns_reads.name
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
            
    def F_ratio_matrix(self, k):
        return self._make_matrices("F_ratios", k)
            
    def SKA_weight_matrix(self, k):
        return self._make_matrices("SKA_weights", k)
        
    def get_optimal_kmer_ranking(self, k):
        from scipy.stats.mstats import gmean
        all_ska_weights = self.SKA_weight_matrix(k)[0]

        #return gmean(all_ska_weights, axis=0).argsort()[::-1]
        return np.median(all_ska_weights, axis=0).argsort()[::-1]

    def store_all_results(self, k):
        #self.logger.info('storing kmer frequencies, R-values and SKA-weights in "{out_path}/{res_file}"'.format(**locals()) )

        order = self.get_optimal_kmer_ranking(k)
        kmers = np.array(list(yield_kmers(k)))
        
        for name in ["SKA_weight", "R_value", "F_ratio"]:
            values, errors = getattr(self, "{name}_matrix".format(name=name) )(k)

            fname = "{self.rbp_name}.{name}.{k}mer.tsv".format(**locals())
            path = os.path.join(self.out_path, fname)

            self.write_kmer_matrix(path, kmers, values.T, errors.T, order)

            
    def write_kmer_matrix(self, out_path, kmers, values, errors, order):
        
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
 

    
    def run_f_value_fit(self, k):
        
        # TODO: needs to fit a global model because top k-mer containing reads show strong saturation effects -> decreased f-value at higher concentration!

        c_in = self.reads[0].fraction_of_reads_with_kmers(k)
        b_pd = np.array([pd_reads.fraction_of_reads_with_kmers(k) for pd_reads in self.reads[1:]])
        rbp_conc = np.array([pd_reads.rbp_conc for pd_reads in self.reads[1:]])
        
        import pickle
        pickle.dump((c_in, rbp_conc, b_pd), file('rbfox2.pkl','w'))
        
        kmers, indices, best_conc = self.select_significant_kmers(k)
        # initial Kd guess from lowest protein concentration
        print b_pd.shape
        print b_pd[0,indices], b_pd[0,:].min()
        print c_in[indices], c_in.min()
        
        K0 = rbp_conc[0] * c_in / b_pd[0]
        
        print "initial K vector", K0.min(), K0.mean(), K0.max()
        
        def to_fit(K):
            occ = rbp_conc[:, np.newaxis] / (rbp_conc[:, np.newaxis] + K[np.newaxis,:])
            bound = occ * c_in[np.newaxis,:]
            f_pd = bound / bound.sum(axis=1)[:, np.newaxis]
            
            err = 1./len(rbp_conc) * ((f_pd - b_pd)**2 ).sum(axis=0)
            print err.mean()
            return np.sqrt(err)
        
        from scipy.optimize import leastsq
        print "fitting"
        K_opt, code = leastsq(to_fit, K0, maxfev=1000)
        
        
        
        
        for kmer,i in zip(kmers, indices):
            print kmer, c_in[i], b_pd[:,i], K_opt[i]
            
        
        
            
    def run_ROC(self):
        
        self.logger.info("computing receiver-operator-characteristic for {0} samples".format(len(self.reads) -1 ) )
        def make_plot(k, order_by="ska_weights", name="SKA"):
            
            pp.figure()
            pp.title('discrimination of {self.rbp_name} pd/input by {name}'.format(**locals()))

            for reads in self.reads[1:]:
                res = self.runs[ (k, reads.rbp_conc) ]

                order = getattr(res, order_by).argsort()[::-1]

                recall_pd = res.pd_reads.recall(order)
                recall_in = res.in_reads.recall(order)
                                
                x = np.array([0,] + list(recall_in.cumsum()))
                y = np.array([0,] + list(recall_pd.cumsum()))
                AUC = np.trapz(y, x)
                pp.step(x, y, where='post', label='{0}mersfloat(in_N) @{1}nM (AUC={2:.3f})'.format(k, reads.rbp_conc, AUC))
                self.AUCs[ (k, reads.rbp_conc) ] = (AUC, name, k, reads.rbp_conc)

            pp.plot([0,1.],[0,1.], color='gray', linestyle = 'dashed')
            
            pp.xlabel('fraction of input explained')
            pp.ylabel('fraction of pulldown explained')
            pp.legend(loc='lower right')
            pp.savefig("{self.out_path}/{self.rbp_name}.{k}mer.{name}.ROC.pdf".format(**locals()))

        for k in self.k_range:
            make_plot(k, order_by="ska_weights", name="SKA")
            make_plot(k, order_by="R_values", name="R")
            make_plot(k, order_by="f_ratios", name="f_ratios")
        
        # TODO: find rank at wich the ROC curve slope drops below 1. 
        # This is where reads with the kmer are no longer more abundant in pd than input
        # (f-value ratio is 1)!
        self.best_auc, self.best_method, self.best_k, self.best_rbp_conc = sorted(self.AUCs.values())[-1]
        print "BEST", self.best_auc, self.best_method, self.best_k, self.best_rbp_conc


    def compare_k(self, z_cut=2):
        import matplotlib.pyplot as pp
        pp.figure()
        k_range = sorted(self.run.results.keys())
        hits_at_k = []
        
        def spread(x, data, min_w=1.1, max_n=20):
            n = len(x)
            lo, hi = x.min(), x.max()
            step = (hi-lo)/float(n-1)
            center = (hi - lo)/2
            
            if n > max_n:
                I = x.argsort()
                data = list(np.array(data)[I])
                x = list(np.array(x)[I])

                data = data[:max_n/2] + ['...'] + data[-max_n/2:]
                x = np.array(x[:max_n/2] + [center,] + x[-max_n/2:])
                
                n = len(x)
                lo, hi = x.min(), x.max()
                step = (hi-lo)/float(n-1)
                center = (hi - lo)/2

            if step < min_w:
                w = min_w * (n-1)
                lo = center - w/2
                hi = lo + w
                
                step = (hi-lo)/float(n-1)

            return np.arange(lo, hi+step, step), data

        def noise(x, amp=.3):
            return x + np.random.random(len(x))*amp - amp/2.
        
        def displace(x, amp=.3):
            n = len(x)
            lo, hi = x.min()- amp/2, x.max()+amp/2
            step = (hi-lo)/float(n-1)
            
            return np.arange(lo, hi+step, step)[:n]
            
        for k in k_range:
            res = self.run.results[k]
            top_i = (res.z_scores_ska > z_cut).nonzero()[0]
            n = len(top_i)
            
            kmers = [cska.ska_kmers.index_to_seq(i,k) for i in top_i]
            scores = res.ska_weights[top_i]
            errors = res.ska_weights_err[top_i]
            
            x = displace((np.ones(len(scores)) * k), amp=.4)
            #print x, len(x), len(scores)
            pp.errorbar(x, scores, yerr=errors, fmt='o', color='r', alpha=.5)
            
            y, kmers = spread(scores, kmers)
            for mer, score, _y in zip(kmers, scores, y):
                pp.gca().text(k - .5, _y, mer, fontsize=6)

            hits_at_k.append( (kmers, scores, errors) )
        pp.savefig(os.path.join(self.run.out_path,"k_comparison.pdf"))

    
    def compute_recall_ratios(self, k):
        kmer_order = self.get_optimal_kmer_ranking(k)
        in_recall = self.reads[0].recall(kmer_order, reorder=False)
 
        kmer_ranks = np.zeros(len(kmer_order), dtype=int)
        kmer_ranks[kmer_order] = np.arange(len(kmer_order), dtype=int)
       
        recall_matrix = []
        for run_key in sorted(self.runs.keys()):
            kr, rbp_conc = run_key
            if kr != k:
                continue
            res = self.runs[run_key]
            ratio = res.pd_reads.recall(kmer_order, reorder=False) / in_recall
            res.recall_ratio = ratio
            recall_matrix.append(ratio)

        recall_matrix = np.array(recall_matrix).T
        with file(os.path.join(self.out_path, "{0}mer_recall.tsv".format(k)), 'w') as f:
            head = ['# kmer','ska_rank' ] + [str(r.rbp_conc) for r in self.reads[1:]]
            f.write("\t".join(head) + '\n')
            
            for kmer, o, ratios in zip(yield_kmers(k), kmer_ranks, recall_matrix):
                cols = [kmer, str(o)] + [str(r) for r in ratios]
                f.write("\t".join(cols) + '\n')

    def store_f_ratios(self, k):
        kmer_order = self.get_optimal_kmer_ranking(k)
        kmer_ranks = np.zeros(len(kmer_order), dtype=int)
        kmer_ranks[kmer_order] = np.arange(len(kmer_order), dtype=int)
       
        f_matrix = []
        for run_key in sorted(self.runs.keys()):
            kr, rbp_conc = run_key
            if kr != k:
                continue
            res = self.runs[run_key]
            ratio = res.f_ratios
            f_matrix.append(ratio)

        f_matrix = np.array(f_matrix).T
        with file(os.path.join(self.out_path, "{0}mer_f_ratio.tsv".format(k)), 'w') as f:
            head = ['# kmer','ska_rank' ] + [str(r.rbp_conc) for r in self.reads[1:]]
            f.write("\t".join(head) + '\n')
            
            for kmer, o, ratios in zip(yield_kmers(k), kmer_ranks, f_matrix):
                cols = [kmer, str(o)] + [str(r) for r in ratios]
                f.write("\t".join(cols) + '\n')

            
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
        
    def get_cooccurrence_tensor(self, k, n_max=20, resume=True):
        import pickle

        tname = "{self.out_path}/cooccurrence_tensor_{k}mers.pkl".format(**locals())
        if os.path.exists(tname):
            self.logger.debug("loading from '{tname}'".format(tname=tname) )
            kmer_list, indices, best_rbp_conc, pd_tensor, in_tensor, R_values, pd_N, in_N = pickle.load(file(tname,'rb'))

        else:
            kmer_list, indices, best_rbp_conc = self.select_significant_kmers(k, n_max=n_max)
            self.logger.info("cooccurrence tensor analysis for k={k} rbp_conc={best_rbp_conc}nM kmers='{kmer_list}'".format(**locals()) )
            
            #kmer_list = sorted(kmer_list) # TODO: re-order by layer correlation
            best_run = self.runs[(k, best_rbp_conc)]

            assert len(kmer_list) == len(indices)
            pd_freqs = best_run.pd_reads.kmer_counts(k)[indices]
            in_freqs = best_run.in_reads.kmer_counts(k)[indices]
            pd_N = best_run.pd_reads.N * (best_run.pd_reads.L - k + 1)
            in_N = best_run.in_reads.N * (best_run.in_reads.L - k + 1)
            

            pd_tensor = best_run.pd_reads.kmer_cooccurrence_distance_tensor(kmer_list)
            in_tensor = best_run.in_reads.kmer_cooccurrence_distance_tensor(kmer_list)
            R_values = best_run.R_values[indices]
            self.logger.debug("saving to '{tname}'".format(tname=tname) )
            pickle.dump( (kmer_list, indices, best_rbp_conc, pd_tensor, in_tensor, R_values, pd_N, in_N), file(tname,'wb'), protocol=pickle.HIGHEST_PROTOCOL )
        
        
        return kmer_list, indices, best_rbp_conc, pd_tensor, in_tensor, R_values, pd_N, in_N

        
    def cooccurrence_tensor_analysis(self, kmer_list, indices, best_rbp_conc, pd_tensor, in_tensor, R_values, pd_N, in_N, pseudo_count=20.):
      
        #print pd_tensor
        n,m,l = pd_tensor.shape
        print n,m,l
        print kmer_list
        k = len(kmer_list[0])
        #print "expected co-occurrences"
        #print in_freqs.shape
        
        exp_in = R_values[:, np.newaxis, np.newaxis] * R_values[np.newaxis, :, np.newaxis] * (np.ones(l) )[np.newaxis, np.newaxis,:]
        #print exp_in.shape
        #exp_in[:,:,0] = 0
        
        #print exp_in
        print "ratio of input tensor to expected"
        
        scale = pd_N / float(in_N)
        print "scale", scale, pd_N, in_N
        enr_tensor = np.log2( (pd_tensor+1) / (in_tensor*exp_in + 1) )
        
        #for s in range(1,l-5):
        for s in range(1,k*2):
            pd_layer = pd_tensor[:,:,s]
            in_layer = in_tensor[:,:,s]
            exp_layer = exp_in[:,:,s]
            enr_layer = enr_tensor[:,:,s]
            #print "input"
            #print in_layer
            #print "expected (independent)"
            #print exp_layer * in_layer
            #print "observed (pull down)"
            #print pd_layer
            #print "log2 ratios"
            #print enr_layer
            ordered = enr_layer.argsort(axis=None)[::-1]
            #print "most co-occuring at spacing",s
            #for n in ordered[:10]:
                ##print n
                #m1,m2 = np.unravel_index(n, enr_layer.shape)
                ##print m1,m2
                #print kmer_list[m1]
                #print " "*(s-1), kmer_list[m2], enr_layer[m1,m2]

            
            pp.figure()
            pp.title("k={0} spacing={1}".format(k, s))
            z = enr_layer.T
            print z.argmax(axis=0)
            import scipy.cluster.hierarchy
            Z = scipy.cluster.hierarchy.linkage(z, method='single')
            print Z[n-2]
            
            y, x = np.mgrid[slice(0, n),slice(0, n)]
            # symmetric, dynamic range of colorbar
            dr = np.fabs(z).max()
            pp.pcolor(x,y,z, cmap=pp.get_cmap('seismic'), vmin=-dr, vmax=dr)

            pp.yticks(np.arange(n)+.5, kmer_list) 
            pp.xticks(np.arange(n)+.5, kmer_list, rotation=90) 
            pp.xlabel("first kmer")
            pp.ylabel("second kmer")
            #pp.xlim(0,)
            cbar = pp.colorbar(orientation="horizontal", fraction=0.1, shrink=0.75, label=r"$\log_2( \frac{pd}{in} )$")
            cbar.ax.tick_params(labelsize=8)
        
            pp.savefig("{self.out_path}/{k}_{s}.heatmap.pdf".format(**locals()))
            
        enr_tensor = np.log2((pd_tensor + pseudo_count) / (in_tensor + pseudo_count))
        
        # length dependence
        #import matplotlib.pyplot as pp
        #return
    
        #pp.figure()
        #pp.title("k={0}".format(k)))
        #pp.plot(enr_tensor.max(axis=0).max(axis=0))
        #pp.show()
        
        #from mayavi import mlab
        #x_max, y_max, z_max = enr_tensor.shape
        
        #x, y, z = np.meshgrid(np.arange(x_max), np.arange(y_max), np.arange(z_max))
        #mlab.points3d(x, y, z, enr_tensor, transparent=True, mode='sphere', colormap='hot')
        
        #for i,mer in enumerate(kmer_list):
            #mlab.text3d(i,0,0, mer, orient_to_camera=False, orientation = ( 0, -2, -90), scale=.5)
            #mlab.text3d(0,i,0, mer, orient_to_camera=False, orientation = ( -2, 0, 190), scale=.5)
        #mlab.xlabel('first kmer')
        #mlab.ylabel('second kmer')
        #mlab.zlabel('spacing')
        #mlab.colorbar()
        #mlab.show()

        
    def find_interactors(self, k, n_top=2, k_flank_max=4):
        
        for core in self.select_significant_kmers(k, n_top):
            for k_int in range(1, k_flank_max+1):
                t0 = time.time()
                screen = PairInteractionScreen(self.runs[(k, self.best_rbp_conc)], core, k_int)
                t1 = time.time()
                self.logger.debug("interaction analysis of {0} with {1}mers took {2:.3f}s".format(core, k_int, (t1-t0)) )
                
                fplot = os.path.join(self.out_path, "{0}_interacting_with_{1}mers.pdf".format(core, k_int) )
                screen.make_plot(fplot)
    

class PairInteractionScreen(object):
    def __init__(self, res, core, k_int, pseudo=10.):
        self.res = res
        self.core = core
        self.k_int = k_int
        self.k_core = len(core)
        
        #print "scanning flanking {0}-mers".format(k_int)
        pd_matrix, pd_mask = self.res.pd_reads.kmer_flank_profiles(core, k_int)
        in_matrix, in_mask = self.res.in_reads.kmer_flank_profiles(core, k_int)

        self.core_density_pd = pd_mask.sum(axis=0)
        self.core_density_bg = in_mask.sum(axis=0)
        
        pd_matrix = np.array(pd_matrix, dtype=np.float32) + pseudo
        in_matrix = np.array(in_matrix, dtype=np.float32) + pseudo

        obsv = pd_matrix / pd_matrix.sum(axis=0)[np.newaxis,:]
        bgnd = in_matrix / in_matrix.sum(axis=0)[np.newaxis,:]

        # Kullback-Leibler (KL) divergence terms
        self._KL = (obsv * np.log2(obsv / bgnd) )
        # and per-position KL
        self.KL = self._KL.sum(axis=0)
        self.log_ratios = np.log2(obsv / bgnd)

        # mask positions overlapping with the core motif
        self.l = self.res.pd_reads.L - self.k_core
        self.KL[self.l-self.k_int+1:self.l+self.k_core] = 0
        self.log_ratios[:,self.l-self.k_int+1:self.l+self.k_core] = 0
        #print "mask",self.l-self.k_int+1,self.l+self.k_core 
        #print "log_ratios after masking", self.log_ratios[:,self.l-self.k_int+1:self.l+self.k_core]

    def top_interactors(self, n_top=10):
        #TODO: cook a set of candidate interacting kmers from significance for now let's just take top 10
        top_i = self._KL.max(axis=1).argsort()[::-1][:n_top]
        # and sort alphabetically to ensure reproducibility across successive runs
        top_i = sorted(top_i)
        top_kmers = [cska.ska_kmers.index_to_seq(i, self.k_int) for i in top_i]
        #print "top interacting kmer candidate list", top_kmers
        return top_i, top_kmers
    
    def make_plot(self, fname, n_top=10):
        import matplotlib as mp
        mp.rcParams['font.family'] = 'Arial'
        mp.rcParams['font.size'] = 8
        mp.rcParams['font.sans-serif'] = 'Arial'
        mp.rcParams['legend.fontsize'] = 'small'
        mp.rcParams['legend.frameon'] = False
        #mp.rcParams['axes.labelsize'] = 8
        
        import matplotlib.pyplot as pp
        fig = pp.figure()
        fig.subplots_adjust(hspace=0.5)
        
        pp.title("{self.res.pd_reads.rbp_name}@{self.res.pd_reads.rbp_conc}nM {self.core} interacting with {self.k_int}-mers".format(self=self))
        pp.subplot(311)
        pp.gca().set_title("density of {0} core".format(self.core.upper()))
        pp.plot(self.core_density_pd, drawstyle='steps-mid', label="pd")
        pp.plot(self.core_density_bg, drawstyle='steps-mid', label="in")
        pp.gca().locator_params(axis='y',nbins=3)
        pp.gca().locator_params(axis='x',nbins=10)

        pp.legend(loc='upper left')
        pp.xlabel("read start pos [nt]")
        pp.ylabel("frequency")
        
        pp.subplot(312)
        pp.title("Kullback-Leibler divergence of flanking kmer composition")
        x = np.arange(len(self.KL)) - len(self.KL)/2 +1.
        pp.plot(x,self.KL, drawstyle='steps-mid', label="{0}mers around {1}".format(self.k_int, self.k_core) )
        pp.xlim(-self.l-.5,self.l+.5)
        pp.xlabel("rel. {0}-mer start pos [nt]".format(self.k_int))
        pp.ylabel("KL [bits]")
     
        pp.subplot(313)
        pp.gca().set_title("enriched {0}-mers".format(self.k_int))
        top_i, top_kmers = self.top_interactors(n_top = n_top)
        z = self.log_ratios[top_i[::-1],:]
        y, x = np.mgrid[slice(0, len(top_i)+1),slice(-(self.l+.5), +self.l+1)]
        
        # symmetric, dynamic range of colorbar
        dr = np.fabs(z).max()
        pp.pcolor(x,y,z, cmap=pp.get_cmap('seismic'), vmin=-dr, vmax=dr)

        pp.yticks(np.arange(len(top_i))+.5, top_kmers[::-1]) # reverse order of kmers so pcolor is not upside down
        pp.xlim(-self.l-.5,self.l+.5)
        cbar = pp.colorbar(orientation="horizontal", fraction=0.1, shrink=0.75, label=r"$\log_2( \frac{pd}{in} )$")
        cbar.ax.tick_params(labelsize=8)
        pp.savefig(fname)
        pp.close()
        


def yield_kmers(k):
    """
    An iterater to all kmers of length k in alphabetical order
    """
    bases = 'ACGT'
    for kmer in itertools.product(bases, repeat=k):
        yield ''.join(kmer)


def main():
    from optparse import OptionParser
    usage = "usage: %prog [options] <input_reads_file> <pulldown_reads_file1> [<pulldown_reads_file2] [...]"

    parser = OptionParser(usage=usage)
    parser.add_option("-k","--min-k",dest="min_k",default=3,type=int,help="min kmer size (default=3)")
    parser.add_option("-K","--max-k",dest="max_k",default=8,type=int,help="max kmer size (default=8)")
    
    parser.add_option("-n","--n-passes",dest="n_passes",default=10,type=int,help="max number of passes (default=10)")
    parser.add_option("-R","--rna-concentration",dest="rna_conc",default=1000.,type=float,help="concentration of random RNA used in the experiment in micro molars (default=100uM)")
    parser.add_option("-p","--rbp-concentration",dest="prot_conc",default="0,320",help="(comma separated list of) protein concentration used in the experiment(s) in nano molars (default=0,300)")
    parser.add_option("","--name",dest="name",default="RBP",help="name of the protein assayed (default=RBP)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimateion (default=5)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("-c","--convergence",dest="convergence",default=0.5,type=float,help="convergence is reached when max. change in absolute weight is below this value (default=0.5)")
    parser.add_option("-B","--background", dest="background", default="", help="path to file with background (input) kmer abundances in the library")
    parser.add_option("-o","--output",dest="output",default=".",help="path where results are to be stored")
    parser.add_option("","--debug",dest="debug",default=False, action="store_true",help="SWITCH: activate debug output")
    parser.add_option("","--no-resume",dest="noresume",default=False, action="store_true",help="SWITCH: disable loading of results from previous runs")
    parser.add_option("","--interactions",dest="interactions",default=False, action="store_true",help="SWITCH: activate combinatorial search")
    parser.add_option("","--n-max",dest="n_max",default=0, type=int,help="TESTING: read at most N reads")
    parser.add_option("","--version",dest="version",default=False, action="store_true",help="SWITCH: show version information and quit")
    options,args = parser.parse_args()

    if options.version:
        print __version__
        print __license__
        print "by", ", ".join(__authors__)
        sys.exit(0)

    if not args:
        parser.error("missing argument: need <reads_file> (or use /dev/stdin)")
        sys.exit(1)

    rbp_concentrations = [float(c) for c in options.prot_conc.split(',')]
    # prepare outout path
    if not os.path.exists(options.output):
        os.makedirs(options.output)

    # set up logging
    log_path = os.path.join(options.output,"run.log")
    if options.debug:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO

    FORMAT = '%(asctime)-20s\t%(levelname)s\t%(name)s\t%(message)s'
    formatter = logging.Formatter(FORMAT)
    logging.basicConfig(level=lvl, format=FORMAT)    
    root = logging.getLogger('')
    fh = logging.FileHandler(filename=log_path, mode='w')
    fh.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(fh)
    
    logger = logging.getLogger("CSKA")
    logger.info("version {0}".format(__version__))
    logger.info("invoked as '{0}'".format(" ".join(sys.argv)) )


    # parametrize SKA algorithm
    ska = SKARunner(
        max_iterations = options.n_passes,
        convergence = options.convergence, 
    )
    
    # setting class-variables
    RBNSReads.out_path = options.output
    RBNSResult.out_path = options.output

    # start a new analysis
    rbns = RBNSAnalysis(
        rbp_name = options.name,
        out_path = options.output,
        ska_runner = ska,
    )
    
    # populate with experimental data
    for fname, rbp_conc in zip(args, rbp_concentrations):
        reads = RBNSReads(
            fname, 
            rbp_conc=rbp_conc,
            rbp_name = options.name,
            n_max=options.n_max, 
            pseudo_count=options.pseudo, 
            rna_conc = options.rna_conc,
            n_subsamples = options.subsamples
        )
        
        rbns.add_reads(reads)
    
    rbns.store_all_results(5)
    
    #print rbns.R_value_matrix(5)
    #print rbns.SKA_weight_matrix(5)
    ## first stage of analysis: actual streaming kmer 
    ## analysis on all samples and for a range of k
    ##rbns.run_ska(
        ##kmin = options.min_k, 
        ##kmax = options.max_k,
        ##max_iterations = options.n_passes, 
        ##convergence = options.convergence, 
        ##subsamples = options.subsamples,
        ##resume = not options.noresume,
    ##)
    ##rbns.run_f_value_fit(5)
    
    ##compute f-values, make overview plots
    ###rbns.compare_k()
    ##rbns.run_ROC()
    #rbns.compute_recall_ratios(5)
    #rbns.compute_recall_ratios(6)
    #rbns.compute_recall_ratios(7)
    #rbns.compute_recall_ratios(8)
    #rbns.compute_recall_ratios(9)

    #rbns.store_f_ratios(5)
    #rbns.store_f_ratios(6)
    #rbns.store_f_ratios(7)
    #rbns.store_f_ratios(8)
    #rbns.store_f_ratios(9)
    
    ## screen for multi-part motifs
    #if options.interactions:
        #tensors = rbns.get_cooccurrence_tensor(2)
        #rbns.cooccurrence_tensor_analysis(*tensors)
        
        #tensors = rbns.get_cooccurrence_tensor(3)
        #rbns.cooccurrence_tensor_analysis(*tensors)

        #tensors = rbns.get_cooccurrence_tensor(4)
        #rbns.cooccurrence_tensor_analysis(*tensors)

        #tensors = rbns.get_cooccurrence_tensor(5)
        #rbns.cooccurrence_tensor_analysis(*tensors)

        #tensors = rbns.get_cooccurrence_tensor(6)
        #rbns.cooccurrence_tensor_analysis(*tensors)

        ##rbns.find_interactors(5, k_flank_max=3, n_top=2)

if __name__ == '__main__':
    main()
