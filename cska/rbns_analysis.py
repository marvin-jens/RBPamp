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
        self.logger.debug("computing R-values")
        def compute_R(sample, control):
            return sample.kmer_frequencies(k) / control.kmer_frequencies(k)
        
        return self._subsampled(compute_R)

    @cached
    def O_values(self, k):
        self.logger.debug("computing approximate occupancies by fitting R-values to linear overlap model")
        R, R_err = self.R_values(k)
        from cska.rbns_model import CrosstalkMatrix
        cm = CrosstalkMatrix(k, self.in_reads)
        
        occ = cm.o_values(R)
        occ_min = cm.o_values(R - R_err)
        occ_max = cm.o_values(R + R_err)
        
        err_m = np.fabs(occ - occ_min)
        err_M = np.fabs(occ - occ_max)
        occ_err = np.where(err_m > err_M, err_m, err_M )
        return occ, occ_err


    @cached
    @pickled
    def SKA_weights(self, k):
        self.logger.debug("computing SKA-weights")
        def compute_SKA(sample, control):
            return self.ska_runner.stream_counts(k, sample, control)
        
        return self._subsampled(compute_SKA)

    @cached
    @pickled
    def F_ratios(self, k):
        self.logger.debug("computing F-ratios")
        def compute_F_ratio(sample, control):
            return sample.fraction_of_reads_with_kmers(k) / control.fraction_of_reads_with_kmers(k)
        
        return self._subsampled(compute_F_ratio)

    @cached
    @pickled
    def recall_ratios(self, k, kmer_order):
        self.logger.debug("computing recall ratios")
        def compute_recall_ratio(sample, control):
            return sample.recall(k, kmer_order, reorder=False) / control.recall(k, kmer_order, reorder=False)
        
        return self._subsampled(compute_recall_ratio)

    @cached
    @pickled
    def pure_F_ratios(self, k, candidates, out_path = "", n_sample = 100000):
        self.logger.debug("computing pure F-ratios")
        def compute_pure_F_ratio(sample, control):
            if not sample.is_subsample and out_path:
                out_file_pd = os.path.join(out_path, "pure_{k}mer_reads_{sample.name}.fa".format(k = k, sample=sample ))
                out_file_in = os.path.join(out_path, "pure_{k}mer_reads_{control.name}.fa".format(k = k, control=control ))
            else:
                out_file_pd = None
                out_file_in = None
                
            f_pd = sample.fraction_of_reads_with_pure_kmers(k, candidates, out_file=out_file_pd, n_sample = n_sample)
            f_in = control.fraction_of_reads_with_pure_kmers(k, candidates, out_file=out_file_in, n_sample = n_sample)
                
            return f_pd / f_in

        return self._subsampled(compute_pure_F_ratio)
            
    def __str__(self):
        return self.name


class RBNSAnalysis(CachedBase):
    def __init__(self, rbp_name ='RBP', out_path='ska_results', ska_runner=None, known_kd="", n_pure_samples = 100000):
        
        CachedBase.__init__(self)
        
        self.reads = []
        self.rbp_name = rbp_name
        self.out_path = out_path
        self.ska_runner = ska_runner
        #self.write_fasta = write_fasta
        self.known_kd = known_kd
        self.n_pure_samples = n_pure_samples
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
    
    def _make_matrices(self, comp_attr, *argc, **kwargs):
        self.logger.debug("gathering data matrices for {0}".format(comp_attr) )

        M = np.array([getattr(comp, comp_attr)(*argc, **kwargs) for comp in self.comparisons])
        values = M[:,0,:]
        errors = M[:,1,:]
        
        return values, errors
        
    def R_value_matrix(self, k):
        return self._make_matrices("R_values", k)

    def O_value_matrix(self, k):
        return self._make_matrices("O_values", k)

        #self.logger.debug("computing M-value matrix by fitting R-values to linear overlap model")
        
        #from cska.rbns_model import CrosstalkMatrix
        #cm = CrosstalkMatrix(k, self.reads[0])
        ##cm.matrix_plot()
        #R_matrix, R_err_matrix = self.R_value_matrix(k)
        
        #I = self.get_optimal_kmer_ranking(k)[:20]
        #print self.rbp_conc
        #kmers, kd, dG = cm.linear_fit(np.array(self.rbp_conc), R_matrix, I)
        
        ##C = np.dot(cm.M_inv, R)
        ##C_err = np.dot(cm.M_inv, R_err)
        ##print "C-range", C.min(), C.max()
        
        ##c_ofs = C.min()
        ##C -= c_ofs
        ##c_scale = 4**k / C.sum()
        ##C *= c_scale
        
        ##C_err = np.abs(np.dot(cm.M_inv, R_err)  * c_scale)
        
        ##print "C-range", C.min(), C.max()
        ##print "Cerr-range", C_err.min(), C_err.max()
        ##for mer, c, err, r in zip(C, C_err, R, cska.ska_kmers.yield_kmers(k) ):
            ##print mer, c, err, r
        ###print C
        ###print C_err
        
        #for mer, k, E in zip(kmers, kd.T, dG):
            #print mer, k.min(), E
            
        #return C, C_err
        
        
        return self._make_matrices("C_values", k)
            
    def SKA_weight_matrix(self, k):
        return self._make_matrices("SKA_weights", k)

    def F_ratio_matrix(self, k):
        return self._make_matrices("F_ratios", k)
            
    def recall_ratio_matrix(self, k):
        kmer_order = self.get_optimal_kmer_ranking(k)
        return self._make_matrices("recall_ratios", k, kmer_order)

    def pure_F_ratio_matrix(self, k):
        self.logger.debug("computing pure F-ratio matrix for k={0}".format(k) )

        # TODO: replace by select_significant_kmers
        order = self.get_optimal_kmer_ranking(k)
        R, R_err = self.F_ratio_matrix(k)
        Rm = np.median(R - R_err, axis=0)[order]
        
        i_cut = (Rm > 2).argmin()
        candidates = np.zeros(4**k, dtype=np.uint32)
        candidates[order[:i_cut]] = np.arange(i_cut) + 1
        
        #for i,o in enumerate(order[:i_cut]):
            #print cska.ska_kmers.index_to_seq(o, k), candidates[o], Rm[i]
            
        #if self.write_fasta:
            #out_path = self.out_path
        #else:
            #out_path = None

        out_path = None
        return self._make_matrices("pure_F_ratios", k, candidates, out_path=out_path, n_sample = self.n_pure_samples)

        
    @cached
    def get_optimal_kmer_ranking(self, k):
        # TODO: factor in consistently elevated scores with increasing protein concentration?
        from scipy.stats.mstats import gmean
        all_ska_weights = self.SKA_weight_matrix(k)[0]

        #return gmean(all_ska_weights, axis=0).argsort()[::-1]
        return np.median(all_ska_weights, axis=0).argsort()[::-1]
    
    def select_significant_kmers(self,k, z_cut=2, n_min=1, n_max=None):
        ska, ska_err = self.SKA_weight_matrix(k)
        
        # be conservative rg. error of SKA weight, but keep it non-negative
        s = np.where(ska > ska_err, ska - ska_err, 0)
        # z-score across kmers, mean across protein concentations
        z = np.mean( (s - ska.mean(axis=1)[:, np.newaxis]) / ska.std(axis=1)[:, np.newaxis], axis=0)
        
        order = z.argsort()[::-1]
        #print z[order][:20]
        rank_cut = max((z[order] < z_cut).argmax(), n_min)
        if n_max:
            rank_cut = min(n_max, rank_cut)
        
        best_sample_i = ska[:,order[0]].argmax()
        kmers = [cska.ska_kmers.index_to_seq(i, k) for i in order[:rank_cut]]
        return kmers, order[:rank_cut], best_sample_i+1

    def cooccurrence_tensor_analysis(self, k):
        kmers, indices, best_sample_i = self.select_significant_kmers(k)
        print kmers
        reads = self.reads[best_sample_i]
        inrds = self.reads[0] # input control
        
        expect = np.array(reads.expected_kmer_cooccurrence_distance_tensor(kmers), dtype=np.float32)
        obsrvd = np.array(reads.kmer_cooccurrence_distance_tensor(kmers), dtype=np.float32)
        inpool = np.array(inrds.kmer_cooccurrence_distance_tensor(kmers), dtype=np.float32)
                          
        lratio =np.log2((obsrvd+1) / (expect+1) )
        sratio =np.log2((obsrvd+1) / (inpool+1) )
        
        #print "expect kmer co-occurrence", expect[0,1,:].sum()
        #print "obsrvd kmer co-occurrence", obsrvd[0,1,:].sum()
        
        #import matplotlib.pyplot as pp
        #pp.plot( lratio[2,2,:] ) 
        #pp.plot( sratio[2,2,:] ) 
        #pp.show()
        
    def compute_results(self, k, results=["pure_F_ratio", "recall_ratio", "SKA_weight", "R_value", "F_ratio"]):
        order = self.get_optimal_kmer_ranking(k)
        all_kmers = np.array(list(cska.ska_kmers.yield_kmers(k)))

        for name in results:
            fname = "{self.rbp_name}.{name}.{k}mer.tsv".format(**locals())
            path = os.path.join(self.out_path, fname)

            if name == 'binding_constants':
                from cska.rbns_model import RBNSKmerModel, RBNSGenerator
                from cska.folding import OpenenHistCollection, ThreadManager

                #fold_path = os.path.join(self.out_path, "openen")
                #oc = OpenenHistCollection.from_pickle(k, name = self.reads[0].name, path=fold_path)

                ###oc = OpenenHistCollection()
                ###oc.load_pickle('/scratch/data/RBNS/RBFOX2/openen/RBP@0.0nM_temp.6mers.pkl')
                ###theta = oc.occ('TGCATGT', 121., 1.8)
                ###print "THETA", theta

                #mdl = RBNSKmerModel.from_analysis(self, k, unfolding_energies = oc )
                ##mdl = RBNSKmerModel.from_file('/scratch/data/RBNS/RBFOX2/ska_RBFOX2/7mer_f_ratio.tsv', col_start=1, col_end=6)
                ##mdl.unfolding = oc

                #if self.known_kd:
                    #mdl.test_known_kds()

                #kmers, betas, k_est = mdl.fit_full()
                #print "full model: omegas", betas
                #print "full model: k_est", k_est.mean(axis=0)
                #print k_est

                gen = RBNSGenerator(k, l=40, seed=47110815)
                gen.assign_experimental_input("bla.reads")
                r, r_err = self.R_value_matrix(k)
                
                print "RBP_CONC", np.array(self.rbp_conc)
                kmers, k_est, dG = gen.linear_fit(np.array(self.rbp_conc), r, n_top=20)
                print "BINDING ENERGIES AFTER FIT", dG
                print "simulated binding energies", gen.kmer_energies[-20:]*gen.RT
                
                import matplotlib.pyplot as pp
                pp.figure()
                pp.plot(gen.kmer_energies[-20:]*gen.RT, dG, 'ok', alpha=.5)
                pp.xlabel("simulated, exact binding energies")
                pp.ylabel("fitted binding energies from measured R values")
                pp.plot([-11,0],[-11,0], '-', linestyle='dashed', color='gray')
                pp.show()

                N = len(kmers)
                values = np.reshape(k_est.mean(axis=0), (N,1) )
                errors = np.reshape(k_est.std(axis=0), (N,1) )
                
                #errors = betas[:, np.newaxis] #np.zeros(values.shape) * np.NaN
                
                header = ['# kmer', 'Kd_est', 'Kd_err']
                self.write_kmer_matrix(path, kmers, values, errors, header=header)
                
            elif name == 'cooccurrence_tensor':
                rbns.cooccurrence_tensor_analysis(k)
                continue

            else:
                values, errors = getattr(self, "{name}_matrix".format(name=name) )(k)
                self.write_kmer_matrix(path, all_kmers, values.T, errors.T, order)


    def write_kmer_matrix(self, out_path, kmers, values, errors, order=[], err_str='error', header=None):
        self.logger.info("writing data matrix '{out_path}'".format(out_path=out_path) )

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

        if header == None:
            header = ['# kmer'] + list(roundrobin(['{0}nM'.format(c) for c in self.rbp_conc], [err_str for c in self.rbp_conc]))

        if not len(order):
            order = np.arange(len(kmers))

        def round_to_2(x):
            if x:
                return round(x, max(-int(np.floor(np.log10(abs(x)))), 2) ) 
            else:
                return x
        
        def round_to_err(x, x_err):
            if x_err:
                n_dig = int(np.ceil(-np.log10(abs(x_err))))+1
                x_err = round(x_err,n_dig)

                n_dig = int(np.ceil(-np.log10(abs(x_err))))+1
                x = round(x,n_dig)
            
            return [str(x), str(x_err)]

        with file(os.path.join(out_path), 'w') as of:

            of.write("\t".join(header) + '\n')
            for mer, values_row, error_row in zip(kmers[order], values[order], errors[order]):
                
                cols = [str(mer), ]
                for x, err in zip(values_row, error_row):
                    cols.extend(round_to_err(x, err))

                of.write("\t".join(cols) + '\n')
            of.close()
 
