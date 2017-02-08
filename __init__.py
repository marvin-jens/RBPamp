#!/usr/bin/env python
__license__ = "MIT"
__version__ = "0.9.5"
__authors__ = ["Marvin Jens","Alex Robertson"]
__email__ = "mjens@mit.edu"

import sys
import itertools
import numpy as np
import copy
import time
import os
import logging
import collections
import ska_kmers


class RBNSReads(object):
    def __init__(self, fname, chunklines=2000000, n_max=0, pseudo_count=10, seqm=[]):
        self.logger = logging.getLogger('RBNSReads')
        self.fname = fname
        self.pseudo_count = pseudo_count
        
        if len(seqm):
            self.seqm = seqm
            self.N, self.L = self.seqm.shape
        else:
            self.logger.info('reading sequences from {fname}'.format(fname=fname) )
            # load and keep all sequences in memory (numerically A=0,...T=3 )
            t0 = time.time()
            self.seqm = ska_kmers.read_raw_seqs_chunked(file(fname), chunklines=chunklines, n_max=n_max)
            self.N, self.L = self.seqm.shape
            t1 = time.time()

            self.logger.info("read {0:.3f}M sequences of length {1} in {2:.1f} seconds".format(self.N/1E6, self.L, (t1-t0) ) )


        self.cached_counts = {}

    def subsample(self, i, N):
        """
        Returns the i-th out of N (i < N) equally sized chunks of the total data. 
        Returned object is again a RBNSReads object.
        """
        chunk_n = self.N / float(N)
        start = int(np.floor(i * chunk_n))
        end = min(self.N, int(np.floor((i+1) * chunk_n)))
        l = end - start
        self.logger.debug("returning subsample of len={l} from {start}:{end}".format(**locals()) )
        
        return RBNSReads("{self.fname}_subsample_{i:02d}".format(**locals()), seqm=self.seqm[start:end], pseudo_count=self.pseudo_count )
        
        
    def kmer_counts(self, k):
        """
        Returns kmer counts. Keeps counts cached so that successive queries for 
        the same k are just a lookup.
        """
        if not k in self.cached_counts:
            self.cached_counts[k] = ska_kmers.seq_set_kmer_count(self.seqm, k)

        return self.cached_counts[k]
    
    def kmer_frequencies(self,k):
        """
        Returns relative kmer frequencies, scaled such that they add up 4**k.
        This means that a uniform kmer distribution would give 1 for every kmer.
        """
        t0 = time.time()
        counts = self.kmer_counts(k) + self.pseudo_count
        t1 = time.time()
        self.logger.debug("counted {0}mer occurrences in {1:.3f} ms".format( k, (t1-t0)*1000. ) )

        N = counts.sum()
        freqs = np.array(counts/float(N) * (4**k), dtype=np.float32)

        return freqs

    def kmer_filter(self, kmer):
        """
        returns the subset of seqm that contains sequences with the desired kmer
        and a boolean matrix with ones at the positions of kmer occurrence
        """
        return ska_kmers.kmer_filter(self.seqm, kmer)


    def recall(self, kmer_order):
        
        kmer_ranks = np.zeros(len(kmer_order))
        kmer_ranks[kmer_order] = np.arange(len(kmer_order))
        
        counts_by_kmer_rank = ska_kmers.count_best_ranked_hits(self.seqm, np.array(kmer_ranks,dtype=np.uint32) )[kmer_order]
        
        return (counts_by_kmer_rank.cumsum() / float(self.N))


    def kmer_flank_profiles(self, kmer, k_flank):
        """
        use kmer_filter first and then compute the average occurrences of kmers
        with k=k_flank (k_flank = 1..k_max) around the desired "central" kmer.
        """
        return ska_kmers.kmer_flank_profiles(self.seqm, kmer, k_flank=k_flank)
    
    def __str__(self):
        return "RBNSReads('{self.fname}' N={self.N} L={self.L})".format(self=self)


            

class SKAResult(object):
    def __init__(self, pd_reads, in_reads, k, R_values, ska_weights, R_values_err=None, ska_weights_err=None):
        self.logger = logging.getLogger('SKAResults')
        self.k = k
        self.pd_reads = pd_reads
        self.in_reads = in_reads
        
        self.R_values = R_values
        self.z_scores_R = (self.R_values - self.R_values.mean() )/ self.R_values.std()
        
        if R_values_err == None:
            self.R_values_err = np.zeros(R_values.shape, dtype=R_values.dtype)
        else:
            self.R_values_err = R_values_err
        
        self.ska_weights = ska_weights
        self.z_scores_ska = (self.ska_weights - self.ska_weights.mean() )/ self.ska_weights.std()
        if ska_weights_err == None:
            self.ska_weights_err = np.zeros(ska_weights.shape, dtype=ska_weights.dtype)
        else:
            self.ska_weights_err = ska_weights_err
        
        self.name = 'SKA:{pd_reads}:{k}mers:bg={in_reads}'.format(**locals())
    
    @staticmethod
    def _res_filename(k, rbp_name, rbp_conc):
        return 'SKA.{rbp_name}.{rbp_conc:.0f}nM.{k}mer.txt'.format(**locals())
    
    @classmethod
    def load(cls, load_path, k, pd_reads, in_reads, rbp_name, rbp_conc):
        path = os.path.join(load_path, cls._res_filename(k, rbp_name, rbp_conc))
        with file(path, 'r') as f:
            rows = []
            kmers = []
            
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.rstrip().split('\t')
                kmers.append(parts[0])
                rows.append(parts[1:])
                
        data = np.array(rows, dtype=float)
        kmers = np.array(kmers)
        I = kmers.argsort()
        
        # undo the sorting by SKA-weight
        data = data[I]
        kmers = kmers[I]
        
        fin, fpd, R_values, R_values_err, ska_weights, ska_weights_err, ska_z, R_z = data.T
        res = cls(pd_reads, in_reads, k, R_values, ska_weights, R_values_err=R_values_err, ska_weights_err=ska_weights_err)
        res.logger.info("successfully loaded from '{0}'".format(path))
        
        return res
                
        
    def store(self, out_path, rbp_name, rbp_conc):
        
        res_file = SKAResult._res_filename(self.k, rbp_name, rbp_conc)
        self.logger.info('storing kmer frequencies, R-values and SKA-weights in "{out_path}/{res_file}"'.format(**locals()) )
       

        order = self.ska_weights.argsort()[::-1]
        results = zip(
            np.array(list(yield_kmers(self.k))) [order],
            self.in_reads.kmer_frequencies(self.k)[order], 
            self.pd_reads.kmer_frequencies(self.k)[order], 
            self.R_values[order],
            self.R_values_err[order],
            self.ska_weights[order],
            self.ska_weights_err[order],
            self.z_scores_ska[order],
            self.z_scores_R[order],
        )

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
                return [round(x,n_dig), x_err]
            else:
                return [x, x_err]

        with file(os.path.join(out_path, res_file), 'w') as of:
            of.write('# kmer\t rel_freq_input\t rel_freq_pulldown\t R_value\tR_err\t ska_weight\tska_err\tska_z_score\t R_z_score\n')
            for mer,fi,fp,r,r_err,ska,ska_err,z_ska,z_r in results:
                #if mer == 'AAA':
                    #print ska,ska_err
                    #print round_to_err(ska,ska_err)
                    
                out = [mer, round_to_2(fi), round_to_2(fp),] \
                    + round_to_err(r,r_err) \
                    + round_to_err(ska,ska_err) \
                    + [z_ska,z_r]

                of.write('%s\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n' % tuple(out))
            of.close()

    def __str__(self):
        I = self.ska_weights.argsort()[::-1]
        buf = [self.name]
        for i in I[:10]:
            buf.append("{0}\t{1:.2f}".format(ska_kmers.index_to_seq(i, self.k), self.ska_weights[i] ) )
        
        return '\n'.join(buf)


class SKARun(object):
    def __init__(self, pd_reads, in_reads, max_iterations=10, convergence=0.5, out_path=".", subsamples=10, rbp_name='RBP', rbp_conc=300., rna_conc=100000.):
        self.logger = logging.getLogger('SKARun')
        self.pd_reads = pd_reads
        self.in_reads = in_reads
        self.max_iterations = max_iterations
        self.convergence = convergence
        self.out_path = os.path.abspath(out_path)
        self.n_subsamples = subsamples
        self.rbp_name = rbp_name
        self.rbp_conc = rbp_conc
        self.rna_conc = rna_conc
        
        self.results = {}
        

    def stream_counts(self, k, pd_reads, in_reads):
        kmers = list(yield_kmers(k))
        #background, bg_source = self.try_load_background_freqs(k)

        pd_freqs = pd_reads.kmer_frequencies(k)
        in_freqs = in_reads.kmer_frequencies(k)
        R_values = pd_freqs / in_freqs # kmer-frequency 'R'-atios

        current_weights = copy.copy(R_values)
        
        weight_history = []
        for iteration_i in range(self.max_iterations):
            new_weights = ska_kmers.seq_set_SKA(pd_reads.seqm, current_weights, in_freqs, k)
                
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
               
        return SKAResult(pd_reads, in_reads, k, R_values, current_weights)


    def run(self, k):
        self.logger.info("streaming {0}-mers".format(k))
        res = self.stream_counts(k, self.pd_reads, self.in_reads)
        
        if self.n_subsamples:
            self.logger.info("subsampling...")
            samples = [self.stream_counts(
                k, 
                self.pd_reads.subsample(i, self.n_subsamples), 
                self.in_reads) 
                for i in range(self.n_subsamples)
            ]
            sample_matrix = np.array([s.ska_weights for s in samples])
            res.ska_weights_err = sample_matrix.std(axis=0)
            
            sample_matrix = np.array([s.R_values for s in samples])
            res.R_values_err = sample_matrix.std(axis=0)
        
        self.results[k] = res
        res.store(self.out_path, self.rbp_name, self.rbp_conc)
        return res
      
    def load(self, k):
        self.logger.info("RESUME: trying to load {0}-mer results from previous run".format(k))
        
        try:
            res = SKAResult.load(self.out_path, k, self.pd_reads, self.in_reads, self.rbp_name, self.rbp_conc)
        except IOError:
            return None
        
        self.results[k] = res
        return res


    
    
    #def precision(self, kmer_order):
        #kmer_ranks = np.zeros(len(kmer_order))
        #kmer_ranks[kmer_order] = np.arange(len(kmer_order))

        #pd_hits = self.pd_reads.count_best_ranked_hits(kmer_ranks)[kmer_order].cumsum()
        #in_hits = self.in_reads.count_best_ranked_hits(kmer_ranks)[kmer_order].cumsum()

        #scale = float(self.pd_reads.N)/self.in_reads.N

        #return (pd_hits / (in_hits*scale + pd_hits))


class PairInteractionScreen(object):
    def __init__(self, run, core, k_int, pseudo=10.):
        self.run = run
        self.core = core
        self.k_int = k_int
        self.k_core = len(core)
        
        #print "scanning flanking {0}-mers".format(k_int)
        pd_matrix, pd_mask = self.run.pd_reads.kmer_flank_profiles(core, k_int)
        in_matrix, in_mask = self.run.in_reads.kmer_flank_profiles(core, k_int)

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
        self.l = self.run.pd_reads.L - self.k_core
        self.KL[self.l-self.k_int+1:self.l+self.k_core] = 0
        self.log_ratios[:,self.l-self.k_int+1:self.l+self.k_core] = 0
        #print "mask",self.l-self.k_int+1,self.l+self.k_core 
        #print "log_ratios after masking", self.log_ratios[:,self.l-self.k_int+1:self.l+self.k_core]

    def top_interactors(self, n_top=10):
        #TODO: cook a set of candidate interacting kmers from significance for now let's just take top 10
        top_i = self._KL.max(axis=1).argsort()[::-1][:n_top]
        # and sort alphabetically to ensure reproducibility across successive runs
        top_i = sorted(top_i)
        top_kmers = [ska_kmers.index_to_seq(i, self.k_int) for i in top_i]
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
        
        pp.title("{0} interacting with {1}-mers".format(self.core, self.k_int))
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
        


class MultiAnalysis(object):
    def __init__(self, runs):
        self.logger = logging.getLogger("MultiAnalysis")
        self.runs = runs
        self.pair_screens = collections.defaultdict(dict)
        self.AUCs = []

    def select_significant_ska_kmers(self,k, n_top=2):
        #TODO: do this based on some statistics, taking into 
        # account the errors of ska weights from subsampling

        res = self.runs[0].results[k]
        kmers = [ska_kmers.index_to_seq(i, k) for i in res.ska_weights.argsort()[::-1][:n_top]]
        return kmers

    def recall_precision_plot(self, k, n_kmers=1000):
        import matplotlib.pyplot as pp

        rbp_name = self.runs[0].rbp_name
        out_path = self.runs[0].out_path

        def make_plot(order_by="ska_weights", name="SKA"):
            
            pp.figure()
            pp.title('discrimination of {rbp_name} pd/input by {name}'.format(**locals()))

            for run in self.runs:
                res = run.results[k]

                order = getattr(res, order_by).argsort()[::-1]

                recall_pd = np.array([0,] + list(run.pd_reads.recall(order)) )
                recall_in = np.array([0,] + list(run.in_reads.recall(order)) )
                
                AUC = np.trapz(recall_pd, recall_in)
                pp.step(recall_in, recall_pd, where='post', label='{0}mers @{1}nM (AUC={2:.3f})'.format(k, run.rbp_conc, AUC))
                self.AUCs.append( (AUC, name, k, rbp_name, run.rbp_conc) )

            pp.plot([0,1.],[0,1.], color='gray', linestyle = 'dashed')
            
            pp.xlabel('fraction of input explained')
            pp.ylabel('fraction of pulldown explained')
            pp.legend(loc='lower right')
            pp.savefig(os.path.join(out_path,"{rbp_name}.{k}mer.{name}.ROC.pdf".format(**locals())))

        make_plot(order_by="ska_weights", name="SKA")
        make_plot(order_by="R_values", name="R")
        
        #print self.AUCs
        print "best discrimination achieved by"
        print sorted(self.AUCs)[-1]

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
            
            kmers = [ska_kmers.index_to_seq(i,k) for i in top_i]
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

        
    def find_interactors(self, k, n_top=2, k_flank_max=4):
        
        for core in self.select_significant_ska_kmers(k, n_top):
            for k_int in range(1, k_flank_max+1):
                t0 = time.time()
                screen = PairInteractionScreen(self.run, core, k_int)
                t1 = time.time()
                self.logger.debug("interaction analysis of {0} with {1}mers took {2:.3f}s".format(core, k_int, (t1-t0)) )
                
                fplot = os.path.join(self.run.out_path, "{0}_interacting_with_{1}mers.pdf".format(core, k_int) )
                screen.make_plot(fplot)
                
                # keep for later?
                #self.pair_screens[core][k_int] = screen
                
                
                


  

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
    parser.add_option("-R","--rna-concentration",dest="rna_conc",default=100.,type=float,help="concentration of random RNA used in the experiment in micro molars (default=100uM)")
    parser.add_option("-p","--rbp-concentration",dest="prot_conc",default="320",help="(comma separated list of) protein concentration used in the pulldown experiment(s) in nano molars (default=300nM)")
    parser.add_option("","--name",dest="name",default="RBP",help="name of the protein assayed (default=RBP)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimateion (default=5)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("-c","--convergence",dest="convergence",default=0.5,type=float,help="convergence is reached when max. change in absolute weight is below this value (default=0.5)")
    parser.add_option("-B","--background", dest="background", default="", help="path to file with background (input) kmer abundances in the library")
    parser.add_option("-o","--output",dest="output",default=".",help="path where results are to be stored")
    parser.add_option("","--debug",dest="debug",default=False, action="store_true",help="SWITCH: activate debug output")
    parser.add_option("","--resume",dest="resume",default=False, action="store_true",help="SWITCH: load results from previous run, to resume with any second stage analyses")
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

    n = len(args) - 1
    protein_concentrations = [float(c) for c in (options.prot_conc.split(',') * n)[:n]]
    print protein_concentrations
    
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
    
    logger = logging.getLogger("SKA")
    logger.info("called as '{0}'".format(" ".join(sys.argv)) )

    # load pull-down and input reads
    in_reads = RBNSReads(args[0], n_max=options.n_max, pseudo_count=options.pseudo)
    pd_reads_list = [RBNSReads(a, n_max=options.n_max, pseudo_count=options.pseudo) for a in args[1:]]
    
    # run streaming kmer analysis
    runs = []
    for pd_reads, p_conc in zip(pd_reads_list, protein_concentrations):
        ska = SKARun(
            pd_reads,
            in_reads,
            max_iterations = options.n_passes, 
            convergence = options.convergence, 
            out_path = options.output,
            subsamples = options.subsamples,
            rbp_name = options.name,
            rbp_conc = p_conc,
            rna_conc = options.rna_conc,
        )

        for k in range(options.min_k, options.max_k+1):
            if options.resume:
                if not ska.load(k):
                    ska.run(k)
            else:
                ska.run(k)
        runs.append(ska)

    multi = MultiAnalysis(runs)
    #multi.compare_k()
    for k in range(options.min_k, options.max_k+1):
        multi.recall_precision_plot(k)

    if options.interactions:
        # TODO: determine good core size!
        multi.find_interactors(3, k_flank_max=3, n_top=5)

if __name__ == '__main__':
    main()
