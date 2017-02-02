#!/usr/bin/env python
__license__ = "MIT"
__version__ = "0.9.3"
__authors__ = ["Marvin Jens","Alex Robertson"]
__email__ = "mjens@mit.edu"

import sys
import itertools
import numpy as np
import copy
import time
import os
import logging
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
        chunk_n = self.N / float(N)
        start = int(np.floor(i * chunk_n))
        end = max(self.N, int(np.floor((i+1) * chunk_n)))
        l = end - start
        self.logger.debug("returning subsample of len={l} from {start}:{end}".format(**locals()) )
        
        return RBNSReads("{self.fname}_subsample_{i:02d}".format(**locals()), seqm=self.seqm[start:end], pseudo_count=self.pseudo_count )
        
        
    def kmer_counts(self, k):
        if not k in self.cached_counts:
            self.cached_counts[k] = ska_kmers.seq_set_kmer_count(self.seqm, k)

        return self.cached_counts[k]
    
    def kmer_frequencies(self,k):
        t0 = time.time()
        counts = self.kmer_counts(k) + self.pseudo_count
        t1 = time.time()
        self.logger.debug("counted kmer occurrences in {0:.1f} seconds".format( (t1-t0) ) )

        N = counts.sum()
        freqs = np.array(counts, dtype=np.float32)/N * (4**k)

        return freqs


    def fraction_reads_with_kmer(self,k):
        t0 = time.time()
        counts_oligos_with_kmer = ska_kmers.get_oligo_counts_with_kmers(self.seqm, k)
        print "here" ###
        t1 = time.time()
        self.logger.debug("counted read fractions attributable to motifs in {0:.1f} seconds".format( (t1-t0) ) )
        
        fractions = np.array(counts_oligos_with_kmer, dtype=np.float32)/self.N
        
        return fractions


    def fraction_reads_with_pattern(self,k,masks):
        t0 = time.time()
        pattern_counts = ska_kmers.get_oligo_counts_with_patterns(self.seqm, k, masks)
        t1 = time.time()
        self.logger.debug("counted read fractions attributable to patterns in {0:.1f} seconds".format( (t1-t0) ) )

        fractions = np.array(pattern_counts, dtype=np.float32)/self.N

        return fractions


    def kmer_filter(self, kmer):
        """
        returns the subset of seqm that contains sequences with the desired kmer
        and a boolean matrix with ones at the positions of kmer occurrence
        """
        return ska_kmers.kmer_filter(self.seqm, kmer)
    
    def kmer_flank_profiles(self, kmer, k_max):
        """
        use kmer_filter first and then compute the average occurrences of kmers
        with k=k_flank (k_flank = 1..k_max) around the desired "central" kmer.
        """
        return ska_kmers.kmer_flank_profiles(self.seqm, kmer, k_max=k_max)
    
    def __str__(self):
        return "RBNSReads('{self.fname}')".format(self=self)

    #def try_load_background_freqs(self,k):
        #if self.input_run:
            #bg_path = os.path.join(self.input_run,"rel_freqs.{0}mer.txt".format(k))
            #background = np.array(read_freqs(bg_path), dtype=np.float32)
            #self.logger.info("loaded background kmer frequencies from {0}".format(bg_path))
        #else:
            #background = np.ones(4 ** k, dtype=np.float32)
            #bg_path='None'
        
        #return background, bg_path


            

class SKAResult(object):
    def __init__(self, pd_reads, in_reads, k, R_values, ska_weights):
        self.logger = logging.getLogger('SKAResults')
        self.k = k
        self.pd_reads = pd_reads
        self.in_reads = in_reads
        
        self.R_values = R_values
        self.R_values_err = np.zeros(R_values.shape, dtype=R_values.dtype)
        
        self.ska_weights = ska_weights
        self.ska_weights_err = np.zeros(ska_weights.shape, dtype=ska_weights.dtype)
        
        self.name = 'SKA:{pd_reads}:{k}mers:bg={in_reads}'.format(**locals())
        
    def store(self, out_path):
        self.logger.info('storing kmer frequencies, R-values and SKA-weights for "{self.name}" in "{out_path}"'.format(self=self, out_path=out_path) )
       
        z_scores_ska = (self.ska_weights - self.ska_weights.mean() )/ self.ska_weights.std()
        z_scores_R = (self.R_values - self.R_values.mean() )/ self.R_values.std()

        results = zip(
            yield_kmers(self.k), 
            self.in_reads.kmer_frequencies(self.k), 
            self.pd_reads.kmer_frequencies(self.k), 
            self.R_values,
            self.R_values_err,
            self.ska_weights,
            self.ska_weights_err,
            z_scores_ska,
            z_scores_R
        )

        def round_to_2(x):
            return round(x, max(-int(np.floor(np.log10(abs(x)))), 2) ) 
        
        def round_to_err(x, x_err):
            n_dig = int(np.ceil(-np.log10(x_err)))+1
            x_err = round(x_err,n_dig)

            n_dig = int(np.ceil(-np.log10(x_err)))+1
            return [round(x,n_dig), x_err]

        with file(os.path.join(out_path, 'SKA_results.{0}mer.txt'.format(self.k)), 'w') as of:
            of.write('# kmer\t rel_freq_input\t rel_freq_pulldown\t R_value\tR_err\t ska_weight\tska_err\tska_z_score\t R_z_score\n')
            for mer,fi,fp,r,r_err,ska,ska_err,z_ska,z_r in results:
                if mer == 'TATC':
                    print ska,ska_err
                    print round_to_err(ska,ska_err)
                    
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

class SecondPass(object):
    def __init__(self, reads, run):
        self.reads = reads
        self.run = run
        

    def analyze_top_kmers(self, k):
        res = self.run.results[k]
        for i in res.ska_weights.argsort()[::-1][:2]:
            kmer = ska_kmers.index_to_seq(i, k)
            print "testing", kmer
            for i, count_matrix in enumerate(self.reads.kmer_flank_profiles(kmer, 4)):
                k_flank = i+1
                
                # assume identical composition at each position as background model
                bg = np.array(self.reads.kmer_counts(k_flank), dtype=float)
                bg /= bg.sum()
                
                #print k_flank, count_matrix.shape
                count_matrix = np.array(count_matrix, dtype=float) + 1. # pseudo count
                
                obs = count_matrix / count_matrix.sum(axis=0)[np.newaxis,:]
                #print obs
                #print "BG",bg
                
                #_KL = (obs * np.log2(obs / bg[:,np.newaxis]) )
                _KL = (bg[:,np.newaxis] * np.log2(bg[:,np.newaxis]/obs) )
                
                KL = _KL.sum(axis=0)
                
                # mask the core motif!
                KL[20-k-k_flank + 1:20+k_flank-1] = 0
                
                # mask the ends TODO: pos-specific background model
                KL[:2] = 0
                KL[-2:] = 0
                
                print "testing for reduction of entropy of {k_flank} mers".format(k_flank=k_flank)
                z = (KL - KL.mean())/KL.std()
                print np.round(z,2)
                print "positions with KL > 1 standard deviations"
                for pos in (z >= 1.).nonzero()[0]:
                    kl = _KL[:,pos]
                    I = kl.argsort()
                    print "{pos} z={z:.2f} most enriched words:".format(pos=pos, z=z[pos])
                    for i in I[:5]:
                        if kl[i] > 0:
                            break
                        
                        word = ska_kmers.index_to_seq(i, k_flank)
                        print "   {word} KL={kl:.3f}".format(word=word, kl=kl[i])
                        
                
                #totals = np.array(profiles.sum(axis=0), dtype=float)
                #freq = profiles / totals
                #ranked_us = []
                #ranked_ds = []
                
                #for i, f in enumerate(freq):
                    #upstream = f[:18-k_flank]
                    #dostream = f[20:]
                    ##print upstream
                    ##print kmer
                    ##print dostream
                    #us = entropy(upstream)
                    #ds = entropy(dostream)
                    #word = ska_kmers.index_to_seq(i,k_flank)
                    ##print "{word} us_entropy={us:.2f} bits ds_entropy={ds:.2f} bits".format(**locals())
                    
                    #ranked_us.append((us, word))
                    #ranked_ds.append((ds, word))
                    
                #print "most associated upstream words"
                #for bits, word in sorted(ranked_us, reverse=True)[:10]:
                    #print "{word} {bits:.5f} bits".format(**locals())
                    
                #print "most associated downstream words"
                #for bits, word in sorted(ranked_ds, reverse=True)[:10]:
                    #print "{word} {bits:.5f} bits".format(**locals())
            
            

class SKARun(object):
    def __init__(self, pd_reads, in_reads, max_iterations=10, convergence=0.5, out_path=".", subsamples=10):
        self.logger = logging.getLogger('SKARun')
        self.pd_reads = pd_reads
        self.in_reads = in_reads
        self.max_iterations = max_iterations
        self.convergence = convergence
        self.out_path = os.path.abspath(out_path)
        self.n_subsamples = subsamples
        
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


    def run(self, min_k, max_k):
        for k in range(min_k, max_k+1):
            self.logger.info("streaming {0}-mers".format(k))
            res = self.stream_counts(k, self.pd_reads, self.in_reads)
            
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
            res.store(self.out_path)
        

def yield_kmers(k):
    """
    An iterater to all kmers of length k in alphabetical order
    """
    bases = 'ACGT'
    for kmer in itertools.product(bases, repeat=k):
        yield ''.join(kmer)


def main():
    from optparse import OptionParser
    usage = "usage: %prog [options] <pulldown_reads_file> <input_reads_file> OR cat <reads_file> | %prog [options] /dev/stdin"

    parser = OptionParser(usage=usage)
    parser.add_option("-k","--min-k",dest="min_k",default=3,type=int,help="min kmer size (default=3)")
    parser.add_option("-K","--max-k",dest="max_k",default=8,type=int,help="max kmer size (default=8)")
    
    parser.add_option("-n","--n-passes",dest="n_passes",default=10,type=int,help="max number of passes (default=10)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimateion (default=5)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("-c","--convergence",dest="convergence",default=0.5,type=float,help="convergence is reached when max. change in absolute weight is below this value (default=0.5)")
    parser.add_option("-B","--background", dest="background", default="", help="path to file with background (input) kmer abundances in the library")
    parser.add_option("-o","--output",dest="output",default=".",help="path where results are to be stored")
    parser.add_option("","--debug",dest="debug",default=False, action="store_true",help="SWITCH: activate debug output")
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
    pd_reads = RBNSReads(args[0], n_max=options.n_max, pseudo_count=options.pseudo)
    in_reads = RBNSReads(args[1], n_max=options.n_max, pseudo_count=options.pseudo)
    
    # run streaming kmer analysis
    ska = SKARun(
        pd_reads,
        in_reads,
        max_iterations = options.n_passes, 
        convergence = options.convergence, 
        out_path = options.output,
        subsamples = options.subsamples,
    )
    
    ska.run(options.min_k, options.max_k)
    
    #pass2 = SecondPass(rbns_reads, ska)
    #print pass2.analyze_top_kmers(3)
    ##print pass2.analyze_top_kmers(4)
    #print pass2.analyze_top_kmers(5)

if __name__ == '__main__':
    main()
