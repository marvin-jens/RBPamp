#!/usr/bin/env python

import sys
import itertools
import numpy as np
import copy
import time
import os
import logging
from ska import RBNSReads
import pdb



def yield_kmers(k):
    """
    An iterater to all kmers of length k in alphabetical order
    """
    bases = 'ACGT'
    for kmer in itertools.product(bases, repeat=k):
        yield ''.join(kmer)




def run_F_values(pd_reads, in_reads, min_k, max_k, out_path=".", subsamples=10):
    logger = logging.getLogger('f_values')
    out_path = os.path.abspath(out_path)
    
    f_value_mean, f_value_err = {}, {}
    
    for k in range(min_k, max_k+1):
        logger.info("calculating f-values for {0}-mers...".format(k))
        
        in_fracs = in_reads.fraction_reads_with_kmer(k)
        
        logger.info("subsampling...")
        samples = [pd_reads.subsample(i, subsamples).fraction_reads_with_kmer(k) \
                       for i in range(subsamples)]
        f_value_matrix = np.array([s/in_fracs for s in samples])
        
        f_value_err[k] = f_value_matrix.std(axis=0)
        f_value_mean[k] = f_value_matrix.mean(axis=0)
        kmers = list(yield_kmers(k))
        write_to_output_file(f_value_mean[k], f_value_err[k], kmers, logger, out_path, pd_reads.fname, k)

    return f_value_mean, f_value_err



def write_to_output_file(f_value_means, f_value_errs, kmers, logger, out_path, fname, k):
    logger.info('storing kmer frequencies and f-values for "{fname}" in "{out_path}"'.format(fname=fname, out_path=out_path) )

    results = zip(
        kmers,
        f_value_means,
        f_value_errs
        )
    
    with file(os.path.join(out_path, 'F_values.{0}mer.txt'.format(k)), 'w') as of:
        of.write('# kmer\tf_value_mean\tf_value_err\n')
        for mer,f_mean,f_err in results:
            out = [mer, f_mean, f_err]
            of.write('%s\t%g\t%g\n' % tuple(out))
    of.close()



def main():
    from optparse import OptionParser
    usage = "usage: %prog [options] <pulldown_reads_file> <input_reads_file>"
    parser = OptionParser(usage=usage)
    parser.add_option("-k","--min-k",dest="min_k",default=3,type=int,help="min kmer size (default=3)")
    parser.add_option("-K","--max-k",dest="max_k",default=8,type=int,help="max kmer size (default=8)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimateion (default=5)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("-o","--output",dest="output",default=".",help="path where results are to be stored")
    parser.add_option("","--debug",dest="debug",default=False, action="store_true",help="SWITCH: activate debug output")
    parser.add_option("","--n-max",dest="n_max",default=0, type=int,help="TESTING: read at most N reads")
    options,args = parser.parse_args()

    if not args:
        parser.error("missing argument: need <reads_file>")
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

    logger = logging.getLogger("f_values")
    logger.info("called as '{0}'".format(" ".join(sys.argv)) )

    # load pull-down and input reads
    pd_reads = RBNSReads(args[0], n_max=options.n_max, pseudo_count=options.pseudo)
    in_reads = RBNSReads(args[1], n_max=options.n_max, pseudo_count=options.pseudo)
    f_value_means, f_value_errors = run_F_values(pd_reads, 
                                                 in_reads, 
                                                 options.min_k,
                                                 options.max_k,
                                                 out_path = options.output,
                                                 subsamples = options.subsamples)

    


if __name__ == '__main__':
    main()
