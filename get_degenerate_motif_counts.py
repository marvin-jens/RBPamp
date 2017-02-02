#!/usr/bin/env python

import sys
import string
import itertools
import numpy as np
import copy
import time
import os
import logging
from ska import RBNSReads
import pdb


IUPAC_DICT={'N':['A','C','G','T'], 'R':['A','G'], 'Y':['C','T'], 'S':['G','C'], \
                'W':['A','T'], 'K':['G','T'], 'M':['A','C'], 'H': ['A','C','T'], \
                'V':['A','C','G'], 'D':['A','G','T'], 'B':['C','G','T']}
NTs=['A','C','G','T']
NTS_TO_BITS={'A':0,'C':1,'G':2,'T':3}


def calculate_pattern_F_values(pd_reads, in_reads, kmer, min_k, \
                                   max_k, padding_code, out_path=".", subsamples=10):
    logger = logging.getLogger('pattern_f_values')
    out_path = os.path.abspath(out_path)

    f_value_mean, f_value_err = {}, {}
    
    for k in range(min_k, max_k+1):
        logger.info("calculating f-values for patterns of {} with {} {}'s...".format(kmer, k, padding_code))
        patterns, patterns_to_motifs, masks = get_degenerate_motifs(kmer, k, padding_code="N")
        masks = np.asarray(masks)
        frac_input = in_reads.fraction_reads_with_pattern(k+len(kmer), masks)
        samples=[pd_reads.subsample(i, subsamples).fraction_reads_with_pattern(k+len(kmer),masks)\
                     for i in range(subsamples)]
        f_value_matrix = np.array([s/frac_input for s in samples])
        f_value_err[k] = f_value_matrix.std(axis=0)
        f_value_mean[k] = f_value_matrix.mean(axis=0)
        write_to_output_file(f_value_mean[k], f_value_err[k], patterns, logger, out_path, pd_reads.fname, kmer, padding_code, k)
    return f_value_mean, f_value_err


def write_to_output_file(f_value_means, f_value_errs, patterns, logger, out_path, fname, \
                             kmer, padding_code, padding_length):
    logger.info('storing kmer frequencies and f-values for "{fname}" in "{out_path}"'.format(fname=fname, out_path=out_path) )

    results = zip(
        patterns,
        f_value_means,
        f_value_errs
        )

    with file(os.path.join(out_path, 'F_values.{0}.{1}.{2}.txt'.format(kmer, padding_code, padding_length)), 'w') as of:
        of.write('# pattern\tf_value_mean\tf_value_err\n')
        for mer,f_mean,f_err in results:
            out = [mer, f_mean, f_err]
            of.write('%s\t%g\t%g\n' % tuple(out))
    of.close()



def get_degenerate_motifs(kmer, padding_length, padding_code="N"):
    # would be 10 choose 5, but it's 8 choose 5 because of edge requirements #
    total_length = len(kmer) + padding_length
    longer_motif = kmer + padding_code*padding_length
    patterns = set([''.join(p) for p in itertools.permutations(longer_motif, total_length)])
    # filter out patterns that did not preserve the original kmer's order #
    patterns = list(set([p for p in patterns if kmer in \
                             p.translate(string.maketrans(padding_code, \
                             "-"*len(padding_code))).replace('-','')] ) )
    patterns = list(set([m for m in patterns if m[0]!=padding_code and \
                             m[-1]!=padding_code]))
    patterns.append( padding_code*padding_length + kmer )
    patterns.append( kmer + padding_code*padding_length )
    # get all kmers for all patterns #
    pattern_to_kmers, pattern_to_masking_kmers = [], []
    for pattern in patterns:
        list_of_chars = []
        for i in range(len(pattern)):
            if pattern[i] in NTs: list_of_chars.append(pattern[i])
            else: list_of_chars.append(IUPAC_DICT[pattern[i]])
        permutations = list(set([''.join(k) for k in itertools.product( *list_of_chars)] ))
        pattern_to_kmers.append( permutations )
        mask_1 = np.asarray([NTS_TO_BITS[char] for char in pattern.replace(padding_code, 'A')], dtype=np.uint8)
        mask_2= np.asarray([NTS_TO_BITS['A'] if char==padding_code else NTS_TO_BITS['T'] for char in pattern], dtype=np.uint8)
        pattern_to_masking_kmers.append((mask_1, mask_2))
    return patterns, pattern_to_kmers, pattern_to_masking_kmers





def main():
    from optparse import OptionParser
    usage = "usage: %prog [options] <pulldown_reads_file> <input_reads_file> <kmer_to_split>"
    parser = OptionParser(usage=usage)
    parser.add_option("-k","--min-padding",dest="min_padding",default=1,type=int,help="min spacer length (default=1)")
    parser.add_option("-K","--max-padding",dest="max_padding",default=5,type=int,help="max spacer length (default=5)")
    parser.add_option("-P","--padding_code",dest="padding_code",default="N",type=str,help="IUPAC code with which to space/pad given kmer (default=N)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimateion (default=5)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("-o","--output",dest="output",default=".",help="path where results are to be stored")
    parser.add_option("","--debug",dest="debug",default=False, action="store_true",help="SWITCH: activate debug output")
    parser.add_option("","--n-max",dest="n_max",default=0, type=int,help="TESTING: read at most N reads")
    options,args = parser.parse_args()

    if not args:
        parser.error("missing argument: need <pulldown_reads_file> <input_reads_file> <kmer>")
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

    logger = logging.getLogger("degenerate pattern search")
    logger.info("called as '{0}'".format(" ".join(sys.argv)) )

    # load pull-down and input reads
    pd_reads = RBNSReads(args[0], n_max=options.n_max, pseudo_count=options.pseudo)
    in_reads = RBNSReads(args[1], n_max=options.n_max, pseudo_count=options.pseudo)
    kmer = args[2].upper().replace('U','T')

    calculate_pattern_F_values(pd_reads, 
                               in_reads,
                               kmer,
                               options.min_padding,
                               options.max_padding,
                               options.padding_code,
                               out_path = options.output,
                               subsamples = options.subsamples)




if __name__ == '__main__':
    main()
