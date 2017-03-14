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

from cska.caching import cached, pickled, CachedBase
from cska.rbns_reads import RBNSReads
from cska.rbns_analysis import RBNSAnalysis
from cska.ska_runner import SKARunner


def main():
    from optparse import OptionParser
    usage = "usage: %prog [options] <input_reads_file> <pulldown_reads_file1> [<pulldown_reads_file2] [...]"

    parser = OptionParser(usage=usage)
    parser.add_option("-k","--min-k",dest="min_k",default=3,type=int,help="min kmer size (default=3)")
    parser.add_option("-K","--max-k",dest="max_k",default=8,type=int,help="max kmer size (default=8)")
    
    parser.add_option("","--ska-max-passes",dest="n_passes",default=10,type=int,help="max number of passes (default=10)")
    parser.add_option("-r","--rna-concentration",dest="rna_conc",default=1000.,type=float,help="concentration of random RNA used in the experiment in micro molars (default=100uM)")
    parser.add_option("-p","--rbp-concentration",dest="prot_conc",default="0,320",help="(comma separated list of) protein concentration used in the experiment(s) in nano molars (default=0,300)")
    parser.add_option("","--name",dest="name",default="RBP",help="name of the protein assayed (default=RBP)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimation (default=10)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("","--ska-convergence",dest="convergence",default=0.5,type=float,help="convergence is reached when max. change in absolute weight is below this value (default=0.5)")
    parser.add_option("-o","--output",dest="output",default=".",help="path where results are to be stored")
    parser.add_option("","--debug",dest="debug",default=False, action="store_true",help="SWITCH: activate debug output")
    parser.add_option("","--interactions",dest="interactions",default=False, action="store_true",help="SWITCH: activate combinatorial search")
    parser.add_option("-w","--write-fasta",dest="write_fasta",default=False, action="store_true",help="SWITCH: write FASTA file for each library with isolated, top-scoring kmers")
    parser.add_option("-n","--n-max",dest="n_max",default=0, type=int,help="TESTING: read at most N reads")
    parser.add_option("","--disable-caching",dest="disable_caching",default=False, action="store_true",help="DEBUG: disable transparent caching (SLOW!)")
    parser.add_option("","--disable-unpickle",dest="disable_unpickle",default=False, action="store_true",help="DEBUG: disable unpickling. Will recompute and overwrite existing pickled data")
    parser.add_option("","--disable-pickle",dest="disable_pickle",default=False, action="store_true",help="DEBUG: disable pickling. Will not create or overwrite any pickled data")
    parser.add_option("","--debug-caching",dest="debug_caching",default=False, action="store_true",help="DEBUG: enable detailed debug output from the caching framework")
    parser.add_option("","--version",dest="version",default=False, action="store_true",help="show version information and quit")
    options,args = parser.parse_args()

    if options.version:
        print __version__
        print __license__
        print "by", ", ".join(__authors__)
        sys.exit(0)

    if not args:
        parser.error("missing argument: need <reads_file> (or use /dev/stdin)")
        sys.exit(1)

    # control caching framework behaviour
    CachedBase.debug_caching = options.debug_caching
    CachedBase._do_not_cache = options.disable_caching
    CachedBase._do_not_pickle = options.disable_pickle
    CachedBase._do_not_unpickle= options.disable_unpickle
        
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
    
    # where to put/find transparent pickle/unpickle objects
    CachedBase.pkl_path = os.path.join(options.output, ".pkl")

    # start a new analysis
    rbns = RBNSAnalysis(
        rbp_name = options.name,
        out_path = options.output,
        ska_runner = ska,
        write_fasta = options.write_fasta,
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
    
    for k in range(options.min_k, options.max_k + 1):
        rbns.store_all_results(k)
        rbns.flush()
    
    ##compute f-values, make overview plots
    ###rbns.compare_k()
    ##rbns.run_ROC()
    
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
