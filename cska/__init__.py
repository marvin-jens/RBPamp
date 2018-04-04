__license__ = "MIT"
__version__ = "0.9.8"
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
import traceback
#import cska.ska_kmers
import matplotlib


def ensure_path(full):
    path = os.path.dirname(full)
    if not os.path.exists(path):
        os.makedirs(path)

    return full

def auto_detect(path='.', exts=["reads","txt"]):
    """
    auto-detect RBP name, reads files and concentrations from files in directory
    """
    from glob import glob
    from collections import defaultdict
    files = []
    for ext in exts:
        pattern = os.path.join(path,'*.{0}'.format(ext))
        hits = list(glob(pattern))
        files.extend(hits)
    
    rbp_names = defaultdict(int)
    rbp_conc = []
    
    for f in files:
        name, conc = os.path.basename(f).split("_")
        conc = conc.rsplit('.',1)[0]
        conc = float(conc.replace('input','0'))
        rbp_names[name] += 1
        rbp_conc.append(conc)
    
    assert len(rbp_names) == 1
    rbp_conc = np.array(rbp_conc)
    files = np.array(files)
    I = rbp_conc.argsort()
    
    return rbp_names.keys()[0], files[I], rbp_conc[I]


def main():
    from optparse import OptionParser
    usage = "usage: %prog [options] <input_reads_file> <pulldown_reads_file1> [<pulldown_reads_file2] [...]"

    parser = OptionParser(usage=usage)
    parser.add_option("","--name",dest="name",default="RBP",help="name of the protein assayed (default=RBP)")
    parser.add_option("-o","--output",dest="output",default="cska",help="path where results are to be stored (default='cska')")
    parser.add_option("","--run-path",dest="run",default="run_{datestr}",help="pattern for run-folder name (default='run_{datestr}')")
    parser.add_option("-a","--auto",dest="auto",default=False, action="store_true",help="SWITCH: attempt to automatically guess RPB name, reads files and concentrations from file names (default=specify manually)")
    parser.add_option("","--overwrite",dest="overwrite",default=False, action="store_true",help="SWITCH: overwrite existing files (default=exit with an error)")
    parser.add_option("","--reports",dest="reports",default=False, action="store_true",help="SWITCH: generate PDF reports (default=off)")
    parser.add_option("","--metrics",dest="results",default="R_value,F_ratio",help="list of RBNS metrics to compute and store (options='*R_value,SKA_weight,F_ratio' *=default)")
    parser.add_option("-m","--model",dest="model",default=False, action="store_true",help="SWITCH: thermodynamic model parameter fit")
    parser.add_option("-s","--seed-analysis",dest="seed_analysis",default=4, type=int, help="activate initial dependent kmer analysis to seed the motifs (default=4,0=off)")
    parser.add_option("","--model-resume",dest="mdl_resume",default=None,help="start with affinity parameters from this file for further optimization")
    parser.add_option("","--model-pwm-init",dest="mdl_pwm_init",default=None,help="start with affinity parameters from this PWM file for further optimization")
    parser.add_option("","--model-global",dest="kmer_opt_global",default=False, action="store_true",help="SWITCH: do global instead of local error optimization when fitting a kmer affinity")
    parser.add_option("","--model-epsilon",dest="mdl_epsilon",default=1e-3, type=float, help="convergence threshold for relative error reduction (default=1e-3)")
    parser.add_option("","--model-sensors",dest="mdl_report_sensors",default="correlation,betas,errors,R_values", help="list of sensors to keep track of optimization progress. default='correlation,betas,errors,R_values'")
    parser.add_option("","--model-report-interval",dest="mdl_report_interval",default=50, type=int, help="generate diagnostic/report PDFs every x iterations of the model fit (default=50)")
    parser.add_option("","--model-report-ignore-trigger",dest="mdl_report_trigger",default="", help="comma separated list of events that should *not* trigger new plots")
    parser.add_option("","--reference",dest="ref_file",default="", help="tab-separated file with measured (reference) Kd values (default=use builtin known_kds.csv)")


    parser.add_option("","--interactions",dest="interactions",default=False, action="store_true",help="SWITCH: activate combinatorial search") # TODO: merge into --compute-results
    parser.add_option("","--debug",dest="debug",default="",help="activate debug output for comma-separated subsystems [root, fold, cache, rbns, opt, model, report]")
    parser.add_option("","--info",dest="info",default="",help="activate info level output for comma-separated subsystems [root, fold, cache, rbns, opt, model, report]")

    parser.add_option("","--version",dest="version",default=False, action="store_true",help="show version information and quit")
    parser.add_option("","--skip-adapters",dest="skip_adap",default=False, action="store_true",help="ignore adapter sequences (default=False)")

    parser.add_option("-k","--min-k",dest="min_k",default=3,type=int,help="min kmer size (default=3)")
    parser.add_option("-K","--max-k",dest="max_k",default=8,type=int,help="max kmer size (default=8)")
    
    parser.add_option("-r","--rna-concentration",dest="rna_conc",default=1000.,type=float,help="concentration of random RNA used in the experiment in nano molars (default=1000 nM)")
    parser.add_option("-p","--rbp-concentration",dest="rbp_conc",default="0,320",help="(comma separated list of) protein concentration used in the experiment(s) in nano molars (default=0,300)")
    parser.add_option("-T","--temperature",dest="temp",default=22.,type=float,help="temperature of the experiment in degrees Celsius (default=22.0)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimation (default=10)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("","--ska-max-passes",dest="n_passes",default=10,type=int,help="max number of passes (default=10)")
    parser.add_option("","--ska-convergence",dest="convergence",default=0.5,type=float,help="convergence is reached when max. change in absolute weight is below this value (default=0.5)")

    parser.add_option("","--disable-caching",dest="disable_caching",default=False, action="store_true",help="DEBUG: disable transparent caching (SLOW!)")
    parser.add_option("","--disable-unpickle",dest="disable_unpickle",default=False, action="store_true",help="DEBUG: disable unpickling. Will recompute and overwrite existing pickled data")
    parser.add_option("","--disable-pickle",dest="disable_pickle",default=False, action="store_true",help="DEBUG: disable pickling. Will not create or overwrite any pickled data")
    
    parser.add_option("-n","--n-max",dest="n_max",default=0, type=int,help="TESTING: read at most N reads")
    parser.add_option("","--format",dest="format",default='raw', help="read file format [raw,fasta,fastq] (default=raw)")
    parser.add_option("","--adap5",dest="adap5",default="gggaguucuacaguccgacgauc", help="5'RNA adapter sequence to add to read sequence")
    parser.add_option("","--adap3",dest="adap3",default="uggaauucucgggugucaagg", help="3'RNA adapter sequence to add to read sequence")
    parser.add_option("","--fold",dest="folding",default=False, action="store_true",help="SWITCH: instead of a normal run, fold all reads and record accessibilities/open-energies")
    parser.add_option("","--openen-discretize",dest="openen_discretize",default="0", choices=["0","8","16"], help="discretize open-energies using <n> bits [8,16] set to 0 to disable (default)")
    parser.add_option("","--parallel",dest="parallel",default=8,type=int,help="number of parallel threads (currently only used for folding. default=8)")
    
    parser.add_option("","--compare",dest="compare",default="", help="compare to literature values for this protein")
    parser.add_option("","--track-kmers",dest="track_kmers",default="", help="comma separated list of kmers to track during optimization.")

    # affinity fit parameters
    #parser.add_option("","--simulate",dest="simulate",choices=["","reads","comparison"],default="",help="simulate RBNS instead of analysis, choices are ['reads','comparison']")    
    #parser.add_option("","--sim-best-Kd",dest="sim_best_Kd",default=10.,type=float,help="best binding dissociation constant for simulation in nM (default=10 nM)")
    #parser.add_option("","--sim-var",dest="sim_var",default=10.,type=float,help="variance for simulated binding energy log-normal distribution (default=)")
    #parser.add_option("","--sim-mean",dest="sim_mean",default=10.,type=float,help="mean for simulated binding energy log-normal distribution (default=)")
    #parser.add_option("","--sim-N-reads",dest="sim_N_reads",default=1000000,type=int,help="number of reads to simulate (default=1,000,000)")
    
    parser.add_option("","--simulate",dest="simulate",choices=["","reads","comparison"],default="",help="simulate RBNS instead of analysis, choices are ['reads','comparison']")    
    parser.add_option("","--seed",dest="seed",default=47110815,type=int,help="seed for fast pseudo-random number generator (for RBNS simulation)")
    parser.add_option("","--sim-best-Kd",dest="sim_best_Kd",default=10.,type=float,help="best binding dissociation constant for simulation in nM (default=10 nM)")
    parser.add_option("","--sim-var",dest="sim_var",default=10.,type=float,help="variance for simulated binding energy log-normal distribution (default=)")
    parser.add_option("","--sim-mean",dest="sim_mean",default=10.,type=float,help="mean for simulated binding energy log-normal distribution (default=)")
    parser.add_option("","--sim-N-reads",dest="sim_N_reads",default=1000000,type=int,help="number of reads to simulate (default=1,000,000)")
    
    options,args = parser.parse_args()

    from cska.caching import cached, pickled, CachedBase
    from cska.reads import RBNSReads
    from cska.analysis import RBNSAnalysis
    from cska.ska_runner import SKARunner

    if options.version:
        print __version__
        print __license__
        print "by", ", ".join(__authors__)
        sys.exit(0)

    if options.auto:
        rbp_name, reads_files, rbp_concentrations = auto_detect('.')
    else:
        rbp_name = options.name
        reads_files = args
        rbp_concentrations = [float(c) for c in options.rbp_conc.split(',')]
    
    if not len(reads_files):
        parser.error("missing arguments: need <input_reads_file> <pulldown_reads1_file> ... (or use --auto)")
        sys.exit(1)
    
    # control caching framework behaviour
    CachedBase._do_not_cache = options.disable_caching
    CachedBase._do_not_pickle = options.disable_pickle
    CachedBase._do_not_unpickle= options.disable_unpickle
        
    # prepare outout path
    import datetime
    run_folder = (options.run+"/").format(datestr=datetime.datetime.now().strftime("%b-%d-%Y_%H:%M:%S"))

    run_path = ensure_path(os.path.join(options.output, run_folder))

    # keep a symlink named "recent" always pointing to last run folder
    recent_path = os.path.join(options.output, "recent")
    try:
        os.remove(recent_path)
    except OSError:
        pass

    os.symlink(run_folder, recent_path)

    # set up logging
    log_path = os.path.join(run_path,"run.log")

    FORMAT = '%(asctime)-20s\t%(levelname)s\t%(name)s\t%(message)s'
    formatter = logging.Formatter(FORMAT)
    logging.basicConfig(level=logging.WARNING, format=FORMAT)    
    root = logging.getLogger('')
    fh = logging.FileHandler(filename=log_path, mode='a')
    fh.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(fh)
    
    logger = logging.getLogger("CSKA")
    logger.setLevel(logging.INFO)
    logger.info("version {0}".format(__version__))
    logger.info("invoked as '{0}'".format(" ".join(sys.argv)) )

    # set info level for specific sub-systems
    for sub in options.info.split(','):
        if not sub:
            continue
        sub = sub.replace('root',"")
        logging.getLogger(sub).setLevel(logging.INFO)

    # set debug log level for specific sub-systems
    for sub in options.debug.split(','):
        if not sub:
            continue
        sub = sub.replace('root',"")
        logging.getLogger(sub).setLevel(logging.DEBUG)
        if sub == 'cache':
            CachedBase.debug_caching = True

    try:
        # parametrize SKA algorithm
        ska = SKARunner(
            max_iterations = options.n_passes,
            convergence = options.convergence, 
        )
        
        # where to put/find transparent pickle/unpickle objects
        CachedBase.pkl_path = os.path.join(options.output, ".pkl")

        # start a new analysis
        rbns = RBNSAnalysis(
            rbp_name = rbp_name,
            out_path = run_path,
            ska_runner = ska,
        )

        # TODO: properly integrate simulation
        if options.simulate == "reads":
            from cska.optimize import RBNSGenerator
            for k in range(options.min_k, options.max_k + 1):
                gen = RBNSGenerator(k,l=40, seed=options.seed)
                gen.assign_experimental_input(reads_files[0])
                gen.energy_plot()

                r_matrix = []
                for P in np.array(rbp_concentrations[1:]):
                    r = gen.predict_r_values(P=P, store="r_{0}.tsv".format(P))
                    r_matrix.append(r)
                    
                    read_path = os.path.join(self.out_path,"sim_bound_{0}.reads".format(P) )
                    gen.generate_bound_reads(read_path, P=P, p_ns=0.00, N=20000000)
                    occ = gen.predict_occupancies(P=P, store="occ_{0}.tsv".format(P))
                    

        # open energy prediction from folding
        fold_path = os.path.join(options.output, "acc")
        storage_kw = dict(overwrite = options.overwrite, T=options.temp, disc_mode='linear')
        if int(options.openen_discretize):
            dtype = getattr(np, "uint{0}".format(options.openen_discretize))
            storage_kw.update(dict(discretize=True, disc_dtype=dtype))
        else:
            storage_kw.update(dict(discretize=False, raw_dtype=np.float32))
        logger.info("populating RBNS analysis with reads")
        # populate with experimental data
        for fname, rbp_conc in zip(reads_files, rbp_concentrations):
            reads = RBNSReads(
                fname, 
                format = options.format,
                rbp_conc=rbp_conc,
                rbp_name = rbp_name,
                n_max=options.n_max, 
                pseudo_count=options.pseudo, 
                rna_conc = options.rna_conc,
                temp = options.temp,
                n_subsamples = options.subsamples,
                adap3=options.adap3,
                acc_storage_path = fold_path,
                storage_kw=storage_kw
            )
            
            rbns.add_reads(reads)
        # first, compute RBNS metrics
        metrics = [m.strip() for m in options.results.strip().split(',') if m.strip()]
        if metrics:
            logger.info("computing RBNS metrics '{0}'".format(metrics))
            for k in range(options.min_k, options.max_k + 1):
                rbns.compute_results(k, options, results=metrics, report=options.reports)
                rbns.flush()

        ### special run modes: 
        # secondary structure prediction and accessibility recording
        if options.folding:
            logger.info("folding reads with '{0}' threads".format(options.parallel))
            from cska.fold import parallel_fold
            # prepare outout path
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)

            # fold the reads
            for reads in rbns.reads:
                logger.info("folding {reads.name} ({reads.fname})".format(reads=reads) )
                parallel_fold(
                    reads.iter_reads(), 
                    reads.acc_storage,
                    temp = reads.temp,
                    adap5 = options.adap5,
                    adap3 = options.adap3,
                    k_min = options.min_k,
                    k_max = options.max_k,
                    n_max = options.n_max,
                    l_insert = rbns.reads[0].L,
                    skip_adap = options.skip_adap,
                    n_parallel= options.parallel,
                )


        # prime the optimization from dependent-kmer analysis
        from cska.pwm import PWMOptimizer, PSAM
        if options.seed_analysis:
            logger.info("performing seed analysis")
            from cska.seed import SeedRefinement
            SR = SeedRefinement(rbns, km=options.seed_analysis)
            k = SR.linear_k

        elif options.mdl_pwm_init:
            logger.info("resuming from PWM: '{0}'".format(options.mdl_pwm_init))
            pwm = PSAM.load(options.mdl_pwm_init)
            k = pwm.n
            print pwm
        else:
            k = options.min_k

        # fit of thermodynamic model parameters (affinities)
        if options.model:
            from cska.optimize import ModelOptimization
            opt = ModelOptimization(
                k, 
                rbns,
                n_subsample=0, 
                sub_replace=False, 
                param_file=options.mdl_resume,
                kmer_opt_global=options.kmer_opt_global,
            )
            pwm_opt = PWMOptimizer(k, options.max_k, opt, eps=options.mdl_epsilon)
            
            from cska.comparison import RefComparison
            if options.compare:
                compare = options.compare
            else:
                compare = rbp_name
            ref = RefComparison(compare, ref_file=options.ref_file)

            from cska.report import OptReporting
            opt.reporter = OptReporting(
                opt, 
                os.path.join(run_path, 'plots'), 
                track=options.mdl_report_sensors.split(','), 
                triggers=options.mdl_report_trigger.split(','),
                ref = ref,
            )

            seed_params = None
            if options.seed_analysis:
                seed_params = SR.linear_seed_params(A0=.1, aff0=1e-5)
                pwm_opt.pwm0 = SR.psam_lin

            if options.mdl_pwm_init:
                seed_params = pwm.kmer_affinity_table(aff0=1e-5)
                pwm_opt.pwm0 = pwm

                # from copy import copy
                # from cska.report import p_bound_plot
                # p_bound_plot(opt.current)
                # import matplotlib.pyplot as pp
                # pp.figure()
                # # Zs = []N 
                # global Zs
                # for reads in rbns.reads:
                #     im = reads.get_index_matrix(opt.k, _do_not_cache=True)
                #     acc = reads.acc_storage.get_raw(opt.k, _do_not_cache=True).acc
                #     Z1 = opt.mdl._spa_partition_function(im, acc, opt.mdl.parameters.affinities)
                #     Zs.append(Z1)
                #     # state = copy(opt.current)
                #     # state.Z1 = Z1

                #     lZ = np.log(Z1)
                #     counts, bins = np.histogram(lZ, bins=1000)
                #     bins = np.exp(bins)
                #     # midpoint integration
                #     aff = (bins[1:] + bins[:-1])/2.
                #     if reads.rbp_conc == 0:
                #         pp.loglog(aff, counts, label=reads.name, color='black')
                #     else:
                #         pp.loglog(aff, counts, label=reads.name)

                # pp.legend(loc = 'upper left')
                # pp.xlabel("total read affinity [1/nM]")
                # pp.ylabel("count")
                # pp.savefig("aff_dist_observed.pdf")
                # pp.close()

                #print "step scale"
                #opt.step_scale(min_scale=.01, max_scale=100.)
                

            try:
                pwm_opt.optimize(seed_params=seed_params)
            except KeyboardInterrupt:
                opt.logger.warning("Keyboard interrupt while in:")
                exc = traceback.format_exc()
                logger.error(exc)
                
            opt.reporter.close()
            pwm_opt.store_params()

            opt.logger.info("converged/interrupted after {0} steps.".format(opt.t))
        

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
    except:        
        logger.error("Caught exception. Gathering traceback")
        exc = traceback.format_exc()
        logger.error(exc)
        sys.stderr.write(exc)
        
        # in case we have child processes, try to end them gracefully
        import cska.fold
        fold.interrupt()
        
        sys.exit(1)
    else:
        logger.info("run completed.")
        sys.exit(0)

Zs = []

if __name__ == '__main__':
    main()
