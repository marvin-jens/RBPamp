__license__ = "MIT"
__version__ = "0.9.9"
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

def parse_cmdline():
    from optparse import OptionParser
    usage = "usage: %prog [options] <input_reads_file> <pulldown_reads_file1> [<pulldown_reads_file2] [...]"

    parser = OptionParser(usage=usage)
    # basic options
    parser.add_option("","--version",dest="version",default=False, action="store_true",help="show version information and quit")
    parser.add_option("","--name",dest="name",default="RBP",help="name of the protein assayed (default=RBP)")
    parser.add_option("-o","--output",dest="output",default="cska",help="path where results are to be stored (default='cska')")
    parser.add_option("","--run-path",dest="run",default="run_{datestr}",help="pattern for run-folder name (default='run_{datestr}')")
    parser.add_option("-a","--auto",dest="auto",default=False, action="store_true",help="SWITCH: attempt to automatically guess RPB name, reads files and concentrations from file names (default=specify manually)")
    parser.add_option("-b","--best",dest="best",default=0, type=int,help="keep only the best n samples (by top R-value) default=0 [off]")
    parser.add_option("","--multi-stage",dest="multi_stage",default=False, action="store_true",help="SWITCH: perform multiple stages of optimization")
    
    parser.add_option("-r","--rna-concentration",dest="rna_conc",default=1000.,type=float,help="concentration of random RNA used in the experiment in nano molars (default=1000 nM)")
    parser.add_option("-p","--rbp-concentration",dest="rbp_conc",default="0,320",help="(comma separated list of) protein concentration used in the experiment(s) in nano molars (default=0,300)")
    parser.add_option("-T","--temperature",dest="temp",default=4.,type=float,help="temperature of the experiment in degrees Celsius (default=4.0)")
    parser.add_option("","--format",dest="format",default='raw', help="read file format [raw,fasta,fastq] (default=raw)")
    parser.add_option("","--adap5",dest="adap5",default="gggaguucuacaguccgacgauc", help="5'RNA adapter sequence to add to read sequence")
    parser.add_option("","--adap3",dest="adap3",default="uggaauucucgggugucaagg", help="3'RNA adapter sequence to add to read sequence")
    parser.add_option("","--skip-adapters",dest="skip_adap",default=False, action="store_true",help="ignore adapter sequences (default=False)")
    parser.add_option("-n","--n-max",dest="n_max",default=0, type=int,help="TESTING: read at most N reads")
    parser.add_option("","--overwrite",dest="overwrite",default=False, action="store_true",help="SWITCH: overwrite existing files (default=exit with an error)")

    # RNA folding
    parser.add_option("","--fold", dest="folding",default="", help="instead of a normal run, fold all reads and record accessibilities/open-energies for k in the given range. example --fold=1-12 (default=off)")
    parser.add_option("","--skip-folded", dest="skip_folded", default=False, action="store_true",help="SWITCH: if files are already in place, do not re-fold")
    parser.add_option("","--acc-scan", dest="acc_scan", default=False, action="store_true",help="SWITCH: scan for high accessibility selection in bound libraries")
    # parser.add_option("","--acc-scale",dest="acc_scale",default=1.,type=float,help="[EXPERIMENTAL] scale unfolding energies")
    parser.add_option("","--openen-discretize", dest="openen_discretize", default="0", choices=["0","8","16"], help="discretize open-energies using <n> bits [8,16] set to 0 to disable (default)")
    parser.add_option("","--parallel", dest="parallel", default=8,type=int,help="number of parallel threads (currently only used for folding. default=8)")
    
    # RBNS metrics
    parser.add_option("","--metrics",dest="results",default="R_value,F_ratio",help="list of RBNS metrics to compute and store (options='*R_value,SKA_weight,F_ratio' *=default)")
    parser.add_option("","--metrics-k", dest="metrics_k", default="3-8", help="range of kmer sizes for which to compute the desired metrics (default: --metrics-k=3-8)")
    parser.add_option("","--pseudo",dest="pseudo",default=10.,type=float,help="pseudo count to add to kmer counts in order to avoid div by zero for large k (default=10)")
    parser.add_option("","--ska-max-passes",dest="n_passes",default=10,type=int,help="max number of passes (default=10)")
    parser.add_option("","--ska-convergence",dest="convergence",default=0.5,type=float,help="convergence is reached when max. change in absolute weight is below this value (default=0.5)")
    parser.add_option("","--subsamples",dest="subsamples",default=10,type=int,help="number of subsamples for error estimation (default=10)")

    # seed motif analysis
    parser.add_option("-s","--seed-analysis",dest="seed_analysis",default=4, type=int, help="activate initial dependent kmer analysis to seed the motifs (default=4,0=off)")

    # accessibility footprint analysis
    parser.add_option("","--footprint-k", dest="footprint", default="5-11", help="size range [nt] to search for ideal accessibility footprint (default: --footprint-k=5-11)")
    
    # affinity model optimization 
    # parser.add_option("","--seed-motif",dest="seed_motif",default="", help="DEBUGGING: override motif from seed analysis with this exact sequence.")
    parser.add_option("-w","--max-width",dest="max_width",default=11, type=int, help="maximum number of nucleotides in PSAM motif (number of columns) default=11)")
    parser.add_option("","--grad-k",dest="grad_k",default=6, type=int, help="k for gradient descent kmer R-value mean squared error objective function (default=6)")
    parser.add_option("","--grad-mdl",dest="grad_mdl",default="", choices=['partfunc', 'meanfield', 'invmeanfield', ''], help="method for gradient descent refinement of PSAM [partfunc, meanfield, invmeanfield, ''=off] default=partfunc")
    parser.add_option("","--grad-maxiter",dest="grad_maxiter",default=500, type=int, help="maximal number of gradient descent iterations (default=500)")

    parser.add_option("","--Z-threshold",dest="Z_thresh",default=0, type=float, help="drop reads that have Boltzmann weight of a factor of Z_thresh below the max weight (default=0/off)")
    parser.add_option("-m","--model",dest="model",default=False, action="store_true",help="SWITCH: thermodynamic model parameter fit")
    parser.add_option("","--no-structure",dest="no_structure",default=False, action="store_true",help="ignore secondary structure folding information (default=False)")
    parser.add_option("","--resume",dest="mdl_psam_init",default=None,help="start with affinity parameters from this PSAM file for further optimization")
    parser.add_option("","--eps",dest="mdl_epsilon",default=1e-3, type=float, help="convergence threshold for relative error reduction (default=1e-3)")

    # TODO: update
    parser.add_option("","--sensors",dest="mdl_report_sensors",default="correlation,betas,errors,R_values", help="list of sensors to keep track of optimization progress. default='correlation,betas,errors,R_values'")
    parser.add_option("","--report-interval",dest="mdl_report_interval",default=50, type=int, help="generate diagnostic/report PDFs every x iterations of the model fit (default=50)")
    parser.add_option("","--report-skip",dest="mdl_report_trigger",default="", help="comma separated list of events that should *not* trigger new plots")

    parser.add_option("","--reference",dest="ref_file",default="", help="tab-separated file with measured (reference) Kd values (default=use builtin known_kds.csv)")
    parser.add_option("","--compare",dest="compare",default="", help="compare to literature values for this protein")

    # infrastructure and logging/debugging control
    parser.add_option("","--disable-caching",dest="disable_caching",default=False, action="store_true",help="DEBUG: disable transparent caching (SLOW!)")
    parser.add_option("","--disable-unpickle",dest="disable_unpickle",default=False, action="store_true",help="DEBUG: disable unpickling. Will recompute and overwrite existing pickled data")
    parser.add_option("","--disable-pickle",dest="disable_pickle",default=False, action="store_true",help="DEBUG: disable pickling. Will not create or overwrite any pickled data")
    
    parser.add_option("","--debug",dest="debug",default="",help="activate debug output for comma-separated subsystems [root, fold, cache, rbns, opt, model, report]")
    parser.add_option("","--info",dest="info",default="",help="activate info level output for comma-separated subsystems [root, fold, cache, rbns, opt, model, report]")

    # parser.add_option("","--track-kmers",dest="track_kmers",default="", help="comma separated list of kmers to track during optimization.")

    # read simulation (currently broken)
    parser.add_option("","--simulate",dest="simulate",choices=["","reads","comparison"],default="",help="simulate RBNS instead of analysis, choices are ['reads','comparison']")    
    parser.add_option("","--rnd-seed",dest="seed",default=47110815,type=int,help="seed for fast pseudo-random number generator (for RBNS simulation)")
    parser.add_option("","--sim-best-Kd",dest="sim_best_Kd",default=10.,type=float,help="best binding dissociation constant for simulation in nM (default=10 nM)")
    parser.add_option("","--sim-var",dest="sim_var",default=10.,type=float,help="variance for simulated binding energy log-normal distribution (default=)")
    parser.add_option("","--sim-mean",dest="sim_mean",default=10.,type=float,help="mean for simulated binding energy log-normal distribution (default=)")
    parser.add_option("","--sim-N-reads",dest="sim_N_reads",default=1000000,type=int,help="number of reads to simulate (default=1,000,000)")
    
    options, args = parser.parse_args()
    
    if options.version:
        print __version__
        print __license__
        print "by", ", ".join(__authors__)
        sys.exit(0)

    return options, args


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
    
    rbp_names = defaultdict(list)
    rbp_conc = []
    
    for f in files:
        try:
            name, conc = os.path.basename(f).split("_")
            conc = conc.rsplit('.',1)[0]
            conc = float(conc.replace('input','0'))
        except ValueError:
            continue

        rbp_names[name].append( (conc, f) )
    
    hits = sorted([(len(rbp_names[name]), name) for name in rbp_names.keys()])[::-1]
    # print "RBP name auto-detect", hits
    rbp_name = hits[0][1]

    results = sorted(rbp_names[hits[0][1]])
    files = np.array([f for conc, f in results])
    rbp_conc = np.array([conc for conc, f in results], dtype=np.float32)

    return rbp_name, files, rbp_conc


def vector_stats(v):
    print getattr(v,"__name__", "no name"), type(v)
    print "shape",v.shape
    print "pos. values", (v > 0).sum()
    print "0 values", (v == 0).sum()
    print "neg. values", (v < 0).sum()
    print "nan values", np.isnan(v).sum()
    print "non-finite values", (~np.isfinite(v)).sum()
    print "min max", v.min(), v.max()
    print "mean median", np.mean(v), np.median(v)


class Run(object):
    def __init__(self, options):
        self.options = options
        if options.auto:
            self.rbp_name, self.reads_files, self.rbp_concentrations = auto_detect('.')
        else:
            self.rbp_name = options.name
            self.reads_files = args
            self.rbp_concentrations = [float(c) for c in options.rbp_conc.split(',')]
        
        if not len(self.reads_files):
            raise ValueError("missing arguments: need <input_reads_file> <pulldown_reads1_file> ... (or use --auto)")

        # control caching framework behavior
        from cska.caching import CachedBase
        CachedBase._do_not_cache = options.disable_caching
        CachedBase._do_not_pickle = options.disable_pickle
        CachedBase._do_not_unpickle= options.disable_unpickle

        self._init_paths()
        self._init_logging()


    def _init_paths(self):
        """prepare and initialize outout paths"""
        import datetime
        self.run_folder = (self.options.run+"/").format(datestr=datetime.datetime.now().strftime("%b-%d-%Y_%H:%M:%S"))
        self.run_path = ensure_path(os.path.join(self.options.output, self.run_folder))
        
        # keep a symlink named "recent" always pointing to last run folder
        recent_path = os.path.join(self.options.output, "recent")
        try:
            os.remove(recent_path)
        except OSError:
            pass

        os.symlink(self.run_folder, recent_path)
        
        # where to put/find transparent pickle/unpickle objects
        from cska.caching import CachedBase
        CachedBase.pkl_path = os.path.join(self.options.output, ".pkl")

        # accessibility prediction from folding
        self.fold_path = os.path.join(self.options.output, "acc")
        if self.options.no_structure:
            self.fold_path = "NOSTRUCTURE"


    def _init_logging(self):
        # set up logging
        self.log_path = os.path.join(self.run_path,"run.log")

        FORMAT = '%(asctime)-20s\t%(levelname)s\t%(name)s\t%(message)s'
        formatter = logging.Formatter(FORMAT)
        logging.basicConfig(level=logging.INFO, format=FORMAT)    
        root = logging.getLogger('')
        fh = logging.FileHandler(filename=self.log_path, mode='a')
        fh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(fh)
        
        self.logger = logging.getLogger('CSKA')
        self.logger.setLevel(logging.INFO)
        import subprocess
        path = os.path.dirname(os.path.realpath(__file__))
        git = subprocess.Popen(["git","describe","--always"], cwd=path, stdout=subprocess.PIPE).communicate()[0]
        self.logger.info("version {0} [git {1}]".format(__version__, git.rstrip()))
        self.logger.info("invoked as '{0}'".format(" ".join(sys.argv)) )

        # set info level for specific sub-systems
        for sub in self.options.info.split(','):
            if not sub:
                continue
            sub = sub.replace('root',"")
            logging.getLogger(sub).setLevel(logging.INFO)

        # set debug log level for specific sub-systems
        for sub in self.options.debug.split(','):
            if not sub:
                continue
            sub = sub.replace('root',"")
            logging.getLogger(sub).setLevel(logging.DEBUG)
            if sub == 'cache':
                from cska.caching import CachedBase
                CachedBase.debug_caching = True


    def make_SKA(self):
        """parametrize SKA algorithm"""
        from cska.ska_runner import SKARunner
        ska = SKARunner(
            max_iterations = self.options.n_passes,
            convergence = self.options.convergence, 
        )
        return ska


    def get_storage_kwargs(self):
        storage_kw = dict(
            overwrite = self.options.overwrite, 
            T=self.options.temp, 
            disc_mode='linear', 
            dummy=self.options.no_structure, 
            # acc_scale=self.options.acc_scale
        )
        if int(self.options.openen_discretize):
            dtype = getattr(np, "uint{0}".format(self.options.openen_discretize))
            storage_kw.update(dict(discretize=True, disc_dtype=dtype))
        else:
            storage_kw.update(dict(discretize=False, raw_dtype=np.float32))

        return storage_kw


    def select_reads(self):
        # start a new analysis
        from cska.analysis import RBNSAnalysis
        from cska.reads import RBNSReads

        storage_kw = self.get_storage_kwargs()

        rbns = RBNSAnalysis(
            rbp_name = self.rbp_name,
            out_path = self.run_path,
            ska_runner = self.make_SKA(),
        )
        self.logger.info("populating RBNS analysis with reads")
        
        for fname, rbp_conc in zip(self.reads_files, self.rbp_concentrations):
            reads = RBNSReads(
                fname, 
                format = self.options.format,
                rbp_conc=rbp_conc,
                rbp_name = self.rbp_name,
                n_max=self.options.n_max, 
                pseudo_count=self.options.pseudo, 
                rna_conc = self.options.rna_conc,
                temp = self.options.temp,
                n_subsamples = self.options.subsamples,
                adap3=self.options.adap3,
                acc_storage_path = self.fold_path,
                storage_kw=storage_kw
            )
            
            rbns.add_reads(reads)
        
        self.rbns = rbns
        self.n_samples = len(self.rbns.reads) - 1 # first is input control
        return rbns


    def compute_metrics(self, metrics):
        self.logger.info("computing RBNS metrics '{0}'".format(metrics))
        kmin, kmax = self.options.metrics_k.split('_')
        for k in range(int(kmin), int(kmax) + 1):
            self.rbns.compute_results(k, self.options, results=metrics)
            self.rbns.flush()


    def keep_best(self):
        if self.options.best:
            self.rbns = self.rbns.keep_best_samples(n=self.options.best)
        return self.rbns


    def fold_reads(self):
        self.logger.info("folding reads with '{0}' threads".format(self.options.parallel))
        from cska.fold import parallel_fold
        # prepare outout path
        if not os.path.exists(self.fold_path):
            os.makedirs(self.fold_path)

        kmin, kmax = self.options.folding.split('-')

        # fold the reads
        for reads in self.rbns.reads:
            if self.options.skip_folded:
                if reads.acc_storage.has_data(self.options.max_k):
                    self.logger.info("skipping {} because accessibilities have already been computed and stored.".format(reads.name))
                    continue

            self.logger.info("folding {reads.name} ({reads.fname})".format(reads=reads) )
            parallel_fold(
                reads.iter_reads(), 
                reads.acc_storage,
                temp = reads.temp,
                adap5 = self.options.adap5,
                adap3 = self.options.adap3,
                k_min = int(kmin),
                k_max = int(kmax),
                n_max = self.options.n_max,
                l_insert = self.rbns.reads[0].L,
                skip_adap = self.options.skip_adap,
                n_parallel= self.options.parallel,
            )


    def simulate(self):
        pass
        # # TODO: properly integrate simulation
        # if options.simulate == "reads":
        #     from cska.optimize import RBNSGenerator
        #     for k in range(options.min_k, options.max_k + 1):
        #         gen = RBNSGenerator(k,l=40, seed=options.seed)
        #         gen.assign_experimental_input(reads_files[0])
        #         gen.energy_plot()

        #         r_matrix = []
        #         for P in np.array(rbp_concentrations[1:]):
        #             r = gen.predict_r_values(P=P, store="r_{0}.tsv".format(P))
        #             r_matrix.append(r)
                    
        #             read_path = os.path.join(self.out_path,"sim_bound_{0}.reads".format(P) )
        #             gen.generate_bound_reads(read_path, P=P, p_ns=0.00, N=20000000)
        #             occ = gen.predict_occupancies(P=P, store="occ_{0}.tsv".format(P))

    
    def init_model_parameters(self):
        ## prime the optimization from dependent-kmer analysis or load PSAM
        from cska.gradient import ModelParametrization
        if self.options.mdl_psam_init:
            self.logger.info("loading params from: '{0}'".format(self.options.mdl_psam_init))
            self.params = ModelParametrization.load(self.options.mdl_psam_init, self.n_samples)

        elif self.options.seed_analysis:
            from cska.seed import SeedRefinement
            SR = SeedRefinement(self.rbns, km=self.options.seed_analysis, max_linear_k=self.options.max_width)
            self.params = SR.seeded_params(self.rbns.n_samples)

            # clean up memory usage
            for reads in self.rbns.reads:
                reads.cache_flush()
        else:
            raise ValueError("need to either load a PSAM using --psam-resume or build one using --seed-analysis")
            sys.exit(1)

        return params


    def calibrate_footprint(self):
        from cska.footprint import FootprintCalibration
        cal = FootprintCalibration(self.rbns, self.params)
        kmin, kmax = self.options.footprint.split('-')
        self.params = cal.calibrate(k_core_range = [int(kmin), int(kmax)])
        self.logger.info("optimal parameters after footprint calibration: acc_k={self.params.acc_k} acc_shift={self.params.acc_shift} acc_scale={self.params.acc_scale}".format(self=self))
        return self.params

    def PSAM_gradient_descent(self, name="opt"):
        if self.options.compare:
            compare = self.options.compare
        else:
            compare = self.rbp_name
        
        from cska.comparison import RefComparison
        ref = RefComparison(compare, ref_file=self.options.ref_file)

        from cska.psamgrad import PSAMGradientDescent
        PGD = PSAMGradientDescent(self.rbns, self.params, ref=ref, k_fit=self.options.grad_k, mdl_name=self.options.grad_mdl, Z_thresh=self.options.Z_thresh, run_name=name, maxiter=self.options.grad_maxiter, eps=self.options.mdl_epsilon)
        PGD.optimize()
        self.params = PGD.descent.params

        return self.params


def main():
    options, args = parse_cmdline()
    run = Run(options)

    try:
        rbns = run.select_reads()
        # first, compute RBNS metrics
        metrics = [m.strip() for m in options.results.strip().split(',') if m.strip()]
        if metrics:
            run.compute_metrics(metrics)
           
        rbns = run.keep_best()
        if options.multi_stage:
            run.logger.info("STAGE0: initialize PSAM")
            params = run.init_model_parameters()

            run.logger.info("STAGE1: PSAM optimization without secondary structure accessibility")
            run.params.acc_k = 0 # disable accessibility
            params = run.PSAM_gradient_descent('opt_nostruct')

            run.logger.info("STAGE2: footprint parameter estimation")
            params = run.calibrate_footprint()

            run.logger.info("STAGE3: PSAM optimization with accessibility footprint")
            params = run.PSAM_gradient_descent('opt_full')
            sys.exit(0)

        if options.folding:
            run.fold_reads()
            sys.exit(0)

        run.init_model_parameters()

        if options.acc_scan:
            run.calibrate_footprint()
            sys.exit(0)

        if options.grad_mdl:
            run.PSAM_gradient_descent()
            sys.exit(0)


    except SystemExit:
        # This is alright
        pass

    except:
        run.logger.error("Caught exception. Gathering traceback")
        exc = traceback.format_exc()
        run.logger.error(exc)
        sys.stderr.write(exc)
        
        # in case we have child processes, try to end them gracefully
        import cska.fold
        fold.interrupt()
    else:
        run.logger.info("run completed.")

if __name__ == '__main__':
    main()
