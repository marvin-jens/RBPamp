# RBPamp
RNA-Binding Protein Affinity Model with Physical constraints A biophysical model and fit for RNA bind'n'seq (RBNS) experiments (Lambert et al. 2014, Dominguez et al. 2018). Yields a compact, versatile, and predictive description of an RNA-binding proteins primary sequence affinity landscape. Publications: Jens, McGurk & Burge 2022 (in preparation)

## Installation

It is not yet in PyPI. But you can pull the latest development version using `git clone https://marjens@bitbucket.org/marjens/RBPamp.git`. 

We recommend that you create a conda environment using the `requirements.txt`:

```
   conda create --name rbpamp --file requirements.txt -c conda-forge
   conda activate rbpamp
```

`RBPamp` uses `setuptools` so

```
   python setup.py build
   python setup.py install
```

will do what you expect. Alternatively, you can run `pip install .` from your git checkout. To uninstall run `pip uninstall RBPamp` .


## Usage

`RBPamp --help`:

```
      Usage: RBPamp [options] [path_to_RBNS_splitreads]

      Options:
      -h, --help            show this help message and exit
      --version             show version information and quit
      --name=NAME           name of the protein assayed (default=RBP)
      -o OUTPUT, --output=OUTPUT
                              path where results are to be stored (default='RBPamp')
      --run-path=RUN        pattern for run-folder name (default='run_{datestr}')
      --ranks=RANKS         analyze these ranks out of the best n samples (by top
                              R-value) default=1,2,3,4
      --min-R=MIN_R         minimal enrichment value of the (overall) most
                              enriched 7-mer required to consider a sample at all
                              (QC pass) default=1.1
      --resume              re-use previous results
      --continue            add more iterations of optimization even if already
                              completed
      --redo                do not re-use previous results at all
      -R RNA_CONC, --rna-concentration=RNA_CONC
                              concentration of random RNA used in the experiment in
                              nano molars (default=1000 nM)
      -P RBP_CONC, --rbp-concentration=RBP_CONC
                              (comma separated list of) protein concentration used
                              in the experiment(s) in nano molars (default=0,300)
      -T TEMP, --temperature=TEMP
                              temperature of the experiment in degrees Celsius
                              (default=4.0)
      --format=FORMAT       read file format [raw,fasta,fastq] (default=raw)
      --adap5=ADAP5         5'RNA adapter sequence to add to read sequence
      --adap3=ADAP3         3'RNA adapter sequence to add to read sequence
      -N N_MAX, --n-max=N_MAX
                              read at most N reads (preserves RAM for very deep
                              sequencing libraries. default=15M, 0=off)
      --no-replace          TESTING: disable drawing with replacement
      --fold-input=FOLD_INPUT
                              instead of a normal run, fold all reads in the INPUT
                              sample and record accessibilities/open-energies for k
                              in the given range. example --fold=1-12 (default=off)
      --fold-samples=FOLD_SAMPLES
                              instead of a normal run, fold all reads from PULL-DOWN
                              samples and record accessibilities/open-energies for k
                              in the given range. example --fold=1-12 (default=off)
      --acc-scan            SWITCH: scan for high accessibility selection in bound
                              libraries
      --openen-discretize=OPENEN_DISCRETIZE
                              discretize open-energies using <n> bits [8,16] set to
                              0 to disable (default)
      --parallel=PARALLEL   number of parallel threads (currently only used for
                              folding. default=8)
      --metrics=RESULTS     list of RBNS metrics to compute and store
                              (options='R_value,SKA_weight,F_ratio' default='')
      --sample-correlations
                              compute sample vs sample correlation matrices
      --metrics-k=METRICS_K
                              range of kmer sizes for which to compute the desired
                              metrics (default: --metrics-k=3-8)
      --pseudo=PSEUDO       pseudo count to add to kmer counts in order to avoid
                              div by zero for large k (default=10)
      --ska-max-passes=N_PASSES
                              max number of passes (default=10)
      --ska-convergence=CONVERGENCE
                              convergence is reached when max. change in absolute
                              weight is below this value (default=0.5)
      --subsamples=SUBSAMPLES
                              number of subsamples for error estimation (default=10)
      --opt-seed            perform initial motif construction (STAGE0: seed-
                              stage)
      --seed-k=K_SEED       kmer size used for seeding PSAM(s) (default=8)
      --z-cut=Z_CUT         Z-score cutoff for R-values of kmers that go into
                              motif building (default=4)
      --max-motifs=MAX_MOTIFS
                              maximal number of individual PSAMs (variant motifs)
                              being fitted (default=5)
      --seed-thresh=SEED_THRESH
                              score threshold for k-mer:PSAM alignment to trigger a
                              new PSAM (default=.75)
      -w MAX_WIDTH, --max-width=MAX_WIDTH
                              maximum number of nucleotides in PSAM motif (number of
                              columns) default=11)
      --seed-pseudo=SEED_PSEUDO
                              'pseudo' affinity for non-cognate bases. Non-zero
                              allows gradient descent to act on all bases
                              (default=.01)
      --seed-keep-weight=SEED_KEEP_WEIGHT
                              how much total PSAM weight to keep when building final
                              --max-width nt wide matrix (default=.99)
      --footprint-k=FOOTPRINT
                              size range [nt] to search for ideal accessibility
                              footprint (default: --footprint-k=5-20)
      --footprint-motif=FP_NUM
                              which motif number to compute the footprint on
                              (default=0 [all])
      --footprint-min-overlap=FP_MIN_OV
                              minimum overlap between footprint and PSAM as fraction
                              of footprint (default=.2)
      --footprint-max-error=FP_MAX_ERR
                              maximum rel. footprint error acceptable to keep
                              footprint
      --top-kmer-acc=TOP_KMER_ACC
                              generate enrichment vs. accessibility data for top
                              kmers (default=0 [off])
      --grad-k=GRAD_K       k for gradient descent kmer R-value mean squared error
                              objective function (default=6)
      --grad-mdl=GRAD_MDL   method for gradient descent refinement of PSAM
                              [partfunc, meanfield, invmeanfield, ''=off]
                              default=partfunc
      --grad-maxiter=GRAD_MAXITER
                              maximal number of gradient descent iterations
                              (default=500)
      --grad-maxtime=GRAD_MAXTIME
                              maximal time to spend for optimization in seconds
                              (default=12 hours)
      --excess-rbp          MODEL: pretend total RBP == free RBP
      --linear-occ          MODEL: pretend no saturation: occ = P/Kd
      -n N_SAMPLES, --n-samples=N_SAMPLES
                              bootstrap sample n reads at regular intervals during
                              gradient descent or when stuck (default=5M)
      -r RESAMPLE_INT, --resample-interval=RESAMPLE_INT
                              re-sample/bootstrap reads every -r iterations of
                              descent (default=1, 0 to disable)
      --opt-nostruct        perform no-struct gradient descent (STAGE1: nostruct
                              stage)
      --opt-footprint       perform footprint calibration (STAGE2: footprint
                              stage)
      --opt-struct          perform structure-aware gradient descent (STAGE3:
                              struct stage)
      --opt-full            perform all stages of optimization (STAGE0 - STAGE3
      --est-errors          perform PSAM error estimation
      --plot=PLOT           (re-) generate plots. Comma-separated items from
                              seed,descent,scatter,fp,lit,logos or 'all'
      --Z-threshold=Z_THRESH
                              drop reads that have Boltzmann weight of a factor of
                              Z_thresh below the max weight (default=0/off)
      --no-structure        ignore secondary structure folding information
                              (default=False)
      --load-psam=MDL_PSAM_INIT
                              start with affinity parameters from this PSAM file
      --fix-A0              do not attempt to optimize A0 at all
      --eps=MDL_EPSILON     convergence threshold for relative error reduction
                              (default=1e-4)
      --tau=MDL_TAU         convergence estimation interval (default=13) [Note,
                              this should be larger than the re-sampling interval
                              -r]
      --reference=REF_FILE  tab-separated file with measured (reference) Kd values
                              (default=use builtin known_kds.csv)
      --compare=COMPARE     compare to literature values for this protein
      --disable-caching     DEBUG: disable transparent caching (SLOW!)
      --disable-unpickle    DEBUG: disable unpickling. Will recompute and
                              overwrite existing pickled data
      --disable-pickle      DEBUG: disable pickling. Will not create or overwrite
                              any pickled data
      --debug=DEBUG         activate debug output for comma-separated subsystems
                              [root, fold, cache, rbns, opt, model, report]
      --debug-grad          compute empirical gradient alongside analytical (for
                              debugging only)
      --info=INFO           activate info level output for comma-separated
                              subsystems [root, fold, cache, rbns, opt, model,
                              report]
      --log-remote=LOG_REMOTE
                              replicate all logging output to this remote server
                              (useful to collect output from multiple runs in
                              parallel)
      --rnd-seed=SEED       seed for fast pseudo-random number generator (for RBNS
                              simulation)
```

## Testing

After everything is compiled and installed, you can run RBPamp on some test-data

```
   RBPamp tests --run-path=t --opt-seed --metrics=R_value
```

This will perform the first stage of the motif generation (seeding the PSAMs). You can find the output of this initial stage in `tests/RBPamp/t/seed/`.
Here, we also asked RBPamp to compute R-values (kmer enrichments) for us (see Lambert et al. 2014). These will be stored in the `metrics` sub-folder.

Next, you can perform stochastic gradient descent optimization on the initial PSAM(s):

```
   RBPamp tests --run-path=t --opt-nostruct
```

Optimized PSAMs are stored in `tests/RBPamp/t/opt_nostruct` . This was last tested successfully on Ubuntu 22.04.1 (AMD64). Enjoy!

