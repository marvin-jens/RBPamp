#!/usr/bin/env python
import sys
import os
import numpy as np
import cska
import cska.ska_kmers as cyska
bases = np.array(list('ACGU'))
base_idx = { 
    'A' : 0,
    'C' : 1,
    'G' : 2,
    'T' : 3,
    'U' : 3 
}

ambig = "-NMRWSYKVHDBACGUT"
ambig_index = dict([(code, n) for n,code in enumerate(ambig)])
ambig_vectors = np.array([
    # A    C    G    T
    [0.0, 0.0, 0.0, 0.0],
    [.25, .25, .25, .25],
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0, 0.5],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.5, 0.5],
    [.33, .33, .33, 0.0],
    [.33, .33, 0.0, .33],
    [.33, 0.0, .33, .33],
    [0.0, .33, .33, .33],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
])

def project_column(col):
    n = col.sum()
    if n:
        col = col / n
    i = (col[np.newaxis,:] * ambig_vectors).sum(axis=1).argmax()
    return ambig[i]
    
def hull(kmer):
    """
    generate all single base substitution variants of a 
    seed motif by bit-operations on the corresponding kmer index.
    """
    k = len(kmer)
    subst = [kmer]
    seed = cyska.seq_to_index(kmer)
    for j in range(k):
        nt = (seed >> j*2) & 3
        mask = seed^ nt << (j*2)
        for l in range(4):
            var = mask | (l << j*2)
            if var != seed:
                subst.append(cyska.index_to_seq(var,k))

    return subst

def expand(kmer_set, left=True):
    for nt in 'acgt':
        for kmer in kmer_set:
            if left:
                yield nt + kmer
            else:
                yield kmer + nt
    
def weblogo_save(counts, fname="pwm.eps", title=""):
        import weblogolib as wl
        from corebio.seq import unambiguous_rna_alphabet
        #data = LogoData(alphabet=unambiguous_rna_alphabet, length=5, counts=counts, entropy=np.ones(5), weight=np.ones(5))
        data = wl.LogoData.from_counts(unambiguous_rna_alphabet, counts)
        #import sys
        #sys.stderr.write(str( data))
        options = wl.LogoOptions(color_scheme=wl.classic, fineprint="", logo_title=title, yaxis_label='A.U.', scale_width=True, resolution=300)
        # options.title = "A Logo Title"
        fmt = wl.LogoFormat(data, options)
        dump = wl.eps_formatter( data, fmt)
        
        if fname:
            with file(fname,'wb') as f:
                f.write(dump)
        
        return fname

class PSAM(object):
    def __init__(self, psam, A0 = 1e-6):
        self.psam = np.array(psam, dtype=np.float32)
        amax = psam.max(axis=1)
        self.psam /= amax[:, np.newaxis]
        assert (self.psam.max(axis=1) == 1).all()
        
        self.A0 = A0
        self.n = len(psam)

    @property
    def Kd(self):
        return 1./self.A0

    @classmethod
    def from_kmer(cls, kmer, **kwargs):
        kmer = kmer.upper()
        cols = []
        for nt in kmer:
            cols.append(ambig_vectors[ambig_index[nt]])
        
        m = np.array(cols, dtype=np.float32)
        psam = cls(m, **kwargs)
        psam.kmer_seed = kmer
        psam.kmer_set = hull(kmer)
        return psam
    
    @classmethod
    def from_kmer_variants(cls, kmers, aff, **kwargs):
        """
        Input HAS TO BE SORTED by descending affinity
        or you will get assertion errors.
        """
        #print ">>> THIS BETTER BE SORTED"
        #for mer, a in zip(kmers, aff):
            #print mer, a
            
        psam = cls.from_kmer(kmers[0], A0=aff[0])
        mask_sum = np.zeros(psam.psam.shape, dtype=float)
        for kmer, a in zip(kmers[1:], aff[1:]):
            new = cls.from_kmer(kmer, A0=a)
            mask = (new.psam == 1) & (psam.psam != 1)
            rel_A = new.A0/psam.A0
            #if rel_A > 1:
                #print "fuckup adding",kmer,a
                #print psam
                #print new
                #print "rel_A", rel_A
                
            assert rel_A <= 1. # should be enforced by prior sorting
            psam.psam += mask * rel_A
            mask_sum += mask
            #print "->merged", kmers[i], rel_A

        #if not (psam.psam <= 1.).all():
            #print mask_sum
            #print psam.psam
        assert (psam.psam <= 1.).all()
        #print psam
        psam.kmer_set = kmers
        return psam


    @property
    def kmer_affinities(self):
        cog = self.psam.argmax(axis=1)
        cognate = bases[cog]
        
        kmers = ["".join(cognate)]
        aff = [1.]
        for i in range(self.n):
            for j in range(4):
                if j == cog[i]:
                    continue
                aff.append(self.psam[i,j])
                mer = np.array(cognate)
                mer[i] = bases[j]
                kmers.append("".join(mer))
        
        kmers = np.array(kmers)
        aff = np.array(aff) * self.A0

        I = aff.argsort()[::-1]
        return kmers[I], aff[I]

    @property
    def affinities(self):
        aff = np.zeros(4**self.n, dtype=np.float32)
        for kmer, a in zip(*self.kmer_affinities):
            aff[cyska.seq_to_index(kmer)] = a

        return aff

    @property
    def consensus(self):
        return "".join([project_column(col) for col in self.psam])
        
    def __add__(self, mdl):
        assert self.n == mdl.n
        
        A0 = self.A0 + mdl.A0
        #w1 = self.A0 / A0
        #w2 = mdl.A0 / A0

        psam = self.A0 * self.psam + mdl.A0 * mdl.psam
        amax = psam.max(axis=1)
        psam /= amax[:, np.newaxis]
        
        return PSAMState(psam, max(self.A0, mdl.A0))

    @property
    def discrimination(self):
        return (self.psam.max(axis=1) / self.psam.sum(axis=1) - .25 ) / .75

    def __str__(self):
        buf = ["PSAM A0={0} n={1}".format(self.A0, self.n)]
        for col,d  in zip(self.psam, self.discrimination):
            buf.append("\t".join(["{0:.6e}".format(s) for s in col] + [project_column(col), str(d)]) )

        kmer_seed = getattr(self, "kmer_seed","")
        kmer_set =  getattr(self, "kmer_set","")
        if kmer_seed:
            buf.append("seeded from '{0}'".format(kmer_seed))
        if len(kmer_set):
            buf.append("built from {0} kmers '{1}'".format(len(kmer_set), ",".join(sorted(kmer_set)) ) )
        return "\n".join(buf)

    def store_params(self, fname):
        file(fname, 'w').write(str(self))
        
    def save_logo(self, fname='pwm.eps', title=""):
        counts = self.psam
        weblogo_save(self.psam, fname=fname, title=title)

from collections import defaultdict 
import logging
class PWMOptimizer(object):
    def __init__(self, k_min, k_max, opt):
        self.k = k_min
        self.k_max = k_max
        self.nA = 4**k_min
        self.opt = opt
        self.pwm_by_kmer = {}
        self.pwms = {}
        self.errors = []
        self.last_improvements = []
        self.t = 0
        self.kmer_queues = defaultdict(set)
        
        self.masked = np.ones(self.nA, dtype=np.uint8)
        self.weights = np.ones(self.nA, dtype=np.uint8)
        self.logger = logging.getLogger("opt.PWMOptimizer")

    def record_cue(self, seed, shift, i, compound):
        # store shifted motifs for future analyses on k+x
        self.kmer_queues[seed.lower()].add( (shift, i, compound) )

    def digest_cues(self, seed):
        """
        Based on stored cues, decide in which direction to expand a PWM (add a column).
        Cues are taken from previously selected k-mers with bad R-value agreement that
        were identified as shifted versions of the PWM associated with seed.
        We add up the affinities assigned to those shifted versions to determine if
        more affinity can be "added to the PWM" by adding a column left or right.
        """
        left = []
        l = 0
        
        right = []
        r = 0
        seed_cues = self.kmer_queues[seed.lower()]
        cues = [(self.opt.mdl.params[i], shift, compound) for i, shift, compound in seed_cues]
        for aff, shift, compound in sorted(cues, reverse=True):
            if shift < 0:
                l += aff
                left.append(compound)
            else:
                r += aff
                right.append(compound)
        
        return l, left, r, right
        
        
    def kmer_residuals(self):
        errors = self.opt.kmer_errors(self.opt.current.R)
        return (errors**2).sum(axis=0) # sum sq. error across concentrations
        #return (errors**2).sum(axis=0) # sum sq. error across concentrations
    
    def kmer_logratios(self):
        lr = np.log2(self.opt.current.R / self.opt.R_obs)
        return lr

    def kmer_logerrors(self):
        le = self.kmer_logratios().mean(axis=0)
        return le

    def dump_logratios(self):
        path = cska.ensure_path(os.path.join(self.opt.out_path,"affinity","{0}mer_logratios.tsv".format(self.k)))
        self.logger.debug("storing kmer prediction error in {0}".format(path) )
        LR = self.kmer_logratios()
        #print LR.shape
        #print self.opt.mdl.param_name.shape
        f = file(path, 'w')
        for kmer, aff, lr in zip(self.opt.mdl.parameters.param_name, self.opt.mdl.params, LR.T):
            row = [kmer, str(aff), ] + [str(l) for l in lr]
            f.write("\t".join(row) + '\n')
        
    def worst_kmer(self, debug=False):
        #residual = self.kmer_residuals()
        residual = self.kmer_logerrors()
        w = self.masked / self.weights
        i = (residual * w).argmin()
        kmer = cyska.index_to_seq(i, self.k)
        self.weights = np.where(self.weights > 1, self.weights - 1, 1)
        self.weights[i] *= 2
        
        if debug:
            R_obs = self.opt.R_obs[:,i]
            R_pred = self.opt.current.R[:,i]
            dR = R_pred - R_obs
            self.logger.debug("{kmer} R_obs={R_obs} R_pred={R_pred} dR={dR}".format(**locals()) )
        
        self.dump_logratios() # THIS IS FOR DEBUGGING

        return i, kmer, residual[i]
    
    def is_shifted(self, kmer, s_max=2):
        k = len(kmer)
        kmer_pwm_map = self.pwm_by_kmer
        #print "mapped", sorted(self.pwm_by_kmer.keys())
        for x in range(1,s_max+1):
            for pad in list(cyska.yield_kmers(x)):
                rshifted = (pad + kmer[:k-x]).lower()
                lshifted = (kmer[x:]+pad).lower()
                #print "l", x, kmer, lshifted
                #print "r", x, kmer, rshifted
                if lshifted in kmer_pwm_map:
                    seed = kmer_pwm_map[lshifted].kmer_seed
                    return -x, seed, kmer[:x] + seed
                elif rshifted in kmer_pwm_map:
                    seed = kmer_pwm_map[rshifted].kmer_seed
                    return x, seed, seed + kmer[-x:]

        return 0, kmer, kmer
        
        
    def optimize_kmer_set_ordered(self, kmers, opt_tick=True):
        #print "optimize_kmer_set_ordered", kmers
        kmers = np.array(kmers)
        kmer_indices = np.array([cyska.seq_to_index(mer) for mer in kmers])
        kmer_res = self.kmer_residuals()[kmer_indices]
    
        # sort descending by residual R-value error
        I = kmer_res.argsort()[::-1]
        kmers = kmers[I]
        kmer_indices = kmer_indices[I]

        err0 = self.opt.global_error(self.opt.current.R)
        self.logger.debug("error before optimizing kmer set {0}".format(err0))
        # optimize kmer and all of its 1-mismatch relatives, ordered by their scheduler scores
        #repeat = True
        #while repeat:
        
        N = float(len(kmers))
        for n, (i, mer) in enumerate(zip(kmer_indices, kmers)):
            param0 = self.opt.current.params[i]
            #if mer.lower() == 'gccca':
                #self.opt.sweep_param(i,x0=param0)
                
            best, err, new_state = self.opt.optimize_single_param(i, local = False)
            better = self.opt.update(new_state, err, "{mer} {param0:.3e} -> {best:.3e}".format(**locals()))
            #if self.opt.t <= (self.k*3)+1:
                #self.opt.reporter.trigger_plots(self.opt.t, occasion="initial_{0}".format(mer.upper()))
            
            cum_change = err0 - err
            rel_change = cum_change / err0
            if rel_change > .1:
                self.logger.warning('substantial changes accumulated during kmer_set fit -> re-freshing')
                self.opt.step_tm_refresh()
                b, new_state, err0 = self.opt.step_betas()

            if n >= 100 and (n % 100 == 0):
                self.logger.warning('{0:.2f}%  of kmer set optimized'.format(100. * n/N) )
                    #repeat = True
                    #break
                #else:
                    #repeat = False

        d_err = err - err0

        # re-sort, descending on fitted kmer affinity
        kmer_aff = new_state.params[kmer_indices]
        I = kmer_aff.argsort()[::-1]

        return kmers[I], kmer_indices[I], kmer_aff[I], d_err
        

    def retrieve_aff_kmer_set(self, kmers, state):
        kmers = np.array(kmers)
        kmer_indices = np.array([cyska.seq_to_index(mer) for mer in kmers])
        kmer_aff = state.params[kmer_indices]
        I = kmer_aff.argsort()[::-1]

        return kmers[I], kmer_indices[I], kmer_aff[I]

    def pwm_optimize_hull(self, kmer, keep_pwm=True):
        
        # create a PWM "centered" on the seeding kmer
        pwm = PSAM.from_kmer(kmer)
        kmers, kmer_indices, kmer_aff, d_err = self.optimize_kmer_set_ordered(pwm.kmer_set)
        
        if kmer.lower() != kmers[0].lower():
            # The hull contains a kmer with higher affinity than our initial seed!
            # Select that kmer as hull instead.
            new = kmers[0]
            self.logger.debug("{kmer} can not be seed, because it is not hull-maximal! Switching to {new} which has higher affinity.".format(**locals()))
            return self.pwm_optimize_hull(kmers[0], keep_pwm=keep_pwm)
            
        self.opt.step_scale()
        # reload these after the step_scale!
        kmer_aff = self.opt.current.params[kmer_indices]
        #self.opt.step_betas()

        pwm = PSAM.from_kmer_variants(kmers, np.array(kmer_aff))
        if keep_pwm:
            for mer in kmers:
                self.pwm_by_kmer[mer.lower()] = pwm
            self.pwms[pwm.kmer_seed] = pwm
        
        self.logger.info("pwm_optimize_hull({pwm.kmer_seed})->Kd={pwm.Kd:.3e} nM d_err={d_err:.3e}".format(pwm=pwm, d_err=d_err) )
        return pwm, d_err
    
    def get_pwms(self, thresh=100.):
        covered_kmers = set()
        names = self.opt.mdl.parameters.param_name
        # kmers in reverse affinity order
        aff = self.opt.current.params[:self.opt.nA]
        I = aff.argsort()[::-1]
        A0 = aff[I[0]]
        last_explained = np.zeros(self.opt.n_conc)

        high_affinity_pwms = []
        for i in I:
            #print "covered", sorted(covered_kmers)
            a = aff[i]
            kmer_seed = names[i].lower()
            #print "checking ",kmer_seed,aff[i]
            if a < A0/thresh:
                break

            kmers, kmer_ind, kmer_aff = self.retrieve_aff_kmer_set(
                PSAM.from_kmer(kmer_seed).kmer_set,
                self.opt.current
            )
            if kmers[0].lower() != kmer_seed:
                continue
            
            #for m,i,a in zip(kmers, kmer_ind, kmer_aff):
                #print m,a
                
            pwm = PSAM.from_kmer_variants(kmers, kmer_aff)

            if pwm.kmer_seed.lower() not in covered_kmers:
                covered_kmers |= set([mer.lower() for mer in pwm.kmer_set])

                fraction_explained = np.array([reads.reads_with_kmer_set(list(pwm.kmer_set)) / float(reads.N) for reads in self.opt.rbns_analysis.reads[1:]])
                pwm.fraction_explained = fraction_explained
                high_affinity_pwms.append(pwm)
        
        pwms_ordered = sorted(high_affinity_pwms, key = lambda pwm : - pwm.fraction_explained.mean())
        f0 = pwms_ordered[0].fraction_explained
        kmer_set_selected = set()
        n = 0
        for pwm in pwms_ordered:
            if (pwm.fraction_explained / f0 > .25).any():
                yield pwm
                kmer_set_selected |= set(pwm.kmer_set)
                n += 1
            else:
                break

        total_fraction_explained = np.array([reads.reads_with_kmer_set(list(kmer_set_selected)) / float(reads.N) for reads in self.opt.rbns_analysis.reads[1:]])
        self.logger.info("{0} selected PWMS explain {1} of RBNS reads".format(n, total_fraction_explained))

    def save_pwms(self):
        for i,pwm in enumerate(self.get_pwms()):
            rank = i+1 # TODO: improve by how much data "is explained"
            pwm_path = cska.ensure_path(
                os.path.join(
                    self.opt.out_path, "motifs", "{pwm.n}mer_rank{rank:02d}_{pwm.consensus}/".format(**locals())
                )
            )
            param_fname = '{pwm.consensus}_t{self.t}.tsv'.format(self=self, pwm=pwm)
            pwm.store_params(os.path.join(pwm_path, param_fname))

            logo_title = 'Kd={pwm.Kd:.2e} nM'.format(**locals())
            logo_fname = '{pwm.consensus}_Kd_{pwm.Kd:.3f}_t={self.t}.eps'.format(**locals())
            self.logger.info("storing PWM {pwm.kmer_seed} Kd={pwm.Kd:.2e} -> '{pwm_path}'".format(pwm=pwm, pwm_path=pwm_path) )

            pwm.save_logo(os.path.join(pwm_path, logo_fname), title=logo_title)
        

    def store_params(self):
        self.opt.mdl.parameters.store(os.path.join(self.opt.out_path, "affinity"))

        # temporary: save partition function samples
        #self.opt.current.store_Z(os.path.join(self.opt.opt_path, '{self.k}mer_Z1.npy'.format(self=self)))

        self.save_pwms()


    def optimize(self, max_iter=1000, eps=1e-2):
        self.opt.reporter.tick(0)
        self.opt.reporter.trigger_plots(self.opt.t, occasion="init")
        for t in range(max_iter):
            if not self.next_move(eps=eps):
                break

        self.opt.reporter.trigger_plots(self.opt.t, occasion="final")
        self.logger.info("ending optimization after {self.t} iterations at k={self.k}".format(self=self))
        
    def next_move(self, lag=3, eps=1e-2):
        # if self.t == 0:
        #     self.increase_k() # force k increase to test degradation of fit
        
        # if self.t == 0:
        #     for pwm in self.get_pwms():
        #         print pwm

        corr = self.opt.correlation()
        self.logger.info("{self.k}mer correlations at t={self.t} {corr}".format(**locals()) )
        last_improvements = ",".join(["{0:.3e}".format(i) for i in self.last_improvements[-lag:]])
        
        err0 = self.opt.global_error(self.opt.current.R)
        self.errors.append(err0)
        self.logger.debug("current_error={err0:.2e} last last_improvements: {last_improvements}".format(**locals()) )
        
        if len(self.last_improvements) >= lag:
            mean_improve = np.mean(np.array(self.last_improvements)[-lag:])
            last_error = self.errors[-1]
            rel_change = - mean_improve / last_error
            self.logger.info("mean_improve={mean_improve:.2e}, last_error={last_error:.2e} rel_change={rel_change:.2e} eps={eps}".format(**locals()))
            
            if rel_change < eps:
                self.logger.warning("t={self.t} no reasonable improvements achieved over past {lag} iterations. Switching to k+1={kn}".format(lag=lag, kn=self.k+1, self=self))
                self.store_params()
                if self.k < self.k_max:
                    self.opt.reporter.trigger_plots(self.opt.t, occasion="before_increase_k")
                    self.increase_k()
                    return True
                else:
                    return False

        i, kmer, res = self.worst_kmer(debug=True)
        keep_pwm = True
        self.logger.info("selected worst kmer {kmer} with residual error={res}".format(**locals()) )
        if kmer in self.pwm_by_kmer:
            pwm = self.pwm_by_kmer[kmer]
            self.logger.info("{kmer} belongs to PWM({pwm.kmer_seed})".format(**locals()) )
            kmer = pwm.kmer_seed
        #else:
            #shift, seed, compound = self.is_shifted(kmer)
            #ashift = abs(shift)
            #if ashift > 0:
                #self.logger.info("{kmer} is {ashift}-SHIFT of HULL({seed}). Recording {compound} for k+{shift}".format(**locals()) )
                #self.record_cue(seed, shift, i, compound)
                ##keep_pwm = False
            #else:
                #self.logger.info("{kmer} does not belong to current PWM set. Starting new PWM".format(**locals()) )

        pwm, d_err = self.pwm_optimize_hull(kmer, keep_pwm=keep_pwm)
        err = self.opt.global_error(self.opt.current.R)

        self.t += 1
        self.last_improvements.append(err-err0)
        
        # test new PWM output
        self.save_pwms()

        return pwm
        
    def params_for_k_increase(self, waterline = None):
        """
        generate kmer parameters for k+1 by scoring k+1 mers with existing  
        k-mer parameters
        """
        k = self.k
        try:
            # try to load k-1 mer ("core") affinities to determine energy contributed by overlap
            # between k-mers
            core_mdl = self.opt.mdl.parameters.resume(os.path.join(self.opt.out_path,"affinities"), k=k-1)
            core = core_mdl.parameters
            self.logger.warning("using '{0}' as core energies".format(core))
        except IOError:
            core = None
        
        new = self.opt.mdl.parameters.params_for_next_k(core_param=core, waterline=10*self.opt.aff0, aff0=self.opt.aff0)
        return new

    def create_optimizer(self, k, params=[]):
        from cska.optimize import ModelOptimization
        # free some memory
        self.opt.input_reads.cache_flush('__cached_get_index_matrix')
        self.opt.input_reads.acc_storage.cache_flush('__cached_get_raw')

        # create new optimizer and model
        new_opt = ModelOptimization(k, self.opt.rbns_analysis,
            mdl_params = params,
            t0 = self.opt.t,
            reporter = self.opt.reporter,
            kmer_opt_global = not self.opt.param_local_fit,
        )
        new_opt.errors = self.opt.errors + new_opt.errors
        new_opt.correlations = self.opt.correlations
        new_opt.rel_improvements = self.opt.rel_improvements

        # some plumbing to make reports/plots contiguous
        self.opt.reporter.set_opt(new_opt)
        return new_opt

    def increase_k(self, cutoff=.01):
        # get new optimizer and model with expanded kmer model parameters
        corr = self.opt.correlation()
        self.logger.warning("{self.k}mer correlations BEFORE k-increase {corr}".format(**locals()) )

        waterline = self.opt.aff0 * 10
        new_params = self.params_for_k_increase(waterline=waterline)
               
        self.opt = self.create_optimizer(self.k+1, params=new_params)
        self.logger.warning("new {0} mer parameters:'{1}'".format(self.k+1, self.opt.mdl.parameters))
        self.k = self.k + 1
        self.nA = 4**self.k

        corr = self.opt.correlation()
        self.logger.warning("new {self.k}mer correlations before scaling {corr}".format(**locals()) )

        rep = self.opt.reporter
        rep.trigger_plots(self.opt.t, occasion="param_expansion")
        # self.opt.step_betas()
        # corr = self.opt.correlation()
        # self.logger.warning("new {self.k}mer correlations after beta {corr}".format(**locals()) )

        self.opt.step_scale(min_scale=.001, max_scale=4.)
        corr = self.opt.correlation()
        self.logger.warning("new {self.k}mer correlations after scaling {corr}".format(**locals()) )

        rep.trigger_plots(self.opt.t, occasion="param_expansion_rescaled")
        
        # all parameters that have been changed from background levels
        need_fit = (new_params[:self.opt.nA] > waterline).nonzero()[0]

        n_fit = len(need_fit)
        self.logger.info("switched to k+1 ={self.k} step 1: re-calibration of {n_fit} parameters.".format(**locals()) )
        
        res = self.kmer_residuals()[need_fit]
        aff = new_params[need_fit]
        kmers = self.opt.mdl.parameters.param_name[need_fit]
        order = aff.argsort()[::-1]
        #order = res.argsort()[::-1]
        #need_fit = new_param[need_fit].argsort()[::-1]

        self.optimize_kmer_set_ordered(kmers[order])
        corr = self.opt.correlation()
        self.logger.warning("new {self.k}mer correlations after param opt {corr}".format(**locals()) )

        rep.trigger_plots(self.opt.t, occasion="param_expansion_optimized")
        self.opt.step_scale()
        corr = self.opt.correlation()
        self.logger.warning("new {self.k}mer correlations after final scaling {corr}".format(**locals()) )

        rep.trigger_plots(self.opt.t, occasion="param_expansion_optimized_scaled")
        
        old_pwms = self.pwms
        n_pwm = len(old_pwms.values())
        pwm_list = sorted(old_pwms.values(), key=lambda x : x.Kd)
        pwm_str = ",".join(["{p.kmer_seed} Kd={p.Kd:.2f}".format(p=pwm) for pwm in pwm_list])
        self.logger.info("switched to k+1 ={self.k} step 2: reconstruction of {n_pwm} PWMs.".format(**locals()) )
        

        #old_kmers = kmer_set_union(self.pwms)
        
        self.pwm_by_kmer = {}
        self.pwms = {}
        self.last_improvements = []
        #self.t = 0
        
        self.masked = np.ones(self.nA, dtype=np.uint8)
        self.weights = np.ones(self.nA, dtype=np.uint8)

        # # migrate PWMs, ordered by affinity
        # for pwm in pwm_list:
        #     l, left, r, right = self.digest_cues(pwm.kmer_seed)

        #     # expand the PWM's seed kmer in the direction indicated by the cues
        #     new_seed_candidates = np.array(list(expand([pwm.kmer_seed,],l > r)))
            
        #     # optimize the 4 versions of the extended (k+1) seed kmer
        #     kmers, indices, aff, d_err = self.optimize_kmer_set_ordered(new_seed_candidates)
        #     self.opt.step_betas()
        #     rep.trigger_plots(self.opt.t, occasion="increase_k_pwm_{pwm.kmer_seed}_update".format(pwm=pwm))
        #     a0 = aff[0] # highest affinity comes first
        #     for kmer, a in zip(kmers, aff)[:1]: # hack, disable pwm split for now!
        #         if a >= a0 * cutoff:
        #             # affinity is within reasonable range of the optimum
        #             #hull_kmers = hull(kmer)
        #             #hull_indices = np.array([cyska.seq_to_index(mer) for mer in hull_kmers])
        #             ##new_pwm = PSAM.from_kmer_variants(hull_kmers, self.opt.mdl.params[hull_indices])
                    
        #             self.pwm_optimize_hull(kmer)
        #     corr = self.opt.correlation()
        #     self.logger.warning("{self.k}mer correlations with new parameters after optimizing PWM {pwm.kmer_seed} corr={corr}".format(**locals()) )

        # reset migration cues
        self.kmer_queues = defaultdict(set)
        rep.trigger_plots(self.opt.t, occasion="increase_k_finished")

        
        
def opt_merge(p0, pk, padding):
    
    pre0, app0, prek, appk = padding

    A0 = p0.A0
    Ak = pk.A0
    
    if prek:
        Ak /= p0.psam[:prek].mean(axis=1).prod() * A0
    if appk:
        Ak /= p0.psam[-appk:].mean(axis=1).prod() * A0

    print "A0", A0, "Ak", Ak
    A = max(A0, Ak)

    
    n_ext = pre0 + app0
    if n_ext:
        A0_avg = (A0/A)**(1./n_ext)
    
    n_ext = prek + appk
    if n_ext:
        Ak_avg = (Ak/A)**(1./n_ext)

    if pre0:
        p0.psam[:pre0,  :] *= A0_avg
    if app0:
        p0.psam[-app0:, :] *= A0_avg
        print app0, A0_avg, p0.psam[-app0:, :]        
    if prek:
        pk.psam[:prek,  :] *= Ak_avg
        print prek, Ak_avg, pk.psam[:prek, :]        
    if appk:
        pk.psam[-app0:, :] *= Ak_avg
        
    print "p0"
    print p0.psam
    print "pk"
    print pk.psam
    
    I0 = (p0.psam == 1).nonzero()
    Ik = (pk.psam == 1).nonzero()
    I = ((p0.psam == 1) | (pk.psam == 1) ).nonzero()

    
    
    #print I
    n = len(I[0])
    #if not n:
        #I = (pk.psam > 0).nonzero() * p0.psam[I0]
        #p0.psam *= p0.A0 / pk.A0
        #p0.psam[I] = 1.
        #p0.A0 = pk.A0
        #return p0
    
    ratio = pk.A0 / p0.A0
    #current = p0.psam[I].prod()
    #if current > 0:
        #ratio /= current
        
    #print "RATIO to distribute", ratio, pk.A0, p0.A0, p0.psam[I].prod()
    #if n == 1:
        #p0.psam[I] = ratio
        #return p0
    
    # find optimal distribution of changes over 
    # elements in p0 such that product == ratio
    # and perturbation of non-zero elements is 
    # minimal
    
    def normed(w):
        p = np.array(p0.psam, dtype=float)
        p[I] = w
        amax = p.max(axis=1)
        p /= amax[:, np.newaxis]
        return p
        
    cost_mask = (p0.psam > 0) | (p0.psam == 1).all(axis=1)[:, np.newaxis]
    
    def to_optimize(w):
        p = normed(w)
        a0 = p[I0].prod() * A # affinity assigned by current weights to best previous sequence
        ak = p[Ik].prod() * A # affinity assigned by current weights to new sequence
        
        if pre0:
            a0 *= p[:pre0, :].mean(axis=1).prod()
        if prek:
            ak *= p[:prek, :].mean(axis=1).prod()
        
        if app0:
            a0 *= p[-app0:, :].mean(axis=1).prod()
        if appk:
            ak *= p[-appk:, :].mean(axis=1).prod()
            
        matrix_dev = ((A*p - A0 * p0.psam )**2 * cost_mask).sum()
        Ak_dev = (Ak - ak)**2
        A0_dev = (A0 - a0)**2
        cost = matrix_dev + 100*Ak_dev + 100*A0_dev 

        #print p
        #print "costs:", matrix_dev, Ak_dev, A0_dev, cost
        return cost

    from scipy.optimize import minimize
    w0 = np.ones(n) * ratio**(1./n)
    bounds = np.array([(1e-10, max(ratio, 1./ratio)),]*n)
    #print "w0", w0
    #print "bounds", bounds
    res = minimize(to_optimize, w0, bounds=bounds)
    
    weights = res.x
    #print res.x, res.fun, res
    #remain = ratio / weights.prod()
    #weights = np.concatenate( ( weights, (remain,)) )
    p0.psam = normed(weights)
    p0.A0 = A
    return p0
    
    
    
if __name__ == "__main__":
    def is_shifted(kmer, s_max=2):
        k = len(kmer)
        kmer_pwm_map = {'TGCATG':1}
        #print "mapped", sorted(self.pwm_by_kmer.keys())
        for x in range(1,s_max+1):
            for pad in list(cyska.yield_kmers(x)):
                rshifted = (pad + kmer[:k-x]).lower()
                lshifted = (kmer[x:]+pad).lower()
                print "l", x, kmer, lshifted
                print "r", x, kmer, rshifted
                if lshifted in kmer_pwm_map:
                    seed = kmer_pwm_map[lshifted].kmer_seed
                    return -x, seed, kmer[:x] + seed
                elif rshifted in kmer_pwm_map:
                    seed = kmer_pwm_map[rshifted].kmer_seed
                    return x, seed, seed + kmer[-x:]

        return 0, kmer, kmer

    "gggcat is 1-shift of TGCATG"
    "ttgggc is 2-shift of GTGCAT"
    print is_shifted('gggcat')
