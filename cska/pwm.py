#!/usr/bin/env python
import sys
import os
import numpy as np
import cska.ska_kmers as cyska
bases = 'ACGU'
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
        import weblogolib as wl
        counts = self.psam
        from corebio.seq import unambiguous_rna_alphabet
        #data = LogoData(alphabet=unambiguous_rna_alphabet, length=5, counts=counts, entropy=np.ones(5), weight=np.ones(5))
        data = wl.LogoData.from_counts(unambiguous_rna_alphabet, counts)
        #import sys
        #sys.stderr.write(str( data))
        options = wl.LogoOptions(color_scheme=wl.classic, fineprint="", logo_title=title, yaxis_label='A.U.', scale_width=False, resolution=300)
        options.title = "A Logo Title"
        fmt = wl.LogoFormat(data, options)
        dump = wl.eps_formatter( data, fmt)
        
        if fname:
            file(fname,'wb').write(dump)
        
        return dump

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
    
    def kmer_logratios(self):
        lr = np.log2(self.opt.current.R / self.opt.R_obs)
        return lr

    def kmer_logerrors(self):
        le = self.kmer_logratios().mean(axis=0)
        return le

    def dump_logratios(self):
        path = os.path.join(self.opt.opt_path,"{0}mer_logratios.tsv".format(self.k))
        self.logger.debug("storing kmer prediction error in {0}".format(path) )
        LR = self.kmer_logratios()
        #print LR.shape
        #print self.opt.mdl.param_name.shape
        f = file(path, 'w')
        for kmer, aff, lr in zip(self.opt.mdl.param_name, self.opt.mdl.params, LR.T):
            row = [kmer, str(aff), ] + [str(l) for l in lr]
            f.write("\t".join(row) + '\n')
        
    def worst_kmer(self, debug=False):
        #residual = self.kmer_residuals()
        residual = self.kmer_logerrors()
        w = self.masked / self.weights
        i = np.fabs((residual * w)).argmax()
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
        
        
    def optimize_kmer_set_ordered(self, kmers):
        #print "optimize_kmer_set_ordered", kmers
        kmers = np.array(kmers)
        kmer_indices = np.array([cyska.seq_to_index(mer) for mer in kmers])
        kmer_res = self.kmer_residuals()[kmer_indices]
    
        # sort descending by residual R-value error
        I = kmer_res.argsort()[::-1]
        kmers = kmers[I]
        kmer_indices = kmer_indices[I]

        err0 = self.opt.global_error(self.opt.current.R)
        # optimize kmer and all of its 1-mismatch relatives, ordered by their scheduler scores
        for i, mer in zip(kmer_indices, kmers):
            param0 = self.opt.mdl.state.params[i]
            best, err, new_state = self.opt.optimize_single_param(i, local = True)
            imp = self.opt.update(new_state, err, "local kmer optimization {mer} -> {best:.3e}".format(**locals()))
            #print self.opt.correlation()
            #print self.opt.correlation(new_state.R)

        d_err = err - err0

        # re-sort, descending on fitted kmer affinity
        kmer_aff = new_state.params[kmer_indices]
        I = kmer_aff.argsort()[::-1]

        return kmers[I], kmer_indices[I], kmer_aff[I], d_err
        

    def pwm_optimize_hull(self, kmer, keep_pwm=True):
        
        # create a PWM "centered" on the seeding kmer
        pwm = PSAM.from_kmer(kmer)
        kmers, kmer_indices, kmer_aff, d_err = self.optimize_kmer_set_ordered(pwm.kmer_set)
        
        if kmer.lower() != kmers[0].lower():
            # The hull contains a kmer with higher affinity than our initial seed!
            # Select that kmer as hull instead.
            new = kmers[0]
            self.logger.warning("{kmer} can not be seed, because it is not hull-maximal! Switching to {new} which has higher affinity.".format(**locals()))
            return self.pwm_optimize_hull(kmers[0], keep_pwm=keep_pwm)
            
        self.opt.step_scale()
        self.opt.step_betas()

        pwm = PSAM.from_kmer_variants(kmers, np.array(kmer_aff))
        if keep_pwm:
            for mer in kmers:
                self.pwm_by_kmer[mer.lower()] = pwm
            self.pwms[pwm.kmer_seed] = pwm
        
        self.logger.info("pwm_optimize_hull({pwm.kmer_seed})->Kd={pwm.Kd:.2f} nM d_err={d_err:.3e}".format(pwm=pwm, d_err=d_err) )
        
        # output/storage of current results
        print pwm
        return pwm, d_err
    
    def store_params(self):
        self.opt.mdl.store_params(os.path.join(self.opt.opt_path, '{self.k}mer_affinities.tsv'.format(self=self)))

        # temporary: save partition function samples
        self.opt.current.store_Z(os.path.join(self.opt.opt_path, '{self.k}mer_Z1.npy'.format(self=self)))

        for pwm in self.pwms.values():
            pwm_path = os.path.join(
                self.opt.opt_path, 
                '{pwm.kmer_seed}_t={self.t}'.format(self=self, pwm=pwm)
            )
            pwm.store_params(pwm_path)

            logo_path = os.path.join(
                self.opt.opt_path, 
                'pwm_{pwm.kmer_seed}_t={self.t}.eps'.format(**locals()) 
            )
            logo_title = 'Kd={pwm.Kd:.2f} nM'.format(**locals())
            pwm.save_logo(logo_path, title=logo_title)

    def optimize(self, max_iter=1000):
        for t in range(max_iter):
            if not self.next_move():
                break

        self.logger.warning("ending optimization after {self.t} iterations at k={self.k}".format(self=self))
        
    def next_move(self, lag=5):
        corr = self.opt.correlation()
        self.logger.info("{self.k}mer correlations at t={self.t} {corr}".format(**locals()) )
        last_improvements = ",".join(["{0:.3e}".format(i) for i in self.last_improvements[-5:]])
        err0 = self.opt.global_error(self.opt.current.R)
        self.logger.warning("current_error={err0:.2e} last last_improvements: {last_improvements}".format(**locals()) )
        
        if len(self.last_improvements) > lag and np.mean(np.array(self.last_improvements)[-lag:]) > 0:
            self.logger.warning("no reasonable improvements achieved over past {lag} iterations. Switching to k+1={kn}".format(lag=lag, kn=self.k+1))
            self.store_params()
            if self.k < self.k_max:
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
            #kmer = pwm.kmer_seed
        else:
            shift, seed, compound = self.is_shifted(kmer)
            ashift = abs(shift)
            if ashift > 0:
                self.logger.info("{kmer} is {ashift}-SHIFT of HULL({seed}). Recording {compound} for k+{shift}".format(**locals()) )
                self.record_cue(seed, shift, i, compound)
                #keep_pwm = False
            else:
                self.logger.info("{kmer} does not belong to current PWM set. Starting new PWM".format(**locals()) )

        pwm, d_err = self.pwm_optimize_hull(kmer, keep_pwm=keep_pwm)
        err = self.opt.global_error(self.opt.current.R)

        self.t += 1
        self.last_improvements.append(err-err0)

        return pwm
        
    def params_for_k_increase(self, k):
        """
        generate kmer parameters for k+1 by expanding an existing table for k
        """
        new = np.zeros(4**(k+1) + len(self.opt.rbp_conc), dtype=np.float32)
        params = self.opt.mdl.params
        
        # copy over beta values
        new[-self.opt.n_conc:] = params[-self.opt.n_conc:]
        
        # copy each kmer value we presently have, to the 8 k+1 
        # values (4 left-padded, 4 right-padded) in the new array
        nts = np.arange(4)

        # fill new parameters in reverse affinity order, overwriting low
        # with high affinity values in case there is a clash
        I = params[:self.nA].argsort()
        for i in I:
            for nt in nts:
                left = i | (nt << (k*2))
                right = (i << 2) | nt
                new[left] = params[i]
                new[right] = params[i]
        
        return new

    def create_optimizer(self, k, params=[]):
        from cska.rbns_model import ModelOptimization
        # create new optimizer and model
        new_opt = ModelOptimization(k, self.opt.rbns_analysis,
            rbp_conc=self.opt.rbp_conc, 
            out_path=self.opt.out_path, 
            mdl_params = params,
            t0 = self.t
        )
        new_opt.errors = self.opt.errors
        new_opt.correlations = self.opt.correlations
        new_opt.rel_improvements = self.opt.rel_improvements
        
        # some plumbing to make reports/plots contiguous
        new_opt.reporter = self.opt.reporter
        self.opt.reporter.opt = new_opt
        
        return new_opt



    def increase_k(self, cutoff=.01):
        # get new optimizer and model with expanded kmer model parameters
        new_params = self.params_for_k_increase(self.k)
        
        # all parameters that have been changed from background levels
        need_fit = (new_params[:self.opt.nA] <= self.opt.aff0).nonzero()[0]
        
        self.opt = self.create_optimizer(self.k+1, params=new_params)
        self.k = self.k + 1
        self.nA = 4**self.k

        # go over need_fit in order of decreasing prediction error
        n_fit = len(need_fit)
        self.logger.info("switched to k+1 ={self.k} step 1: re-calibration of {n_fit} parameters.".format(**locals()) )
        
        res = self.kmer_residuals()[need_fit]
        need_fit = need_fit[res.argsort()[::-1]]
        self.optimize_kmer_set_ordered(self.opt.mdl.param_name[need_fit])
        
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

        # migrate PWMs, ordered by affinity
        for pwm in pwm_list:
            l, left, r, right = self.digest_cues(pwm.kmer_seed)

            # expand the PWM's seed kmer in the direction indicated by the cues
            new_seed_candidates = np.array(list(expand([pwm.kmer_seed,],l > r)))
            
            # optimize the 4 versions of the extended (k+1) seed kmer
            kmers, indices, aff, d_err = self.optimize_kmer_set_ordered(new_seed_candidates)
            self.opt.step_betas()
            
            a0 = aff[0] # highest affinity comes first
            for kmer, a in zip(kmers, aff)[:1]: # hack, disable pwm split for now!
                if a >= a0 * cutoff:
                    # affinity is within reasonable range of the optimum
                    #hull_kmers = hull(kmer)
                    #hull_indices = np.array([cyska.seq_to_index(mer) for mer in hull_kmers])
                    ##new_pwm = PSAM.from_kmer_variants(hull_kmers, self.opt.mdl.params[hull_indices])
                    
                    self.pwm_optimize_hull(kmer)

        # reset migration cues
        self.kmer_queues = defaultdict(set)

        
        
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
