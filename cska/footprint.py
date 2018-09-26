import numpy as np
import gc, os, sys
import logging
from cska import ensure_path
import cska.caching

# gc.enable()
# gc.set_debug(gc.DEBUG_LEAK)

def dump_garbage():
    """
    show us what's the garbage about
    """
        
    # force collection
    print "\nGARBAGE:"
    gc.collect()

    print "\nGARBAGE OBJECTS:"
    for x in gc.garbage:
        s = str(x)
        if len(s) > 80: s = s[:80]
        print type(x),"\n  ", s


class FootprintCalibration(object):
    def __init__(self, rbns, params, pad=5, thresh=1e-3):
        self.path = ensure_path(os.path.join(rbns.out_path, 'footprint/'))
        self.params = params.copy()
        self.params.acc_k = 0
        self.punp_profiles = []
        self.Z1_in_noacc = None
        self.input_reads = rbns.reads[0]
        self.rbp_conc = rbns.rbp_conc
        self.pad = pad
        self.params.acc_scale = 0
        self.params.non_specific = 0
        self.logger = logging.getLogger('opt.FootprintCalibration')

        Z1 = np.array([reads.PSAM_partition_function(self.params) for reads in rbns.reads])
        self.Z1_in_noacc = Z1[0]
        self.punp_profiles = np.array([
            reads.weighted_accessibility_profile(z, self.params.k, pad=self.pad)[0]
            for reads, z in zip(rbns.reads, Z1)])

        self.logger.debug("plotting punp profiles")
        
        import matplotlib.pyplot as plt
        plt.figure()
        for reads, profile in zip(rbns.reads, self.punp_profiles):
            plt.plot(profile, label=reads.name)

        plt.legend()
        plt.savefig(os.path.join(self.path, 'punp_noacc.pdf'))
        plt.close()

        self.logger.debug("done, freeing some memory")
        for reads in rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()
        
        self.pd_names = [reads.name for reads in rbns.reads[1:]]

        self.I = (self.Z1_in_noacc > thresh).any(axis=1)
        N = self.I.sum()
        self.logger.debug("subsetting to {} reads with Z1 > {}".format(N, thresh) )
        self.Z1 = self.Z1_in_noacc[self.I,:]

    def calibrate(self, k_core_range=[3, None], plot=True):
        # TODO: smarter way to guess footprint size from motif?
        kmin, kmax = k_core_range
        if kmax is None:
            kmax = self.params.k + 2

        results = []
        self.logger.debug("scanning acc_k = {} .. {}".format(kmin, kmax) )
        try:
            for k in range(kmin, kmax+1):
                d = self.params.k - k
                for s in range( 2 - k , d + 2):
                    self.logger.debug("optimizing acc_k={} acc_shift={}".format(k, s) )
                    res, punp_predict = self.optimize(k, s)
                    results.append( (res.fun, k, s, res, punp_predict)  )
                    self.logger.debug("a={res.x[0]} A0={res.x[1]} err={res.fun}".format(res = res) )

                    if plot:
                        self.plot_profiles(punp_predict, k, s, res)
        
        except KeyboardInterrupt:
            pass

        results = sorted(results)
        self.results = results
        self.store_footprints()
        if plot:
            self.matrix_plots(results)

        err, k, s, res, punp_predict = results[0]
        self.params.acc_k = k
        self.params.acc_shift = s
        self.params.acc_scale = res.x[0]
        self.params.A0 = res.x[1]

        file(os.path.join(self.path, 'calibrated.tsv'), 'w').write(str(self.params) + '\n')
        return self.params
    

    def store_footprints(self):
        with file(os.path.join(self.path, 'footprints.tsv'), 'w') as f:
            f.write('acc_k\tacc_shift\tacc_scale\tA0\terror')
            for opt in self.results:
                err, k, s, res, punp_predict = opt
                out = [k, s, res.x[0], res.x[1], err]
                f.write("\t".join([str(o) for o in out]) + "\n")


    def plot_profiles(self, punp_expect, acc_k, acc_shift, res):
        import seaborn as sns
        import matplotlib.pyplot as plt
        pwm = self.params.as_PSAM()
        x = np.arange(-self.pad, pwm.n + self.pad )

        plt.figure()
        plt.title("acc_k = {acc_k} acc_shift = {acc_shift}".format(**locals()))
        colors = sns.color_palette("husl", 8)
        for pred, obs, name, color in zip(punp_expect, self.punp_profiles[1:], self.pd_names, colors):
            plt.plot(x, obs, '-', color=color, label=name)
            plt.plot(x, pred, ':', color=color, label="fit a={res.x[0]:.2f} A0={res.x[1]:.2f} err={res.fun:.2e}".format(res=res))

        plt.legend()

        cons = pwm.consensus
        plt.xticks(x, [str(p) for p in range(-self.pad,0)] + list(cons) + [str(p) for p in range(1, self.pad+1)])
        plt.axvline( - .5)
        plt.axvline(pwm.n - .5)

        plt.ylabel(r"$P_{unpaired}$ (motif-weighted)")
        plt.xlabel('pos. rel to motif (consensus) [nt]')

        fname = os.path.join(self.path, '{acc_k}_{acc_shift}.pdf'.format(**locals()))
        self.logger.debug("saving plot: '{}'".format(fname))
        plt.savefig(fname)
        plt.close()

    def matrix_plots(self, results):
        import seaborn as sns
        import matplotlib.pyplot as plt

        results = [(acc_k, acc_shift, res.fun, res.x[0]) for err, acc_k, acc_shift, res, punp in results]
        acc_k, acc_shift, err, a = np.array(results).T
        k_base = int(acc_k.min())
        n_k = int(acc_k.max()) - k_base + 1
        shift = int(np.fabs(acc_shift).max())
        n_shift = shift * 2 + 1 
        mid_shift = shift

        mat_a = np.zeros((n_k, n_shift), dtype=float) + np.NaN
        mat_err = np.zeros((n_k, n_shift), dtype=float) + np.NaN
        for acc_k, acc_shift, err, a in results:
            mat_a[acc_k - k_base, acc_shift + mid_shift] = a
            mat_err[acc_k - k_base, acc_shift + mid_shift] = np.log10(err)

        fig = plt.figure(figsize=(6,6))
        # fig.suptitle("accessibility footprint analysis")
        plt.subplot(211)
        plt.pcolor(mat_err, cmap="viridis")
        plt.colorbar(label=r'accessibility error [$\log_{10}$]', fraction=.05)

        plt.ylabel("size [nt]")
        plt.xlabel("shift [nt]")

        plt.xticks(np.arange(n_shift)+.5, [str(s) for s in range(-shift, shift+1)])
        plt.yticks(np.arange(n_k)+.5, [str(k) for k in range(k_base, k_base + n_k)])

        plt.subplot(212)
        plt.pcolor(mat_a, cmap="inferno")
        plt.colorbar(label=r'accessibility scaling', fraction=.05)

        plt.ylabel("size [nt]")
        plt.xlabel("shift [nt]")

        plt.xticks(np.arange(n_shift)+.5, [str(s) for s in range(-shift, shift+1)])
        plt.yticks(np.arange(n_k)+.5, [str(k) for k in range(k_base, k_base + n_k)])

        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'footprint.pdf'))



    def optimize(self, acc_k, acc_shift):
        import cska.cyska as cyska
        from cska.sc import SelfConsistency
        from time import time
        from pympler.tracker import SummaryTracker
        tracker = SummaryTracker()

        openen = self.input_reads.acc_storage.get_raw(acc_k)
        openen_punp = self.input_reads.acc_storage.get_raw(1)

        self.input_reads.acc_storage.cache_flush() # free up memory
        acc0 = openen.acc[self.I, :]
        punp = openen_punp.acc[self.I, :]
        # print "log"
        ofs = openen.ofs - self.params.k + 1
        zw = self.Z1.shape[1]
        lacc0 = np.log(acc0)

        self.params.acc_k = acc_k
        self.params.acc_shift = acc_shift
        self.params.non_specific = 0.

        def predict_profiles(a, A0):
            t0 = time()
            acc1 = np.exp(lacc0 * a) # scale accessibilities
            ofs = openen.ofs - self.params.k + 1 + acc_shift
            Z1_acc = self.Z1 * acc1[:, ofs:ofs + zw]
            t1 = time()
            Z1_read, Z1_read_max = cyska.clipped_sum_and_max(Z1_acc, clip=1E6) # aggregate to read-level
            t2 = time()
            sc = SelfConsistency(Z1_read, self.input_reads.rna_conc, bins=1000)
            rbp_free = sc.free_rbp_vector(self.rbp_conc, Z_scale=A0)
            # print "rbp_free", rbp_free
            t3 = time()
            psi = cyska.p_bound(Z1_read, rbp_free*A0)
            # print "psi", psi.min(axis=1), psi.max(axis=1)
            # print psi.shape
            t4 = time()
            # punp_expect = self.input_reads.weighted_accessibility_profile(self.Z1, self.params.k, pad=self.pad, row_w=psi)
            punp_expect = cyska.acc_footprints(self.Z1, punp, self.params.k, 1, openen_punp.ofs - self.params.k + 1, pad=self.pad, row_w = psi)
            t_prof = time() - t4

            times = 1000 * np.array([t1-t0, t2-t1, t3-t2, t4-t3, t_prof])
            # print "t_partfunc={:.2f} t_Zread={:.2f} t_sc={:.2f} t_psi={:.2f} t_prof={:.2f}".format(*times)
            return np.array(punp_expect)

        def to_opt(args):
            a, A0 = args

            punp_expect = predict_profiles(a, A0)
            err = np.sum((punp_expect - self.punp_profiles[1:])**2)
            return err

        from scipy.optimize import minimize
        res = minimize(
            to_opt, 
            (0, self.params.A0),  # start with no secondary structure data A0
            bounds= [ (0, 1.), (1e-3, 1000.)], 
            options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
        )
        a, A0 = res.x
        punp_expect = predict_profiles(a, A0)
        gc.collect()
        return res, punp_expect

        


        
        

        


