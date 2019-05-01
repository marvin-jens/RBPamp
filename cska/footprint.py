import numpy as np
import gc
import os
import sys
import shelve
import logging
from cska import ensure_path
from cska.caching import pickled, cached, monitored, CachedBase, get_cache_sizes
import cska.cyska as cyska
from cska.sc import SelfConsistency

from scipy.optimize import minimize
from time import time
from copy import deepcopy
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


def dump_caches():
    for size, cache in get_cache_sizes():
        print cache, size/1024.


class RowOptimization(object):
    def __init__(self, cal, acc_k):
        self.logger = logging.getLogger("opt.footprint.row_opt")
        self.cal = cal
        self.acc_k = acc_k
        self.params = deepcopy(self.cal.params)
        self.params.acc_k = acc_k
        
        self.lacc0, self.punp1 = self.cal.get_lacc_punp_cached(acc_k)
        self.openen = self.cal.get_input_openen_cached(acc_k)
        self.openen_punp = self.cal.get_input_openen_cached(1)
        self.Z1 = self.cal.Z1
        self.zw = self.Z1.shape[1]

    def predict_profiles(self, acc_shift, a, A0):
        from time import time
        t0 = time()
        # scale accessibilities
        acc1 = np.exp(self.lacc0 * np.float32(a))
        
        # select shifted accessibilities
        self.params.acc_shift = acc_shift
        ofs = self.openen.ofs - self.params.k + 1 + acc_shift
        if ofs < 0:
            self.logger.warning("ofs underflow")

        # compute partition function with scaled acc.
        Z1_acc = self.Z1 * acc1[:, ofs:ofs + self.zw]

        # aggregate to read-level
        t1 = time()
        Z1_read, Z1_read_max = cyska.clipped_sum_and_max(Z1_acc, clip=1E6)
        t2 = time()

        # predict binding probability
        sc = SelfConsistency(Z1_read, self.cal.input_reads.rna_conc, bins=1000)
        rbp_free = sc.free_rbp_vector(self.cal.rbp_conc, Z_scale=A0)
        t3 = time()

        psi = cyska.p_bound(Z1_read, np.array(rbp_free*A0,dtype=np.float32))
        t4 = time()

        # compute base-p_unpaired profiles
        punp_expect = cyska.acc_footprints(
            self.Z1,
            self.punp1,
            self.params.k, 
            1, 
            self.openen_punp.ofs - self.params.k + 1, 
            pad=self.cal.pad, 
            row_w = np.ascontiguousarray(psi.T),
            n_threads=1
        )
        t_prof = time() - t4

        times = 1000 * np.array([t1-t0, t2-t1, t3-t2, t4-t3, t_prof])
        self.logger.debug("t_partfunc={:.2f} t_Zread={:.2f} t_sc={:.2f} t_psi={:.2f} t_prof={:.2f}".format(*times))
        return np.array(punp_expect)

    def optimize_a(self, acc_shift, A0=None):
        if A0 is None:
            A0 = self.params.A0
        
        self.logger.debug("optimize_a({acc_shift}, A0={A0})".format(**locals()))
        def to_opt_a(a):
            punp_expect = self.predict_profiles(acc_shift, a, A0)
            err = np.sum((punp_expect - self.cal.punp_profiles[1:])**2)
            return err

        from scipy.optimize import minimize
        res = minimize(
            to_opt_a,
            (0, ),  # start with no secondary structure data A0
            bounds= [ (0, 1.), ], 
            options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
        )
        res.a = res.x
        res.A0 = A0
        punp_expect = self.predict_profiles(acc_shift, res.a, res.A0)
        return res, punp_expect

    def optimize_A0(self, acc_shift, a=1):
        self.logger.debug("optimize_A0({acc_shift}, a={a})".format(**locals()))
        def to_opt_A0(A0):
            punp_expect = self.predict_profiles(acc_shift, a, A0)
            err = np.sum((punp_expect - self.cal.punp_profiles[1:])**2)
            return err

        res = minimize(
            to_opt_A0,
            (1., ),
            bounds= [ (1e-4, 1e3), ], 
            options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
        )
        res.a = a
        res.A0 = res.x
        punp_expect = self.predict_profiles(acc_shift, res.a, res.A0)
        return res, punp_expect

    def optimize(self, acc_shift):
        self.logger.debug("optimize({acc_shift})".format(**locals()))
        def to_opt(args):
            a, A0 = args
            punp_expect = self.predict_profiles(acc_shift, a, A0)
            err = np.sum((punp_expect - self.cal.punp_profiles[1:])**2)
            return err

        res = minimize(
            to_opt,
            (0, 1., ),
            bounds= [ (0, 1), (1e-4, 1e3), ], 
            options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
        )
        res.a = res.x[0]
        res.A0 = res.x[1]
        punp_expect = self.predict_profiles(acc_shift, res.a, res.A0)
        return res, punp_expect



class FootprintCalibration(CachedBase):
    def __init__(self, rbns, params, pad=5, thresh=1e-3, subsample=False, redo=False):
         
        CachedBase.__init__(self)

        # print ">>> before initialization"
        # dump_caches()
        self.path = ensure_path(os.path.join(rbns.out_path, 'footprint/'))
        self.params = params.copy()
        self.params.acc_k = 0
        self.params.acc_scale = 0
        self.params.non_specific = 0
        self.consensus = self.params.as_PSAM().consensus
        self.consensus_ul = self.params.as_PSAM().consensus_ul
        self.rbns = rbns
        self.input_reads = rbns.reads[0]
        self.subsample = subsample
        self.pd_names = [reads.name for reads in rbns.reads[1:]]
        self.rbp_conc = rbns.rbp_conc
        self.pad = pad
        self.thresh = thresh
        self.logger = logging.getLogger('opt.FootprintCalibration({})'.format(self.consensus))
        self.result_log = logging.getLogger('results.footprint')

        self.shelve = shelve.open(
            os.path.join(self.path, "history"), 
            protocol=-1, 
            flag='n' if redo else 'c'
        )
        self.shelve["rbp_conc"] = self.rbp_conc

        self.results = {}
        self._openen_cache = {}
        self._lacc_cache = {}

        fp = os.path.join(self.path, 'footprints_{}.tsv'.format(self.consensus))
        # if os.path.exists(fp):
        #     self.load_footprints(fp)
        # no need to load these, as we now keep pickled results from optimize()

        self.fp_file = file(fp, 'w')
        self.fp_file.write('acc_k\tacc_shift\tacc_scale\tA0\terror\n')

        # Z1 = np.array([reads.PSAM_partition_function(self.params) for reads in rbns.reads])
        self.logger.debug("evaluating partition function")
        from cska.params import ModelSetParams
        self.Z1_full = np.array([reads.PSAM_partition_function(ModelSetParams([self.params, ]), subsample=self.subsample) for reads in rbns.reads])
        self.Z1_in_noacc = self.Z1_full[0]
        
        Z1_read = self.Z1_in_noacc.sum(axis=1)
        thresh = self.thresh * Z1_read.max()
        self.I = Z1_read > thresh
        N = self.I.sum()
        self.logger.debug("subsetting to {} reads with Z1 > {}".format(N, thresh) )
        self.Z1 = self.Z1_in_noacc[self.I,:]

        for reads in rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()

        self.punp_profiles, self.naive_profiles = self.compute_initial_profiles()

        self.store_shelve("params_initial", params)
        self.store_shelve("punp_profiles", self.punp_profiles)
        self.store_shelve("naive_profiles", self.naive_profiles)

        # self.logger.debug("plotting naive punp profiles")
        # self.plot_profiles(self.naive_profiles, 0, 0, None)
        self.err0 = np.sum((self.naive_profiles - self.punp_profiles[1:])**2)
        self.logger.debug("naive error: {}".format(self.err0))

        un_opt = (self.err0, 0, 0, 0, self.params.A0)
        self.results[(0, 0)] = un_opt
        self.store_footprint(un_opt)

        self.logger.debug("done, freeing some memory")
        for reads in rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()

    def store_shelve(self, key, value):
        self.shelve["{0}_{1}".format(self.consensus_ul, key)] = value
        self.shelve.sync()        

    def load_shelve(self, key, default=None):
        k = "{0}_{1}".format(self.consensus_ul, key)
        if k in self.shelve:
            return self.shelve[k]
        return default

    def load_profile(self, k, s):
        key = "opt_profile_{k}_{s}".format(k=k, s=s)
        return self.load_shelve(key)

    def store_profile(self, k, s, value):
        key = "opt_profile_{k}_{s}".format(k=k, s=s)
        return self.store_shelve(key, value)

    @property
    def cache_key(self):
        return "{self.params}.{self.rbp_conc}.{self.input_reads.cache_key}.{self.subsample}.{self.thresh}".format(self=self)

    def optimize_row(self, acc_k, shift_range, from_scratch=False):
        from multiprocessing.pool import ThreadPool
        
        row = RowOptimization(self, acc_k)
        pool = ThreadPool()

        def _optimize(s):
            x = self.load_profile(acc_k, s)
            if not x or from_scratch:
                x = row.optimize(s)

            return (acc_k, s, x)

        results = pool.map(_optimize, shift_range, chunksize=1)
        pool.terminate()
        pool.close()
        pool.join()
        return results
        # return map(_optimize, shift_range)


    def calibrate(self, k_core_range=[3, None], plot=True, pad=5, from_scratch=False, heuristic=0):
        kmin, kmax = k_core_range
        if kmax is None:
            kmax = self.params.k + 2

        self.logger.debug("scanning acc_k = {} .. {}".format(kmin, kmax) )
        for acc_k in range(kmax, kmin - 1, -1):
            d = self.params.k - acc_k + 1
            shift_range = range(-pad, d + pad)

            for k, s, (res, punp_predict) in self.optimize_row(acc_k, shift_range, from_scratch):
                err = res.fun
                rel_err = err / self.err0
                a = res.a
                A0 = res.A0

                if not res.success:
                    self.logger.warning("unable to optimize footprint k={k}, s={s}.".format(**locals()))
                    a = 0.
                    A0 = self.params.A0
                    err = self.err0
                
                opt = (err, k, s, a, A0)
                self.results[(k, s)] = opt
                self.store_profile(k, s, (punp_predict, res))
                self.store_footprint(opt)
                self.result_log.info("{self.consensus} k={k} s={s} a_opt={a} A0_opt={A0} err={err} rel_err={rel_err}".format(**locals()) )


        results = sorted(self.results.values())
        err, k, s, a, A0 = results[0]
        rel_err = err/self.err0
        self.result_log.critical("OPTIMUM {self.consensus} k={k} s={s} a_opt={a} A0_opt={A0} err={err} rel_err={rel_err}".format(**locals()) )
        self.params.acc_k = k
        self.params.acc_shift = s
        self.params.acc_scale = a
        self.params.A0 = A0
        self.store_shelve("params_calibrated", self.params)

        # for the optimum, also compute profile for a=1
        row = RowOptimization(self, k)
        res_a_one, punp_a_one = row.optimize_A0(s, a=1)
        self.store_shelve("opt_profile_a_one_{k}_{s}".format(**locals()), (punp_a_one, res_a_one))

        return self.params

    def store_footprint(self, opt):
        err, k, s, a, A0 = opt
        out = [k, s, a, A0, err]
        self.fp_file.write("\t".join([str(o) for o in out]) + "\n")
        self.fp_file.flush()
        self.store_shelve("{0}_{1}".format(k, s), opt)

    # @monitored
    @pickled
    def compute_initial_profiles(self):

        punp_profiles = np.array([
            reads.weighted_accessibility_profile(z, self.params.k, pad=self.pad, subsample=self.subsample)[0]
            for reads, z in zip(self.rbns.reads, self.Z1_full)])

        # predict profiles w/o accessibility footprint
        Z1_read, Z1_read_max = cyska.clipped_sum_and_max(self.Z1_in_noacc, clip=1E6) # aggregate to read-level
        sc = SelfConsistency(Z1_read, self.input_reads.rna_conc, bins=1000)
        rbp_free = sc.free_rbp_vector(self.rbp_conc, Z_scale=self.params.A0)
        psi = cyska.p_bound(Z1_read, rbp_free*self.params.A0)

        openen_punp = self.get_input_openen_cached(1)
        punp_acc = openen_punp.acc
        if self.subsample:
            punp_acc = self.input_reads.sub_sampler.draw(data=punp_acc)

        naive_profiles = cyska.acc_footprints(
            self.Z1_in_noacc, 
            punp_acc, 
            self.params.k, 
            1, 
            openen_punp.ofs - self.params.k + 1, 
            pad=self.pad, 
            row_w = np.ascontiguousarray(psi.T),
        )

        return punp_profiles, naive_profiles

    def get_input_openen_cached(self, k):
        if not k in self._openen_cache:
            self.logger.debug("get_input_openen_cached({}) not found".format(k))
            self._openen_cache[k] = self.input_reads.acc_storage.get_raw(k, _do_not_cache=True)
            # self.input_reads.acc_storage.cache_flush() # free up memory
        
        for x in self._openen_cache.keys():
            # drop everything that's not p-unpaired or current k
            if x > 1 and x != k and k > 1:
                self.logger.debug("get_input_openen_cached({}) dropping {}".format(k, x))
                self._openen_cache[x].cache_flush()
                del self._openen_cache[x]

        return self._openen_cache[k]

    def get_lacc_punp_cached(self, k):
        if not k in self._lacc_cache:
            self.logger.debug("get_lacc_punp_cached({}) not found".format(k))
            openen = self.get_input_openen_cached(k)
            openen_punp = self.get_input_openen_cached(1)

            acc0 = openen.acc
            punp = openen_punp.acc

            if self.subsample:
                acc0 = self.input_reads.sub_sampler.draw(data=acc0)
                punp = self.input_reads.sub_sampler.draw(data=punp)

            acc0 = acc0[self.I, :]
            punp = punp[self.I, :]

            lacc0 = np.log(acc0)
            self._lacc_cache = { k : (lacc0, punp) }  # always keep only one item!
        
        return self._lacc_cache[k]

    def close(self):
        for reads in self.rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()

        self.shelve.close()
        import cska.caching
        cska.caching._dump_cache_sizes()
        import gc
        gc.collect()