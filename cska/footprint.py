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
        self.Z1_full = np.array([reads.PSAM_partition_function(ModelSetParams([self.params,]), subsample=self.subsample) for reads in rbns.reads])
        self.Z1_in_noacc = self.Z1_full[0]

        self.I = (self.Z1_in_noacc > self.thresh).any(axis=1)
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
        self.shelve["{0}_{1}".format(self.consensus, key)] = value
        self.shelve.sync()        

    @property
    def cache_key(self):
        return "{self.params}.{self.rbp_conc}.{self.input_reads.cache_key}.{self.subsample}.{self.thresh}".format(self=self)

    def calibrate(self, k_core_range=[3, None], plot=True, pad=5, from_scratch=False, heuristic=2):
        # TODO: smarter way to guess footprint size from motif?
        kmin, kmax = k_core_range
        if kmax is None:
            kmax = self.params.k + 2

        self.logger.debug("scanning acc_k = {} .. {}".format(kmin, kmax) )
        try:
            if from_scratch:
                for k in range(kmin, kmax+1):
                    d = self.params.k - k
                    for s in range( -pad , d + pad):
                        self.drop_pickle("optimize", k, s)

            lowest_err = np.Inf
            h_map = None
            for k in range(kmax, kmin-1, -1):
                d = self.params.k - k + 1
                l = d + 2*pad + 1
                k_row = np.zeros(l, dtype=float) + np.Inf
                for s in range(-pad, d + pad):
                    # if not (k, s) in self.results:
                    if not h_map is None:
                        print "testing", s, h_map[s + pad]
                        if not h_map[s + pad]:
                            print "skip"
                            continue

                    self.logger.debug("optimizing acc_k={} acc_shift={}".format(k, s) )
                    res, punp_predict = self.optimize(k, s)
                    err = res.fun
                    rel_err = err / self.err0
                    a = res.x[0]
                    A0 = res.x[1]
                    if not res.success:
                        self.logger.warning("unable to optimize footprint k={k}, s={s}.".format(**locals()))
                        a = 0.
                        A0 = self.params.A0
                        err = self.err0
                    
                    opt = (err, k, s, a, A0)

                    self.results[(k, s)] = opt
                    self.store_shelve("opt_profile_{k}_{s}".format(**locals()), (punp_predict, res))
                    self.store_footprint(opt)
                    # self.logger.debug("a={a} A0={A0} err={err}".format(**locals()) )
                    self.result_log.info("{self.consensus} k={k} s={s} a_opt={a} A0_opt={A0} err={err} rel_err={rel_err}".format(**locals()) )
                    #     self.plot_profiles(punp_predict, k, s, res)

                    k_row[s + pad] = err
                    lowest_err = min(lowest_err, err)

                if heuristic:
                    print k_row
                    I = (k_row <= 1.1 * k_row.min()).nonzero()[0]
                    print I
                    h_map = np.zeros(l+1, dtype=bool)
                    for i in I:
                        # set the optimum and nearest neighbors to True
                        h_map[max(0, i-1):min(l+2, i + 3)] = True
                        print "heuristic map: good site at", i - pad, k_row[i]

                    print "heuristic map for k=", k-1
                    for x in range( -pad , d + pad + 1):
                        print x, h_map[x + pad]

                # if k_row.min() > lowest_err and heuristic:
                #     # we went past the optimum. Everything smaller than this is 
                #     # not interesting
                #     self.logger.debug("breaking early because optimum was already seen at higher k")
                #     break
    
        except KeyboardInterrupt:
            self.logger.warning("received KeyboardInterrupt")
            raise
            return False

        results = sorted(self.results.values())
        # self.store_footprints()
        # if plot:
        #     self.matrix_plots(results)

        err, k, s, a, A0 = results[0]
        rel_err = err/self.err0
        self.result_log.critical("OPTIMUM {self.consensus} k={k} s={s} a_opt={a} A0_opt={A0} err={err} rel_err={rel_err}".format(**locals()) )
        self.params.acc_k = k
        self.params.acc_shift = s
        self.params.acc_scale = a
        self.params.A0 = A0
        self.store_shelve("params_calibrated", self.params)

        # for the optimum, also compute profile for a=1
        res_a_one, punp_a_one = self.optimize(k, s, a_fixed=True)
        self.store_shelve("opt_profile_a_one_{k}_{s}".format(**locals()), (punp_a_one, res_a_one))

        # path = os.path.join(self.path, '{}_calibrated.tsv'.format(self.consensus))
        # self.logger.info("storing footprinted model in '{}'".format(path))
        # file(path, 'w').write(str(self.params) + '\n')
        return self.params
    

    # def load_footprints(self, fp):
    #     for line in file(fp).readlines()[1:]:
    #         k, s, a, A0, err = line.split('\t')
    #         k = int(k)
    #         s = int(s)
    #         self.results[(k, s)] = ( float(err), k, s, float(a), float(A0) )

    #     self.logger.debug("loaded {} footprint records from '{}'".format(len(self.results),fp))


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

        # from cska import vector_stats
        # print "Z_read"
        # vector_stats(Z1_read)
        # print "rbp_free * A0", rbp_free*self.params.A0
        # print "PSIs"
        # [vector_stats(p) for p in psi]

        # openen_punp = self.input_reads.acc_storage.get_raw(1)
        openen_punp = self.get_input_openen_cached(1)
        punp_acc = openen_punp.acc
        if self.subsample:
            punp_acc = self.input_reads.sub_sampler.draw(data=punp_acc)
        naive_profiles = cyska.acc_footprints(self.Z1_in_noacc, punp_acc, self.params.k, 1, openen_punp.ofs - self.params.k + 1, pad=self.pad, row_w = psi)

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

    # @monitored
    @pickled
    def optimize(self, acc_k, acc_shift, a_fixed=False):
        from time import time
        self.logger.info("performing calibration for k={} s={}".format(acc_k, acc_shift))
        # from pympler.tracker import SummaryTracker
        # tracker = SummaryTracker()
        if a_fixed:
            acc0 = self.get_input_openen_cached(acc_k).acc
            punp = self.get_input_openen_cached(1).acc
            if self.subsample:
                acc0 = self.input_reads.sub_sampler.draw(data=acc0)
                punp = self.input_reads.sub_sampler.draw(data=punp)
            acc0 = acc0[self.I, :]
            punp = punp[self.I, :]
        else:
            lacc0, punp = self.get_lacc_punp_cached(acc_k)

        openen_punp = self.get_input_openen_cached(1)
        openen = self.get_input_openen_cached(acc_k)
        ofs = openen.ofs - self.params.k + 1
        zw = self.Z1.shape[1]

        from copy import deepcopy
        params = deepcopy(self.params)
        params.acc_k = acc_k
        params.acc_shift = acc_shift

        def predict_profiles(a, A0):
            t0 = time()
            if not a_fixed:
                acc1 = np.exp(lacc0 * a) # scale accessibilities
            else:
                acc1 = acc0

            ofs = openen.ofs - params.k + 1 + acc_shift

            Z1_acc = self.Z1 * acc1[:, ofs:ofs + zw]
            t1 = time()
            Z1_read, Z1_read_max = cyska.clipped_sum_and_max(Z1_acc, clip=1E6) # aggregate to read-level
            t2 = time()
            sc = SelfConsistency(Z1_read, self.input_reads.rna_conc, bins=1000)
            rbp_free = sc.free_rbp_vector(self.rbp_conc, Z_scale=A0)
            # print "rbp_free", rbp_free
            t3 = time()
            psi = cyska.p_bound(Z1_read, np.array(rbp_free*A0,dtype=np.float32))
            # print "psi", psi.min(axis=1), psi.max(axis=1)
            # print psi.shape
            t4 = time()
            # punp_expect = self.input_reads.weighted_accessibility_profile(self.Z1, self.params.k, pad=self.pad, row_w=psi)
            punp_expect = cyska.acc_footprints(self.Z1, punp, self.params.k, 1, openen_punp.ofs - params.k + 1, pad=self.pad, row_w = psi)
            t_prof = time() - t4
            # print self.Z1.shape, punp.shape
            times = 1000 * np.array([t1-t0, t2-t1, t3-t2, t4-t3, t_prof])
            print "t_partfunc={:.2f} t_Zread={:.2f} t_sc={:.2f} t_psi={:.2f} t_prof={:.2f}".format(*times)
            return np.array(punp_expect)

        def to_opt(args):
            if a_fixed:
                A0 = args[0]
                punp_expect = predict_profiles(1., A0)
            else:
                a, A0 = args
                punp_expect = predict_profiles(a, A0)
            
            err = np.sum((punp_expect - self.punp_profiles[1:])**2)
            return err

        from scipy.optimize import minimize
        if a_fixed:
            res = minimize(
                to_opt, 
                (params.A0,),  
                bounds= [ (1e-3, 1000.), ], 
                options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
            )
            A0 = res.x
            a = 1.
            res.x = (1., A0)
        else:
            res = minimize(
                to_opt, 
                (0, params.A0),  # start with no secondary structure data A0
                bounds= [ (0, 1.), (1e-3, 1000.)], 
                options=dict(eps=1e-4, maxiter=100, ftol=1e-5)
            )
            a, A0 = res.x

        punp_expect = predict_profiles(a, A0)
        gc.collect()
        return res, punp_expect

    def close(self):
        for reads in self.rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()

        self.shelve.close()
        import cska.caching
        cska.caching._dump_cache_sizes()
        import gc
        gc.collect()