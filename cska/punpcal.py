import numpy as np

class PunpairedCalibrate(object):
    def __init__(self, rbns, params, pad=5, thresh=1e-4, k_core_range=[3, None]):
        self.params = params.copy()
        self.params.acc_k = 0
        self.punp_profiles = []
        self.Z1_in_noacc = None
        self.input_reads = rbns.reads[0]
        self.rbp_conc = rbns.rbp_conc
        self.pad = pad
        print "getting punp profiles"
        for reads in rbns.reads:
            Z1 = reads.PSAM_partition_function(self.params)
            if reads.rbp_conc == 0:
                self.Z1_in_noacc = Z1
            
            self.punp_profiles.append(reads.weighted_accessibility_profile(Z1, self.params.k, pad=self.pad))
        self.punp_profiles = np.array(self.punp_profiles)

        print "done, freeing some memory"
        for reads in rbns.reads[1:]:
            reads.cache_flush()
            reads.acc_storage.cache_flush()

        print "thresholding for reads with useful sites"
        # self.I = (self.Z1_in_noacc > thresh).any(axis=1)
        self.I = np.ones(self.Z1_in_noacc.shape[0], dtype=np.bool)
        N = self.I.sum()
        print "getting Z1 and seqm for {} sites".format(N)
        self.Z1 = self.Z1_in_noacc[self.I,:]
        self.seqm = self.input_reads.get_padded_seqm(params.k)[self.I,:]

        # TODO: smarter way to guess footprint size from motif?
        kmin, kmax = k_core_range
        if kmax is None:
            kmax = params.k + 2

        results = []
        print kmin, kmax+1
        for k in range(kmin, kmax+1):
            d = self.params.k - k
            # for s in range( -1, self.params.k - k + 2):
            for s in range(1,2):
                print "testing acc_k={} acc_shift={} getting acc".format(k, s)

                res, punp_predict = self.optimize(k, s)
                print res
                results.append( (res.fun, k, s, res, punp_predict)  )
        
        results = sorted(results)
        for opt in results:
            err, k, s, res, punp_predict = opt
            print k, s, '->', err
        
        err, k, s, res, punp_predict = results[0]
        import matplotlib.pyplot as plt
        for pred, obs, color in zip(punp_predict, self.punp_profiles[1:], ['r','b','g']):
            plt.plot(obs, color+'-')
            plt.plot(pred, color+':')

        plt.savefig('yada.pdf')
    
    def optimize(self, acc_k, acc_shift):
        import cska.cyska as cyska
        from cska.sc import SelfConsistency

        openen = self.input_reads.acc_storage.get_raw(acc_k)
        acc0 = openen.acc[self.I,:]
        # print "log"
        ofs = openen.ofs - self.params.k + 1
        zw = self.Z1.shape[1]
        lacc0 = np.log(acc0)

        self.params.acc_k = acc_k
        self.params.acc_shift = acc_shift
        self.params.non_specific = 0.

        def predict_profiles(a, A0):
            # Z1_acc = self.Z1 * np.exp(lacc0[:, ofs + acc_shift:ofs + acc_shift + zw] * a)
            # acc1 = cyska.pow_scale(acc0, a)
            self.params.acc_scale = a
            Z1_acc = self.input_reads.PSAM_partition_function(self.params)
            Z1_read, Z1_read_max = cyska.clipped_sum_and_max(Z1_acc, clip=1E6) # aggregate to read-level

            sc = SelfConsistency(Z1_acc, self.input_reads.rna_conc, bins=1000)

            rbp_free = np.array([sc.free_rbp(total, Z_scale=A0) for total in self.rbp_conc], dtype=np.float32)
            psi = cyska.p_bound(Z1_read, rbp_free*A0)

            punp_expect = []
            for i, conc in enumerate(rbp_free):
                p = psi[i]
                w = p[:,np.newaxis] * self.Z1
                punp_expect.append( self.input_reads.weighted_accessibility_profile(w, self.params.k, pad=self.pad) )

            return np.array(punp_expect)

        def to_opt(args):
            a, A0 = args

            punp_expect = predict_profiles(a, A0)
            err = np.sum((punp_expect - self.punp_profiles[1:])**2)

            # print 
            print 'a,A0', a, A0, '->', err
            return err

        from scipy.optimize import minimize
        res = minimize(to_opt, (.25, .8), bounds= [ (1e-3, 1.), (1e-3, 1000.)], options=dict(eps=1e-4, maxiter=100))
        a, A0 = res.x
        punp_expect = predict_profiles(a, A0)

        # class bla(object):
        #     pass

        # res = bla()
        # a, A0 = (.25, .8)
        # res.x = (a, A0)
        # res.fun = -1
        # punp_expect = predict_profiles(a, A0)

        return res, punp_expect

        


        
        

        


