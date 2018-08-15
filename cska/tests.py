import sys
import numpy as np
import unittest
from cska.reads import RBNSReads
from cyska import *

def get_test_reads(adap5='CCCCCCC', adap3='TTTTTTTT'):
    return RBNSReads("""GAGGTCACTCTCTTGCATGTATGCATGCAGTCTCAACGAA
CATTTTTTTTGAAACTACGTGCATGTACAATAGGCGACGA
ATACGACTGCTTATCCACTGCATGCTGCAGGAAGTGCATG
TTGTAGCATGTCCGTCAACAGAAACTGCATGTTTCTAATA
CATCCAATGCATGTTTCGGTTCTCTGACAAACCTTCCCTT
AATATATCTCGACGTTGCATGTAATTACCACATAAAAACG
ATTAACCGTAACGCCATAAGCGCGCACCTACTGCATGTTT
ATCCGATGAGCAATAGTGAGAAAAAACATACTGCATGTAT
ATTAAATGCAAGTCGTCGGAGCCTGCATGTACTAAATTAG
AATTAACCGTCTGCATGTGGACGTCCAATTAAACAAATGT
GACGGGTGTTACTGCATGTTAGGATCCCCGTGCATGACAG
TAAATGGCGGGTACCCTTGTGAACGGTACGGATGCATGTG
GGTGGCAGCTTGAATTCTGCGAGTGCATGTGAATACATAT
GGAGCAGATGCATGTGTCCCCGGAGTGGAAATAGGGTCCA
TTTCGTATCTCATGCATGTACACTATCAGCTGTGAAAAAT
GAATTAAACATTGCATGTATGGGTAGGATGGAAATCCCAC
TCATGCATGTTTGATTATAACTGGTAAGTCCTGTACACGT
TCCAGAATTAGTGCATGTAGGAGAAACACACGATATTGAT
GAGGAAAATAACGTGCATGTCCCACTTTAAATATATAGCA
GAATGCAGTCCGGCGCTTTAATGCATGTGCATCCTATACT
""".split('\n'), adap5=adap5, adap3=adap3)
    

real_reads = {}
def get_real_reads(N =100000):
    if not N in real_reads:
        real_reads[N] = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=N, pseudo_count=1e-3)

    return real_reads[N]

class TestPhysModel(unittest.TestCase):

    def test_clipped_sum_and_max(self, N = 100000, L = 28, thresh=10000., rnd_seed=47110815, n = 100):
        from time import time
        np.random.seed(rnd_seed)
        Z = np.array(np.exp(4*np.random.random( (N,L))), dtype=np.float32)
        
        t0 = time()
        for i in range(n):
            Zsum = np.clip(Z.sum(axis=1), None, thresh)
            Zmax = Zsum.max()
        t1 = time()

        from cska.cyska import clipped_sum_and_max
        for i in range(n):
            zsum, zmax = clipped_sum_and_max(Z, clip=thresh)

        t2 = time()

        # print "timing {0:.2f} ms vs {1:.2f} ms".format(1000./n* (t1-t0), 1000./n * (t2-t1))
        self.assertTrue(np.allclose(Zsum, zsum))
        self.assertTrue(np.allclose(zmax, Zmax))


    def test_weighted_kmer_counts(self, N = 100000, k=6, rnd_seed=47110815, n = 100):
        from time import time
        np.random.seed(rnd_seed)
        reads = get_real_reads(N)
        w = np.array(np.random.random( reads.N ), dtype=np.float32)
        
        im = reads.get_index_matrix(k)
        from cska.cyska import weighted_kmer_counts
        t0 = time()
        prev = None
        for i in range(n):
            counts = weighted_kmer_counts(im, w, k)
            if not prev is None:
                self.assertTrue((counts == prev).all()) # test for consistency
            else:
                prev = counts
        t1 = time()
        # print "took {0:.2f} ms".format(1000./n * (t1-t0))


from cska.pwm import *
from cska.gradient import *
class TestGradientMethods(unittest.TestCase):


    @staticmethod
    def params_from_motif(motif, betas = [], a0=1e-6):
        psam = motif.psam + a0
        params = ModelParametrization(motif.n, max(1, len(betas)), psam=psam, A0=motif.A0, betas=betas)
        return params

    def from_kmers(self, *argc, **kwargs):
        return TestGradientMethods.params_from_motif(PSAM.from_kmer_variants(*argc), **kwargs)

    def setup_model(self, correct_params, k_monitor=5, rbp_conc=None, N=1000000):
        # generating reference state
        R0 = np.ones((correct_params.n_samples, 4**k_monitor), dtype=np.float32)
        from cska.partfunc import PartFuncModel
        if rbp_conc is None:
            rbp_conc = [1.,] * correct_params.n_samples
        
        reads = get_real_reads(N)
        model = PartFuncModel(reads, correct_params, R0, rbp_conc=rbp_conc, Z_thresh=0)
        state0 = model.predict(correct_params, beta_fixed=True)
        R0 = state0.R
        model.set_R0(R0)
        state0 = model.predict(correct_params, beta_fixed=True)

        return model, state0

    def setUp(self):
        # params = self.from_kmers(['GCATG', 'GCACG', ], [5., .5, ], betas = [.001,.005,.009])
        # # params = self.from_kmers(['GCATG',], [5., ], betas = [.01,.05,.09])
        # self.model, self.state0 = self.setup_model(params, rbp_conc=[.2, 20., 2000.])
        kmers = ['TGCATG', 'TGCACG', 'AGCATG', 'CGCATG', 'GGCATG']
        aff = np.array([1.,.1,.5, .42, .4]) * 5
        params = self.from_kmers(kmers, aff, betas = [.001,.005,.025])
        self.model, self.state0 = self.setup_model(params, rbp_conc=[1.,5.,25.])

    def get_default(self):
        return self.model, self.state0

    def run_descent(self, correct_params, initial_params, k_monitor=5, dec=.75, rbp_conc=None):
        model, state0 = self.setup_model(correct_params, k_monitor=k_monitor, rbp_conc=rbp_conc)
        from cska import vector_stats
        print ">>> reference state"
        print state0

        state = model.tune(initial_params)
        print ">>> TUNED"
        print state
        # print ">>> ORACLE0"
        # print correct_params
        # print ">>> INITIAL"
        # print initial_params
        
        print R0.min(), R0.max(), R0.mean(), cyska.index_to_seq(R0.argmax(), k_monitor)
        G = GradientDescent(model, initial_params, dec=dec)
        res = G.optimize(initial_params, maxiter=1, debug=True)
        print ">>> ORACLE"
        print correct_params
        print ">>> INITIAL"
        print initial_params
        print ">>> FINAL"
        print G.params
        print ">>> FINAL GRADIENT"
        print G.last_state.grad
        print ">>> FINAL EMP. GRADIENT"
        print emp_grad(G.last_state)

        d = np.fabs(G.params.data - correct_params.data)
        print "maximal parameter deviation:", d.max(), d.argmax()
        self.assertTrue(G.status.startswith('CONVERGED'))
        # self.assertLess(G.t, 30)
        self.assertTrue(np.allclose(G.params.data, correct_params.data, rtol=1e-3, atol=1e-2))


    def noisy_variant(self, motif, noise=.01, A0=None, betas=None, seed=4711):
        if seed:
            np.random.seed(seed)

        delta = motif.copy()
        delta.data[:] = np.array(np.random.randn(motif.n) * noise, dtype=np.float32)[:]
        variant = GradientDescent.apply_delta(motif, delta)

        if not A0 is None:
            variant.A0 = A0

        if not betas is None:
            variant.betas[:] = betas[:]

        return variant

    # @unittest.skip("")
    def test_beta_opt(self):
        model, state0 = self.get_default()
        state = model.predict(state0.params, beta_fixed=False)
        print state0.params.betas
        print state.params.betas
        self.assertTrue( np.allclose(state.params.betas, state0.params.betas) )

    def test_fit_A0(self):
        from cska import vector_stats
        model, state0 = self.get_default()

        # state2 = model.fit_A0_and_betas(state1, plot="test_fit_A0.pdf")
        # est = state2.beta_estimators
        import matplotlib.pyplot as pp
        # pp.figure()
        # pp.title('after fit of A0 and beta')
        # for i, per_sample in enumerate(est):
        #     # print "sample", i
        #     q = np.linspace(0,100,10)
        #     perc = np.percentile(per_sample, q)
        #     # print "percentiles", perc
        #     pp.semilogx(model.R0[i], per_sample, '.')
        #     pp.semilogx(model.R0[i][model.top_Ri], per_sample[model.top_Ri], 'xr')
        #     pp.axhline(state0.params.betas[i], color='black')
        #     pp.axhline(state2.params.betas[i], color='red')

        # pp.ylim(0, .03)
        def one_round_betas(state, n=10):
            if n < 1:
                return state

            est = state.beta_estimators
            est_b = np.median(est[:,state.mdl.top_Ri],axis=1)
            print "quick'n'dirty estimates for beta at round", est_b, n
            state.params.betas[:] = est_b
            state._update_betas() # recompute everything after changing the beta values

            return one_round_betas(state, n=n-1)

        subopt_params = state0.params.copy()
        subopt_params.psam_matrix[4,1] = .01 # set C4 too low
        subopt_params.A0 = 1
        # subopt_params.betas[:] = [.1,.5,.1]
        state1 = model.predict(subopt_params, beta_fixed=True)
        # print state1

        a0s = [subopt_params.A0]
        beta0s = [subopt_params.betas[0]]

        from scipy.stats import sem
        def one_round_A0(state, n=20):
            if n < 1:
                return state

            est = state.beta_estimators
            est_b = np.median(est[:,state.mdl.top_Ri],axis=1)
            # est_b = state.params.betas
            beta0s.append(est_b[0])
            print "quick'n'dirty estimates for beta at round", est_b, n
            print "SEM(betas)", sem(est, axis=1)
            state.params.betas[:] = est_b

            est = state.A0_estimators
            est_A0 = np.median(est[:,state.mdl.top_Ri],axis=1).mean()
            # est_A0 = state.params.A0
            a0s.append(est_A0)

            print "quick'n'dirty estimates for A0 at round", est_A0, n
            print "SEM(A0)", sem(est, axis=1), sem(est, axis=None)
            state.params.A0 = est_A0

            # state._update_betas() # recompute everything after changing the beta values

            state = state.mdl.predict(state.params, beta_fixed=True)

            return one_round_A0(state, n=n-1)

        def opt_A0(state):
            from scipy.optimize import minimize_scalar
            from cska.gradient import minimize_logspaced
            opt_A0.state = state
            mask = np.fabs(state.mdl.R0 - 1) > .001
            print mask.shape

            def err(A0):
                opt_A0.state.params.A0 = A0
                opt_A0.state = opt_A0.state.mdl.predict(opt_A0.state.params, beta_fixed=False)
                # est = opt_A0.state.beta_estimators
                # est_b = np.median(est[:,opt_A0.state.mdl.top_Ri],axis=1)
                # opt_A0.state._update_betas()

                a0s.append(A0)
                beta0s.append(opt_A0.state.params.betas[0])

                est = opt_A0.state.A0_estimators
                err = sem(est[mask], axis=None)
                print A0, "->", err, opt_A0.state.error
                return err + opt_A0.state.error * opt_A0.state.mdl.nA

            # res = minimize_scalar(err, bounds=(1e-3, 100), method='Bounded')
            res = minimize_logspaced(err, bounds=np.array((1e-3, 1000)), n_samples=7, nested=2, plot='minimize_A0_est_SEM.pdf')
            print res
            state.params.A0 = res.x
            state = state.mdl.predict(state.params, beta_fixed=False)

            return state

        # state = one_round_A0(state1)
        state = opt_A0(state1)
        pp.figure()
        pp.loglog(state.mdl.R0.T, state.R.T, 'x')
        m = min(state.mdl.R0.min(), state.R.min())
        M = max(state.mdl.R0.max(), state.R.max())
        print m,M
        pp.loglog([m,m],[M,M], 'k', linestyle='dashed')

        pp.figure()
        pp.loglog(a0s, beta0s, '-b')
        pp.loglog(a0s, beta0s, '.k')
        pp.plot([a0s[0],], [beta0s[0],],'^k') # start
        pp.plot([a0s[-1],], [beta0s[-1],],'xk') # end
        pp.plot([state0.params.A0,], [state0.params.betas[0],],'.r')

        # subopt_params.A0 = 5.
        
        # state1 = model.predict(subopt_params, beta_fixed=True)
        # print state1
        # print one_round_betas(state1, n=1)

        state1 = state
        est = state1.beta_estimators
        pp.figure()
        pp.title('beta estimators')
        for i, per_sample in enumerate(est):
            pp.semilogx(model.R0[i], per_sample, '.')
            pp.semilogx(model.R0[i][model.top_Ri], per_sample[model.top_Ri], 'xr')
            pp.axhline(state0.params.betas[i], color='black')
            pp.axhline(state1.params.betas[i], color='red')

        # pp.ylim(0, .03)
        pp.ylabel('beta estimator')

        est = state1.A0_estimators
        est_A0 = np.median(est[:,model.top_Ri],axis=1).mean()
        print "quick'n'dirty estimates for A0", est_A0

        pp.figure()
        pp.title('A0 estimators')
        for i, per_sample in enumerate(est):
            pp.semilogx(model.R0[i], per_sample, '.')
            pp.semilogx(model.R0[i][model.top_Ri], per_sample[model.top_Ri], 'xr')
            pp.axhline(state0.params.A0, color='black')
            pp.axhline(state1.params.A0, color='red')

        # pp.ylim(0, 10)
        pp.ylabel('A0 estimator [nM]')
        state1 = state.mdl.predict(state1.params, beta_fixed=True)
        print state1
        print state1.grad
        # print state2.grad
        pp.show()
        return


    def test_suboptimal(self):
        from cska import vector_stats
        model, state0 = self.get_default()

        subopt_params = state0.params.copy()
        subopt_params.psam_matrix[4,1] = .01 # set C4 too low
        print subopt_params

        state = model.predict(subopt_params, beta_fixed=False)
        # print "ANALYTICAL GRADIENT AT SUB-OPTIMUM"
        # from time import time

        # t0 = time()
        # print state.grad
        # dt = 1000. * (time() - t0)
        # print "# gradient computation took {0:.2f} ms".format(dt)
        
        # from cska.gradient import emp_grad
        # print "EMPIRICAL GRADIENT AT SUB-OPTIMUM"
        # t0 = time()
        # print emp_grad(state)
        # dt = 1000. * (time() - t0)
        # print "# gradient computation took {0:.2f} ms".format(dt)

        print "performing gradient descent optimization"
        G = GradientDescent(model, subopt_params)
        def callback(descent):
            state = descent.last_state
            # descent.print_state(state)
            # import matplotlib.pylab as pp
            # pp.figure()
            # print state0.R.shape, state.R.shape
            # pp.loglog(state0.R[0], state.R[0], 'x', label="A0={0:.2f} beta={1:.2e}".format(state.params.A0, state.params.betas[0]))
            # pp.legend(loc='lower right')
            # pp.savefig('dA0_{}.pdf'.format(descent.t))
            # pp.show()
            # pp.close()

        res = G.optimize(subopt_params, maxiter=20, debug=True, ls_plot="ls_subopt_{self.t}.pdf", A0_plot="A0_subopt_{self.t}.pdf", tune=True, callback=callback)
        # res = G.optimize(subopt_params, maxiter=50, debug=True, tune=True, callback=callback)
        print res.last_state.params

    def test_dA0(self):
        model, state0 = self.get_default()

        print "correct reference state"
        print state0
        subopt_params = state0.params.copy()
        subopt_params.A0 *= .1 # set A0 too low

        state = model.predict(subopt_params, beta_fixed=False)
        print "perturbed state"
        print state
        import matplotlib.pylab as pp
        pp.loglog(state0.R, state.R, 'x')
        pp.savefig('dA0.pdf')
        pp.close()

        # print "ERROR", state.error
        print "ANALYTICAL GRADIENT"
        from time import time

        t0 = time()
        print state.grad
        dt = 1000. * (time() - t0)
        print "# gradient computation took {0:.2f} ms".format(dt)
        
        from cska.gradient import emp_grad
        print "EMPIRICAL GRADIENT"
        t0 = time()
        print emp_grad(state, eps=1e-4)
        dt = 1000. * (time() - t0)
        print "# gradient computation took {0:.2f} ms".format(dt)

        print "performing gradient descent optimization"
        G = GradientDescent(model, subopt_params)
        res = G.optimize(subopt_params, maxiter=10, debug=True)
        print res.last_state.params


    # @unittest.skip("")
    def test_consistency(self):
        """
        Subsequent evaluations of the same model should yield the exact same results.
        """
        # from cska import vector_stats
        model, state0 = self.get_default()
        states = [model.predict(state0.params) for i in range(3)]

        def check(getter):
            for i in range(1,len(states)):
                self.assertTrue( (getter(states[0]) == getter(states[i])).all() )

        check(lambda x : x.psi)
        check(lambda x : x.w )
        check(lambda x : x.rbp_free )
        check(lambda x : x.params.betas )
        check(lambda x : x.Q )
        check(lambda x : x.W )
        check(lambda x : x.R )
        check(lambda x : x.mdl.R0 )
        check(lambda x : x.R_errors )
        check(lambda x : x.grad.data )

    # @unittest.skip("")
    def test_grad_optimum(self):
        """
        At the optimal parameters, the gradient should be 0 in every element.
        """
        model, state0 = self.get_default()
        grad = state0.grad
        from cska.gradient import emp_grad
        egrad = emp_grad(state0, eps=1e-6)
        
        print "\nANALYTICAL GRADIENT AT OPTIMUM"
        print grad
        print "EMP. GRADIENT AT OPTIMUM"
        print egrad
        self.assertTrue(np.allclose(grad.data,0))

    # @unittest.skip("")
    def test_grad_matrix(self, d=1e-4):
        """
        Small perturbations of the parameter matrix should result in gradients pointing
        in the opposite direction.
        """
        from cska.gradient import GradientDescent, emp_grad

        model, state0 = self.get_default()
        grad0 = state0.grad
        n = len(state0.params.psam_vec)

        ratios = np.ones(n, dtype=np.float32)
        for i in range(1, n):
        # for i in [1,2,3,4, 14,]:
            if i and state0.params.data[i] >= 1:
                continue # only non-cognate

            delta = state0.params.copy()
            delta.data[:] = 0
            delta.data[i] = d

            params = GradientDescent.apply_delta(state0.params, delta)
            pert = params.copy()
            pert.data[:] -= state0.params.data
            print "perturbation", i, "params[i] =", state0.params.data[i]
            print pert

            state = model.predict(params, beta_fixed=True)
            grad = state.grad
            print "gradient"
            print grad.unity()
            egrad = emp_grad(state)
            print "emp. gradient"
            print egrad.unity()

            i_pert = pert.data.argmax()
            print "i_pert", i_pert
            assert i_pert == i
            I = grad.data.argsort()[::-1]
            i_grad = I[0]
            i_next = I[1]
            # self.assertTrue(i_grad == i_pert)
            # ratio of highest value in gradient to second-highest.
            
            if i_grad == i_pert:
                ratios[i] = grad.data[i_pert] / grad.data[i_next]
                print "SUCCESS"
            else:
                ratios[i] = grad.data[i_pert] / grad.data[i_grad]
                print "FAILED"

        print "summary", ratios
        self.assertTrue((ratios >= 1.).all())

    @unittest.skip("")
    def test_5mer_noise(self):
        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        self.run_descent(motif, self.noisy_variant(motif, noise=.05), k_monitor=5, rbp_conc=[.1,.5,2.])

    @unittest.skip("")
    def test_5mer_1off(self):
        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        off = motif.copy()
        off.A0 = .5
        self.run_descent(motif, off, k_monitor=5, rbp_conc=[.1,.5,2.])

    @unittest.skip("")
    def test_5mer_betas(self):
        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        variant = self.noisy_variant(motif, noise=.05)
        variant.psam_vec[:] = motif.psam_vec[:]
        self.run_descent(motif, variant, k_monitor=5, rbp_conc=[.5,50.,200.])

    @unittest.skip("")
    def test_5mer_optimum(self):
        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        print motif
        init = motif.copy()
        init.A0 = .5
        self.run_descent(motif, init, k_monitor=5, rbp_conc=[.5,50.,200.])
        # self.run_descent(motif, motif, k_monitor=5, rbp_conc=[15., 50.,200.])

    # def test_5mer_A0(self):
    #     motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [5., 3.0, .1,])
    #     self.run_descent(motif, self.noisy_variant(motif, noise=0, A0=6.), k_monitor=5)

    # def test_5mer_high_noise(self):
    #     motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,])
    #     self.run_descent(motif, self.noisy_variant(motif, noise=.1), k_monitor=5)

    # def test_7mer_noise(self):
    #     motif = PSAM.from_kmer_variants(['UGCAUGU', 'UGCACGU', 'UGCAGGC', ], [1., .5, .02])
    #     self.run_descent(motif, self.noisy_variant(motif))

    # def test_7mer_high_noise(self):
    #     motif = PSAM.from_kmer_variants(['UGCAUGU', 'UGCACGU', 'UGCAGGC', ], [1., .5, .02])
    #     self.run_descent(motif, self.noisy_variant(motif, noise=.2), k_monitor=5)

    # def test_5mer_off_matrix(self):
    #     motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GCATG'], [1., 1., 1.])
    #     motif_variant = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG','GGGGG'], [1., .2, .8,.1])
    #     self.run_descent(motif_correct, motif_variant)


if __name__ == '__main__':
    # import gzip
    # path = os.path.join(os.path.dirname(__file__), '../tests/reads_20.txt.gz')
    # TODO: include small amount of raw data in git repo for testing!
    # import logging
    # logging.basicConfig(level=logging.WARNING)
    # reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=1000000)
    unittest.main(verbosity=2)



# i0 = seq_to_index('TGCATGT')
# i1 = seq_to_index('TGCATGC')
# i2 = seq_to_index('TGCATGA')
# i3 = seq_to_index('TGCATGG')
# i4 = seq_to_index('ATGCATG')
# i5 = seq_to_index('CTGCATG')
# i6 = seq_to_index('GTGCATG')
# i7 = seq_to_index('TTGCATG')

# R = RBNSReads(reads, adap5='CCCCCCC', adap3='TTTTTTTT')
# im = R.get_index_matrix(5)
# print reads[0]
# print im.shape, len(reads[0])
# for x in im[0]:
#     print index_to_seq(x, 5),
    
# sys.exit(0)
# counts = R.kmer_counts(7)
# print counts[i0]

# F = R.reads_with_kmers(7)
# print F[i0]

# k = 7

# candidates = np.zeros(4**k, dtype=np.uint32)
# candidates[i0] = 1
# candidates[i1] = 2
# candidates[i2] = 3
# candidates[i3] = 4
# candidates[i4] = 5
# candidates[i5] = 6
# candidates[i6] = 7
# candidates[i7] = 8
# R.write_pure_reads_fasta(sys.stdout, k, candidates, n_sample=100000)
