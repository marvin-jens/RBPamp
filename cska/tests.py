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
        real_reads[N] = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=N)

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


    def test_weighted_kmer_counts(self, N = 1000000, k=6, rnd_seed=47110815, n = 100):
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
        print "took {0:.2f} ms".format(1000./n * (t1-t0))


from cska.pwm import *
from cska.gradient import *
class TestGradientMethods(unittest.TestCase):

    @staticmethod
    def params_from_motif(motif, betas = [], a0=1e-6):
        psam = motif.psam + a0
        params = ModelParametrization(motif.n, max(1, len(betas)), psam=psam, A0=motif.A0, betas=betas)
        return params

    def setup_model(self, correct_params, k_monitor=5, rbp_conc=None):
        # generating reference state
        R0 = np.ones((correct_params.n_samples, 4**k_monitor), dtype=np.float32)
        from cska.partfunc import PartFuncModel
        if rbp_conc is None:
            rbp_conc = [1.,] * correct_params.n_samples
        model = PartFuncModel(reads, correct_params, R0, rbp_conc=rbp_conc)
        state0 = model.predict(correct_params, beta_fixed=True)
        R0 = state0.R
        model.set_R0(R0)

        return model, state0

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


    def from_kmers(self, *argc, **kwargs):
        return TestGradientMethods.params_from_motif(PSAM.from_kmer_variants(*argc), **kwargs)

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
    def test_consistency(self):
        from cska import vector_stats

        motif = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,], betas = [.08, .11, .03])
        model, state0 = self.setup_model(motif, rbp_conc=[.5,50.,400.])

        states = [model.predict(motif) for i in range(3)]
        i = 0
        for i in range(1,len(states)):
            print "psi", self.assertTrue(np.allclose(states[0].psi, states[i].psi))
            print "psi", self.assertTrue( (states[0].psi == states[i].psi).all() )
            print "b", self.assertTrue(np.allclose(states[0].b, states[i].b))
            print "rbp_free", self.assertTrue(np.allclose(states[0].rbp_free, states[i].rbp_free))
            print "betas", self.assertTrue(np.allclose(states[0].params.betas, states[1].params.betas))
            print "Qs", self.assertTrue(np.allclose(states[0].Q, states[i].Q))
            print "Ws", self.assertTrue(np.allclose(states[0].W, states[i].W))
            print "R", self.assertTrue(np.allclose(states[0].R, states[i].R))
            print "R0", self.assertTrue(np.allclose(states[0].mdl.R0, states[i].mdl.R0))
            print "R_errors", self.assertTrue(np.allclose(states[0].R_errors, states[i].R_errors))

            delta = states[0].R - states[i].R
            vector_stats(delta)
            states[i].R_errors = states[0].R_errors
            print "grads", self.assertTrue(np.allclose(states[0].grad.data, states[i].grad.data))

    @unittest.skip("")
    def test_grad_optimum(self, dec=.75):
        from cska.partfunc import PartFuncModel
        import cska.cyska as cyska
        params = self.from_kmers(['GCATG', 'GCACG', 'GCAGG', ], [5., 3.0, .1,])
        print "INITIAL PARAMS"
        print params
        R0 = np.ones((3,4**5), dtype=np.float32)
        mdl = PartFuncModel(reads, params, R0)
        state = mdl.predict(params)
        mdl.R0 = state.R

        G = GradientDescent(mdl, params, dec=dec)
        # # generating reference state
        # R0, dR0 = G.predict_R(params)
        # G.set_reference(R0)

        # dR_dA0 = dR0[:,0]
        # I = dR_dA0.argsort()
        # print "gaining"
        # for i in I[::-1][:10]:
        #     print cyska.index_to_seq(i, 5), dR_dA0[i], R0[i]

        # print "losing"
        # for i in I[:10]:
        #     print cyska.index_to_seq(i, 5), dR_dA0[i], R0[i]

        G.optimize()
        grad = G.last_state.grad
        self.assertTrue(np.allclose(grad.data,0))

        # reduce A0
        params.A0 -= 1
        # ana = G.ana_grad(params)
        # emp = G.emp_grad(params)
        # print ">>>> analytical"
        # print ana.unity()
        # print ">>>> empirical"
        # print emp.unity()

        
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

    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=1000000)
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
