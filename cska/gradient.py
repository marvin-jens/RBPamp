import os
import unittest
import copy
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from cska import auto_detect
from cska.reads import RBNSReads
import cska.ska_kmers as cyska
from cska.pwm import PSAM

class Proxy(object):
    def __init__(self, data, start, end, shape=None, unpack=True):
        self.data = data
        self.start = start
        self.end = end
        self.shape = shape
        self.unpack = unpack
    
    def get_values(self):
        # print "get_values", self.start, self.end
        d = self.data[self.start:self.end]
        if not self.shape is None:
            d = np.reshape(d, self.shape)
        
        if len(d) == 1 and self.unpack:
            return d[0]
        else:
            return d

    def set_values(self, d):
        # print "set_values", self.start, self.end, d
        
        if hasattr(d, '__len__'):
            if not isinstance(d, np.ndarray):
                d = np.array(d)
            d = d.flatten()
            assert len(d) == self.end - self.start
            self.data[self.start:self.end] = d[:]
        else:
            assert self.end - self.start == 1
            self.data[self.start] = d

        return d

class ModelParametrization(object):
    def __init__(self, k, n_samples, psam=[], A0=1., betas = [], data = [], dtype=np.float32):
        self.k = k
        self.n_samples = n_samples
        self.n = 4*k + 1 + n_samples
        self.psam_start = 0
        self.psam_end = 4*k + 1
        self.betas_start = self.psam_end
        self.betas_end = self.n
        self.dtype = dtype

        self.data = np.zeros(self.n, dtype=np.float32)

        # self.psam_vec = Proxy(self.data, self.psam_start, self.psam_end)
        # self.psam_matrix = Proxy(self.data, self.psam_start+1, self.psam_end, shape=(k,4))
        self.attrs = {
            'psam_vec' : Proxy(self.data, self.psam_start, self.psam_end),
            'psam_matrix' : Proxy(self.data, self.psam_start+1, self.psam_end, shape=(k,4)),
            'A0' : Proxy(self.data, 0, 1),
            'beta' : Proxy(self.data, self.betas_start, self.betas_start + 1),
            'betas' : Proxy(self.data, self.betas_start, self.betas_end, unpack=False)
        }

        if len(psam):
            self.psam_matrix = psam
        if A0:
            self.A0 = A0
        if len(betas):
            self.betas = betas

        if len(data):
            self.set_vector(data)

    @classmethod
    def from_vector(cls, vec, k, n_samples=1):
        return cls(k, n_samples, data = vec)

    def as_vector(self, dtype=np.float32):
        return self.data
    
    def copy(self):
        new = ModelParametrization(self.k, self.n_samples, dtype=self.dtype, data=self.data)
        if not np.allclose(new.data, self.data):
            d = np.fabs(new.data - self.data)
            i = d.argmax()
            print "OFFENDING PARAMETER:", i, new.data[i], self.data[i]
            print self.data
            1/0
        return new

    def set_vector(self, vec):
        self.data[:len(vec)] = vec[:]
        return self

    def unity_bounded(self):
        p = self.copy()
        
        v = p.psam_vec
        i = np.fabs(v).argmax()
        x = v[i]
        if x > 0:
            p.psam_vec = v / x
        elif x < 0:
            p.psam_vec = - v / x
        
        # print 'unity_bounded', i, x
        return p

    def unity(self):
        p = self.copy()
        n = np.linalg.norm(self.data)
        if n > 0:
            p.data /= n
        return p


    def __getattr__(self, a):
        if hasattr(self, 'attrs'):
            attrs = object.__getattribute__(self, 'attrs') 
        else:
            attrs = {}
        # print 'getattr', a
        if a in attrs:
            return attrs[a].get_values()
        else:
            return object.__getattribute__(self, a)
        # return super(ModelParametrization, self).__getattr__(a)

    def __setattr__(self, a, v):
        # attrs = super(ModelParametrization, self).__getattr__('attrs') 
        if hasattr(self, 'attrs'):
            attrs = object.__getattribute__(self, 'attrs') 
        else:
            attrs = {}
        # print "setattr", a, v
        if a in attrs:
            return attrs[a].set_values(v)
        else:
            return object.__setattr__(self, a, v)
        # return super(ModelParametrization, self).__setattr__(a, v)

    def __str__(self):
        buf = []
        buf.append("PSAM")
        buf.append("A0={0:.4f}".format(self.A0))
        buf.append("\tA\t\tC\t\tG\t\tU")
        for row in self.psam_matrix:
            buf.append("\t".join(["{0:>10.3f}".format(x) for x in row]))

        buf.append("BACKGROUND")
        for i, beta in enumerate(self.betas):
            buf.append('beta{0}\t{1:.3f}'.format(i, beta))
        return '\n'.join(buf)

    def __add__(self, x):
        c = self.copy()
        if isinstance(x, ModelParametrization):
            c.data += x.data
        else:
            c.data += x
        return c
    
    def __sub__(self, x):
        c = self.copy()
        if isinstance(x, ModelParametrization):
            c.data += x.data
        else:
            c.data -= x
        return c

    def __mul__(self, x):
        c = self.copy()
        if isinstance(x, ModelParametrization):
            c.data *= x.data
        else:
            c.data *= x
        return c

    def __div__(self, x):
        c = self.copy()
        if isinstance(x, ModelParametrization):
            c.data /= x.data
        else:
            c.data /= x
        return c

    def __neg__(self):
        return ModelParametrization.from_vector(- self.data, self.k, self.n_samples)

class KmerModelState(object):
    def __init__(self, mdl, params):
        self.mdl = mdl
        self.params = params
        
        # print "psam params vector", psam
        cyZ = cyska.PSAM_partition_function(mdl.sub_padded, mdl.sub_acc, params.psam_matrix, openen_ofs=mdl.acc_ofs)
        
        cyZ *= params.A0
        Zj = cyZ.sum(axis=1)
        psi = P*Zj/ (P*Zj+1)

        # print "PSI", psi, psi.min(), psi.mean(), psi.max()
        self.cyZ = cyZ
        self.Zj = Zj
        # avoid zeros in Zj by all means!
        np.clip(self.Zj, 1e-9, None, out=self.Zj)
        self.psi = psi
        self.pi = cyska.weighted_kmer_counts(mdl.sub_im, psi + params.beta, mdl.k_monitor)

        self.R = self.pi / self.pi.sum() / mdl.f0
        self.error = self.mdl.error(self.R) 
        self.mdl.n_fev += 1

    @property
    def grad(self):
        N = len(cyZ)
        pi, d_pi = cyska.PSAM_kmer_gradient(self.mdl.sub_padded, self.cyZ, self.Zj, self.psi, self.mdl.sub_im, self.params.psam_vec, self.mdl.k_monitor)
        pi += N * self.f0 * params.beta 

        # print R.shape, d_pi.shape
        dR_dM = (self.R/self.pi)[:,np.newaxis] * (d_pi - (self.f0 * self.R)[:,np.newaxis] * d_pi.sum(axis=0)[np.newaxis,:])
        dR_dbeta =  self.f0 / self.pi * (self.R - self.R**2)

        dR = np.hstack( (dR_dM, dR_dbeta[:,np.newaxis]) )

        # TODO: refactor the R^2 part of gradient into optimizer?
        _grad = 2 * ((1. * (self.R - self.mdl.opt.R0))[:,np.newaxis] * dR).sum(axis=0)
        param_grad = self.params.copy().set_vector(_grad)
        self.mdl.n_grad += 1
        return param_grad

def emp_grad(state, eps=1e-4):
    v0 = state.params.as_vector()
    var = state.params.copy()
    grad = state.params.copy()

    err0 = state.error

    for i in range(state.params.n):
        d = v0[i] * eps
        var.data[i] = v0[i] + d
        state = state.mdl.predict(var)
        derr = state.error - err0
        grad.data[i] = derr/d
        var.data[i] = v0[i]
    
    return grad

class PartFuncModel(object):
    def __init__(self, reads, params, R0, k_monitor=5, subsample=1.):
        self.logger = logging.getLogger('PartFuncModel')
        self.reads = reads
        self.opt = None
        self.params = params
        self.R0 = R0
        self.openen = self.reads.acc_storage.get_raw(self.params.k)
        self.acc_ofs = self.reads.l5 - params.k + 1
        print "acc_ofs", self.acc_ofs
        self.acc = self.openen.acc
        print self.acc.shape
        self.n = params.k
        adap5 = cyska.seq_to_bits(reads.adap5)
        adap3 = cyska.seq_to_bits(reads.adap3)
        self.padded = cyska.seqm_pad_adapters(reads.seqm, adap5, adap3, self.n)

        self.k_monitor = k_monitor
        self.im = reads.get_index_matrix(k_monitor)

        f0 = reads.kmer_frequencies(k_monitor)
        self.f0 = f0 / f0.sum()

        self.subsample = subsample
        self.new_subsample()
    
        self.n_fev = 0
        self.n_grad = 0

    def new_subsample(self):
        self.logger.info('new subsample')
        n = int(self.subsample * self.reads.N)
        if n == self.reads.N:
            self.sub_indices = np.arange(n)
        else:
            self.sub_indices = np.random.permutation(self.reads.N)[:n]

        self.sub_padded = self.padded[self.sub_indices]
        self.sub_im = self.im[self.sub_indices]
        self.sub_acc = self.acc[self.sub_indices]
        
    def predict(self, params):
        self.n_fev += 1
        return KmerModelState(self, params)

    def error(self, R):
        return ((self.R0 - R)**2).mean()



class GradientDescent(object):
    def __init__(self, model, params0, dec=.75):
        self.logger = logging.getLogger('GradientDescent')
        self.model = model
        self.params = params0
        self.model.opt = self # link model to this optimizer instance so it can find out R0 etc.

        # momentum smoothing of the gradient
        self.past_grad = None
        self.past_sqg = 1
        self.past_var = 1
        self.dec = dec

        # records
        self.errors = []
        self.history = []
        self.ls_nfev = []
        self.ls_step = []
        self.t = 0

        # optimization result/status
        self.status = None
        
    @staticmethod
    def apply_delta(params, delta):
        new = params.copy()
        p = np.clip(params.psam_matrix + delta.psam_matrix, 1e-9, None)
        M = p.max(axis=1)
        p /= M[:,np.newaxis]
        new.psam_matrix = p
        new.A0 *= M.max() # keep matrix elements <= 1 and absorb excess into A0
        new.A0 = max(1e-9, new.A0 + delta.A0) # prevent underflow

        new.betas = np.clip(params.betas + delta.betas, 0, None)
        return new

    def backtracking_LS(self, params, grad, local_grad, min_step = 1e-6, max_step = 5., max_iter=21, tau=.5, c=.5, debug=False):

        grad.data /= np.linalg.norm(grad.data)
        print grad
        def err(s):
            m = self.apply_delta(params, grad * s)
            # d = delta(m.data, correct_params.data)
            # print s, d, m.data, 
            R, dR = self.predict_R(m, grad=False)
            e = self.error(R)
            # print s,"->", e - e0, d - d0
            return e

        e0 = err(0)

        print local_grad
        a = max_step
        m = np.dot(grad.data, local_grad.data)
        print m
        t = - c * m
        print t
        for n in range(max_iter):
            e = err(a)
            print n, '->', a, e0 - e, a*t, 'crit', (e0-e) - (a*t)
            # Armijo-Goldstein criterion
            if e0 - e >= a*t:
                break
            # if e < e0:
            #     break

            a *= tau

        return a, n

    def line_search(self, params, grad, debug=False, min_step = 1e-6, max_step = 10., maxiter=50, xatol=1e-1, e0=None):
        from scipy.optimize import minimize_scalar

        if e0 is None:
            state = self.model.predict(params)
            e0 = state.error

        def err(x):
            s = np.exp(x)
            m = self.apply_delta(params, grad * s)
            state = self.model.predict(m)
            # print s,"->", e - e0
            return state.error

        res = minimize_scalar(err, method='Bounded', bounds=np.log(np.array([min_step, max_step])), options=dict(maxiter=maxiter, xatol=xatol))
        return np.exp(res.x), res.nfev

    def momentum_grad(self, local_grad):
        if self.past_grad is None:
            self.past_grad = local_grad

        grad = self.past_grad * self.dec + local_grad * (1- self.dec)
        self.past_grad = grad
        return grad

    def RMSprop(self, local_grad):
        if self.past_grad is None:
            self.past_grad = local_grad.data

        m = self.dec * self.past_grad + (1 - self.dec) * local_grad.data
        s = self.dec * self.past_sqg + (1 - self.dec) * local_grad.data**2

        self.past_grad = m
        self.past_sqg = s

        upd = local_grad.copy()
        upd.data = m / (np.sqrt(s) + .001)

        return upd

    def converged(self, rtol=1e-6, atol=1e-8, tau=10):
        if len(self.errors):
            if self.errors[-1] < atol:
                return 'CONVERGED_ERR_MINIMAL'

        if len(self.errors) < tau:
            return self.status
           
        last_errs = np.array(self.errors[-tau:])
        mean = last_errs.mean()
        if (mean - self.errors[-1]) / mean < rtol:
            return 'CONVERGED_NO_MORE_DECREASE'

        return self.status


    def optimize(self, maxiter=100, debug=False, callback=None):
        state = self.model.predict(self.params)
        self.errors.append(state.error)
        self.history.append(state)

        try:
            while not self.converged() and self.t < maxiter:
                local_grad = state.grad
                descent = self.RMSprop(- local_grad).unity()

                s,n = self.line_search(self.params, descent, e0=self.errors[-1])
                self.ls_step.append(s)
                self.ls_nfev.append(n)

                upd = descent * s
                self.params = self.apply_delta(self.params, upd)

                state = self.model.predict(self.params)
                self.errors.append(state.error)
                self.t += 1
                self.history.append(state)
                if debug:
                    print ">>>>>>>>>UPDATE, scale=",s
                    print descent
                    print ">>>>>>>>>PARAMS"
                    print self.params
                    print "=" * 50
                    print "step:", self.t, self.errors[-1], self.model.n_fev, self.model.n_grad, s

                if callback:
                    callback(self)

                print "n_fev={self.model.n_fev} t_aff={t_aff:.3f} t_fev={t_fev:.3f}ms n_grad={self.model.n_grad} t_grad={t_grad:.3f}ms".format(
                    self=self,
                    t_aff = 1000. * self.model.t_aff/self.model.n_fev,
                    t_fev = 1000. * self.model.t_fev/self.model.n_fev,
                    t_grad = 1000. * self.model.t_grad/self.model.n_grad,
                )

        # except ValueError: #KeyboardInterrupt
        except KeyboardInterrupt:
            self.status = "KEYBOARD_INTERRUPT"
        
        if self.t < maxiter:
            self.status = self.converged()
        else:
            self.status = "MAX_ITER"

        print "optimization ended with status", self.status
        # print "last gradient"
        # print self.past_grad
        # print "squared"
        # print self.past_sqg
        
        return self



class TestGradientMethods(unittest.TestCase):

    @staticmethod
    def params_from_motif(motif, betas = [], a0=1e-6):
        psam = motif.psam + a0
        params = ModelParametrization(motif.n, 1, psam=psam, A0=motif.A0, betas=betas)
        return params

    def run_descent(self, motif_correct, motif_variant, k_monitor=6, dec=.75):
        correct_params = self.params_from_motif(motif_correct)
        initial_params = self.params_from_motif(motif_variant)

        print "ORACLE"
        print correct_params
        print "INITIAL"
        print initial_params
        
        G = GradientDescent(reads, initial_params, None, dec=dec, k_monitor=k_monitor, subsample=1.)
        # generating reference state
        R0, dR0 = G.predict_R(correct_params)
        G.set_reference(R0)

        res = G.optimize(maxiter=200)
        print "final parametrization"
        print G.params
        d = np.fabs(G.params.data - correct_params.data)
        print "maximal parameter deviation:", d.max(), d.argmax()
        self.assertTrue(G.status.startswith('CONVERGED'))
        # self.assertLess(G.t, 30)
        self.assertTrue(np.allclose(G.params.data, correct_params.data, rtol=1e-3, atol=1e-2))


    def noisy_variant(self, motif, noise=.01, A0=None, seed=4711):
        if seed:
            np.random.seed(seed)
        variant = copy.deepcopy(motif)
        variant.psam += np.array(np.random.rand(*motif.psam.shape) * noise, dtype=np.float32)
        variant.psam = variant.psam.clip(variant.psam, 1e-9, None)
        variant.psam /= variant.psam.max(axis=1)[:,np.newaxis]

        if not A0 is None:
            variant.A0 = A0

        print "noisy variant"
        print variant
        return variant

    @unittest.skip("")
    def test_grad_optimum(self, dec=.75):
        motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [5., 3.0, .1,])
        params = self.params_from_motif(motif)
        G = GradientDescent(reads, params, None, dec=dec, k_monitor=5, subsample=1.)
        # generating reference state
        R0, dR0 = G.predict_R(params)
        G.set_reference(R0)

        dR_dA0 = dR0[:,0]
        I = dR_dA0.argsort()
        print "gaining"
        for i in I[::-1][:10]:
            print cyska.index_to_seq(i, 5), dR_dA0[i], R0[i]

        print "losing"
        for i in I[:10]:
            print cyska.index_to_seq(i, 5), dR_dA0[i], R0[i]

        grad = G.ana_grad(params)
        self.assertTrue(np.allclose(grad.data,0))

        # reduce A0
        params.A0 -= 1
        ana = G.ana_grad(params)
        emp = G.emp_grad(params)
        print ">>>> analytical"
        print ana.unity()
        print ">>>> empirical"
        print emp.unity()

        
    @unittest.skip("")
    def test_5mer_noise(self):
        motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,])
        self.run_descent(motif, self.noisy_variant(motif), k_monitor=5)

    @unittest.skip("")
    def test_5mer_optimum(self):
        motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [5., 3.0, .1,])
        print motif
        self.run_descent(motif, motif, k_monitor=5)

    def test_5mer_A0(self):
        motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [5., 3.0, .1,])
        self.run_descent(motif, self.noisy_variant(motif, noise=0, A0=6.), k_monitor=5)

    def test_5mer_high_noise(self):
        motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GCAGG', ], [1., .6, .02,])
        self.run_descent(motif, self.noisy_variant(motif, noise=.1), k_monitor=5)

    def test_7mer_noise(self):
        motif = PSAM.from_kmer_variants(['UGCAUGU', 'UGCACGU', 'UGCAGGC', ], [1., .5, .02])
        self.run_descent(motif, self.noisy_variant(motif))

    def test_7mer_high_noise(self):
        motif = PSAM.from_kmer_variants(['UGCAUGU', 'UGCACGU', 'UGCAGGC', ], [1., .5, .02])
        self.run_descent(motif, self.noisy_variant(motif, noise=.2), k_monitor=5)

    def test_5mer_off_matrix(self):
        motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GCATG'], [1., 1., 1.])
        motif_variant = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG','GGGGG'], [1., .2, .8,.1])
        self.run_descent(motif_correct, motif_variant)

if __name__ == '__main__':
    # import gzip
    # path = os.path.join(os.path.dirname(__file__), '../tests/reads_20.txt.gz')
    # TODO: include small amount of raw data in git repo for testing!

    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=100000)
    unittest.main(verbosity=2)
