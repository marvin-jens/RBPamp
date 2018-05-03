import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from cska import auto_detect
from cska.reads import RBNSReads
import cska.ska_kmers as cyska
import matplotlib.pyplot as pp

# CachedBase.debug_caching=True
test_reads = [
    "AAAAAAAAGCAGGAAAAAAAAAA",
    "AAAAAAAAGCAGGAAAAAAAAAA",
    "AAAAAAAAGCAGGAAAAAAAAAA",
    "AAAAAAAAGCAGGAAAAAAAAAA",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    "CCCCCCCCGCATGCCCCCCCCCC",
    # "CGCACGCGCCCCGCCCGCGCCGC",
    # "AGAGGACGGAGAGAGTCGCGCGA",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTCTT",
    "TTTTTTTTGGACGTTTTTTTTTT",
    "TTTTTTTTGGACGTTTTTTATTT",
    "TTTTTTTTGGACGTTTTTTATTT",
    "TTTTTTTTGGACGTTTAATTTTT",
    "GGGGGGACGGGGGGGGGGGGGGG",
    "GGGGGGATGGGGGGGGGGGGGGG",
    "GGGGGTAGGGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGGGGGGGGG",
    "GGGGGGGTCGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGGGGGGGGG",
]
# reads = RBNSReads.from_seqs(test_reads, pseudo_count=1)
reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=100000)
print "5'adapter len", reads.l5
np.random.seed(4711)

def unity_matrix(M):
    F = M.flatten()
    i = np.fabs(F).argmax()
    x = F[i] 
    if x > 0:
        return M / F[i]
    elif x < 0:
        return -M / F[i]
    else:
        return M # it's all zeros, we're done

def delta(m1,m2):
    return np.fabs(m1-m2).sum()


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



class GradientDescent(object):
    def __init__(self, reads, params, R0, k_monitor=5, dec=.75, subsample=1.):
        self.logger = logging.getLogger('GradientDescent')
        self.reads = reads
        self.params = params
        self.openen = self.reads.acc_storage.get_raw(self.params.k)
        self.acc_ofs = self.reads.l5 - params.k + 1
        print "acc_ofs", self.acc_ofs
        self.acc = self.openen.acc
        print self.acc.shape
        self.R0 = R0
        self.n = params.k
        self.subsample = subsample
        adap5 = cyska.seq_to_bits(reads.adap5)
        adap3 = cyska.seq_to_bits(reads.adap3)
        self.padded = cyska.seqm_pad_adapters(reads.seqm, adap5, adap3, self.n)

        self.k_monitor = k_monitor
        self.im = reads.get_index_matrix(k_monitor)
        f0 = reads.kmer_frequencies(k_monitor)
        self.f0 = f0 / f0.sum()

        # momentum smoothing of the gradient
        self.past_grad = None
        self.past_sqg = 1
        self.past_var = 1
        self.dec = dec

        # records
        self.errors = []
        self.residuals = []
        self.nfevs = []
        self.n_part_func = 0
        self.n_grad = 0
        self.scales = []
        self.t = 0
        self.new_subsample()
    
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

    def set_reference(self,R0):
        self.R0 = R0
        self.f = self.R0/self.f0

    def error(self, R):
        # return (np.log2(R / self.R0)**2).mean()
        return ((self.R0 - R)**2).mean()
    
    def predict_R(self, params, P=1., grad=True, debug=False):
        psam = params.psam_vec
        # print "psam params vector", psam
        cyZ = cyska.PSAM_partition_function(self.sub_padded, self.sub_acc, params.psam_matrix, openen_ofs=self.acc_ofs)
        N = len(cyZ)
        cyZ *= params.A0
        Zj = cyZ.sum(axis=1)
        psi = P*Zj/ (P*Zj+1)

        # print "PSI", psi, psi.min(), psi.mean(), psi.max()
        self.n_part_func += 1

        if grad:
            # assert np.isnan(cyZ).any() == False
            # assert np.isnan(psi).any() == False
            # if (Zj <= 0).any():
            #     I = (Zj <= 0).nonzero()
            #     print "detected non-positive partition functions", len(I[0])
            #     print I
            #     nts = np.array(list('ACGU'))
            #     for i in I[0]:
            #         print i, Zj[i], ''.join(nts[self.sub_padded[i]])
            
            # assert (Zj > 0).all()

            # avoid zeros in Zj by all means!
            np.clip(Zj, 1e-9, None, out=Zj)

            pi, d_pi = cyska.PSAM_kmer_gradient(self.sub_padded, cyZ, Zj, psi, self.sub_im, psam, self.k_monitor)
            pi += N * self.f0 * params.beta 
            self.n_grad += 1
            # assert np.isnan(pi).any() == False
            # if np.isnan(d_pi).any():
            #     NAN =  np.isnan(d_pi)
            #     print "total NaN elements", NAN.sum()
            #     print "corresponding kmers"
            #     I = NAN.nonzero()
            #     distinct = np.array(sorted(set(I[0])))
            #     for i in distinct[:10]:
            #         print cyska.index_to_seq(i, self.k_monitor)
            #         print d_pi[i]


                # print I
                

            R = pi / pi.sum() / self.f0
            # print R.shape, d_pi.shape
            dR_dM = (R/pi)[:,np.newaxis] * (d_pi - (self.f0 * R)[:,np.newaxis] * d_pi.sum(axis=0)[np.newaxis,:])
            dR_dbeta =  self.f0 / pi * (R - R**2)

            return R, np.hstack( (dR_dM, dR_dbeta[:,np.newaxis]) )
        else:
            pi = cyska.weighted_kmer_counts(self.sub_im, psi + params.beta, self.k_monitor)
            R = pi / pi.sum() / self.f0
            return R, None

    def grad_from_R(self, R1, dR, debug=False):
        grad = 2 * ((1. * (R1 - self.R0))[:,np.newaxis] * dR).sum(axis=0)
        param_grad = self.params.copy().set_vector(grad)
        return param_grad #.unity_bounded()


    def ana_grad(self, params, debug=False):
        R, dR = self.predict_R(params, debug=debug)
        return self.grad_from_R(R,dR, debug=debug)

    def emp_grad(self, params, eps=1e-4):
        v0 = params.as_vector()
        k = params.k

        var = params.copy()
        grad = params.copy()
        R0, dR0 = self.predict_R(params, grad=False)
        err0 = self.error(R0)

        for i in range(params.n):
            var.data[i] = v0[i] + eps
            R, dR = self.predict_R(var, grad=False)
            var.data[i] = v0[i]
            derr = self.error(R) - err0

            grad.data[i] = derr/eps
        
        grad.beta = 0
        return grad


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


    def line_search(self, params, grad, debug=False, min_step = 1e-6, max_step = 10., e0=None):
        from scipy.optimize import minimize_scalar

        if e0 is None:
            r0, bla = self.predict_R(params, grad=False)
            e0 = self.error(r0)

        def err(x):
            s = np.exp(x)
            m = self.apply_delta(params, grad * s)
            R, dR = self.predict_R(m, grad=False)
            e = self.error(R)
            # print s,"->", e - e0
            return e

        res = minimize_scalar(err, method='Bounded', bounds=np.log(np.array([min_step, max_step])), options=dict(maxiter=50, xatol=1e-1))
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
            return False
           
        last_errs = np.array(self.errors[-tau:])
        if last_errs.max() / last_errs.min() < rtol:
            return 'CONVERGED_NO_MORE_DECREASE'

        return False


    def optimize(self, maxiter=100):
        R, dR = self.predict_R(self.params)
        self.errors.append(self.error(R))
        self.residuals.append(np.log2(R/self.R0))

        # print self.params
        
        try:
            while not self.converged() and self.t < maxiter:
                local_grad = self.grad_from_R(R, dR)
                local_grad.A0 *= self.params.k * 4 #psam_matrix.sum()
                # local_grad.A0 = 0
                # local_grad.beta = 0

                descent = self.RMSprop(- local_grad).unity()

                s,n = self.line_search(self.params, descent)
                self.scales.append(s)
                self.nfevs.append(n)

                upd = descent * s
                print upd
                print ">>>>UPDATE", self.params.A0, upd.A0
                self.params = self.apply_delta(self.params, upd)
                print "after apply", self.params.A0
                # if not self.t % 5:
                #     self.new_subsample()
                #     R0, dR0 = self.predict_R(correct_params, grad=False)
                #     self.set_reference(R0)

                print "=" * 50
                R, dR = self.predict_R(self.params)
                self.errors.append(self.error(R))
                self.residuals.append(np.log2(R/self.R0))
                self.t += 1
                print "step:", self.t, self.errors[-1], self.n_part_func, self.n_grad, s

        except KeyboardInterrupt:
            pass
        
        if self.t < maxiter:
            self.status = self.converged()
        else:
            self.status = "MAX_ITER"

        print "n_part_func={self.n_part_func} n_grad={self.n_grad}".format(self=self)
        print "optimization ended with status", self.status
        # print "last gradient"
        # print self.past_grad
        # print "squared"
        # print self.past_sqg
        
        return self

    def plot_gradients(self, psam_correct, local_grad, grad):
        print descent
        pp.figure()
        pp.subplot(131)
        self.plot_psam(unity_matrix(psam_correct - self.psam),'actual delta')
        pp.subplot(132)
        self.plot_psam(unity_matrix(local_grad),'local gradient')
        pp.subplot(133)
        self.plot_psam(unity_matrix(grad),'RMSprop')
        pp.show()
        pp.close()

        
    def plot_psam(self, psam, title):
        pp.pcolor(psam.T, cmap='bwr', vmin=-1, vmax=+1)
        pp.xlabel(title)
        pp.ylabel("base")
        pp.yticks(np.arange(0.5,4.5,1), ['A','C','G','U'])
        pp.colorbar(label='weight', shrink=.5, orientation='horizontal')

from cska.pwm import PSAM
# # motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG'], [1., .12, .1])
# motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GCATG'], [1., 1., 1.])
# motif_variant = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG','GGGGG'], [1., .2, .8,.1])
# # motif_correct = PSAM.from_kmer_variants(['GCATG', ], [1., ])
# # motif = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG'], [5., .06, .05])
# # sm = reads.seqm

# psam_correct = motif_correct.psam * motif_correct.A0 + 1e-6 
# k = len(psam_correct)
# psam_variant = motif_variant.psam * motif_variant.A0 + 1e-6 

# correct_params = ModelParametrization(k, 1, psam=psam_correct, A0=1., betas=[.0])
# initial_params = ModelParametrization(k, 1, psam=psam_variant, A0=1.)

# print "REFERENCE"
# print correct_params
# print "SEED"
# print initial_params
# sys.exit(0)
def emp_grad_Z(seqm, psam0, eps=1e-4):
    Z0 = SPA_part_func(seqm, psam0)
    psam = np.array(psam0)
    k = len(psam0)
    l = len(Z0)
    grad = np.zeros((l,k,4), dtype=np.float64)

    for i in range(len(psam0)):
        for j in range(4):
            psam[i,j] = psam0[i,j] + eps
            Z = SPA_part_func(seqm, psam)
            psam[i,j] = psam0[i,j]
            dZ = Z - Z0

            grad[:,i,j] = dZ/eps
    
    return grad

def grad_Z(seqm, psam0):
    Z0 = SPA_part_func(seqm, psam0)
    psam = np.array(psam0)
    k = len(psam0)
    l = len(Z0)
    grad = np.zeros((l,k,4), dtype=np.float64)

    for i in range(l):
        for d in range(k):
            n = seqm[i+d]
            grad[i,d,n] = Z0[i] / psam[d,n]
    
    return grad
    

def SPA_part_func(seqm, psam):
    L = len(seqm)
    k = len(psam)
    l = L - k + 1
    Z = np.ones(l, dtype=float)
    for i in range(l):
        for d in range(k):
            n = seqm[i+d]
            Z[i] *= psam[d,n]

    return Z

def test_grad():
    seqm = G.sub_padded[90]
    # psam = G.psam

    eps = 1e-4
    # Z = SPA_part_func(seqm, psam)
    # emp = emp_grad_Z(seqm, psam, eps=eps).sum(axis=0)
    # ana = grad_Z(seqm, psam).sum(axis=0)
    print ">>>>> at optimum"
    print "empirical"
    print G.emp_grad(correct_params, eps=1e-4).unity_bounded()
    print "analytical"
    print G.ana_grad(correct_params).unity_bounded()

    print ">>>>> at start point"
    print "empirical"
    print G.emp_grad(initial_params, eps=1e-4).unity_bounded()
    print "analytical"
    print G.ana_grad(initial_params).unity_bounded()




    





# G = GradientDescent(reads, initial_params, None, dec=.7, k_monitor=5, subsample=1.)
# print "generating reference state"
# R0, dR0 = G.predict_R(correct_params)
# I = R0.argsort()[::-1]
# for i in I[:10]:
#     print "simulated R-value", cyska.index_to_seq(i, G.k_monitor), R0[i]

# G.set_reference(R0)
# test_grad()
# _R0 = np.array(R0)
# print "R0:", R0
# R, dR = G.predict_R(G.params, grad=True)
# grad = G.grad_from_R(R, dR)
# print "initial error", G.error(R)
# print "initial gradient"
# print grad


# # sanity checks
# R_test, dR_test = G.predict_R(correct_params)
# grad0 = G.grad_from_R(R_test, dR_test)
# print "gradient at optimum"
# print grad0
# assert np.allclose(R_test, R0)
# assert np.allclose(grad0.data, 0)
# assert G.error(R_test) == 0

# G.optimize()    # for m in G.past_grad:
# # sys.exit(0)
#     #     print np.round(m,3)


# from scipy.optimize import minimize

# def loss(params):
#     psam = np.array(params.reshape(G.psam.shape), dtype=np.float32)
#     R, dR = G.predict_R(psam, grad=True)

#     err = G.error(R)
#     jac = np.array(G.grad_from_R(R, dR).flatten(), dtype=float)
#     return float(err), jac


# def scipy_minimize():
#     n = len(psam_variant)*4
#     bounds = [ (1e-9, 1), ]*n
#     print bounds
#     x0 = np.array(psam_variant.flatten(), dtype=float)
#     print x0
#     res = minimize(loss, x0, jac=True, bounds=bounds, options={ 'eps' : 1e-4 })

#     print res
#     sys.exit(1)

# # scipy_minimize()
# # sys.exit(0)

# # sys.exit(0)

# pp.figure(figsize=(6,12))
# pp.subplot(411)
# pp.pcolor(np.array(G.residuals).T, cmap='RdBu', vmin=-.1, vmax=.1)
# pp.xlabel('time step')
# pp.ylabel('kmer index')
# pp.colorbar(label='R-value difference', orientation='horizontal', shrink=.5)

# pp.subplot(412)
# pp.semilogy(np.array(G.scales), label='step size')
# pp.legend(loc='upper right')
# pp.subplot(413)
# pp.semilogy(np.array(G.errors), label='mean squared R-value error')
# pp.legend(loc='upper right')
# pp.subplot(414)
# pp.semilogy(G.nfevs, label='no. function evaluations')
# pp.legend(loc='upper right')
# pp.tight_layout()
# # pp.legend()
# # print "R0:", R0
# # print "R1:", R1

# # print "d_R"
# # for nt, grad, f in zip('ACGT', dR, R1-R0):
# #     print ">>>", nt, f
# #     print np.round(grad, 3)

# # print "resulting gradient"
# # print np.round(grad, 3)
# # print np.round(unity(grad), 3)

# # print delta(psam_correct, psam_variant)
# # print delta(psam_correct, psam_variant)

# # print "R0:", R0
# # print "R1:", R1
# # R1, dR = predict_R(psam_variant)
# # print "R1:", R1
# # pp.plot(R1, '.')
# # grad = -2*((R1 - R0)[:,np.newaxis,np.newaxis] * dR).sum(axis=0)
# # psam_variant -= .1*unity(grad)
# # R1, dR = predict_R(psam_variant)
# # pp.plot(R1, '.')
# pp.show()

# psam_variant += .01*unity(grad)
# print delta(psam_correct, psam_variant)



# grad = np.array(grad)
# emp = np.array(emp)
# pp.figure()
# pp.plot(grad.sum(axis=0), emp.sum(axis=0),'.')
# pp.show()

import os
import unittest
import copy

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

        # self.assert

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    import gzip
    path = os.path.join(os.path.dirname(__file__), '../tests/reads_20.txt.gz')
    # TODO: include small amount of raw data in git repo for testing!

    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=100000)
    unittest.main(verbosity=2)
