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


class ModelParametrization(object):
    def __init__(self, psam = [], A0=1., per_sample = []):
        self.psam = psam
        self.k = len(psam)
        self.A0 = A0
        self.per_sample = np.array(per_sample)
        if len(psam):
            self.update_len()
    
    def update_len(self):
        self.n = 1 + self.psam.size + self.per_sample.size

    def as_vector(self, dtype=np.float32):
        v = np.zeros(self.n, dtype=dtype)
        v[0] = self.A0
        v[1:1+self.psam.size] = self.psam.flatten()[:]
        v[1+self.psam.size:] = self.per_sample.flatten()[:]

        return v
    
    @classmethod
    def from_vector(cls, vec, k):
        n = len(vec)
        psam = np.reshape(vec[1:1+k*4], (k,4))
        return cls(psam=psam, A0= vec[0], per_sample=vec[1+k*4:])

    def set_vector(self, vec, dtype = np.float32):
        self.A0 = vec[0]
        self.psam = np.reshape(self.psam.shape, vec[1:1+self.psam.size])
        self.per_sample = np.reshape(self.per_sample.shape, vec[1+self.psam.size:])

    @property
    def psam_vec(self):
        return self.as_vector()[:1+self.psam.size]

    def set_psam_vec(self, vec):
        self.A0 = vec[0]
        self.psam = np.reshape(self.psam.shape, vec[1:])

    def __str__(self):
        buf = []
        buf.append("PSAM")
        buf.append("A0={0:.4f}".format(self.A0))
        buf.append("\tA\t\tC\t\tG\t\tU")
        for row in self.psam:
            buf.append("\t".join(["{0:>10.3f}".format(x) for x in row]))

        return '\n'.join(buf)


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
    
    def predict_R(self, params, P=1., grad=True):
        psam = params.psam_vec
        # print "psam params vector", psam
        cyZ = cyska.PSAM_partition_function(self.sub_padded, self.sub_acc, params.psam, openen_ofs=self.acc_ofs)
        cyZ *= params.A0
        Zj = cyZ.sum(axis=1)
        psi = P*Zj/ (P*Zj+1)
        # print "PSI", psi, psi.min(), psi.mean(), psi.max()
        self.n_part_func += 1

        if grad:
            pi, d_pi = cyska.PSAM_kmer_gradient(self.sub_padded, cyZ, Zj, psi, self.sub_im, psam, self.k_monitor)
            self.n_grad += 1
            R = pi / pi.sum() / self.f0
            # print R.shape, d_pi.shape
            dR = (R/pi)[:,np.newaxis] * (d_pi - (self.f0 * R)[:,np.newaxis] * d_pi.sum(axis=0)[np.newaxis,:])

            # if not self.R0 is None:
                # worst = ((R - self.R0)**2).argmax()
                # print "worst kmer:", worst, cyska.index_to_seq(worst, self.k_monitor), R[worst], R0[worst]
                # print "corresponding gradient"
                # print "R0 constant?", np.allclose(R0, _R0)
                # print d_pi[worst]
                # def dump(seq):
                #     i = cyska.seq_to_index(seq)
                #     print 'dR[{seq}]'.format(seq=seq), (R-self.R0)[i]
                #     print np.round(unity_matrix(dR[i]),3)
                # dump('gcacg')
                # dump('gcagg')
                # dump('gcaug')




            return R, dR
        else:
            pi = cyska.weighted_kmer_counts(self.sub_im, psi, self.k_monitor)
            R = pi / pi.sum() / self.f0
            return R, None

    def grad_from_R(self, R1, dR):
        grad = 2 * ((1. * (R1 - self.R0))[:,np.newaxis] * dR).sum(axis=0)
        return unity_matrix(grad)

    def ana_grad(self, params):
        R, dR = self.predict_R(params)
        return self.grad_from_R(R,dR)

    def emp_grad(self, params, eps=1e-4):
        v0 = params.as_vector()
        k = params.k
        v = np.array(v0, dtype=np.float32)

        grad = np.zeros(k*4 + 1, dtype=np.float32)
        R0, dR0 = self.predict_R(params, grad=False)
        err0 = self.error(R0)

        for i in range(k*4+1):
            v[i] = v0[i] + eps
            R, dR = self.predict_R(ModelParametrization.from_vector(v,k), grad=False)
            v[i] = v0[i]
            derr = self.error(R) - err0

            grad[i] = derr/eps
        
        return grad


    @staticmethod
    def apply_delta(params, delta0):
        psam = params.psam
        delta = np.reshape(delta0[1:], psam.shape)

        p = np.clip(psam + delta, 1e-9, None)
        p /= p.max(axis=1)[:,np.newaxis]
        
        A0 = max(1e-9, params.A0 + delta0[0])
        return ModelParametrization(psam=p, A0=A0)

    def line_search(self, params, grad, debug=False):
        from scipy.optimize import minimize_scalar

        r0, bla = self.predict_R(params, grad=False)
        e0 = self.error(r0)
        psam0 = np.array(params.psam_vec)
        res0 = r0 - self.R0
        if debug:
            scales = 10**(np.linspace(-3,0,20))
            errs = []
            deltas = []
            residuals = []
            for s in scales:
                m = self.apply_delta(psam,s * grad)    
                R, dR = self.predict_R(m, grad=False)
                e = self.error(R)
                d = delta(m, psam_correct)
                print "changes", m - psam
                print s, '->', R[582], self.R0[582], "mean sq. error change", e - e0, "mean delta change", d - d0
                res = R - self.R0
                worst = (res**2).argmax()

                error_change = (res - res0)**2
                most = (error_change**2).argmax()
                print s, 'kmer with largest error', cyska.index_to_seq(worst, self.k_monitor), "change in error", res0[worst], "->", res[worst]
                print s, 'kmer with largest error change', cyska.index_to_seq(most, self.k_monitor), "change in error", res0[most], "->", res[most]
                deltas.append(d)
                # print psam_
                errs.append(e)
                residuals.append( np.log2(res/res0) )
            
            errs = np.array(errs)
            deltas = np.array(deltas)
            residuals = np.array(residuals)

            i = errs.argmin()
            print "start point"
            print psam
            print "grad"
            print grad
            print "'best' matrix?"
            m = self.apply_delta(psam,scales[i] * grad)
            print m
            print "known optimum"
            print psam_correct

            pp.figure(figsize=(6,12))
            pp.subplot(311)
            pp.pcolor(residuals.T, cmap='RdBu', vmin=-.1, vmax=.1)
            pp.xlabel('line search parameter')
            pp.ylabel('kmer index')
            pp.colorbar(label='R-value difference', orientation='horizontal', shrink=.5)

            pp.subplot(312)
            pp.semilogx(scales, errs, label='R^2 error')
            pp.axhline(e0)

            pp.legend(loc='upper right')
            pp.subplot(313)
            pp.semilogx(scales, deltas, label='matrix element deviation')
            pp.axhline(d0)
            pp.legend(loc='upper right')
            pp.tight_layout()
            pp.show()


        def err(x):
            s = np.exp(x)
            m = self.apply_delta(params, s * grad)
            R, dR = self.predict_R(m, grad=False)
            e = self.error(R)
            # print s,"->", e - e0
            return e

        min_step = 1e-6
        max_step = 10.
        # force some movement at the beginning
        # if self.t < 10:
        #     min_step = 1e-3

        res = minimize_scalar(err, method='Bounded', bounds=np.log(np.array([min_step, max_step])), options=dict(maxiter=50))
        return np.exp(res.x), res.nfev

    def momentum_grad(self, local_grad):

        if self.past_grad is None:
            grad = local_grad
        else:
            grad = self.dec * self.past_grad + (1- self.dec) * local_grad

        self.past_grad = grad

        # self.past_grad.append(local_grad)
        # if len(self.past_grad) > self.tau:
        #     self.past_grad.pop(0)

        # past = np.array(self.past_grad)
        # grad = past.mean(axis=0)
        # past *= self.dec
        # self.past_grad = list(past)

        # return unity_matrix(grad)
        return .01 * grad

    def RMSprop(self, local_grad):
        # grad = local_grad + self.past_grad * self.dec
        # sqg = local_grad**2 + self.past_sqg * self.dec
        if self.past_grad is None:
            self.past_grad = local_grad

        # if self.past_sqg is None:
            # self.past_sqg = local_grad**2


        m = self.dec * self.past_grad + (1 - self.dec) * local_grad
        s = self.dec * self.past_sqg + (1 - self.dec) * local_grad**2
        v = self.dec * self.past_var + (1 - self.dec) * (local_grad - self.past_grad)**2

        self.past_grad = m
        self.past_sqg = s
        self.past_var = v

        upd = unity_matrix(local_grad * (np.sqrt(s) + .001))
        
        # self.past_sqg = v
        # print "local gradient"
        # print np.round(local_grad, 3)
        # print "running mean gradient"
        # print np.round(m, 3)
        # print "running mean squared gradient"
        # print np.round(s, 3)
        # print "estimated variance"
        # print np.round(v, 3)


        # print "current update"
        # print np.round(upd, 3)
        # print "="*50

        return upd
        # return 0.1* upd



    def step(self):
        R, dR = self.predict_R(self.params)
        self.errors.append(self.error(R))
        self.residuals.append(np.log2(R/self.R0))

        local_grad = - unity_matrix(self.grad_from_R(R, dR))
        local_grad[0] *= 20
        # emp_grad = - unity_matrix(self.emp_grad(self.psam))

        # print "local grad"
        # print local_grad
        # print "emp_grad"
        # print emp_grad

        # grad = self.momentum_grad(local_grad)
        grad = self.RMSprop(local_grad)
        # pp.figure()
        # pp.subplot(131)
        # self.plot_psam(unity_matrix(psam_correct - self.psam),'actual delta')
        # pp.subplot(132)
        # self.plot_psam(unity_matrix(local_grad),'local gradient')
        # pp.subplot(133)
        # self.plot_psam(unity_matrix(grad),'RMSprop')

        # pp.show()
        # pp.close()

        s,n = self.line_search(self.params, grad)
        #s, n = 1,1
        self.scales.append(s)
        self.nfevs.append(n)

        self.params = self.apply_delta(self.params, s*grad)
        self.t += 1
        if not self.t % 5:
            self.new_subsample()
            R0, dR0 = G.predict_R(correct_params, grad=False)
            self.set_reference(R0)

        print "current"
        print self.params
        print "step()", self.t, self.errors[-1], self.n_part_func, self.n_grad
        return self.errors[-1]

    def converged(self, tol=1e-6, tau=10):
        if len(self.errors) < tau:
            return False
        
        last_errs = np.array(self.errors[-tau:])
        if last_errs.max() - last_errs.min() < tol:
            return True

    def optimize(self, maxiter=100):
        try:
            for t in range(maxiter):
                err = G.step()
                if self.converged():
                    break
        except KeyboardInterrupt:
            pass
        
        if self.t < maxiter:
            self.status = "CONVERGED"
        else:
            self.status = "MAX_ITER"

        print "optimization ended with status", self.status
        print "final parametrization"
        print self.params
        print "last gradient"
        print self.past_grad
        print "squared"
        print self.past_sqg
        print "n_part_func={self.n_part_func} n_grad={self.n_grad}".format(self=self)
        
        return self

    def plot_psam(self, psam, title):
        pp.pcolor(psam.T, cmap='bwr', vmin=-1, vmax=+1)
        pp.xlabel(title)
        pp.ylabel("base")
        pp.yticks(np.arange(0.5,4.5,1), ['A','C','G','U'])
        pp.colorbar(label='weight', shrink=.5, orientation='horizontal')

from cska.pwm import PSAM
# motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG'], [1., .12, .1])
motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GCATG'], [1., 1., 1.])
motif_variant = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG','GGGGG'], [1., .2, .8,.1])
# motif_correct = PSAM.from_kmer_variants(['GCATG', ], [1., ])
# motif = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG'], [5., .06, .05])
# sm = reads.seqm

psam_correct = motif_correct.psam * motif_correct.A0 + 1e-6 
k = len(psam_correct)
psam_variant = motif_variant.psam * motif_variant.A0 + 1e-6 

correct_params = ModelParametrization(psam=psam_correct, A0=5., per_sample=[0.,.1,.1,0])
initial_params = ModelParametrization(psam=psam_variant, A0=1., per_sample=[0.,.1,.1,0])

# np.random.seed(4711)
# psam_variant = psam_correct + np.array(np.random.rand(k,4) * .01, dtype=np.float32)
# psam_variant = GradientDescent.apply_delta(psam_variant,0)
print "REFERENCE"
print correct_params
print "SEED"
print initial_params

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
    print "empirical"
    print unity_matrix(G.emp_grad(G.params, eps=1e-5))
    print "analytical"
    print unity_matrix(G.ana_grad(G.params))
    # print ana
    # print np.allclose(emp, ana, 10*eps)



    





G = GradientDescent(reads, initial_params, None, dec=.7, k_monitor=6, subsample=1.)
print "generating reference state"
R0, dR0 = G.predict_R(correct_params)
I = R0.argsort()[::-1]
for i in I[:10]:
    print "simulated R-value", cyska.index_to_seq(i, G.k_monitor), R0[i]

G.set_reference(R0)
# test_grad()
# sys.exit(0)
_R0 = np.array(R0)
print "R0:", R0
R, bla = G.predict_R(G.params, grad=False)
print "initial error", G.error(R)


# sanity checks
R_test, dR_test = G.predict_R(correct_params)
grad0 = G.grad_from_R(R_test, dR_test)
print grad0
assert np.allclose(R_test, R0)
assert np.allclose(grad0, 0)
assert G.error(R_test) == 0

G.optimize()    # for m in G.past_grad:
    #     print np.round(m,3)


from scipy.optimize import minimize

def loss(params):
    psam = np.array(params.reshape(G.psam.shape), dtype=np.float32)
    R, dR = G.predict_R(psam, grad=True)

    err = G.error(R)
    jac = np.array(G.grad_from_R(R, dR).flatten(), dtype=float)
    return float(err), jac


def scipy_minimize():
    n = len(psam_variant)*4
    bounds = [ (1e-9, 1), ]*n
    print bounds
    x0 = np.array(psam_variant.flatten(), dtype=float)
    print x0
    res = minimize(loss, x0, jac=True, bounds=bounds, options={ 'eps' : 1e-4 })

    print res
    sys.exit(1)

# scipy_minimize()
# sys.exit(0)

# sys.exit(0)

pp.figure(figsize=(6,12))
pp.subplot(411)
pp.pcolor(np.array(G.residuals).T, cmap='RdBu', vmin=-.1, vmax=.1)
pp.xlabel('time step')
pp.ylabel('kmer index')
pp.colorbar(label='R-value difference', orientation='horizontal', shrink=.5)

pp.subplot(412)
pp.semilogy(np.array(G.scales), label='step size')
pp.legend(loc='upper right')
pp.subplot(413)
pp.semilogy(np.array(G.errors), label='mean squared R-value error')
pp.legend(loc='upper right')
pp.subplot(414)
pp.semilogy(G.nfevs, label='no. function evaluations')
pp.legend(loc='upper right')
pp.tight_layout()
# pp.legend()
# print "R0:", R0
# print "R1:", R1

# print "d_R"
# for nt, grad, f in zip('ACGT', dR, R1-R0):
#     print ">>>", nt, f
#     print np.round(grad, 3)

# print "resulting gradient"
# print np.round(grad, 3)
# print np.round(unity(grad), 3)

# print delta(psam_correct, psam_variant)
# print delta(psam_correct, psam_variant)

# print "R0:", R0
# print "R1:", R1
# R1, dR = predict_R(psam_variant)
# print "R1:", R1
# pp.plot(R1, '.')
# grad = -2*((R1 - R0)[:,np.newaxis,np.newaxis] * dR).sum(axis=0)
# psam_variant -= .1*unity(grad)
# R1, dR = predict_R(psam_variant)
# pp.plot(R1, '.')
pp.show()

# psam_variant += .01*unity(grad)
# print delta(psam_correct, psam_variant)



# grad = np.array(grad)
# emp = np.array(emp)
# pp.figure()
# pp.plot(grad.sum(axis=0), emp.sum(axis=0),'.')
# pp.show()


