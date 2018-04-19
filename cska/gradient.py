import logging
# logging.basicConfig(level=logging.DEBUG)
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
reads = RBNSReads.from_seqs(test_reads, pseudo_count=1)
# reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', n_max=50000)


def unity_matrix(M):
    F = M.flatten()
    i = np.fabs(F).argmax()
    if F[i] > 0:
        return M / F[i]
    else:
        return -M / F[i]

class GradientDescent(object):
    def __init__(self, reads, psam_seed, R0, k_monitor=2, dec=.75, tau=5):
        self.reads = reads
        self.R0 = R0
        self.psam = psam_seed
        self.n = len(psam_seed)
        adap5 = cyska.seq_to_bits(reads.adap5)
        adap3 = cyska.seq_to_bits(reads.adap3)
        self.padded = cyska.seqm_pad_adapters(reads.seqm, adap5, adap3, self.n)

        self.k_monitor = k_monitor
        self.im = reads.get_index_matrix(k_monitor)
        f0 = reads.kmer_frequencies(k_monitor)
        self.f0 = f0 / f0.sum()

        # momentum smoothing of the gradient
        self.past_grad = []
        self.dec = dec
        self.tau = tau

        # records
        self.errors = []
        self.residuals = []
        self.nfevs = []
        self.scales = []

    def error(self, R):
        return ((self.R0 - R)**2).mean()
    
    def predict_R(self, psam, P=1., grad=True):

        cyZ = cyska.PSAM_partition_function(self.padded, np.ones(self.padded.shape, dtype=np.float32), psam)
        Zj = cyZ.sum(axis=1)
        psi = P*Zj/ (P*Zj+1)
        # print "PSI", psi, psi.min(), psi.mean(), psi.max()

        if grad:
            pi, d_pi = cyska.PSAM_kmer_gradient(self.padded, cyZ, Zj, psi, self.im, psam, self.k_monitor)
            R = pi / pi.sum() / self.f0
            # print R.shape, d_pi.shape
            dR = (R/pi)[:,np.newaxis,np.newaxis] * (d_pi - (self.f0 * R)[:,np.newaxis, np.newaxis] * d_pi.sum(axis=0)[np.newaxis,:,:])
            return R, dR
        else:
            pi = cyska.weighted_kmer_counts(self.im, psi, self.k_monitor)
            R = pi / pi.sum() / self.f0
            return R, None

    def grad_from_R(self, R1, dR):
        grad = -2 * ((R1 - self.R0)[:,np.newaxis,np.newaxis] * dR).sum(axis=0)
        return unity_matrix(grad)

    def apply_delta(self, psam, delta):
        p = np.clip(psam + delta, 1e-9, None)
        p /= p.max(axis=1)[:,np.newaxis]
        return p

    def line_search(self, psam, grad):
        from scipy.optimize import minimize_scalar
        def err(x):
            s = np.exp(x)
            R, dR = self.predict_R(self.apply_delta(psam,s * grad), grad=False)
            e = self.error(R)
            return e

        res = minimize_scalar(err, method='Bounded', bounds=np.log(np.array([1e-8, .5])), options=dict(maxiter=40))
        return np.exp(res.x), res.nfev

    def momentum_grad(self, local_grad):
        self.past_grad.append(local_grad)
        if len(self.past_grad) > self.tau:
            self.past_grad.pop(0)

        past = np.array(self.past_grad)
        grad = past.mean(axis=0)
        past *= self.dec
        self.past_grad = list(past)

        return unity_matrix(grad)
        
    def step(self):
        R, dR = self.predict_R(self.psam)
        self.errors.append(self.error(R))
        self.residuals.append(R - self.R0)

        local_grad = self.grad_from_R(R, dR)
        grad = self.momentum_grad(local_grad)

        s,n = self.line_search(self.psam, grad)
        self.scales.append(s)
        self.nfevs.append(n)

        self.psam = self.apply_delta(self.psam, s*grad)

        return self.errors[-1]


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
# np.random.seed(4711)
# psam_variant = np.array(np.random.rand(k,4) * .01 + psam_correct, dtype=np.float32)

print "REFERENCE"
print psam_correct
print "SEED"
print psam_variant


G = GradientDescent(reads, psam_variant, None, tau=5, k_monitor=2)
R0, dR0 = G.predict_R(psam_correct)

G.R0 = R0
print "R0:", R0

for t in range(200):
    print t, G.step()
    # for m in G.past_grad:
    #     print np.round(m,3)
print G.psam
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


