import logging
logging.basicConfig(level=logging.DEBUG)
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
    "GGGGGGGGGGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGGGGGGGGG",
    "GGGGGTAGGGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGGGGGGGGG",
    "GGGGGGGTCGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGGGGGGGGG",
]
reads = RBNSReads.from_seqs(test_reads)


from cska.pwm import PSAM
# motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG'], [1., .12, .1])
motif_correct = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GCATG'], [1., 1., 1.])
motif_variant = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG'], [1., .5, .8])
# motif_correct = PSAM.from_kmer_variants(['GCATG', ], [1., ])
# motif = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG'], [5., .06, .05])
# sm = reads.seqm

psam_correct = motif_correct.psam + 1e-6
psam_variant = motif_variant.psam + 1e-6

k = len(psam_correct)
L = reads.L - k + 1
def part_func(seq, psam):
    L = len(seq) -  k + 1
    Z = np.ones(L, dtype=np.float32)
    for i in range(L):
        for d in range(k):
            n = seq[i+d]
            Z[i] *= psam[d,n]
    
    return Z

def gradient(seq, psam, Z):
    grad = np.zeros(psam.shape, dtype=np.float32)
    L = len(seq) -  k + 1
    for i in range(L):
        for d in range(k):
            n = seq[i+d]
            grad[d,n] += Z[i] / psam[d,n]
    return grad

def emp_grad(seq, psam, Z0, delta=1e-5):
    grad = np.zeros(psam.shape, dtype=np.float32)

    for d in range(k):
        for n in range(4):
            p = np.array(psam)
            p[d,n] += delta
            Z = part_func(seq, p)

            grad[d,n] += (Z-Z0).sum()/delta

    return grad


adap5 = cyska.seq_to_bits(reads.adap5)
adap3 = cyska.seq_to_bits(reads.adap3)
    
padded = cyska.seqm_pad_adapters(reads.seqm, adap5, adap3, k)

# grad = []
# emp = []
# for read, seq, z in zip(test_reads, padded, cyZ):
#     print read
#     Z = part_func(seq, psam)
#     print Z.shape, z.shape
#     print Z
#     print z
#     print Z.sum(), z.sum()
#     grad.append( gradient(seq, psam, Z).flatten() )
#     emp.append( emp_grad(seq, psam, Z).flatten() )


def predict_R(psam, k=2, P=1.):
    f0 = reads.kmer_frequencies(k)
    f0 /= f0.sum()

    cyZ = cyska.PSAM_partition_function(padded, np.ones(padded.shape, dtype=np.float32), psam)
    Zj = cyZ.sum(axis=1)
    psi = P*Zj/ (P*Zj+1)
    # print "PSI", psi, psi.min(), psi.mean(), psi.max()
    im = reads.get_index_matrix(k)

    pi, d_pi = cyska.PSAM_kmer_gradient(padded, cyZ, Zj, psi, im, psam, k)

    # print "d_pi"
    # for nt, grad, f in zip('ACGT', d_pi, pi):
    #     print ">>>", nt, f
    #     print PSAM(grad, A0=1)

    R = pi / pi.sum() / f0
    # print R.shape, d_pi.shape
    dR = (R/pi)[:,np.newaxis,np.newaxis] * (d_pi - (f0 * R)[:,np.newaxis, np.newaxis] * d_pi.sum(axis=0)[np.newaxis,:,:])

    return R, dR

def unity(M):
    F = M.flatten()
    i = np.fabs(F).argmax()
    if F[i] > 0:
        return M / F[i]
    else:
        return -M / F[i]

def grad_from_R(R0, R1, dR):
    grad = -2*((R1 - R0)[:,np.newaxis,np.newaxis] * dR).sum(axis=0)
    return unity(grad)

def apply_delta(psam, delta):
    p = np.clip(psam + delta, 1e-6, None)
    p /= p.max(axis=1)[:,np.newaxis]
    return p

def line_search(R0, psam, grad):
    # grid = 10**np.linspace(-6,-.001)
    # errs = []
    # for s in grid:
    #     trial = np.clip(psam + s * grad, 1e-6, None)
    #     R, dR = predict_R(trial)
    #     errs.append( ((R-R0)**2).sum() )


    from scipy.optimize import minimize_scalar
    # print "line_search"
    # print np.round(grad,2)
    def err(x):
        s = np.exp(x)
        R, dR = predict_R(apply_delta(psam,s * grad))
        e = ((R-R0)**2).sum()
        # print s,"->",e
        return e

    res = minimize_scalar(err, method='Bounded', bounds=np.log(np.array([1e-6, .5])))
    # print res
    # pp.figure()
    # pp.loglog(grid, errs)
    # pp.axvline(np.exp(res.x),color='r')

    return np.exp(res.x)

def delta(M1, M2):
    return np.fabs(M1 - M2).sum()

# pp.figure(0)
R0, dR0 = predict_R(psam_correct)
# pp.plot(R0, 'x', label='correct')
R1, dR = predict_R(psam_variant)
# pp.plot(R1, '.k', label='initial')

errs = [(R1-R0),]
print "R0:", R0
for t in range(100):
    
    print "R1:", R1
    # if not t % 10:
        # pp.plot(R1, '.')
    
    grad = grad_from_R(R0, R1, dR)
    s = line_search(R0, psam_variant, grad)
    psam_variant = apply_delta(psam_variant, s*grad)
    print s
    R1, dR = predict_R(psam_variant)
    errs.append((R1-R0))
    # pp.figure(0)
    # pp.plot(R1, '.', label=str(t+1))

pp.figure()
pp.pcolor(np.array(errs), cmap='seismic', vmin=-2, vmax=2)
# pp.legend()
print motif_correct
print PSAM(psam_variant)
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


