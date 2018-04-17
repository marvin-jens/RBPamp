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
    # "CGCACGCGCCCCGCCCGCGCCGC",
    # "AGAGGACGGAGAGAGTCGCGCGA",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
    "TTTTTTTTGCACGTTTTTTTTTT",
]
reads = RBNSReads.from_seqs(test_reads)


from cska.pwm import PSAM
motif = PSAM.from_kmer_variants(['GCAGG', 'GCACG', 'GGATG'], [1., .12, .002])
sm = reads.seqm

psam = motif.psam + 1e-6

k = motif.n
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
cyZ = cyska.PSAM_partition_function(padded, np.ones(padded.shape, dtype=np.float32), psam)

grad = []
emp = []
for read, seq, z in zip(test_reads, padded, cyZ):
    print read
    Z = part_func(seq, psam)
    print Z.shape, z.shape
    print Z
    print z
    print Z.sum(), z.sum()
    grad.append( gradient(seq, psam, Z).flatten() )
    emp.append( emp_grad(seq, psam, Z).flatten() )

Zj = cyZ.sum(axis=1)

P = 1.
psi = P*Zj/ (P*Zj+1)
print "PSI", psi, psi.min(), psi.mean(), psi.max()

im = reads.get_index_matrix(1)
kmer_grad = cyska.PSAM_kmer_gradient(padded, cyZ, Zj, psi, im, psam, 1)

for grad in kmer_grad:
    print grad


# grad = np.array(grad)
# emp = np.array(emp)
# pp.figure()
# pp.plot(grad.sum(axis=0), emp.sum(axis=0),'.')
# pp.show()


