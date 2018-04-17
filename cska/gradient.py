import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from cska import auto_detect
from cska.reads import RBNSReads


# CachedBase.debug_caching=True
test_reads = [
    "TAATTTTTGCATGAAAAATCGAT",
    "AGAGGACGGAGAGAGTCGCGCGA",
    "CGCACGCGTCGCGATAGCGTCGA",
]
reads = RBNSReads.from_seqs(test_reads)


from cska.pwm import PSAM
motif = PSAM.from_kmer_variants(['GCATG', 'GCACG', 'GGATG'], [1., .12, .002])
sm = reads.seqm

psam = motif.psam + 1e-6

k = motif.n
L = reads.L - k + 1
def part_func(seq, psam):
    Z = np.ones(L, dtype=np.float32)
    for i in range(L):
        for d in range(k):
            n = seq[i+d]
            Z[i] *= psam[d,n]
    
    return Z

def gradient(seq, psam, Z):
    grad = np.zeros(psam.shape, dtype=np.float32)
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


grad = []
emp = []
for read, seq in zip(test_reads, reads.seqm):
    print read
    Z = part_func(seq, psam)
    print Z.sum()
    grad.append( gradient(seq, psam, Z).flatten() )
    emp.append( emp_grad(seq, psam, Z).flatten() )

grad = np.array(grad)
emp = np.array(emp)
import matplotlib.pyplot as pp
pp.figure()
pp.plot(grad.sum(axis=0), emp.sum(axis=0),'.')
pp.show()


