#!/usr/bin/env python
import numpy as np

bases = 'ACGU'
base_idx = { 
    'A' : 0,
    'C' : 1,
    'G' : 2,
    'T' : 3,
    'U' : 3 
}

ambig = "-NMRWSYKVHDBACGU"
ambig_vectors = np.array([
    # A    C    G    T
    [0.0, 0.0, 0.0, 0.0],
    [.25, .25, .25, .25],
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0, 0.5],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.5, 0.5],
    [.33, .33, .33, 0.0],
    [.33, .33, 0.0, .33],
    [.33, 0.0, .33, .33],
    [0.0, .33, .33, .33],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

def project_column(col):
    n = col.sum()
    if n:
        col /= n
    i = (col[np.newaxis,:] * ambig_vectors).sum(axis=1).argmax()
    return ambig[i]
    

class PSAMModel(object):
    def __init__(self, psam, A0 = 1e-6):
        self.psam = np.array(psam, dtype=np.float32)
        self.A0 = A0
        self.n = len(psam)
        
    @classmethod
    def from_kmer(cls, kmer, **kwargs):
        kmer = kmer.upper()
        cols = []
        for nt in kmer:
            col = np.zeros(4)
            col[base_idx[nt]] = 1
            cols.append(col)
        
        psam = np.array(cols, dtype=np.float32)

        return cls(psam, **kwargs)
    
    def scan(self, mdl, max_shift=2):
        #print "aligning",
        #print mdl
        idx = mdl.psam

        L = mdl.n
        l = L - self.n + 1
        x = np.arange(-max_shift, l + max_shift)
        #print x
        #print idx
        scores = []
        for i in x:
            start_psam = max(0, -i)
            end_psam = self.n - max(0, i - l + 1 )
            start_idx = max(0, i)
            
            l_psam = end_psam - start_psam
            
            #print "i={i} start_psam={start_psam} end_psam={end_psam} start_idx={start_idx} l_psam={l_psam}".format(**locals())
            
            I = idx[start_idx:start_idx+l_psam]
            
            #print i, "self", self.psam[start_psam:end_psam]
            #print i, "window", I
            
            choice = self.psam[start_psam:end_psam]*I
            
            score = choice.sum()
            #print i, seq[i], choice, score
            scores.append(score)
        
        return x, np.array(scores)
        
    def merge(self, mdl):
        x, scores = self.scan(mdl)
        print "scores", scores
        best_i = x[scores.argmax()]
        
        prepend = max(0, best_i)
        append = max(0, mdl.n - self.n - best_i)
        
        # pad the PSAM with ones if we need to add columns
        n = self.n + prepend + append
        A_ = np.ones( (n, 4), dtype=np.float32)
        
        # similarly pad the other PSAM to compute the delta
        K_ = np.zeros( (self.n + prepend + append, 4), dtype=np.float32)
        l = mdl.n - self.n + 1
        start_psam = max(0, -best_i)
        end_psam = self.n - max(0, best_i - l + 1 )
        start_idx = max(0, best_i-prepend)
        
        l_psam = end_psam - start_psam
        
        #print "i={i} start_psam={start_psam} end_psam={end_psam} start_idx={start_idx} l_psam={l_psam}".format(**locals())
        
        start_k = max(-best_i,best_i - prepend)
        end_k = min(start_k + mdl.n, n)
        print "best_i={best_i} mdl.n={mdl.n} self.n={self.n}: prepend={prepend} append={append} start_k={start_k} end_k={end_k} start_idx={start_idx}".format(**locals())

        n_other = end_k - start_k
        K_[start_k:end_k] = mdl.psam[start_idx:start_idx+n_other]
        A_[prepend:self.n+prepend] = self.psam
    
        return PSAMModel(A_, A0 = self.A0), PSAMModel(K_, A0 = mdl.A0)
        
    def __add__(self, mdl):
        assert self.n == mdl.n
        
        A0 = self.A0 + mdl.A0
        #w1 = self.A0 / A0
        #w2 = mdl.A0 / A0

        psam = self.A0 * self.psam + mdl.A0 * mdl.psam
        amax = psam.max(axis=1)
        
        psam /= amax[:, np.newaxis]
        
        return PSAMModel(psam, A0)

    def affinity_lookup_table(self):
        N = 4**self.n
        A = np.ones(N, dtype=np.float32)
        for i in xrange(N):
            n = i
            for j in range(self.n - 1, -1, -1):
                A[i] *= self.psam[j, n & 3]
                n = n >> 2

        return A * self.A0
    
    def __str__(self):
        buf = []
        for col in self.psam:
            buf.append("\t".join(["{0:.3f}".format(s) for s in col] + [project_column(col),]) )
                                                          
        return "\n".join(buf)

    #def merge(self, other):

if __name__ == "__main__":
    psam = PSAMModel.from_kmer('GCAUG', A0= 1.)
    for s in [
        #'GCACG',
        #'UGCAU',
        #'UUGCACG',
        #'CACGC',
        #'GCACGCA',
        'UGCACGU',
        'UGCAUGU',
        'UUGCACGU'
    ]:
        
        print s
        other = PSAMModel.from_kmer(s, A0=.5)
        a,b = psam.merge(other)
        print "merged"
        c = a + b
        print c
        from cska.ska_kmers import seq_to_index
        #print "making tabke"
        tbl = c.affinity_lookup_table()
        print tbl
        #print tbl[seq_to_index('UGCAUGU')]
        #print tbl[seq_to_index('UGCACGU')]
        
    

        
