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

ambig = "-NMRWSYKVHDBACGUT"
ambig_index = dict([(code, n) for n,code in enumerate(ambig)])
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
    [0.0, 0.0, 0.0, 1.0],
])

def project_column(col):
    n = col.sum()
    if n:
        col = col / n
    i = (col[np.newaxis,:] * ambig_vectors).sum(axis=1).argmax()
    return ambig[i]
    

class PSAMState(object):
    def __init__(self, psam, A0 = 1e-6):
        self.psam = np.array(psam, dtype=np.float32)
        amax = psam.max(axis=1)
        self.psam /= amax[:, np.newaxis]
        assert (self.psam.max(axis=1) == 1).all()
        
        self.A0 = A0
        self.n = len(psam)
        
    @classmethod
    def from_kmer(cls, kmer, **kwargs):
        kmer = kmer.upper()
        cols = []
        for nt in kmer:
            cols.append(ambig_vectors[ambig_index[nt]])
        
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
        K_ = np.ones( (self.n + prepend + append, 4), dtype=np.float32)
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
    
        padding = (prepend, append, start_k, n - end_k)
        return PSAMState(A_, A0 = self.A0), PSAMState(K_, A0 = mdl.A0), padding
        
    def __add__(self, mdl):
        assert self.n == mdl.n
        
        A0 = self.A0 + mdl.A0
        #w1 = self.A0 / A0
        #w2 = mdl.A0 / A0

        psam = self.A0 * self.psam + mdl.A0 * mdl.psam
        amax = psam.max(axis=1)
        psam /= amax[:, np.newaxis]
        
        return PSAMState(psam, max(self.A0, mdl.A0))

    def nmer_affinities(self):
        N = 4**self.n
        A = np.ones(N, dtype=np.float32) * self.A0
        
        for i in xrange(N):
            n = i
            for j in range(self.n - 1, -1, -1):
                A[i] *= self.psam[j, n & 3]
                n = n >> 2

        return A
    
    def __str__(self):
        buf = ["PSAM A0={0} n={1}".format(self.A0, self.n)]
        for col in self.psam:
            buf.append("\t".join(["{0:.3f}".format(s) for s in col] + [project_column(col),]) )
                                                          
        return "\n".join(buf)

    #def merge(self, other):


def opt_merge(p0, pk, padding):
    
    A0 = p0.A0
    Ak = pk.A0
    A = max(A0, Ak)

    pre0, app0, prek, appk = padding
    n_ext = pre0 + app0
    A0_avg = (A0/A)**(1./n_ext)
    Ak_avg = (Ak/A)**(1./n_ext)
    print "A0_avg", A0_avg
    print p0.psam[:pre0,  :]
    if pre0:
        p0.psam[:pre0,  :] *= A0_avg
    if app0:
        p0.psam[-app0:, :] *= A0_avg
    if prek:
        pk.psam[:prek,  :] *= Ak_avg
    if appk:
        pk.psam[-app0:, :] *= Ak_avg
        
    print "p0"
    print p0.psam
    print "pk"
    print pk.psam
    
    I0 = (p0.psam == 1).nonzero()
    Ik = (pk.psam == 1).nonzero()
    I = ((p0.psam == 1) | (pk.psam == 1) ).nonzero()

    
    
    #print I
    n = len(I[0])
    #if not n:
        #I = (pk.psam > 0).nonzero() * p0.psam[I0]
        #p0.psam *= p0.A0 / pk.A0
        #p0.psam[I] = 1.
        #p0.A0 = pk.A0
        #return p0
    
    ratio = pk.A0 / p0.A0
    #current = p0.psam[I].prod()
    #if current > 0:
        #ratio /= current
        
    #print "RATIO to distribute", ratio, pk.A0, p0.A0, p0.psam[I].prod()
    #if n == 1:
        #p0.psam[I] = ratio
        #return p0
    
    # find optimal distribution of changes over 
    # elements in p0 such that product == ratio
    # and perturbation of non-zero elements is 
    # minimal
    
    def normed(w):
        p = np.array(p0.psam, dtype=float)
        p[I] = w
        amax = p.max(axis=1)
        p /= amax[:, np.newaxis]
        return p
        
    cost_mask = (p0.psam > 0) | (p0.psam == 1).all(axis=1)[:, np.newaxis]
    
    def to_optimize(w):
        p = normed(w)
        a0 = p[I0].prod() * A # affinity assigned by current weights to best previous sequence
        ak = p[Ik].prod() * A # affinity assigned by current weights to new sequence
        
        if pre0:
            a0 *= p[:pre0, :].mean(axis=1).prod()
        if prek:
            ak *= p[:prek, :].mean(axis=1).prod()
        
        if app0:
            a0 *= p[-app0:, :].mean(axis=1).prod()
        if appk:
            ak *= p[-appk:, :].mean(axis=1).prod()
            
        matrix_dev = ((A*p - A0 * p0.psam )**2 * cost_mask).sum()
        Ak_dev = (Ak - ak)**2
        A0_dev = (A0 - a0)**2
        cost = matrix_dev + 100*Ak_dev + 100*A0_dev 

        #print p
        print "costs:", matrix_dev, Ak_dev, A0_dev, cost
        return cost

    from scipy.optimize import minimize
    w0 = np.ones(n) * ratio**(1./n)
    bounds = np.array([(1e-10, max(ratio, 1./ratio)),]*n)
    print "w0", w0
    print "bounds", bounds
    res = minimize(to_optimize, w0, bounds=bounds)
    
    weights = res.x
    print res.x, res.fun, res
    #remain = ratio / weights.prod()
    #weights = np.concatenate( ( weights, (remain,)) )
    p0.psam = normed(weights)
    p0.A0 = A
    return p0
    
    
    
if __name__ == "__main__":
    psam = PSAMState.from_kmer('GCAUG', A0= 2.)
    print psam
    print "="*20
    for s, a0 in [
        #('GCAUG', 2.),
        #('GCACG', .1),
        #('GUACG', .05),
        ('UGCAUGU', 3.)
        #'CACGC',
        #'GCACGCA',
        #('UGCAUGU', .8),
        #('UGCAUGU', 2.),
        #('UUGCACGU', 2.3),
    ]:
        
        print s
        other = PSAMState.from_kmer(s, A0=a0)
        a,b, padding = psam.merge(other)

        print "a", padding
        print a
        print a.psam
        print "b"
        print b
        #print b, prepend, append
        print "!!!merged!!!"
        c = opt_merge(a, b, padding)
        #c = a + b
        print c
        from cska.ska_kmers import seq_to_index
        #print "making tabke"
        tbl = c.nmer_affinities()
        print "tabulated affinity of {0}={1}, A0={2}".format(s, tbl[seq_to_index(s)], a0)
        print "tabulated affinity of {0}={1}, A0={2}".format('GCAUG', tbl[seq_to_index('GCAUG')], a0)
        
        #print tbl[seq_to_index('UGCAUGU')]
        #print tbl[seq_to_index('UGCACGU')]
        
    

# when merging two PSAM without change of dimension (point mutations):
# -> merged should reproduce desired best score of both
# -> change w as little as possible (least sq)

# when merging two PSAM with change of dimension (shifted):
# -> ensure that best score is maintained over *average* of added columns
# -> change w as little as possible
        
