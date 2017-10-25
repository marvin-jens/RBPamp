#!/usr/bin/env python
import sys
import numpy as np
import cska.ska_kmers as cyska
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
    
    @classmethod
    def from_kmer_affinities(cls, k, aff, min_aff=1e-9):
        kmers = np.array(list(cyska.yield_kmers(k)))
        
        n = (aff >= min_aff).sum()
        I = list(aff.argsort()[::-1][:n])
        
        # start with highest affinity/enrichment kmer
        #psam = cls.from_kmer(kmers[I[0]], A0=aff[I[0]])
        
        
        i = I[0]
        #i = cyska.seq_to_index('TGCAT')
        j = I.index(i)
        I.pop(j)

        print "starting with", kmers[i], aff[i]
        psam = cls.from_kmer(kmers[i], A0=aff[i])
        
        # first merge all single nucleotide variants
        print len(I), "kmers left to merge"
        print "single nt variants"
        J = []
        for j,i in enumerate(I):

            new = cls.from_kmer(kmers[i], A0=aff[i])
            x, scores = psam.scan(new, max_shift=0)
            s = scores[0]

            if s >= k - 1:
                mask = (new.psam == 1) & (psam.psam != 1)
                rel_A = new.A0/psam.A0
                psam.psam += mask * rel_A
                print "->merged", kmers[i], rel_A
            else:
                J.append(i)
        I,J = J, []
        print psam
        print psam.psam.shape
        print len(I), "kmers left to merge"

        def grow(psam, I, thresh = 1, debug=False):
            J = []
            Al = np.zeros(4,dtype=float)
            Ar = np.zeros(4,dtype=float)

            left = []
            right = []
            print "first col", psam.psam[0], psam.psam[0].mean()
            print "last col", psam.psam[-1], psam.psam[-1].mean()
            
            aff_l = psam.psam[-1].mean()
            aff_r = psam.psam[0].mean()

            #print "background affinity expected for shifted kmers", aff_l, aff_r
            # merge shifted kmer
            for j,i in enumerate(I):

                new = cls.from_kmer(kmers[i], A0=aff[i])
                dbg = False
                if kmers[i] == 'TTGCA' and debug:
                    dbg = True
                x, scores = psam.scan(new, max_shift=1, debug=dbg)
                #print kmers[i], x, scores
                sr = scores[0]
                sl = scores[-1]

                if sr >= k - thresh:
                    idx = base_idx[kmers[i][0]]
                    Al[idx] = aff[i]
                    #print "left", x, scores
                    left.append(kmers[i])
                    
                elif sl >= k - thresh:
                    idx = base_idx[kmers[i][-1]]
                    Ar[idx] = aff[i]
                    #print "right", x, scores
                    right.append(kmers[i])
                else:
                    J.append(i)

            print "left shifted"
            for seq, a in zip("ACGU", Al):
                print seq, a, a/aff_l

            print "right shifted"
            for seq, a in zip("ACGU", Ar):
                print seq, a, a/aff_r

            A0 = psam.A0
            Al /= aff_l
            Ar /= aff_r
            Alm = Al.max()
            Arm = Ar.max()
            Al = (Al + A0)/ (Alm + A0)
            Ar = (Ar + A0)/ (Arm + A0)
                
            psam.A0 = A0 + Alm + Arm
            psam.psam = np.vstack( (Al, psam.psam, Ar))
            psam.n += 2
            
            return psam, J
        
        print "growing"
        psam, I = grow(psam, I, debug=False)
        print psam
        print len(I), "kmers left to merge"

        print "after pruning"
        print psam.prune(1e-5)

        print "growing"
        psam, I = grow(psam, I, debug=False)
        print psam
        print len(I), "kmers left to merge"
        
        print "after pruning"
        print psam.prune(1e-5)
        
        print "after pruning"
        print psam.prune(1e-5)
        psam = psam.prune(1e-5)
        print "final"
        print psam
        
        for j,i in enumerate(I):
            x, scores = psam.scan(new, max_shift=1, debug=False)
            #print x, scores.argmax()
            print kmers[i], aff[i], scores.max(), x[scores.argmax()], scores


    def prune(self, cutoff):
        start = 0
        disc = self.discrimination
        end = self.n
        
        while disc[start] < cutoff and start < end:
            start += 1
        
        while disc[end - 1] < cutoff and end > 0:
            end -= 1
        
        left_avg = 1
        if start > 0:
            self.psam[:start].mean(axis=1).prod()

        right_avg = 1
        if end < self.n:
            right_avg = self.psam[end:].mean(axis=1).prod()

        pruned = PSAMState(self.psam[start:end], A0 = self.A0 * left_avg * right_avg)
        return pruned
        
    def scan(self, mdl, max_shift=2, debug=False):
        if debug:
            print "aligning",
            print mdl
        idx = mdl.psam

        L = self.n
        l = L - mdl.n + 1
        x = np.arange(-max_shift, l + max_shift)
        if debug:
            print x, mdl.n, self.n
        #print idx
        scores = []
        for i in x:
            start_mdl = max(0, -i)
            end_mdl = mdl.n - max(0, i - l + 1 )
            start_self = max(0, i)
            
            n_cols = end_mdl - start_mdl
            if debug:
                print "i={i} start_mdl={start_mdl} end_mdl={end_mdl} start_self={start_self} n_cols={n_cols}".format(**locals())
            
            I = mdl.psam[start_mdl:start_mdl + n_cols]
            
            if debug:
                print i, "self", self.psam[start_self:start_self+n_cols]
                print i, "window", I
            
            choice = self.psam[start_self:start_self+n_cols]*I
            score = choice.sum()
            scores.append(score)

        if debug:
            print "scan results:", x, scores
        
        return x, np.array(scores)
        
    def merge(self, mdl):
        x, scores = self.scan(mdl)
        #print "score s", scores
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

    def score_kmer(self, kmer_i, k, shift=0):
        index = kmer_i
        n_cognate = 0
        for j in range(k):
            nt = index >> ((k - j - 1) * 2) & 3
            #index = index >> 2
            
            if j + shift < 0:
                continue
            
            if j + shift >= self.n:
                break
            
            #print "score_kmer", shift, j, nt, self.psam[j + shift, nt]
            if self.psam[j + shift,nt] == 1:
                n_cognate += 1
        
        return n_cognate
            
    @property
    def consensus(self):
        return "".join([project_column(col) for col in self.psam])
        
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

    @property
    def discrimination(self):
        return (self.psam.max(axis=1) / self.psam.sum(axis=1) - .25 ) / .75

    def __str__(self):
        buf = ["PSAM A0={0} n={1}".format(self.A0, self.n)]
        for col,d  in zip(self.psam, self.discrimination):
            buf.append("\t".join(["{0:.6e}".format(s) for s in col] + [project_column(col), str(d)]) )
                                                          
        return "\n".join(buf)

    #def merge(self, other):


def opt_merge(p0, pk, padding):
    
    pre0, app0, prek, appk = padding

    A0 = p0.A0
    Ak = pk.A0
    
    if prek:
        Ak /= p0.psam[:prek].mean(axis=1).prod() * A0
    if appk:
        Ak /= p0.psam[-appk:].mean(axis=1).prod() * A0

    print "A0", A0, "Ak", Ak
    A = max(A0, Ak)

    
    n_ext = pre0 + app0
    if n_ext:
        A0_avg = (A0/A)**(1./n_ext)
    
    n_ext = prek + appk
    if n_ext:
        Ak_avg = (Ak/A)**(1./n_ext)

    if pre0:
        p0.psam[:pre0,  :] *= A0_avg
    if app0:
        p0.psam[-app0:, :] *= A0_avg
        print app0, A0_avg, p0.psam[-app0:, :]        
    if prek:
        pk.psam[:prek,  :] *= Ak_avg
        print prek, Ak_avg, pk.psam[:prek, :]        
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
        #print "costs:", matrix_dev, Ak_dev, A0_dev, cost
        return cost

    from scipy.optimize import minimize
    w0 = np.ones(n) * ratio**(1./n)
    bounds = np.array([(1e-10, max(ratio, 1./ratio)),]*n)
    #print "w0", w0
    #print "bounds", bounds
    res = minimize(to_optimize, w0, bounds=bounds)
    
    weights = res.x
    #print res.x, res.fun, res
    #remain = ratio / weights.prod()
    #weights = np.concatenate( ( weights, (remain,)) )
    p0.psam = normed(weights)
    p0.A0 = A
    return p0
    
    
    
if __name__ == "__main__":
    
    aff5 = np.array([float(line.split('\t')[1]) for line in file('/scratch/data/RBNS/RBFOX2/best5mersofar.txt')])[:-4]
    PSAMState.from_kmer_affinities(5, aff5)

    #aff7 = np.array([float(line.split('\t')[1]) for line in file('/scratch/data/RBNS/RBFOX2/blup7.tsv')])[:-4]
    #PSAMState.from_kmer_affinities(7, aff7)
    sys.exit(0)
    monitor = ['GCAUG','GCACG','GUACG']
    psam = PSAMState.from_kmer('GCAUG', A0= 1.)
    #psam = PSAMState.from_kmer('NNNNN', A0= .001)
    print psam
    print "="*20
    for s, a0 in [
        #('GCAUG', 2.),
        #('CAUGU', 1.), 
        ('GCACG', .1),
        ('GUACG', .05),
        ('UGCAUGU', 3.),
        #'CACGC',
        ('UGCACGCA', 3.5),
        #('UGCAUGU', .8),
        #('UGCAUGU', 2.),
        #('UUGCACGU', 2.3),
    ]:
        
        print "->", s, a0
        other = PSAMState.from_kmer(s, A0=a0)
        a,b, padding = psam.merge(other)

        print "padding", padding
        #print a
        #print a.psam
        #print "b"
        #print b
        #print b, prepend, append
        #print "!!!merged!!!"
        c = opt_merge(a, b, padding)
        #c = a + b
        psam = c
        print c
        from cska.ska_kmers import seq_to_index
        #print "making tabke"
        tbl = c.nmer_affinities()
        for m in monitor:
            print "tabulated affinity of {0}={1}, A0={2}".format(m, tbl[seq_to_index(m)], a0)
        
        #print tbl[seq_to_index('UGCAUGU')]
        #print tbl[seq_to_index('UGCACGU')]
        
    

# when merging two PSAM without change of dimension (point mutations):
# -> merged should reproduce desired best score of both
# -> change w as little as possible (least sq)

# when merging two PSAM with change of dimension (shifted):
# -> ensure that best score is maintained over *average* of added columns
# -> change w as little as possible
        
