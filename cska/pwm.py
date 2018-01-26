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
    

class PSAM(object):
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
        
        m = np.array(cols, dtype=np.float32)
        psam = cls(m, **kwargs)
        psam.kmer_seed = kmer
        return psam
    
    @classmethod
    def from_kmer_variants(cls, kmers, aff, **kwargs):
        #kmers = np.array(list(cyska.yield_kmers(k)))
        
        #n = (aff >= min_aff).sum()
        I = list(aff.argsort()[::-1][:n])
        
        # start with highest affinity/enrichment kmer
        i = I[0]
        j = I.index(i)
        I.pop(j)

        #print "starting with", kmers[i], aff[i]
        psam = cls.from_kmer(kmers[i], A0=aff[i])
        
        # first merge all single nucleotide variants
        #print len(I), "kmers left to merge"
        #print "single nt variants"
        J = []
        for j,i in enumerate(I):

            new = cls.from_kmer(kmers[i], A0=aff[i])

            mask = (new.psam == 1) & (psam.psam != 1)
            rel_A = new.A0/psam.A0
            psam.psam += mask * rel_A
            #print "->merged", kmers[i], rel_A

        #print psam
        psam.kmer_set = set(kmers)
        return psam

            
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

    @property
    def discrimination(self):
        return (self.psam.max(axis=1) / self.psam.sum(axis=1) - .25 ) / .75

    def __str__(self):
        buf = ["PSAM A0={0} n={1}".format(self.A0, self.n)]
        for col,d  in zip(self.psam, self.discrimination):
            buf.append("\t".join(["{0:.6e}".format(s) for s in col] + [project_column(col), str(d)]) )

        kmer_seed = getattr(self, "kmer_seed","")
        kmer_set =  getattr(self, "kmer_set","")
        if kmer_seed:
            buf.append("seeded from '{0}'".format(kmer_seed))
        if kmer_set:
            buf.append("built from {0} kmers '{1}'".format(len(kmer_set), ",".join(sorted(kmer_set)) ) )
        return "\n".join(buf)

    def save_logo(self, fname='pwm.pdf', title=""):
        import weblogolib as wl
        counts = self.psam
        from corebio.seq import unambiguous_rna_alphabet
        #data = LogoData(alphabet=unambiguous_rna_alphabet, length=5, counts=counts, entropy=np.ones(5), weight=np.ones(5))
        data = wl.LogoData.from_counts(unambiguous_rna_alphabet, counts)
        import sys
        sys.stderr.write(str( data))
        options = wl.LogoOptions(color_scheme=wl.classic, fineprint="", logo_title=title, yaxis_label='A.U.')
        options.title = "A Logo Title"
        fmt = wl.LogoFormat(data, options)
        dump = wl.pdf_formatter( data, fmt)
        
        if fname:
            file(fname,'wb').write(dump)
        
        return dump

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
        
