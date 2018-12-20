import numpy as np
import logging


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
    def __init__(self, k, n_samples, nt=1, psam=[], A0=1., betas = [], data = [], acc_shift=0, acc_k=None, acc_scale=1.):
        self.k = k
        self.nt = nt
        self.depth = 4**nt
        self.n_samples = n_samples
        self.n = self.depth * k + 1 + n_samples
        self.n_psam = self.depth * k+1
        self.Nk = 4**k
        self.psam_start = 0
        self.psam_end = self.depth * k + 1
        self.betas_start = self.psam_end
        self.betas_end = self.n
        self.dtype = np.float32
        
        # accessibility might be selected in a shifted region of size != k
        self.acc_shift = acc_shift
        self.acc_scale = acc_scale
        if acc_k is None: 
            self.acc_k = self.k
        else:
            self.acc_k = acc_k

        self.data = np.zeros(self.n, dtype=np.float32)

        # self.psam_vec = Proxy(self.data, self.psam_start, self.psam_end)
        # self.psam_matrix = Proxy(self.data, self.psam_start+1, self.psam_end, shape=(k,4))
        self.attrs = {
            'psam_vec' : Proxy(self.data, self.psam_start, self.psam_end),
            'psam_matrix' : Proxy(self.data, self.psam_start+1, self.psam_end, shape=(k, self.depth)),
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

        self.names =['A0']
        for i in range(self.k):
            self.names.extend(['{0}{1}'.format(nt, i+1) for nt in 'ACGU'])
        for i in range(self.n_samples):
            self.names.append('beta{0}'.format(i))

    @classmethod
    def from_vector(cls, vec, k, n_samples=1):
        return cls(k, n_samples, data=vec)

    @classmethod
    def from_PSAM(cls, psam, n_samples=1, **kwargs):
        params = cls(psam.n, n_samples, psam=psam.psam, **kwargs)
        params.A0 = psam.A0
        return params

    @classmethod
    def load(cls, fname, n_samples, beta0=1e-6, mina=1e-6):
        aff = []
        attrs = {}
        with file(fname) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if line.startswith('PSAM'):
                    # parse attributes
                    for kw in line.split()[1:]:
                        if not kw.strip():
                            continue
                        k,v = kw.split('=')
                        attrs[k] = float(v)

                elif line.startswith('seeded'):
                    break
                else:
                    parts = line.split('\t')
                    aff.append(parts[:4])

        psam = np.array(aff, dtype=np.float32)
        psam = np.where(psam > 0, psam, mina)
        params = cls(len(psam), n_samples, psam=psam, A0=attrs.get('A0', 1))
        params.acc_k = int(attrs.get('acc_k', len(psam)))
        params.acc_shift = int(attrs.get('acc_shift', 0))
        params.acc_scale = attrs.get('acc_scale', 1)
        params.betas[:] = beta0
        return params

    def save(self, fname):
        file(fname, 'w').write(str(self) + '\n')

    def as_vector(self, dtype=np.float32):
        return self.data
    
    def as_PSAM(self):
        from cska.pwm import PSAM
        return PSAM(self.psam_matrix, A0=self.A0)

    def copy(self):
        new = ModelParametrization(self.k, self.n_samples, data=self.data, acc_k=self.acc_k, acc_shift=self.acc_shift, acc_scale=self.acc_scale, nt=self.nt)
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
        from cska.pwm import project_column
        import cska.cyska as cyska
        buf = []
        buf.append("PSAM A0={self.A0} n={self.k} acc_k={self.acc_k} acc_shift={self.acc_shift} acc_scale={self.acc_scale}".format(self=self))
        buf.append("#\t{}\tcons".format("\t".join(cyska.yield_kmers(self.nt))))
        
        for row in self.psam_matrix:
            buf.append("\t".join(["{0:>10.5f}".format(x) for x in row] + [project_column(row)]))

        # buf.append("BACKGROUND")
        for i, beta in enumerate(self.betas):
            buf.append('# beta{0}={1:.3e}'.format(i, beta))
        
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


if __name__ == "__main__":
    psam = np.identity(4)
    print psam

    params = ModelParametrization(4, 3, psam=psam, A0=2.)
    params.save('bla.tsv')

    params = ModelParametrization.load('bla.tsv', 3)
    print params