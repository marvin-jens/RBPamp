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

class ModelSetParams(object):
    def __init__(self, param_set, forward = ['k', 'n_samples', 'k_mdl', 'acc_shift', 'acc_scale']):
        self._forward = set(forward)
        self.param_set = param_set
        # for name in forward:
        #     setattr(ModelSetParams, name, property(fget = lambda : getattr(self.param_set[0], name)))
        
    def __getattr__(self, attr):
        fw = object.__getattribute__(self, '_forward')
        p0 = object.__getattribute__(self, 'param_set')[0]
        # print p0, attr
        if attr in fw:
            return getattr(p0, attr)
        else:
            return object.__getattribute__(self, attr)

    # @property
    # def n_samples(self):
    #     return self.param_set[0].n_samples

    # @property
    # def n_samples(self):
    #     return self.param_set[0].n_samples

    @property
    def A0(self):
        return self.param_set[0].A0

    @A0.setter
    def A0(self, value):
        # change all motif A0s in proportion
        ratio = value / self.param_set[0].A0
        for par in self.param_set:
            par.A0 *= ratio

    @property
    def acc_k(self):
        return self.param_set[0].A0

    @acc_k.setter
    def acc_k(self, value):
        # change all motif acc_k's (esp. for acc_k=0)
        for par in self.param_set:
            par.acc_k = value

    @property
    def betas(self):
        return self.param_set[0].betas

    @betas.setter
    def betas(self, value):
        self.param_set[0].betas = value

    def copy(self):
        return ModelSetParams([p.copy() for p in self.param_set])

    def get_data(self):
        "return one np.ndarray containing all model parameters"
        all_data = [p.data for p in self.param_set]
        # print all_data
        return np.concatenate(all_data)

    def set_data(self, data):
        "broadcast raw data write across all model parameters"
        i = 0
        for p in self.param_set:
            l = len(p.data)
            p.data[:] = data[i:i+l]
            i += l

        assert i == len(data)
        return self
    
    def unity(self):
        p = self.copy()
        n = np.linalg.norm(self.get_data())
        if n > 0:
            p /= n
        return p

    @classmethod
    def load(cls, fname, n_samples, max_motifs=4):
        param_set = list(ModelParametrization.load(fname, n_samples))
        if len(param_set) > max_motifs:
            param_set = param_set[:max_motifs]
        return cls(param_set)

    def save(self, fname):
        for i, params in enumerate(self.param_set):
            params.save(fname, append=(i > 0) )

    def __str__(self):
        buf = ["# ModelSetParams with {} PSAMs\n".format(len(self.param_set))]
        for i, params in enumerate(self.param_set):
            buf.append("# PSAM {}".format(i))
            buf.append(str(params))
        
        return "\n".join(buf)

    def __iter__(self):
        for params in self.param_set:
            yield params   

    def __getitem__(self, i):
        return self.param_set[i]

    def __setitem__(self, i, params):
        self.param_set[i] = params

    def __add__(self, x):
        c = self.copy()
        if isinstance(x, ModelSetParams):
            c.set_data(self.get_data() + x.get_data() )
        else:
            c.set_data(self.get_data() + x)
        return c
    
    def __sub__(self, x):
        c = self.copy()
        if isinstance(x, ModelSetParams):
            c.set_data(self.get_data() - x.get_data() )
        else:
            c.set_data(self.get_data() - x)
        return c

    def __mul__(self, x):
        c = self.copy()
        if isinstance(x, ModelSetParams):
            c.set_data(self.get_data() * x.get_data() )
        else:
            c.set_data(self.get_data() * x)
        return c

    def __div__(self, x):
        c = self.copy()
        if isinstance(x, ModelSetParams):
            c.set_data(self.get_data() / x.get_data() )
        else:
            c.set_data(self.get_data() / x)
        return c

    def __neg__(self):
        c = self.copy()
        c.set_data( - self.get_data())
        return c

    def apply_delta(self, delta_set, min_rel_A0=1e-3):
        c = self.copy()
        new = []
        for i, (params, delta) in enumerate(zip(c.param_set, delta_set)):
            p = params.psam_matrix + delta.psam_matrix
            # print i, "after applying update of magnitude", np.fabs(delta.data).max(), "min/max", p.min(), p.max()
            # m = p.min(axis=1) # find out if we dropped below zero
            # m = np.where(m < 0, -m + 1e-6, 0)
            # # print "raise", m
            # p += m[:, np.newaxis] # and raise the level in these columns accordingly
            p = np.clip(p, 1e-6, None)
            M = p.max(axis=1) # increases above 1 on cognate should increase A0
            p /= M[:,np.newaxis]
            params.psam_matrix = np.clip(p, 1e-6, 1)

            params.A0 *= M.prod() # keep matrix elements <= 1 and absorb excess into A0
            # print i, "increasing A0 by", M.prod()
            params.A0 = max(1e-6, params.A0 + delta.A0) # prevent underflow

            params.betas = np.clip(params.betas + delta.betas, 1e-9, None)
            new.append(params)

        a0s = np.array([params.A0 for params in new])
        min_a0 = a0s.max() * min_rel_A0
        a0s = np.where(a0s > min_a0, a0s, min_a0)
        for a0, params in zip(a0s, new):
            params.A0 = a0
        # print "params.A0", params.A0
        c.param_set = new
        return c


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
        def make_params():
            psam = np.array(aff, dtype=np.float32)
            psam = np.where(psam > 0, psam, mina)
            params = cls(len(psam), n_samples, psam=psam, A0=attrs.get('A0', 1))
            params.acc_k = int(attrs.get('acc_k', len(psam)))
            params.acc_shift = int(attrs.get('acc_shift', 0))
            params.acc_scale = attrs.get('acc_scale', 1)
            params.betas[:] = beta0
            return params

        with file(fname) as f:
            for line in f:
                if line.startswith('#'):
                    if aff:
                        yield make_params()
                        aff = []
                        attrs = {}
                    continue

                if line.startswith('PSAM'):
                    # parse attributes
                    for kw in line.split()[1:]:
                        if not kw.strip():
                            continue
                        k,v = kw.split('=')
                        attrs[k] = float(v)

                elif line.startswith('seeded'):
                    continue
                else:
                    parts = line.split('\t')
                    aff.append(parts[:4])

        if aff:
            yield make_params()

    def save(self, fname, append=False):
        if append:
            mode = 'a'
        else:
            mode = 'w'
        file(fname, mode).write(str(self) + '\n')

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
            c.data -= x.data
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