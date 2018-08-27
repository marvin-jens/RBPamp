import os
import unittest
import time
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from cska import auto_detect
from cska.reads import RBNSReads
import cska.cyska as cyska
from cska.pwm import PSAM

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
    def __init__(self, k, n_samples, psam=[], A0=1., betas = [], data = [], dtype=np.float32):
        self.k = k
        self.n_samples = n_samples
        self.n = 4*k + 1 + n_samples
        self.n_psam = 4*k+1
        self.Nk = 4**k
        self.psam_start = 0
        self.psam_end = 4*k + 1
        self.betas_start = self.psam_end
        self.betas_end = self.n
        self.dtype = dtype

        self.data = np.zeros(self.n, dtype=np.float32)

        # self.psam_vec = Proxy(self.data, self.psam_start, self.psam_end)
        # self.psam_matrix = Proxy(self.data, self.psam_start+1, self.psam_end, shape=(k,4))
        self.attrs = {
            'psam_vec' : Proxy(self.data, self.psam_start, self.psam_end),
            'psam_matrix' : Proxy(self.data, self.psam_start+1, self.psam_end, shape=(k,4)),
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

    def as_vector(self, dtype=np.float32):
        return self.data
    
    def as_PSAM(self):
        return PSAM(self.psam_matrix, A0=self.A0)

    def copy(self):
        new = ModelParametrization(self.k, self.n_samples, dtype=self.dtype, data=self.data)
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
        buf = []
        buf.append("PSAM")
        buf.append("A0={0:.4e}".format(self.A0))
        buf.append("\tA\t\tC\t\tG\t\tU")
        for row in self.psam_matrix:
            buf.append("\t".join(["{0:>10.3f}".format(x) for x in row] + [project_column(row), ]))

        buf.append("BACKGROUND")
        for i, beta in enumerate(self.betas):
            buf.append('beta{0}\t{1:.3e}'.format(i, beta))
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


def emp_grad(state, eps=1e-6):
    v0 = state.params.as_vector()
    var = state.params.copy()
    grad = state.params.copy()

    err0 = state.error
    # print "err0", err0
    kw = dict()#.predict_kwargs)
    kw['beta_fixed'] = True

    for i in range(state.params.n):
        # print ">>> EMP GRAD", state.params.names[i]
        # d = max(v0[i] * eps,1e-6)
        d = eps
        var.data[i] = v0[i] + d
        state = state.mdl.predict(var, **kw)
        derr = state.error - err0
        grad.data[i] = derr/d
        # print "derr", derr
        var.data[i] = v0[i]
        # print ">>>GRAD ELEMENT", grad.data[i]
    
    return grad


def emp_gradi(state, eps=1e-6):
    v0 = state.params.as_vector()
    var = state.params.copy()
    Nk = state.mdl.nA
    n_data = len(state.params.data)
    n_samples = state.params.n_samples
    gradi = np.zeros((n_samples, Nk, n_data), dtype=np.float32)

    R0 = state.R
    state0 = state

    # print "err0", err0
    kw = dict()  # .predict_kwargs)
    kw['beta_fixed'] = True
    kw['rbp_free'] = state0.rbp_free

    for i in range(state.params.n):
        # print ">>> EMP GRAD", state.params.names[i]
        # d = max(v0[i] * eps,1e-6)
        d = eps
        var.data[i] = v0[i] + d
        state = state.mdl.predict(var, **kw)
        dR = state.R - R0
        from cska.cyska import index_to_seq
        if i == 12:
            print "funky gradient element, should be zero for aaaaaa"
            print "dR(aaaaa)_dU3", dR[:,0]
            print "dpsi_dU3", state.psi - state0.psi
            print "dq(aaaaa)_dU3", state.q[:,0] - state0.q[:, 0]
            print "dQ_dU3", state.Q - state0.Q
            print "kmers with changes in dq at conc 1"
            dq = state.q[1, :] - state0.q[1, :]
            for j in dq.argsort()[::-1][:10]:
                print index_to_seq(j, 5), dq[j], state.mdl.f0[j]
            print "partition function elements reacting to change"
            dZ = state.Z1 - state0.Z1
            _dZ = state0.Z1 / var.data[i]
            for z, a in zip(dZ, _dZ):
                if (z == 0).all():
                    continue
                print "emp", ["{0:.2e}".format(x) for x in z]
                print "ana", ["{0:.2e}".format(x) for x in a]
        
            print "dZ_read", state.Z1_read - state0.Z1_read

        print "dRBP_free", state.rbp_free - state0.rbp_free, state.mdl._last_sc.last_error

        gradi[:,:,i] = dR/d
        # print "derr", derr
        var.data[i] = v0[i]
        # print ">>>GRAD ELEMENT", grad.data[i]
    
    return gradi


def minimize_logspaced(func, bounds=[], n_samples=7, debug=False, nested=2, plot="", **kwargs):
    """
    first evaluate at log-spaced sampling points along parameter range
    then select at most 3 orders of magnitude around the lowest observed value
    for Brent optimization. Requires pos. valued bounds!
    """
    from scipy.optimize import minimize_scalar
    import time
    t0 = time.time()
    bmin = bounds.min()
    bmax = bounds.max()

    known = {}

    def func_or_lookup(x):
        if not x in known:
            known[x] = func(x)
        return known[x]

    def logsearch(bmin, bmax):
        
        lmin = np.log10(bmin)
        lmax = np.log10(bmax)
        
        sample_x = 10**np.linspace(lmin, lmax, n_samples)
        samples = np.array([func_or_lookup(x) for x in sample_x])
            
        if debug:
            print "logspaced sample", zip(sample_x, samples)

        i = samples.argmin()
        li = max(0, i -1)
        ri = min(n_samples-1, i+1)
        
        brent_min = sample_x[li]
        brent_max = sample_x[ri]
        if debug:
            print "search optimum between", brent_min, brent_max
    
        return brent_min, brent_max

    for i in range(nested):
        bmin, bmax = logsearch(bmin, bmax)

    if debug:
        print "minimize_scalar(bounds=[{bmin}, {bmax}])".format(**locals())

    res = minimize_scalar(func_or_lookup, bounds = np.array([bmin, bmax]), method='Bounded') #, **kwargs)
    t1 = time.time()

    if plot:
        import matplotlib.pyplot as pp
        x = sorted(known.keys())
        y = [known[i] for i in x]
        pp.title("minimize_logspaced ->{}".format(plot))
        pp.axhline(0, color='gray')
        pp.semilogx(x,y)
        pp.semilogx(x,y,'xr')
        pp.axvline(res.x,color='red')
        pp.xlabel('variable')
        pp.ylabel('change in error')
        pp.savefig(plot)
        # pp.show()
        pp.close()

    # self.logger.debug("minimize_logspaced took {dt:.2f}ms".format(dt= 1000. * (t1-t0)) )

    return res



class GradientDescent(object):
    def __init__(self, model, params0, dec=.5, ref_state=None, predict_kwargs=dict(beta_fixed=False, tune=True)):
        self.logger = logging.getLogger('opt.GradientDescent')
        self.model = model
        self.params = params0
        self.ref_state = ref_state # used for simulations, where true values are known.
        self.predict_kwargs = predict_kwargs
        self.model.opt = self # link model to this optimizer instance so it can find out R0 etc.

        # momentum smoothing of the gradient
        self.past_grad = None
        self.past_sqg = 1
        self.past_var = 1
        self.dec = dec

        # records
        self.errors = []
        self.history = []
        self.ls_nfev = []
        self.ls_step = []
        self.t = 0
        self.last_quantile = 0

        # optimization result/status
        self.status = None
        
    @staticmethod
    def apply_delta(params, delta):
        new = params.copy()
        p = np.clip(params.psam_matrix + delta.psam_matrix, 1e-6, None)
        M = p.max(axis=1)
        p /= M[:,np.newaxis]
        new.psam_matrix = np.clip(p, 1e-6, 1)
        new.A0 *= M.prod() # keep matrix elements <= 1 and absorb excess into A0
        new.A0 = max(1e-6, new.A0 + delta.A0) # prevent underflow

        new.betas = np.clip(params.betas + delta.betas, 1e-9, None)
        return new

    def backtracking_LS(self, params, grad, local_grad, min_step = 1e-6, max_step = 5., max_iter=21, tau=.5, c=.5, debug=False):

        grad.data /= np.linalg.norm(grad.data)
        print grad
        def err(s):
            m = self.apply_delta(params, grad * s)
            # d = delta(m.data, correct_params.data)
            # print s, d, m.data, 
            R, dR = self.predict_R(m, grad=False, **self.predict_kwargs)
            e = self.error(R)
            # print s,"->", e - e0, d - d0
            return e

        e0 = err(0)

        print local_grad
        a = max_step
        m = np.dot(grad.data, local_grad.data)
        print m
        t = - c * m
        print t
        for n in range(max_iter):
            e = err(a)
            print n, '->', a, e0 - e, a*t, 'crit', (e0-e) - (a*t)
            # Armijo-Goldstein criterion
            if e0 - e >= a*t:
                break
            # if e < e0:
            #     break

            a *= tau

        return a, n        

    def line_search(self, state, vec, debug=False, min_step = 1e-6, max_step = 10., maxiter=10, xatol=1e-1, e0=None, plot=""):
        from scipy.optimize import minimize_scalar

        e0 = state.error
        assert e0 == state.mdl.predict(state.params, **self.predict_kwargs).error

        params0 = state.params
        t0 = time.time()
        N = {'fev' : 0}
        kw = dict(self.predict_kwargs)
        kw['beta_fixed'] = True
        kw['tune'] = False

        self.model.set_mask( state.Z1_read > self.model.Z_thresh * state.Z1_read_max)
        def err(s):
            # s = np.exp(x)
            m = self.apply_delta(params0, vec * s)
            new = self.model.predict(m, **kw)
            N['fev'] += 1
            if debug:
                print s,"->", new.error - e0
            return new.error - e0

        assert np.fabs(err(0)) < 1e-6
        # def exponential_backtrack(func, e0, a,b, dec=.5):
        #     x = b
        #     nfev = 0
        #     while x > a:
        #         e = func(x)
        #         nfev += 1
        #         # print x, e
        #         if e < e0:
        #             return x,e,nfev

        #         x *= dec

        #     return 0,e0,nfev

        res = minimize_logspaced(err, bounds = np.array([min_step, max_step]), plot=plot, options=dict(maxiter=maxiter) )
        # res = minimize_scalar(err, method='Bounded', bounds=np.log(np.array([min_step, max_step])), options=dict(maxiter=maxiter, xatol=xatol))
        # self.logger.debug("minimize_logspaced took {dt:.2f}ms".format(dt= 1000. * (t1-t0)) )

        self.model.set_mask()
        self.logger.debug("line_search took {0:.3f} seconds for {1} iterations".format(time.time() - t0, N['fev']))
        if not res.success or res.fun > 0:
            self.logger.warning("line_search could not decrease error!")
            x = 0
        else:
            x = res.x
            # x = np.exp(res.x)
        # x, e, nfev = exponential_backtrack(err, e0, min_step, max_step)
        return x, res.nfev

    def momentum_grad(self, local_grad):
        if self.past_grad is None:
            self.past_grad = local_grad

        grad = self.past_grad * self.dec + local_grad * (1- self.dec)
        self.past_grad = grad
        return grad

    def RMSprop(self, local_grad, delta=.00001):
        if self.past_grad is None:
            self.past_grad = local_grad.data

        m = self.dec * self.past_grad + (1 - self.dec) * local_grad.data
        s = self.dec * self.past_sqg + (1 - self.dec) * local_grad.data**2

        self.past_grad = m
        self.past_sqg = s

        upd = local_grad.copy()
        upd.data = m / (np.sqrt(s) + delta)

        return upd

    def converged(self, rtol=1e-6, atol=1e-7, tau=10):
        if len(self.errors):
            if self.errors[-1] < atol:
                return 'CONVERGED_ERR_MINIMAL'

        if len(self.errors) < tau:
            return self.status
           
        last_errs = np.array(self.errors[-tau:])
        mean = last_errs.mean()
        if (mean - self.errors[-1]) / mean < rtol:
            return 'CONVERGED_NO_MORE_DECREASE'

        return self.status

    @property
    def error_reduction(self):
        if len(self.errors) < 2:
            return 0.
        
        return self.errors[-1] / self.errors[0]

    def print_state(self, state):
        print ">>>>>>>>>PARAMS"
        print state.params
        s = 0
        if len(self.ls_step):
            s = self.ls_step[-1]

        print "=" * 50
        last_err = self.errors[-1]
        print "step={self.t} error={last_err:.5e} n_fev={self.model.n_fev} n_grad={self.model.n_grad} scale={s} corr={state.correlations[0]}".format(**locals())


    def optimize(self, params, maxiter=100, debug=False, callback=None, ls_plot="", A0_plot="", tune=True):
        if debug:
            print "INITIAL PARAMETERS"
            print params
        
        # state = self.model.tune(self.params, A0_plot=A0_plot.format(**locals()), debug=debug)
        state = self.model.predict(self.params, **self.predict_kwargs)

        self.errors.append(state.error)
        self.history.append(state.archive())
        self.last_state = state

        if debug:
            print "INITIAL STATE AFTER FIRST EVAL kwargs=", self.predict_kwargs
            self.print_state(state)

        if callback:
            callback(self)

        try:
            while not self.converged() and self.t < maxiter:
                local_grad = state.grad.unity()
                if debug:
                    print "LOCAL GRAD"
                    print local_grad
                local_grad.A0 = 0
                local_grad.betas *= 0
                # local_grad = local_grad.unity()
                descent = self.RMSprop( -local_grad ).unity()
                # descent = self.momentum_grad( - local_grad).unity()
                # descent = - local_grad.unity()


                s,n = self.line_search(state, descent, e0=self.errors[-1], plot=ls_plot.format(**locals()))
                if s == 0:
                    # # and tune parameters
                    # state = self.model.tune(state.params) # recent addition, needs testing!
                    # take local gradient instead
                    local_grad = state.grad
                    # TODO: Clean up my act and handle this gracefully
                    local_grad.A0 *= 0
                    local_grad.betas *= 0

                    descent = - local_grad.unity()
                    s,n = self.line_search(state, descent, e0=self.errors[-1], plot=ls_plot.format(**locals()))
                    # and reset RMSProp
                    self.past_grad = None
                    self.past_sqg = 1
                    if s == 0:
                        self.status = "CONVERGED_NO_DECREASE_ALONG_GRADIENT"
                        break

                self.ls_step.append(s)
                self.ls_nfev.append(n)

                upd = descent * s
                self.params = self.apply_delta(state.params, upd)
                # if tune:
                #     state = self.model.tune(self.params, A0_plot=A0_plot.format(**locals()), debug=debug)
                #     self.params = state.params

                self.model.params = self.params

                # if s < 1e-4 and self.t-self.last_quantile > 5:
                #     # res = self.model.quantile_fit(state)
                #     print "optimal parameters from quantile fit"
                #     # self.params.A0 = res[0]
                #     # self.params.betas[:] = self.model.optimal_betas(state)
                #     # self.last_quantile = self.t


                state = self.model.predict(self.params, **self.predict_kwargs)
                self.errors.append(state.error)
                self.t += 1
                self.history.append(state.archive())
                self.last_state = state
                if debug:
                    print ">>>>>>>>>UPDATE, scale=",s
                    print descent
                    self.print_state(state)

                if callback:
                    callback(self)

                print "n_fev={self.model.n_fev} t_aff={t_aff:.3f} t_fev={t_fev:.3f}ms n_grad={self.model.n_grad} t_grad={t_grad:.3f}ms".format(
                    self=self,
                    t_aff = 1000. * self.model.t_aff/self.model.n_fev,
                    t_fev = 1000. * self.model.t_fev/self.model.n_fev,
                    t_grad = 1000. * self.model.t_grad/self.model.n_grad,
                )

        # except ValueError: #KeyboardInterrupt
        except KeyboardInterrupt:
            self.status = "KEYBOARD_INTERRUPT"
        
        if self.t < maxiter:
            self.status = self.converged()
        else:
            self.status = "MAX_ITER"

        print "optimization ended with status {self.status} after {self.t} iterations".format(self=self)
        # print "last gradient"
        # print self.past_grad
        # print "squared"
        # print self.past_sqg
        
        return self




if __name__ == '__main__':
    # import gzip
    # path = os.path.join(os.path.dirname(__file__), '../tests/reads_20.txt.gz')
    # TODO: include small amount of raw data in git repo for testing!

    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=1000000)
    unittest.main(verbosity=2)
