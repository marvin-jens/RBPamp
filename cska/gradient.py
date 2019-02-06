import os
import unittest
import time
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class Tracked(object):
    """
    Used as a container to hold data on line-search intermediate results, A0 fitting, beta fitting, etc.
    """
    def __init__(self, **kwargs):
        self._kw = kwargs
        for k,v in kwargs.items():
            setattr(self, k,v)


def emp_grad(state, eps=1e-6):
    v0 = state.params.get_data()
    v = np.array(v0)
    var = state.params.copy()
    grad = np.array(v0)
    state0 = state
    # print "err0", err0
    kw = dict()#.predict_kwargs)
    kw['beta_fixed'] = True
    kw['tune'] = False
    kw['rbp_free'] = state0.rbp_free

    for i in range(len(v0)):
        # print ">>> EMP GRAD", state.params.names[i]
        # d = max(v0[i] * eps,1e-6)
        d = eps
        v[i] = v0[i] + d
        var.set_data(v)
        state = state.mdl.predict(var, **kw)
        derr = state.error - state0.error
        grad[i] = derr/d
        # print "derr", derr
        v[i] = v0[i]

        # print ">>>GRAD ELEMENT", grad.data[i]
    
    var.set_data(grad)
    return var


def emp_gradi(state, eps=1e-6):
    v0 = state.params.as_vector()
    var = state.params.copy()
    Nk = state.mdl.nA
    n_data = len(state.params.data)
    n_samples = state.params.n_samples
    gradi = np.zeros((n_samples, Nk, n_data), dtype=np.float32)

    R0 = state.R
    state0 = state

    kw = dict()
    kw['beta_fixed'] = True
    kw['rbp_free'] = state0.rbp_free

    for i in range(state.params.n):
        d = eps
        var.data[i] = v0[i] + d
        state = state.mdl.predict(var, **kw)
        dR = state.R - R0
        gradi[:,:,i] = dR/d

        var.data[i] = v0[i]
    
    return gradi


def minimize_logspaced(func, bounds=[], n_samples=7, debug=False, nested=2, options=None, **kwargs):
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

    res = minimize_scalar(func_or_lookup, bounds = np.array([bmin, bmax]), method='Bounded', options=options) #, **kwargs)
    t1 = time.time()
    x = sorted(known.keys())
    y = [known[i] for i in x]
    data = Tracked(x=x, y=y)
    res.opt_data = data
    # self.logger.debug("minimize_logspaced took {dt:.2f}ms".format(dt= 1000. * (t1-t0)) )
    return res


class GradientDescent(object):
    def __init__(self, model, params0, dec=.75, ref_state=None, maxiter=1000, maxtime=11.5*3600, eps=1e-6, predict_kwargs=dict(beta_fixed=False, tune=True)):
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
        # self.history = []
        self.ls_nfev = [0,]
        self.ls_step = [0,]
        self.t = 0
        self.last_quantile = 0

        # optimization result/status
        self.status = None
        self.maxiter = maxiter
        self.maxtime = maxtime
        self.eps = eps
        
    # @staticmethod
    # def apply_delta(params, delta):
    #     new = params.copy()
    #     p = np.clip(params.psam_matrix + delta.psam_matrix, 1e-6, None)
    #     M = p.max(axis=1)
    #     p /= M[:,np.newaxis]
    #     new.psam_matrix = np.clip(p, 1e-6, 1)
    #     new.A0 *= M.prod() # keep matrix elements <= 1 and absorb excess into A0
    #     new.A0 = max(1e-6, new.A0 + delta.A0) # prevent underflow

    #     new.betas = np.clip(params.betas + delta.betas, 1e-9, None)
    #     return new


    def line_search(self, state, vec, debug=False, min_step = 1e-6, max_step = 10., maxiter=10, xatol=1e-1, e0=None, plot=""):
        from scipy.optimize import minimize_scalar

        e0 = state.error
        # assert e0 == state.mdl.predict(state.params, **self.predict_kwargs).error

        params0 = state.params
        t0 = time.time()
        N = {'fev' : 0}
        kw = dict(self.predict_kwargs)
        kw['beta_fixed'] = True
        kw['tune'] = False

        scales = []
        errors = []

        self.model.set_mask( state.Z1_read > self.model.Z_thresh * state.Z1_read_max)
        def err(s):
            # s = np.exp(x)
            m = params0.apply_delta(vec * s)
            new = self.model.predict(m, **kw)
            N['fev'] += 1
            scales.append(s)
            new_err = new.error
            errors.append(new_err)
            if debug:
                print s,"->", new_err - e0
            return new_err - e0

        assert np.fabs(err(0)) < 1e-6

        res = minimize_logspaced(err, bounds = np.array([min_step, max_step]), options=dict(maxiter=maxiter) )
        # res = minimize_scalar(err, method='Bounded', bounds=np.log(np.array([min_step, max_step])), options=dict(maxiter=maxiter, xatol=xatol))
        # self.logger.debug("minimize_logspaced took {dt:.2f}ms".format(dt= 1000. * (t1-t0)) )

        self.model.set_mask()
        self.logger.debug("line_search took {0:.3f} seconds for {1} iterations".format(time.time() - t0, N['fev']))
        # print res.success, res.fun, res
        if res.fun > 0:
            self.logger.warning("line_search could not decrease error!")
            s = 0
        else:
            s = res.x
        
        scales = np.array(scales)
        errors = np.array(errors)
        I = scales.argsort()
        return s, Tracked(scales = scales[I], errors=errors[I], res=res, s_opt=s, err0=e0)

    def momentum_grad(self, local_grad):
        if self.past_grad is None:
            self.past_grad = local_grad

        grad = self.past_grad * self.dec + local_grad * (1- self.dec)
        self.past_grad = grad
        return grad

    def RMSprop(self, local_grad, delta=.00001):
        if self.past_grad is None:
            self.past_grad = local_grad.get_data()

        m = self.dec * self.past_grad + (1 - self.dec) * local_grad.get_data()
        s = self.dec * self.past_sqg + (1 - self.dec) * local_grad.get_data()**2

        self.past_grad = m
        self.past_sqg = s

        upd = local_grad.copy()
        upd.set_data(m / (np.sqrt(s) + delta))

        return upd

    def converged(self, atol=1e-7, tau=10):
        if len(self.errors):
            if self.errors[-1] < atol:
                return 'CONVERGED_ERR_MINIMAL'

        if len(self.errors) < tau:
            return self.status
           
        last_errs = np.array(self.errors[-tau:])
        mean = last_errs.mean()
        if not self.past_grad is None:
            mag = np.sqrt((self.past_grad**2).sum())
            if np.allclose(self.past_grad, 0, atol=atol):
                return 'CONVERGED_GRAD_NULL'

        elif (mean - self.errors[-1]) / mean < self.eps:
            return 'CONVERGED_NO_MORE_DECREASE'
            
        else:
            return self.status

    def reached_maxtime(self, dt):
        return self.maxtime and (dt >= self.maxtime)
    
    def reached_maxiter(self, t):
        return self.maxiter and (t >= self.maxiter)

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


    def optimize(self, params, debug=True, callback=None):
        if debug:
            print "INITIAL PARAMETERS"
            print params
        
        state = self.model.predict(self.params, **self.predict_kwargs)

        self.errors.append(state.error)
        # self.history.append(state.archive())
        self.last_state = state

        if debug:
            print "INITIAL STATE AFTER FIRST EVAL kwargs=", self.predict_kwargs
            self.print_state(state)

        if callback:
            callback(self)

        t0 = time.time()
        dt = 0

        try:
            while not self.converged() and not self.reached_maxiter(self.t) and not self.reached_maxtime(dt):
                # print "computing gradient"
                local_grad = state.grad #.unity()
                if debug:
                    print "LOCAL GRAD, EMP. GRAD"
                    print local_grad
                    # for lcl, emp in zip(local_grad, emp_grad(state)):
                    #     print "LCL"
                    #     print lcl
                    #     print "EMP"
                    #     print emp


                local_grad.betas *= 0
                # local_grad.A0 = 0
                # local_grad = local_grad.unity()
                descent = self.RMSprop( - local_grad ).unity()
                # descent = self.momentum_grad( - local_grad).unity()
                # descent = - local_grad.unity()
                # print descent

                s, ls_data = self.line_search(state, descent, e0=self.errors[-1], debug=debug)
                if s == 0:
                    self.logger.warning("line_search could not decrease error! Resetting search direction to local gradient ...")
                    # # and tune parameters
                    # state = self.model.tune(state.params) # recent addition, needs testing!
                    # take local gradient instead
                    local_grad = state.grad
                    # TODO: Clean up my act and handle this gracefully
                    # local_grad.A0 *= 0
                    # local_grad.betas *= 0

                    descent = - local_grad.unity()
                    s, data = self.line_search(state, descent, e0=self.errors[-1])
                    # and reset RMSProp
                    self.past_grad = None
                    self.past_sqg = 1
                    if s == 0:
                        self.logger.warning("line_search unable to reduce error using local gradient")
                        self.status = "CONVERGED_NO_DECREASE_ALONG_GRADIENT"
                        break

                self.ls_step.append(s)
                self.ls_nfev.append(ls_data.res.nfev)

                upd = descent * s
                self.params = state.params.apply_delta(upd)
                self.model.params = self.params

                state = self.model.predict(self.params, **self.predict_kwargs)
                state._ls_data = ls_data
                self.errors.append(state.error)
                self.t += 1
                # self.history.append(state.archive())
                self.last_state = state
                if debug:
                    print ">>>>>>>>>UPDATE, scale=",s
                    # print descent
                    self.print_state(state)

                if callback:
                    callback(self)

                self.logger.debug("n_fev={self.model.n_fev} t_aff={t_aff:.3f} t_fev={t_fev:.3f}ms n_grad={self.model.n_grad} t_grad={t_grad:.3f}ms".format(
                    self=self,
                    t_aff = 1000. * self.model.t_aff/self.model.n_fev,
                    t_fev = 1000. * self.model.t_fev/self.model.n_fev,
                    t_grad = 1000. * self.model.t_grad/self.model.n_grad,
                ))
                dt = time.time() - t0

        # except ValueError: #KeyboardInterrupt
        except KeyboardInterrupt:
            self.status = "KEYBOARD_INTERRUPT"
        else:
            if self.reached_maxtime(dt):
                self.status = "MAX_TIME"

            elif self.reached_maxiter(self.t):
                self.status = "MAX_ITER"

            else:
                self.status = self.converged()

        self.logger.info("optimization ended with status {self.status} after {self.t} iterations".format(self=self))
        # print "last gradient"
        # print self.past_grad
        # print "squared"
        # print self.past_sqg
        
        return self




if __name__ == '__main__':
    # import gzip
    # path = os.path.join(os.path.dirname(__file__), '../tests/reads_20.txt.gz')
    # TODO: include small amount of raw data in git repo for testing!
    from cska.reads import RBNSReads
    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt', acc_storage_path='cska/acc', n_max=1000000)
    unittest.main(verbosity=2)
