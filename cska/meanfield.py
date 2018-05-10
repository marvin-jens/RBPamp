import cska.gradient
from cska.crosstalk_matrix import CrosstalkMatrix
import logging, os, sys
import cska.ska_kmers as cyska
from cska.pwm import PSAM
import numpy as np
import time

def debug_kmer_vector(vec, top=10, ref=None, header='values'):
    k = int(np.log(len(vec))/ np.log(4))
    
    I = vec.argsort()
    print "highest {0}".format(header)
    for i in I[::-1][:top]:
        out = [cyska.index_to_seq(i, k), str(vec[i])]
        if not ref is None:
            out.append(str(ref[i]))
        print "\t".join(out)
        

class MeanFieldModelState(object):
    def __init__(self, mdl, params):
        t0 = time.time()
        self.mdl = mdl
        self.params = params
        self.rbp_conc = mdl.rbp_conc

        self.A = cyska.params_from_pwm(params.psam_matrix, A0=params.A0, aff0=mdl.aff0)
        # TODO: make protein concentration self-consistent
        cyska.PSAM_mean_field_eval(self) # This call populates self.error, self.occ, self.pi etc...

        self.mdl.n_fev += 1
        self.mdl.t_fev += time.time() - t0
        
    @property
    def grad(self):
        t0 = time.time()
        self.mdl.n_grad += 1
        _grad = self.params.copy()
        t1 = time.time() - t0
        print "dt params.copy=", t1
        _grad.data[:] = cyska.PSAM_mean_field_gradient(self)
        self.mdl.t_grad += time.time() - t0
        return _grad
    

class MeanFieldModel(object):
    def __init__(self, reads, params0, rbp_conc=[], aff0=1e-6, **kwargs):
        self.rbp_conc = np.array(rbp_conc, dtype=np.float32)
        self.reads = reads
        self.params = params0
        self.aff0 = aff0
        self.opt = None
        k = self.params.k
        self.xm = CrosstalkMatrix(k, reads)
        
        f0 = reads.kmer_frequencies(k)
        self.f0 = f0 / f0.sum()
        
        self.n_fev = 0
        self.t_fev = 0
        self.n_grad = 0
        self.t_grad = 0

    def predict(self, params, aff0=1e-6, debug=False):
        state = MeanFieldModelState(self, params)
        return state

    def estimate_betas(self, state, q=5):
        R_ns = np.percentile(self.opt.R0, q, axis=1)
        # print "R_ns,j", R_ns
        beta = R_ns / (1 - R_ns)
        # print "beta?", beta
        # print "sum_pi with current beta estimate", state.sum_pi
        return state.sum_pi * beta

        
class MeanFieldAnalysis(object):
    def __init__(self, rbns, pwm, k):
        self.rbns = rbns
        self.out_path = cska.ensure_path(os.path.join(rbns.out_path, "meanfield/"))
        self.k = k
        self.R, self.R_err = rbns.R_value_matrix(k)
        
        params = cska.gradient.ModelParametrization(k, len(rbns.reads) - 1, psam=pwm.psam, A0=.1)
        params.betas[:] = .01
        # initial guess
        # params.betas[:] = [.013,.023,.062,.18,.19]
        # params.psam_matrix[3,1] = .1
        
        model = MeanFieldModel(rbns.reads[0], params, rbp_conc = rbns.rbp_conc)
        self.descent = cska.gradient.GradientDescent(model, params, self.R)
        state = model.predict(params)
        params.betas[:] = model.estimate_betas(state)
        import matplotlib.pyplot as pp
        pp.figure()
        pp.loglog(self.R[0],state.R[0],'x')
        pp.loglog(self.R[1],state.R[1],'x')
        pp.loglog(self.R[2],state.R[2],'x')
        pp.loglog(self.R[3],state.R[3],'x')
        pp.savefig(os.path.join(self.out_path, 'unoptimized.pdf'))
        pp.close()
        # pp.show()

        debug_kmer_vector(self.R[0], ref=state.R[0])
        debug_kmer_vector(self.R[1], ref=state.R[1])
        debug_kmer_vector(self.R[2], ref=state.R[2])
        debug_kmer_vector(self.R[3], ref=state.R[3])
        debug_kmer_vector(self.R[4], ref=state.R[4])
        # print "error", state.error
        # print "PARAMS"
        # print params
        # print "ANALYTICAL"
        # print state.grad
        # print "EMPIRICAL"
        # print cska.gradient.emp_grad(state)
        # print "-"*50
        from cska.report import GradientDescentReport
        def make_plots(descent):
            rep = GradientDescentReport(descent, path=self.out_path)
            rep.plot_report()
            rep.plot_param_hist()
            pwm = PSAM(psam= descent.params.psam_matrix, A0 = descent.params.A0)
            name = 'mean_field_{0}mer_PSAM'.format(descent.params.k)
            pwm.save_logo(os.path.join(self.out_path, name + '.eps' ))
            pwm.store_params(os.path.join(self.out_path, name + '.tsv'))
            
        def callback(descent):
            if not descent.t % 10:
                make_plots(descent)

        self.descent.optimize(maxiter=1000, debug=True, callback=callback)
        print "OPTIMIZATION RESULTS"
        print self.descent.params

        
        
        state = model.predict(self.descent.params)
        import matplotlib.pyplot as pp
        pp.figure()
        pp.loglog(self.R[0],state.R[0],'x')
        pp.loglog(self.R[1],state.R[1],'x')
        pp.loglog(self.R[2],state.R[2],'x')
        pp.loglog(self.R[3],state.R[3],'x')
        pp.savefig(os.path.join(self.out_path, 'optimized.pdf'))
        pp.close()

        make_plots(self.descent)

if __name__ == "__main__":
    
    from cska.reads import RBNSReads
    reads = RBNSReads('/scratch/data/RBNS/RBFOX3/RBFOX3_input.txt')
    xm = CrosstalkMatrix(6, reads)
    occ = np.zeros(4**6)
    occ[cyska.seq_to_index('UGCAUG')] = .5
    occ[cyska.seq_to_index('AGCAUG')] = .3
    occ[cyska.seq_to_index('CGCAUG')] = .25
    occ[cyska.seq_to_index('GGCAUG')] = .05
    occ[cyska.seq_to_index('UGCACG')] = .1
    t0 = time.time()
    r = xm.r_values_from_occupancies(occ, .01)
    print time.time() - t0
    debug_kmer_vector(r)