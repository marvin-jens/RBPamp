import cska.gradient
import logging, os, sys
import cska.cyska as cyska
from cska.pwm import PSAM
import numpy as np
import time
import shelve
from cska.meanfield import MeanFieldModel, InvMeanFieldModel
from cska.affinitylogo import nice_conc

class PSAMGradientDescent(object):
    def __init__(self, rbns, params, ref=None, k_fit=6, mdl_name='partfunc', run_name='meanfield', maxiter=1000, maxtime=11.5*3600, eps=1e-5, redo=False, debug_grad=False, resample_int=0, **kwargs):
        self.rbns = rbns
        self.ref = ref
        self.out_path = cska.ensure_path(os.path.join(rbns.out_path, "{}/".format(run_name)))
        self.logger = logging.getLogger('opt.PSAMGradientDescent')
        self.results = logging.getLogger('results.PSAMGradientDescent')

        fname = os.path.join(self.out_path, "descent.tsv")
        sname = os.path.join(self.out_path, "history")
        self.shelve = shelve.open(
            sname, 
            protocol=-1, 
            flag='n' if redo else 'c'
        )
        self.logger.info("storing states in shelve '{}'".format(sname))

        self.t_ofs = 0
        if not os.path.exists(fname) or redo:
            self.logger.info("tracking progress in new file '{}'".format(fname))
            self.track_file = file(fname, 'w', 0)
            MSE_samples = ["MSE{}".format(i) for i in range(params.n_samples)]
            corr_samples = ["corr{}".format(i) for i in range(params.n_samples)]
            self.track_file.write('# t\tA0\tMSE\t{0}\t{1}\tnfev\tstep\n'.format("\t".join(MSE_samples), "\t".join(corr_samples)))
        else:
            lines = file(fname).readlines()
            try:
                self.t_ofs = int(lines[-1].split('\t')[0]) + 1
            except IndexError, ValueError:
                pass
            self.logger.info("resuming track file '{0}' with {1} lines at t={2}".format(fname, len(lines), self.t_ofs))
            self.track_file = file(fname, 'a', 0)

        self.k = params.k
        self.k_fit = k_fit
        self.R, self.R_err = rbns.R_value_matrix(self.k_fit)
        self.logR = np.log2(self.R)
        # print "k_fit", k_fit, "rbnd.reads", [str(r) for r in rbns.reads]
        # params.betas[:] = .0001
        # initial guess
        # params.betas[:] = [.013,.023,.062,.18,.19]
        # params.psam_matrix[3,1] = .1
        
        from cska.partfunc import PartFuncModel
        # model = MeanFieldModel(rbns.reads[0], params, self.R, rbp_conc = rbns.rbp_conc)
        # model = InvMeanFieldModel(rbns.reads[0], params, self.R, rbp_conc = rbns.rbp_conc)
        mdl = {
            'partfunc' : PartFuncModel,
            '' : PartFuncModel,
            'meanfield' : MeanFieldModel,
            'invmeanfield' : InvMeanFieldModel,
        } [mdl_name]
        self.model = mdl(rbns.reads[0], params, self.R, rbp_conc = rbns.rbp_conc, **kwargs)
        self.model.init_subsample()
        self.rbns.flush(all=True)

        self.resample_int = resample_int
        self.last_resample = 0

        self.descent = cska.gradient.GradientDescent(self.model, params, maxiter=maxiter, maxtime=maxtime, eps=eps, debug_grad=debug_grad)
        self.params = params
    
    def optimize(self, debug=False):
        # params.betas[:] = model.estimate_betas(state)
        # params.betas[:] = model.optimal_betas(state)
        # res = model.quantile_fit(state)
        # print "optimal parameters from quantile fit"
        # params.A0 = res[0]
        # params.betas[:] = res[1:]
        self.shelve["R_exp"] = self.R
        self.shelve["rbp_conc"] = self.descent.model.rbp_conc
        # from cska.report import GradientDescentReport, LiteratureComparisonReport

        # lrep = LiteratureComparisonReport(self.descent, self.ref, path=self.out_path)

        def make_plots(descent, dt=None):
            # rep = GradientDescentReport(descent, path=self.out_path)
            reset = False
            if dt > 5 or dt is None:
                # lrep.plot_scatter(debug=False)
                # rep.plot_report()
                # rep.plot_param_hist()
                # rep.plot_line_search()
                # rep.plot_A0_fit()
                for i, params in enumerate(descent.params):
                    pwm = PSAM(psam=params.psam_matrix, A0 = params.A0)
                    name = 'rank_{i}_{pwm.consensus}.svg'.format(**locals())
                    pwm.save_logo(os.path.join(self.out_path, name))
                # pwm.store_params(os.path.join(self.out_path, name + '.tsv'))
                    # params.save(os.path.join(self.out_path, 'parameters.tsv'), append=(i > 0))
                # state = descent.model.predict(descent.params)
                # self.store_affinities(state)
                # self.store_residuals(descent.last_state)
                reset = True

            # if (descent.t % 10) == 0 or dt is None:
            #     rep.plot_scatter()

            return reset

        def callback(descent, state):
            self.shelve["params_t{}".format(descent.t + self.t_ofs)] = state.params
            self.shelve["grad_t{}".format(descent.t + self.t_ofs - 1)] = descent.past_grad
            self.shelve["stats_t{}".format(descent.t + self.t_ofs)] = state.stats
            self.shelve["R_t{}".format(descent.t + self.t_ofs)] = state.R
            self.shelve["linesearch_t{}".format(descent.t + self.t_ofs)] = (descent.ls_nfev[-1], descent.ls_step[-1])
            self.shelve.sync()
            descent.params.save(os.path.join(self.out_path, 'parameters.tsv'))

            dt = time.time() - self.t0
            if make_plots(descent, dt):
                self.t0 = time.time()
                
            # collect and write data on the gradient descent progress
            pR, pval = state.correlations
            out = [descent.t + self.t_ofs, state.params[0].A0, descent.errors[-1],] \
                + list((state.R_errors**2).mean(axis=1)) + list(pR) \
                + [descent.ls_nfev[-1], descent.ls_step[-1]]

            line = "\t".join([str(o) for o in out])
            print line
            self.track_file.write(line)
            self.track_file.write('\n')

            if self.resample_int and (descent.t - self.last_resample) >= self.resample_int:
                # it's time to draw a new sub-sample
                state = descent.new_subsample()
                self.last_resample = descent.t

            return state

        self.t0 = time.time()
        self.descent.optimize(self.params, debug=debug, callback=callback)

        stats_first = self.shelve["stats_t0"]
        stats_last = self.descent.last_state.stats
        err_reduction = stats_first.error / stats_last.error # x-fold reduced

        corr_first = stats_first.pearsonR.max()
        corr_last = stats_last.pearsonR.max()
        
        Kd_first = 1/self.shelve["params_t0"].A0
        Kd_last = 1/self.descent.last_state.params.A0

        self.logger.info("finished with status {0} and relative improvement of {1} ".format(self.descent.status, self.descent.error_reduction))
        self.logger.info("optimized parameters {0}".format(self.descent.params))        
        self.results.critical("GRAD err={stats_first.error:.2e} -> {stats_last.error:.2e} ({err_reduction:.2f} -fold) corr={corr_first:.3f} -> {corr_last:.3f} Kd={Kd_first} -> {Kd_last} t={self.descent.t} steps".format(**locals()))
        self.track_file.close()
        
        state = self.descent.last_state
        # import matplotlib.pyplot as pp
        # pp.figure()
        # pp.loglog(self.R[0],state.R[0],'x')
        # pp.loglog(self.R[1],state.R[1],'x')
        # pp.loglog(self.R[2],state.R[2],'x')
        # pp.loglog(self.R[3],state.R[3],'x')
        # pp.savefig(os.path.join(self.out_path, 'optimized.pdf'))
        # pp.close()

        make_plots(self.descent)
        # self.store_affinities(state)
        self.store_residuals(state)

        return state

    def store_affinities(self, state):
        pass
        self.logger.error("store_affinities() deprecated")
        # with file(os.path.join(self.out_path, '{0}mer_affinities.tsv'.format(state.params.k)),'w') as f:
        #     f.write('#kmer\taffinity[1/nM]\n')
        #     for i, aff in enumerate(state.mdl.affinities):
        #         kmer = cyska.index_to_seq(i, state.params.k)
        #         f.write('{0}\t{1}\n'.format(kmer, aff))

    def store_residuals(self, state):
        with file(os.path.join(self.out_path, '{0}mer_residuals.tsv'.format(self.descent.model.k)),'w') as f:
            f.write('#kmer\tlog2(R_pred/R_obs)\n')
            res = np.log2(state.R/state.mdl.R0)
            for i in range(state.mdl.nA):
                kmer = cyska.index_to_seq(i, self.descent.model.k)
                out = [kmer,] + ["{0:.3f}".format(r) for r in res[:,i]]
                f.write('\t'.join(out) + '\n')