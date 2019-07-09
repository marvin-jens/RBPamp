import os
import logging
import shelve
import cska.gradient
import cska.cyska as cyska
import numpy as np

class PSAMGradientDescent(object):
    def __init__(self, rbns, params, ref=None, k_fit=6, run_name='opt_grad', maxiter=1000, maxtime=11.5*3600, eps=1e-5, tau=13, redo=False, debug_grad=False, resample_int=0, fix_A0=False, **kwargs):
        self.rbns = rbns
        self.ref = ref
        self.out_path = cska.ensure_path(os.path.join(rbns.out_path, "{}/".format(run_name)))
        self.logger = logging.getLogger('opt.PSAMGradient')
        self.results = logging.getLogger('results.PSAMGrad')

        self.k = params.k
        self.k_fit = k_fit
        self.R, self.R_err = rbns.R_value_matrix(self.k_fit)
        self.logR = np.log2(self.R)

        fname = os.path.join(self.out_path, "descent.tsv")
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
            except (IndexError, ValueError):
                pass
            self.logger.info("resuming track file '{0}' with {1} lines at t={2}".format(fname, len(lines), self.t_ofs))
            self.track_file = file(fname, 'a', 0)
        
        from cska.partfunc import PartFuncModel
        # from cska.meanfield import MeanFieldModel, InvMeanFieldModel
        # model = MeanFieldModel(rbns.reads[0], params, self.R, rbp_conc = rbns.rbp_conc)
        # model = InvMeanFieldModel(rbns.reads[0], params, self.R, rbp_conc = rbns.rbp_conc)
        # mdl = {
        #     'partfunc' : PartFuncModel,
        #     '' : PartFuncModel,
        #     'meanfield' : MeanFieldModel,
        #     'invmeanfield' : InvMeanFieldModel,
        # } [mdl_name]
        # self.model = mdl(rbns.reads[0], params, self.R, rbp_conc = rbns.rbp_conc, **kwargs)
        self.model = PartFuncModel(
            rbns.reads[0],
            params,
            self.R,
            rbp_conc = rbns.rbp_conc,
            **kwargs
        )
        self.model.init_subsample()
        self.rbns.flush(all=True)  # save some memory

        self.descent = cska.gradient.GradientDescent(
            self.model,
            params,
            maxiter = maxiter,
            maxtime = maxtime,
            eps = eps,
            tau = tau,
            debug_grad = debug_grad,
            fix_A0 = fix_A0
        )
        self.resample_int = resample_int
        self.last_resample = 0
        self.params = params

        sname = os.path.join(self.out_path, "history")
        self.shelve = shelve.open(
            sname, 
            protocol=-1, 
            flag='n' if redo else 'c'
        )
        self.logger.info("storing states in shelve '{}'".format(sname))
        self.shelve["R_exp"] = self.R
        self.shelve["rbp_conc"] = self.descent.model.rbp_conc


    def optimize(self, debug=False):
        def callback(descent, state):
            self.shelve["params_t{}".format(descent.t + self.t_ofs)] = state.params
            self.shelve["grad_t{}".format(descent.t + self.t_ofs - 1)] = descent.past_grad
            self.shelve["stats_t{}".format(descent.t + self.t_ofs)] = state.stats
            self.shelve["R_t{}".format(descent.t + self.t_ofs)] = state.R
            self.shelve["linesearch_t{}".format(descent.t + self.t_ofs)] = (descent.ls_nfev[-1], descent.ls_step[-1])
            self.shelve.sync()

            # collect and write data on the gradient descent progress
            pR, pval = state.correlations
            out = [descent.t + self.t_ofs, state.params.A0, state.error] \
                + list(state.sample_errors) + list(pR) \
                + [descent.ls_nfev[-1], descent.ls_step[-1]]

            line = "\t".join([str(o) for o in out])
            print line
            self.track_file.write(line)
            self.track_file.write('\n')
            descent.params.save(os.path.join(self.out_path, 'parameters.tsv'))

            self.logger.debug("t={0} error={1:.3f}% max_corr={2:.4f}".format(descent.t + self.t_ofs, 100 * state.error/descent.errors[0], np.array(pR).max()))
            if self.resample_int and (descent.t - self.last_resample) >= self.resample_int:
                # it's time to draw a new sub-sample
                state = descent.new_subsample()
                self.last_resample = descent.t

            return state

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
        self.store_residuals(state)
        return state

    def store_residuals(self, state):
        with file(os.path.join(self.out_path, '{0}mer_residuals.tsv'.format(self.descent.model.k)),'w') as f:
            f.write('#kmer\tlog2(R_pred/R_obs)\n')
            res = np.log2(state.R/state.mdl.R0)
            for i in range(state.mdl.nA):
                kmer = cyska.index_to_seq(i, self.descent.model.k)
                out = [kmer,] + ["{0:.3f}".format(r) for r in res[:,i]]
                f.write('\t'.join(out) + '\n')