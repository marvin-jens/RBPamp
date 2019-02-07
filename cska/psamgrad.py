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
    def __init__(self, rbns, params, ref=None, k_fit=6, mdl_name='partfunc', run_name='meanfield', maxiter=1000, maxtime=11.5*3600, eps=1e-5, redo=False, **kwargs):
        self.rbns = rbns
        self.ref = ref
        self.out_path = cska.ensure_path(os.path.join(rbns.out_path, "{}/".format(run_name)))

        fname = os.path.join(self.out_path, "descent.tsv")
        self.shelve = shelve.open(
            os.path.join(self.out_path, "history"), 
            protocol=-1, 
            flag='n' if redo else 'c'
        )

        self.t_ofs = 0
        if not os.path.exists(fname) or redo:
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
            self.track_file = file(fname, 'a', 0)

        self.k = params.k
        self.k_fit = k_fit
        self.R, self.R_err = rbns.R_value_matrix(self.k_fit)
        self.logR = np.log2(self.R)
        self.logger = logging.getLogger('opt.PSAMGradientDescent')
        self.results = logging.getLogger('results.PSAMGradientDescent')
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
        model = mdl(rbns.reads[0], params, self.R, rbp_conc = rbns.rbp_conc, **kwargs)
        # print self.descent.params.acc_k, self.descent.model.acc_k

        self.descent = cska.gradient.GradientDescent(model, params, maxiter=maxiter, maxtime=maxtime, eps=eps)
        self.model = model
        self.params = params
    
    def optimize(self, debug=True):
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

        def callback(descent):
            # ugcacgu = cyska.seq_to_index('ugcacgu')
            # print "UGCACGU", descent.model.affinities[ugcacgu]
            state = descent.last_state
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
            # from scipy.stats import pearsonr
            # pR, pval = np.array([pearsonr(lr0, lr) for lr0, lr in zip(self.logR,np.log2(descent.last_state.R))]).T
            pR, pval = descent.last_state.correlations
            out = [descent.t + self.t_ofs, descent.last_state.params[0].A0, descent.errors[-1],] \
                + list((descent.last_state.R_errors**2).mean(axis=1)) + list(pR) \
                + [descent.ls_nfev[-1], descent.ls_step[-1]]

            line = "\t".join([str(o) for o in out])
            print line
            self.track_file.write(line)
            self.track_file.write('\n')

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