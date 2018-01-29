import os
import cska
import numpy as np
import matplotlib
import matplotlib.pyplot as pp
import logging
from collections import defaultdict

class EnrichmentBarPlot(object):
    def __init__(self, rbns_comparison):
        self.rbns_comparison = rbns_comparison
        
    def make_plot(self, k, fname=None, dest="./", fmt='svg', Z_cut=2, figsize=(7,5) ):
        import matplotlib
        #matplotlib.use(fmt)
        import matplotlib.pyplot as pp
        import numpy as np
        if not fname:
            fname = "r_values.{self.rbns_comparison.pd_reads.name}.{k}mers".format(self=self, k=k)
        
        R, R_err = self.rbns_comparison.R_values(k)
        kmers = np.array(list(cska.ska_kmers.yield_kmers(k)))

        I = R.argsort()
        R = R[I]
        R_err = R_err[I]
        kmers = kmers[I]
        
        Z = (R - np.mean(R))/ np.std(R)
        enriched_i = (Z > Z_cut).argmax()
        depleted_i = (Z < -Z_cut).argmin()
        
        
        pp.figure(figsize=figsize)

        n = len(R)
        n_enriched = len(R) - enriched_i
        n_ns = enriched_i
        #x_ns = np.linspace(0, 1., num=n_ns)
        x_ns = np.arange(n_ns)
        x_enr = np.arange(n_ns,n)
        pp.fill_between(x_ns, 1, R[:enriched_i], color="0.75", label=None)
        pp.fill_between(x_enr, 1, R[enriched_i:], color='k', label="Z-score > {0}".format(Z_cut))
        
        #xlabel = 
        for x, y, mer in zip(x_enr, R[enriched_i:], kmers[enriched_i:]):
            #pp.text(x, r, mer, fontdict=dict(size=6), withdash=True)
            pp.annotate(mer, xy=(x, y), xytext=(x-.05*n, y), arrowprops=dict(facecolor='black', arrowstyle="->, head_length = .2, head_width = .2"), horizontalalignment='right', verticalalignment='center', fontsize=6)
        
        pp.ylabel('{0}mer enrichment [R-value]'.format(k) )
        pp.xlabel('rank')
        pp.axhline(1, color='k', linewidth=.5)
        pp.xlim(0,n)
        pp.gca().set_yscale('log')
        pp.legend(loc='upper left')

        #x_enriched = np.linspace(0, 1,n_enriched) + 1.1
        #print x_enriched.shape, R[enriched_i:].shape
        #pp.bar(x_enriched, R[enriched_i:], width = .05, color='k')
        
        
        path = os.path.join(dest, "{0}.{1}".format(fname, fmt) )
        pp.savefig(path)
        
class TrackedValues(object):
    def __init__(self):
        self.d0 = None
        self.last = None
        self.updates = []
        self.times = []
        self.N = 0
        self.logger = logging.getLogger("TrackedValues")
        
    def store(self, t, d):
        
        if not self.times:
            self.d0 = d
            self.last = d
            self.times.append(t)
            return
        
        delta = self.last - d
        self.last = d

        if (delta == 0).all():
            return
        
        ind = delta.nonzero()[0]
        #self.logger.error(delta)
        #self.logger.error(ind)
        self.updates.append( (ind, d[ind]) )
        self.times.append(t)
        
    def read(self):
        if self.d0 != None:
            data = [self.d0,]
        else:
            data = []
        last = self.d0
        for ind, vals in self.updates:
            d = np.array(last)
            d[ind] = vals
            data.append(d)
            last = d
        
        data = np.array(data)
        #print len(self.times), len(self.updates), data.shape

        assert len(self.times) == len(data)
        return self.times, data
        
        
class OptReporting(object):
    def __init__(self, opt, path='./', track=[], report_interval=50, comp=None):
        self.opt = opt
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        
        from matplotlib.backends.backend_pdf import PdfPages
        self.sweep_pdf = PdfPages(os.path.join(self.path,'local_fits.pdf') )
        self.descent_pdf = PdfPages(os.path.join(self.path,'gradient_descent.pdf') )
        self.R_pdf = PdfPages(os.path.join(self.path,'R_value_fit.pdf') )
        self.invkd_pdf = PdfPages(os.path.join(self.path,'invkd_fit.pdf') )
        self.err_pdf = PdfPages(os.path.join(self.path,'err_fit.pdf') )
        
        self.comp = comp
        if comp and not track:
            track = sorted(comp.uniq_kmers)

        self.tracked_kmers = [t for t in track if len(t)== self.opt.k]
        import cska.ska_kmers
        self.tracked_indices = [cska.ska_kmers.seq_to_index(t) for t in self.tracked_kmers]
        self.tracked = set(self.tracked_indices)
        self.tracked_history = defaultdict(list)
        self.tracked_updated = defaultdict(list)
        for param_i in self.tracked_indices:
            self.tracked_history[param_i].append( (self.opt.t, self.opt.current.params[param_i]) )

        self.report_interval = report_interval
        self.last_report = 0
        self.logger = logging.getLogger('OptReporting')
        self.betas = TrackedValues()
        #self.affinities = TrackedValues()
    
    def tick(self, t):
        self.betas.store(t, self.opt.current.params[self.opt.nA:])
        #self.affinities.store(t, self.opt.current.params[:self.opt.nA])
        
        if t > self.last_report + self.report_interval:
            self.plot_errors()
            self.plot_correlations()
            self.plot_betas()
            #self.plot_tracked_kmer_histories()
            #self.plot_affinity_history()
            
            self.plot_R_value_agreement()
            self.plot_known_comparison()
            
            for param_i in self.tracked_indices:
                self.plot_sweep(param_i)
            self.last_report = t
        
        for param_i in self.tracked_indices:
            self.tracked_history[param_i].append( (t, self.opt.current.params[param_i]) )
            if param_i == self.opt.sched.last_param_update:
                self.tracked_updated[param_i].append(t)

    def close(self):
        self.plot_errors()
        self.plot_correlations()
        self.plot_betas()
        #self.plot_affinity_history()
        
        for pdf in [self.sweep_pdf, self.descent_pdf, self.R_pdf, self.invkd_pdf, self.err_pdf]:
            try:
                pdf.close()
            except AttributeError:
                pass

    def plot_errors(self):
        pp.figure()
        pp.semilogy(self.opt.errors)
        pp.xlabel("time")
        pp.ylabel("total error")
        pp.savefig(os.path.join(self.path,'error.pdf'))
        pp.close()
        
    def plot_correlations(self):
        pp.figure()
        for conc, corr in zip(self.opt.rbp_conc, np.array(self.opt.correlations).T )[::-1]:
            pp.plot(corr, label='{0:.2f} nM'.format(conc))
        
        pp.xlabel('time step')
        pp.ylabel('correlation coefficient')
        pp.legend(loc='lower right')
        pp.savefig(os.path.join(self.path,'corr.pdf'))
        pp.close()

    def plot_betas(self):
        pp.figure()
        t, betas = self.betas.read()

        for conc, beta in zip(self.opt.rbp_conc, betas.T )[::-1]:
            pp.semilogy(t, beta, label='{0:.2f} nM'.format(conc))
        
        pp.xlabel('time step')
        pp.ylabel('background (beta)')
        pp.legend(loc='lower right')
        pp.savefig(os.path.join(self.path,'betas.pdf'))
        pp.close()

    def plot_affinity_history(self, n=10):
        pp.figure()
        t, aff_matrix = self.affinities.read()
        last = aff_matrix[-1]
        top_kmer_ind = last.argsort()[::-1][:n]
        top_kmers = self.opt.kmers[top_kmer_ind]

        for kmer, aff in zip(top_kmers, aff_matrix.T[top_kmer_ind] ):
            pp.semilogy(t, aff, label=kmer)
        
        pp.xlabel('time step')
        pp.ylabel('affinity [1/nM]')
        pp.legend(loc='lower right')
        pp.savefig(os.path.join(self.path,'affinity_history.pdf'))
        pp.close()
        
        
    def plot_tracked_kmer_histories(self):
        self.logger.info("rendering tracked kmer affinity history plots")
        pp.figure()
        for i, param_i in enumerate(self.tracked_indices):
            kmer = self.tracked_kmers[i]
            t, param = np.array(self.tracked_history[param_i]).T
            
            pp.semilogy(t, param, label=kmer)
            known = self.opt.known_params[param_i]
            if np.isfinite(known):
                pp.axhline(known, label='{0} reference'.format(kmer))

            #for t_update in self.tracked_updated[param_i]:
                #pp.axvline(t_update)
                
        pp.xlabel('optimization step')
        pp.ylabel('affinity [1/nM]')
        pp.legend(loc='lower right')
        pp.savefig(os.path.join(self.path, "tracked_kmers.pdf"))
        pp.close()

    def plot_known_comparison(self):
        if not self.comp:
            return
        
        x = self.comp.observed_affinities
        y = self.comp.expected_affinities
        R = np.corrcoef(np.log(x), np.log(y))[0][1]
        pp.figure()
        pp.loglog(x, y, '.r', label="R={:.3f}".format(R))
        pp.xlabel("observed/known affinity [1/nM]")
        pp.ylabel("expected from fit [1/nM]")
        pp.legend(loc='lower right')
        pp.savefig(os.path.join(self.path, "known_affinity_comparison.pdf"))
        pp.close()
        
        
    def plot_gradient_descent(self, n_top=100):
        I = self.opt.R_obs.argsort()[::-1][:n_top]
        
        x = np.arange(len(I))
        pp.figure()
        t = self.opt.t
        pp.title("gradient-descent at step {0}".format(t))
        
        plot = pp.semilogy
        plot(x, self.opt.known_invkd[I], 'k', label='known affinities')
        plot(x, self.opt.prev_invkd[I], '.b', label='prev. delta')
        plot(x, self.opt.trial_invkd[I], '.r', label='last delta')
        pp.xlim(-1, len(x))
        self.descent_pdf.savefig()
        pp.savefig(os.path.join(self.path, "descent_t{0}.pdf".format(t)) )

        
    def plot_jacobi(self):
        
        for conc, jac in zip(self.opt.rbp_conc, self.opt.jacobi_new):
            pp.figure()
            pp.title("Jacobi matrix @{1}nM at t={0}".format(self.opt.t, conc))
            pp.imshow(np.arcsinh(jac), cmap='hot')
            pp.colorbar(label='arcsinh(jacobi matrix)')
            pp.savefig('jacobi_{1}nM_t{0}.pdf'.format(self.opt.t, conc))

        pp.show()
        pp.close()
        
    def plot_sweep(self, param_i=None, param= None, min_val = 1e-12, max_val = 1e3, steps=100, errors = [], x = []):
  
        if param_i == None:
            param_i = self.opt.sched.last_param_update

        if param == None:
            param = self.opt.mdl.param_name[param_i]

        if not len(errors):
            params = np.copy(self.opt.current.params)
            x = np.exp(np.linspace(np.log(min_val), np.log(max_val), steps))
            if param_i < self.opt.mdl.nA:
                # evaluate thermodynamic model, but only on the subset of sequences containing the kmer
                tm_update = True
                from cska.rbns_model import SPAPartition
                opt = SPAPartition(self.opt.mdl, param_i)
            else:
                # do not evaluate the thermodynamic model, only re-compute R-values
                tm_update = force_tm
                opt = self.opt.mdl

            for ikd in x:
                params[param_i] = ikd
                state = opt.evaluate(params, tm_update=tm_update)
                err = self.opt.global_error(state.R)
                errors.append(err)

            errors = np.array(errors)

        pp.figure()
        pp.title("param-fit for {0} at step {1}".format(param, self.opt.t) )
        
        print x
        print errors
        pp.semilogx(x, errors)

        #pp.axvline(self.opt.mdl.known_params[param_i], color='r', label="correct value")
        #pp.axvline(self.opt.trial_val[param_i], color='k', label="fitted root")

        pp.axhline(0, color='k')
        #if opt.param_updates[param] > 1:
            #pp.axvline(self.opt.prev_val[param_i], color='gray', label="previous value")
                
        pp.xlabel(r"$\frac{1}{K_d}$ [nM]")
        #pp.ylabel(r"expected R - observed R")
        pp.ylabel(r"global error")
        
        pp.legend(loc='upper left')
        pp.tight_layout()
        #self.sweep_pdf.savefig()
        pp.savefig(os.path.join(self.path, "sweep_{0}_t{1}.pdf".format(param, self.opt.t)) )
        pp.show()
        pp.close()

    def plot_R_value_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]
        
        self.logger.info('rendering R-value agreement plot')
        kmer_i = self.opt.sched.last_param_update
        if kmer_i == None:
            kmer = "none"
        else:
            kmer = self.opt.mdl.param_name[kmer_i]
        
        pp.figure()
        pp.title('R-value fit after step {0}'.format(self.opt.t) )
        R_a = self.opt.R_obs
        R_b = self.opt.current.R
        for i,rbp_conc in reversed(list(enumerate(self.opt.rbp_conc))):
            corr = np.corrcoef(np.log(R_a[i]), np.log(R_b[i]))[0][1]
            patches = pp.loglog(R_a[i], R_b[i], 'o', markeredgecolor='none', markersize=3, alpha=.75, label="P={0:.2f}nM (R={1:.3f})".format(rbp_conc, corr) )

        if kmer_i < self.opt.nA and kmer_i != None:
            pp.loglog(R_a[:, kmer_i], R_b[:, kmer_i], 'o', markersize=6, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        #m = min(R_a.min(), R_b.min())
        #M = max(R_a.max(), R_b.max())
        m = R_a.min() * .75 # always use experiment as reference
        M = R_a.max() * 1.25

        pp.plot([m,M],[m,M], '--k', zorder=np.inf)
        pp.xlim(m,M)
        pp.ylim(m,M)
        pp.xlabel(r'{0} [R-value]'.format("observed/simulated") )
        pp.ylabel(r'{0} [R-value]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
        pp.legend(loc='upper left')
        pp.tight_layout()
        self.R_pdf.savefig()
        pp.savefig(os.path.join(self.path, "predicted_vs_obs_R_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        pp.close()
        
    def plot_invkd_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]
        
        kmer_i = self.opt.sched.last_param_update
        kmer = self.opt.mdl.param_name[kmer_i]
        
        
        pp.figure()
        pp.title('affinity agreement after step {0}'.format(self.opt.t) )
        A_a = self.opt.known_invkd
        A_b = self.opt.trial_invkd
        x = A_a
        y = A_b
        
        max_error_conc = np.fabs(self.opt.kmer_error_new).argmax(axis=0)
        for i,rbp_conc in enumerate(self.opt.rbp_conc):
            ind = max_error_conc == i
            #print ind.shape, ind, x[ind]
            patches = pp.loglog(x[ind], y[ind], 'o', markeredgecolor='none', markersize=5, label="max error at P={0:.2f}nM".format(rbp_conc) )

        corr = np.corrcoef(np.log(A_a), np.log(A_b))[0][1]
        pp.loglog(x[kmer_i], self.opt.prev_invkd[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='gray', label="previous values" )
        pp.loglog(A_a[kmer_i], A_b[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        m = min(A_a.min(), A_b.min())
        M = max(A_a.max(), A_b.max())
        pp.plot([m,M],[m,M], '--k', zorder=np.inf)
        pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
        pp.ylabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
        pp.legend(loc='lower right')
        #pp.xlim(1e-1,1e2)
        #pp.ylim(1e-1,1e2)
        pp.tight_layout()
        self.invkd_pdf.savefig()
        pp.savefig(os.path.join(self.path, "predicted_vs_obs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        pp.close()
        
    #def plot_errors(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]

        #kmer_i = self.opt.last_param_update
        #kmer = self.opt.kmers[kmer_i]
        
        #pp.figure()
        #pp.title('residual errors after step {0}'.format(self.opt.t) )

        #abs_err = np.arcsinh(self.opt.kmer_error_new)
        ##abs_err = np.where(abs_err > 0, np.arcsinh(abs_err), -np.arcsinh(-abs_err) )
        
        #for i,rbp_conc in enumerate(self.opt.rbp_conc):
            #patches = pp.semilogx(self.opt.known_invkd, abs_err[i,:], 'o', markeredgecolor='none', markersize=3, alpha=.75, label="P={0}nM".format(rbp_conc) )
            
        #mark_x = self.opt.known_invkd[to_mark_i]
        #mark_y = abs_err[:,to_mark_i].max(axis=0)
        #for mer, index, x, y in zip(to_mark, to_mark_i, mark_x, mark_y):
            #pp.text(x*2, y, mer)

        ##pp.plot(mark_x, mark_y, 'o', markersize=10, markeredgecolor = 'black' , markerfacecolor='none')

        #pp.semilogx(np.repeat(self.opt.known_invkd[kmer_i], len(self.opt.rbp_conc)), abs_err[:, kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        
        #pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
        #pp.ylabel(r'arcsinh(residual error) [a.u.]')
        #pp.legend(loc='upper left')
        ##pp.xlim(1e-1,1e2)
        ##pp.ylim(1e-1,1e2)
        #pp.tight_layout()
        #self.err_pdf.savefig()
        #pp.savefig(os.path.join(self.path, "err_vs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        #pp.close()

