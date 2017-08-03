import os
import cska
import numpy as np
import matplotlib
import matplotlib.pyplot as pp

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
        

class OptReporting(object):
    def __init__(self, opt, path='./'):
        self.opt = opt
        self.path = path
        from matplotlib.backends.backend_pdf import PdfPages
        self.sweep_pdf = PdfPages(os.path.join(self.path,'local_fits.pdf') )
        self.descent_pdf = PdfPages(os.path.join(self.path,'gradient_descent.pdf') )
        self.R_pdf = PdfPages(os.path.join(self.path,'R_value_fit.pdf') )
        self.invkd_pdf = PdfPages(os.path.join(self.path,'invkd_fit.pdf') )
        self.err_pdf = PdfPages(os.path.join(self.path,'err_fit.pdf') )

    def close(self):
        self.sweep_pdf.close()
        self.descent_pdf.close()
        self.R_pdf.close()
        self.invkd_pdf.close()
        self.err_pdf.close()

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
        
    def plot_sweep(self, kmer_index=None, min_invkd = 1e-12, max_invkd = 1e2, steps=100):
  
        if kmer_index == None:
            kmer_index = self.opt.last_kmer_update

        invkd = np.copy(opt.trial_invkd)
        x = np.exp(np.linspace(np.log(min_invkd), np.log(max_invkd), steps))
        errors = []
        for ikd in x:
            invkd[kmer_index] = ikd
            R_trial = self.opt.predict_R(invkd)
            err = R_trial[:,kmer_index] - self.opt.R_obs[:,kmer_index]
            errors.append(err)
            
        errors = np.array(errors).T
        
        kmer = self.opt.kmers[kmer_index]
        pp.figure()
        pp.title("kmer-fit for {0} at step {1}".format(kmer, self.opt.t) )
        
        for P, err in zip(self.opt.rbp_conc, errors):
            pp.semilogx(x, err, label="P={0}nM".format(P))

        pp.axvline(self.opt.known_invkd[kmer_index], color='r', label="correct value")
        pp.axvline(self.opt.trial_invkd[kmer_index], color='k', label="fitted root")
        pp.axhline(0, color='k')
        if opt.kmer_updates[kmer] > 1:
            pp.axvline(self.opt.prev_invkd[kmer_index], color='gray', label="previous value")
                
        pp.xlabel(r"$\frac{1}{K_d}$ [nM]")
        pp.ylabel(r"expected R - observed R")
        pp.legend(loc='upper left')
        pp.tight_layout()
        self.sweep_pdf.savefig()
        pp.savefig(os.path.join(self.path, "sweep_{0}_t{1}.pdf".format(kmer, self.opt.t)) )
        #pp.show()
        pp.close()

    def plot_R_value_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]
        
        kmer_i = self.opt.last_kmer_update

        if kmer_i != None:
            kmer = self.opt.kmers[kmer_i]
        else:
            kmer = "none"
        
        
        pp.figure()
        pp.title('R-value fit after step {0}'.format(self.opt.t) )
        R_a = self.opt.R_obs
        R_b = self.opt.current.R
        for i,rbp_conc in enumerate(self.opt.rbp_conc):
            corr = np.corrcoef(np.log(R_a[i]), np.log(R_b[i]))[0][1]
            patches = pp.loglog(R_a[i], R_b[i], 'o', markeredgecolor='none', markersize=5, alpha=.75, label="P={0:.2f}nM (R={1:.3f})".format(rbp_conc, corr) )

        if kmer_i != None:
            pp.loglog(R_a[:, kmer_i], R_b[:, kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        m = min(R_a.min(), R_b.min())
        M = max(R_a.max(), R_b.max())
        pp.plot([m,M],[m,M], '--k', zorder=np.inf)
        pp.xlabel(r'{0} [R-value]'.format("observed/simulated") )
        pp.ylabel(r'{0} [R-value]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
        pp.legend(loc='upper left')
        pp.tight_layout()
        self.R_pdf.savefig()
        pp.savefig(os.path.join(self.path, "predicted_vs_obs_R_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        pp.close()
        
    def plot_invkd_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        #to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]
        
        kmer_i = self.opt.last_kmer_update
        kmer = self.opt.kmers[kmer_i]
        
        
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
        
    def plot_errors(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]

        kmer_i = self.opt.last_kmer_update
        kmer = self.opt.kmers[kmer_i]
        
        pp.figure()
        pp.title('residual errors after step {0}'.format(self.opt.t) )

        abs_err = np.arcsinh(self.opt.kmer_error_new)
        #abs_err = np.where(abs_err > 0, np.arcsinh(abs_err), -np.arcsinh(-abs_err) )
        
        for i,rbp_conc in enumerate(self.opt.rbp_conc):
            patches = pp.semilogx(self.opt.known_invkd, abs_err[i,:], 'o', markeredgecolor='none', markersize=3, alpha=.75, label="P={0}nM".format(rbp_conc) )
            
        mark_x = self.opt.known_invkd[to_mark_i]
        mark_y = abs_err[:,to_mark_i].max(axis=0)
        for mer, index, x, y in zip(to_mark, to_mark_i, mark_x, mark_y):
            pp.text(x*2, y, mer)

        #pp.plot(mark_x, mark_y, 'o', markersize=10, markeredgecolor = 'black' , markerfacecolor='none')

        pp.semilogx(np.repeat(self.opt.known_invkd[kmer_i], len(self.opt.rbp_conc)), abs_err[:, kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        
        pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
        pp.ylabel(r'arcsinh(residual error) [a.u.]')
        pp.legend(loc='upper left')
        #pp.xlim(1e-1,1e2)
        #pp.ylim(1e-1,1e2)
        pp.tight_layout()
        self.err_pdf.savefig()
        pp.savefig(os.path.join(self.path, "err_vs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        pp.close()

