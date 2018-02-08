import os
import cska
import numpy as np
import matplotlib
import matplotlib.pyplot as pp
import logging
from collections import defaultdict


def density_scatter_plot(
    x,y, 
    outlier_percentile=10, 
    density_kw = dict(cmap=pp.cm.hot_r, nbins=100), 
    plot_kw = dict(style=".k"), 
    contour=False, 
    plot_outliers=True,
    label="none", data_labels=[]
    ):
    from scipy.stats import kde

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde([x,y])
    
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    nbins = density_kw['nbins']
    xi, yi = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    pp.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=density_kw['cmap'])
    pp.colorbar()

    if contour:
        pp.contour(xi, yi, zi.reshape(xi.shape))

    if plot_outliers and outlier_percentile > 0:
        data = np.vstack([x,y])
        dens_at_points = k(data)
        
        lower = np.percentile(dens_at_points, outlier_percentile)
        out = dens_at_points < lower

        out_x = x[out]
        out_y = y[out]
        pp.plot(out_x, out_y, plot_kw['style'], markersize=3, label=label)

    m = xmin + np.log10(3./4.) # always use experiment as reference
    M = xmax + np.log10(4./3.)

    if len(data_labels):
        # annotate the most enriched and most off-diagonal k-mers
        top = x.argsort()[::-1][:5]
        pp.plot(x[top], y[top], 'o', markersize=6, markerfacecolor='none', markeredgecolor='red', label="most enriched", alpha=.75 )

        for _x, _y, mer in zip(x[top], y[top], data_labels[top]):
            mer = mer.upper().replace('T','U')
            #pp.text(x, r, mer, fontdict=dict(size=6), withdash=True)
            pp.annotate(mer, xy=(_x, _y), xytext=(_x-.05*xmax, _y), arrowprops=dict(facecolor='red', arrowstyle="->, head_length = .2, head_width = .2"), horizontalalignment='right', verticalalignment='center', fontsize=6)
            
        off = np.fabs(x-y).argsort()[::-1][:10]
        pp.plot(x[off], y[off], 'o', markersize=6, markerfacecolor='none', markeredgecolor='blue', alpha=.75, label="highest error" )

        for _x, _y, mer in zip(x[off], y[off], data_labels[off]):
            #pp.text(x, r, mer, fontdict=dict(size=6), withdash=True)
            mer = mer.upper().replace('T','U')
            pp.annotate(mer, xy=(_x, _y), xytext=(_x-.05*xmax, _y), arrowprops=dict(facecolor='blue', arrowstyle="->, head_length = .2, head_width = .2"), horizontalalignment='right', verticalalignment='center', fontsize=6)

    pp.plot([m,M],[m,M], '--k', zorder=np.inf, linewidth=.1)
    pp.xlim(m,M)
    pp.ylim(m,M)
        

        
class TrackedValues(object):
    def __init__(self):
        self.d0 = None
        self.last = None
        self.updates = []
        self.times = []
        self.N = 0
        self.logger = logging.getLogger("reports.TrackedValues")
        
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
        if self.d0 is not None:
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
        

class Sensor(object):
    def __init__(self, rep, name, plot_interval=10, data_interval=1, get_func=lambda this : 0, labels=[], multipage=False, snapshot=True, xlabel="optimization step", ylabel="data", fname="{self.name}.pdf", mp_fname="mp_{self.name}.pdf", plot_func=pp.plot, mode='temporal', description=""):
        self.name = name.replace(' ','_')
        self.description = description
        self.get_func = get_func
        self.plot_func = plot_func
        self.mode = mode
        self.data_interval = data_interval
        self.plot_interval = plot_interval
        self.rep = rep
        self.opt = rep.opt
        
        self.logger = logging.getLogger('report.Sensor.{name}'.format(name=name))
        self.t_data = 0
        self.t_plot = 0
        self.data = TrackedValues()
        
        self.multipage = multipage
        self.snapshot = snapshot
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.labels = labels
        self.snap_path = os.path.join(self.rep.path, fname)
        self.mp_path = os.path.join(self.rep.path, mp_fname.format(**locals()))

        from matplotlib.backends.backend_pdf import PdfPages
        if self.multipage:
            self.pdf = PdfPages(self.mp_path)

    def tick(self, t):
        if self.mode == 'temporal':
            if t - self.t_data >= self.data_interval:
                self.record_data(t)
        
        if t - self.t_plot >= self.plot_interval:
            self.update_plot(t)
            
    def record_data(self, t):
        data = np.array(self.get_func(self))
        self.logger.debug('recording data of shape {0}'.format(data.shape))
        self.data.store(t, data)
        self.t_data = t
        
    def start_plot(self, t):
        pp.figure()
        if not self.description:
            pp.title(self.name)
        else:
            pp.title(self.description)
        
    def end_plot(self, t):
        pp.xlabel(self.xlabel)
        pp.ylabel(self.ylabel)
        pp.legend(loc='lower right')
        if self.snapshot:
            path = self.snap_path.format(**locals())
            self.logger.debug("saving snapshot in '{0}'".format(path) )
            pp.savefig(path)
        
        if self.multipage:
            self.logger.debug("adding page to '{0}'".format(self.mp_path) )
            self.pdf.savefig()

        pp.close()
        self.t_plot = t
        
    def do_plot(self, t):
        if self.mode == 'temporal':
            from itertools import izip_longest
            times, cols = self.data.read()
            for label, row in izip_longest(self.labels, cols.T, fillvalue="none" ):
                self.plot_func(times, row, label=label)

        elif self.mode == 'scatter':
            pp.title("{0}mer R-value scatter plot".format(self.opt.k) )
            x, y = self.get_func(self)
            corr = np.corrcoef(x,y)[0][1]
            density_scatter_plot(x, y, label="{0} R={1:.3f}".format(self.labels[0], corr), data_labels=self.opt.mdl.param_name)
        
    def update_plot(self, t):
        self.start_plot(t)
        self.do_plot(t)
        self.end_plot(t)

    def close(self):
        if self.multipage:
            self.pdf.close()
    
        
class OptReporting(object):
    def __init__(self, opt, path='./', track=[], report_interval=200, comp=None):
        self.opt = opt
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.logger = logging.getLogger('report.OptReporting')
        self.conc_labels = ['{0:.2f} nM'.format(conc) for conc in self.opt.rbp_conc]
        self.sensors = []
        # populate with sensors
        for name in track:
            adder = getattr(self, "add_sensor_{0}".format(name))
            sensors = adder()
            self.sensors.extend(sensors)
            
    def close(self):
        self.logger.info('broadcasting close() to {0} sensors'.format(len(self.sensors)) )
        # broadcast close to all attached sensors
        for sensor in self.sensors:
            sensor.close()

    def tick(self, t):
        self.logger.debug('broadcasting tick() to {0} sensors'.format(len(self.sensors)) )
        # broadcast to all attached sensors
        for sensor in self.sensors:
            sensor.tick(t)

    def set_opt(self, opt):
        self.logger.debug('broadcasting set_opt() to {0} sensors'.format(len(self.sensors)) )
        # broadcast to all attached sensors
        for sensor in self.sensors:
            sensor.opt = opt
        
    def add_sensor_correlation(self):
        sensor = Sensor(
            self, "kmer_correlation", 
            get_func = lambda this : this.opt.correlation(), 
            ylabel=r"log(R-value) correlation coefficient", 
            labels=self.conc_labels,
        )
        return [sensor,]
    
    def add_sensor_betas(self):
        sensor = Sensor(
            self, "beta",
            get_func = lambda this : this.opt.current.params[this.opt.nA:],
            plot_func = pp.semilogy,
            ylabel=r"estimated sample background ($\beta$)",
            labels=self.conc_labels
        )
        return [sensor,]
        
    def add_sensor_errors(self):
        sensor = Sensor(
            self, "errors",
            get_func = lambda this : [this.opt.global_error(this.opt.current.R),],
            plot_func = pp.semilogy,
            ylabel=r"global error of the model"
        )
        return [sensor,]
        
    def add_sensor_R_values(self):
        sensors = []
        for i, label in enumerate(self.conc_labels):
            # add one sensor per experiment
            sensor = Sensor(
                self, "R_values_{0}".format(label),
                get_func = lambda this : (np.log10(this.opt.R_obs[i]), np.log10(this.opt.current.R[i])),
                ylabel=r"predicted kmer enrichment [R-value]",
                xlabel=r"observed kmer enrichment [R-value]",
                mode='scatter',
                labels=[label,],
                fname="{self.name}_{t}.pdf",
                plot_interval=100,
                multipage=True# 100
            )
            sensors.append(sensor)
        return sensors
    
        #self.comp = comp
        #if comp and not track:
            #track = sorted(comp.uniq_kmers)

        #self.tracked_kmers = [t for t in track if len(t)== self.opt.k]
        #import cska.ska_kmers
        #self.tracked_indices = [cska.ska_kmers.seq_to_index(t) for t in self.tracked_kmers]
        #self.tracked = set(self.tracked_indices)
        #self.tracked_history = defaultdict(list)
        #self.tracked_updated = defaultdict(list)
        #for param_i in self.tracked_indices:
            #self.tracked_history[param_i].append( (self.opt.t, self.opt.current.params[param_i]) )

        #self.report_interval = report_interval
        #self.last_report = 0
        #self.logger = logging.getLogger('OptReporting')
        #self.betas = TrackedValues()
        ##self.affinities = TrackedValues()
    
    #def tick(self, t):
        #self.betas.store(t, self.opt.current.params[self.opt.nA:])
        ##self.affinities.store(t, self.opt.current.params[:self.opt.nA])
        
        #if t > self.last_report + self.report_interval:
            #self.plot_errors()
            #self.plot_correlations()
            #self.plot_betas()
            ##self.plot_tracked_kmer_histories()
            ##self.plot_affinity_history()
            
            #self.plot_R_value_agreement()
            #self.plot_known_comparison()
            
            #for param_i in self.tracked_indices:
                #self.plot_sweep(param_i)
            #self.last_report = t
        
        #for param_i in self.tracked_indices:
            #self.tracked_history[param_i].append( (t, self.opt.current.params[param_i]) )
            #if param_i == self.opt.sched.last_param_update:
                #self.tracked_updated[param_i].append(t)

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
        
    #def plot_invkd_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
        ##to_mark_i = [cska.ska_kmers.seq_to_index(x) for x in to_mark]
        
        #kmer_i = self.opt.sched.last_param_update
        #kmer = self.opt.mdl.param_name[kmer_i]
        
        
        #pp.figure()
        #pp.title('affinity agreement after step {0}'.format(self.opt.t) )
        #A_a = self.opt.known_invkd
        #A_b = self.opt.trial_invkd
        #x = A_a
        #y = A_b
        
        #max_error_conc = np.fabs(self.opt.kmer_error_new).argmax(axis=0)
        #for i,rbp_conc in enumerate(self.opt.rbp_conc):
            #ind = max_error_conc == i
            ##print ind.shape, ind, x[ind]
            #patches = pp.loglog(x[ind], y[ind], 'o', markeredgecolor='none', markersize=5, label="max error at P={0:.2f}nM".format(rbp_conc) )

        #corr = np.corrcoef(np.log(A_a), np.log(A_b))[0][1]
        #pp.loglog(x[kmer_i], self.opt.prev_invkd[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='gray', label="previous values" )
        #pp.loglog(A_a[kmer_i], A_b[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

        #m = min(A_a.min(), A_b.min())
        #M = max(A_a.max(), A_b.max())
        #pp.plot([m,M],[m,M], '--k', zorder=np.inf)
        #pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
        #pp.ylabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
        #pp.legend(loc='lower right')
        ##pp.xlim(1e-1,1e2)
        ##pp.ylim(1e-1,1e2)
        #pp.tight_layout()
        #self.invkd_pdf.savefig()
        #pp.savefig(os.path.join(self.path, "predicted_vs_obs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
        #pp.close()
     
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


if __name__ == "__main__":
    pass
