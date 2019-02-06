# -*- coding: utf-8 -*-
import os
import numpy as np
import logging
import time
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
# matplotlib.rc('xtick.major', size = .5)
# matplotlib.rc('ytick.major', size = .5)

sns_style = { 
    'axes.linewidth': .5, 
    'axes.grid' : False, 
    # 'ticks.xtick.major.linewidth' : .5,
    # 'xtick.major.linewidth' : .5, # Whut is the right one? Seaborn docs, where are u?
    'legend.frameon' : False,
    'legend.fancybox' : False,
}

import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import cska

def pval_str(p):
    if p > 0:
        return "P < {0:.3e}".format(p)
    else:
        return u"P ≈ 0"

def roundmax(x, m):
    i = int(x)
    remain = x - i
    if remain:
        return np.round(x, m)
    else:
        return i

def repel_labels(x, y, labels, **kwargs):
    from adjustText import adjust_text
    texts = []
    for x_, y_, label in zip(x, y, labels):
        texts.append(pp.text(x_, y_, label, horizontalalignment='center', color='k'))
        
    adjust_text(
        texts, 
        #add_objects=lines, 
        #autoalign='y', 
        #expand_objects=(0.1, 1),
        #only_move={'points':'', 'text':'y', 'objects':'y'}, force_text=0.75, force_objects=0.1,
        arrowprops=dict(arrowstyle="-", color='k', lw=0.5)
    )


def sparse_y(ax, nth=2):
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % nth != 0:
            label.set_visible(False)

def repel_labels_nx(x, y, labels, k=0.15, ax=None):
    import networkx as nx
    if ax == None:
        ax = plt.gca()
    G = nx.DiGraph()
    print x.shape, y.shape, len(labels), ax
    
    data_nodes = []
    init_pos = {}
    for xi, yi, label in zip(x, y, labels):
        data_str = 'data_{0}'.format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k, iterations=200)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
    scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val*scale) + shift

    for label, data_str in G.edges():
        ax.annotate(label,
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[label], textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                                    shrinkA=0, shrinkB=0,
                                    #connectionstyle="arc3", 
                                    color='k'), )
    # expand limits
    all_pos = np.vstack(pos.values())
    x_span, y_span = np.ptp(all_pos, axis=0)
    mins = np.min(all_pos-x_span*0.15, 0)
    maxs = np.max(all_pos+y_span*0.15, 0)
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])


def density_scatter_plot(
    x,y, 
    outlier_percentile=10, 
    density_kw = dict(cmap=pp.cm.YlGn, nbins=100), 
    plot_kw = dict(style='.', color='#4495c3'), 
    contour=False, 
    plot_outliers=True,
    label="none", data_labels=[],
    dens_thresh=1000,
    x_ref=True,
    tick_exp=0,
    ):
    from scipy.stats import kde
    import seaborn as sns
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    t0 = time.time()
    N = len(x)
    
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    if N > dens_thresh and x_ref:
        # use experiment as reference
        m = xmin  
        M = xmax
    else:
        # show full range
        m = min(xmin, ymin)
        M = max(xmax, ymax)
    
    # add margin in log-space
    m += np.log10(3./4.)
    M += np.log(4./3.)

    nbins = density_kw['nbins']
    t1 = time.time()
    #Z = zi.reshape((len(yi), len(xi)))
    #print Z.shape
    #pp.imshow(Z, interpolation='none', cmap=density_kw['cmap'], origin='lower', extent=[xmin,xmax,ymin,ymax])
    import matplotlib
    with sns.axes_style("ticks", sns_style):
        if N > dens_thresh:
            k = kde.gaussian_kde([x,y])
            xi, yi = np.mgrid[m:M:nbins*1j, m:M:nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            zi[zi < 1e-3] = np.nan

            z_min = np.nanmin(zi)
            z_max = np.nanmax(zi)
            # print "zmin/max", z_min, z_max
            # pca().set_facecolor('w')
            pm = pp.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=density_kw['cmap'], edgecolors='None', linewidth=0, rasterized=True, vmin=0, vmax=z_max)
            pm.set_rasterized(True)
            cb = pp.colorbar(pm, shrink=.3) #orientation='horizontal', fraction=.05)
            cb.set_label('density')
            zt = np.array([z_min, (z_max + z_min)/2., z_max])
            ztr = np.round(zt, 1)
            cb.outline.set_linewidth(.5)
            # cb.ax.yaxis.set_ticks_position('right')
            cb.set_ticks(zt)
            cb.ax.set_yticklabels([str(z) for z in ztr])
            cb.ax.tick_params(axis='y', direction='out', length=3, width=.5, )
            # cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            # cb.locator = matplotlib.ticker.MaxNLocator(nbins=4)
            # cb.update_ticks()

            if contour:
                pp.contour(xi, yi, zi.reshape(xi.shape))
            
            pp.grid(False)

        t2 = time.time()

        if plot_outliers and outlier_percentile > 0:
            data = np.vstack([x,y])
            if N <= dens_thresh:
                print "plotting all data points"
                out = np.arange(N)
            else:
                dens_at_points = k(data)
                lower = np.percentile(dens_at_points, outlier_percentile)
                out = dens_at_points < lower

            out_x = x[out]
            out_y = y[out]
            pp.plot(out_x, out_y, plot_kw['style'], color=plot_kw['color'], markersize=3, label=label, rasterized=True)


        t3 = time.time()
        
        if len(data_labels):
            if N < dens_thresh*.1:
                print "just add the damn labels"
                print x,y, data_labels
                repel_labels_nx(x, y, data_labels)
            else:
                # annotate the most enriched and most off-diagonal k-mers
                top = x.argsort()[::-1][:5]
                print "top", top
                pp.plot(x[top], y[top], 'o', markersize=6, markerfacecolor='none', markeredgecolor='red', label="most enriched", alpha=.75 )

                repel_labels_nx(x[top], y[top], data_labels[top])
                # for _x, _y, mer in zip(x[top], y[top], data_labels[top]):
                #     mer = mer.upper().replace('T','U')
                #     #pp.text(x, r, mer, fontdict=dict(size=6), withdash=True)
                #     pp.annotate(mer, xy=(_x, _y), xytext=(_x-.05*xmax, _y), arrowprops=dict(facecolor='red', arrowstyle="->, head_length = .2, head_width = .2"), horizontalalignment='right', verticalalignment='center', fontsize=6)
                    
                off = np.fabs(x-y).argsort()[::-1][:10]
                pp.plot(x[off], y[off], 'o', markersize=6, markerfacecolor='none', markeredgecolor='blue', alpha=.75, label="highest error" )

                repel_labels_nx(x[off], y[off], data_labels[off])
                # for _x, _y, mer in zip(x[off], y[off], data_labels[off]):
                #     #pp.text(x, r, mer, fontdict=dict(size=6), withdash=True)
                #     mer = mer.upper().replace('T','U')
                #     pp.annotate(mer, xy=(_x, _y), xytext=(_x-.05*xmax, _y), arrowprops=dict(facecolor='blue', arrowstyle="->, head_length = .2, head_width = .2"), horizontalalignment='right', verticalalignment='center', fontsize=6)

        # draw guides through zero and the diagonal
        mM = np.array([m,M])
        mM = (mM - mM.mean()) *.95 + mM.mean()
        pp.plot(mM, mM, '-k', linewidth=.5, alpha=.3)
        # pp.plot([0,0],mM, '-k', linewidth=.5, alpha=.3)
        # pp.plot(mM,[0,0], '-k', linewidth=.5, alpha=.3)
        
        pp.xlim(m,M)
        pp.ylim(m,M)
        if tick_exp:
            xlocs, labels = pp.xticks()
            xlocs = xlocs[1:-1]
            pp.xticks(xlocs, [roundmax(tick_exp**l, 1) for l in xlocs])

            ylocs, labels = pp.yticks()
            ylocs = ylocs[1:-1]
            pp.yticks(ylocs, [roundmax(tick_exp**l, 1) for l in ylocs])

        t4 = time.time()
        logger = logging.getLogger('timing.density_plot')
        t_kde = 1000. * (t1-t0)
        t_mesh = 1000. * (t2-t1)
        t_out = 1000. * (t3-t2)
        t_label = 1000. * (t4-t3)
        logger.debug('KDE={t_kde:.2f}ms pcolormesh={t_mesh:.2f}ms outliers={t_out:.2f}ms labels={t_label:.2f}ms'.format(**locals()) )
        
        sns.despine(trim=True)

        
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
        

# def p_bound_plot(state, fname = "pbound.pdf", bins=1000):
#     pp.figure()
    
#     lZ = np.log(state.Z1)
#     counts, bins = np.histogram(lZ, bins=bins)
#     bins = np.exp(bins)
#     # midpoint integration
#     aff = (bins[1:] + bins[:-1])/2.
            
#     N = counts.sum()
#     expected = []
#     state.betas = [0.0008, 0.003,0.02,0.1,0.2]
#     for conc, beta in zip(state.rbp_free, state.betas):
#         Z = conc * aff
#         pb = Z/(Z + 1.)
#         pp.semilogx(aff, pb, label="{0:.2f} nM free RBP".format(conc) )
        
#         x = counts * (pb + beta)
#         print x.min(), x.max(), np.median(x)
#         expected.append( N * x/x.sum() )

#         print conc, beta, "pb", pb.min(), pb.max(), np.median(pb)


#     pp.axhline(1, color="k")
#     pp.axhline(.5, color="gray", linestyle='dashed')
#     pp.legend(loc='upper center', faceolor='white', frameon=False)
#     pp.xlabel("total read affinity [1/nM]")
#     pp.ylabel(r"$\psi$")
#     pp.savefig(fname)
#     pp.close()

#     pp.figure()
    
#     pp.loglog(aff, counts, color="black", label="random RNA pool")
#     for conc, x in zip(state.rbp_free, expected):
#         pp.loglog(aff, np.clip(x, a_min=1, a_max=None) , label="predicted @ {0:.2f}nM free RBP".format(conc))

#     pp.legend(loc = 'upper left')
#     pp.xlabel("total read affinity [1/nM]")
#     pp.ylabel("count")
#     pp.savefig("aff_dist.pdf")
#     pp.close()
    
#     # cska -a --run-path=blup --n-max=1000000 --metrics="" --model-report-ignore-trigger="init" --seed-analysis --model

# class Sensor(object):
#     def __init__(self, rep, name, plot_interval=100, data_interval=1, get_func=lambda this : 0, setup_func = None, labels=[], multipage=False, snapshot=True, xlabel="optimization step", ylabel="data", fname="{self.name}.pdf", mp_fname="mp_{self.name}.pdf", plot_func=pp.plot, mode='temporal', description="", scatter_func = density_scatter_plot):
#         self.name = name.replace(' ','_')
#         self.description = description
#         self.get_func = get_func
#         self.plot_func = plot_func
#         self.setup_func = setup_func
#         self.mode = mode
#         self.scatter_func = scatter_func
#         self.data_interval = data_interval
#         self.plot_interval = plot_interval
#         self.rep = rep
#         self.opt = rep.opt
        
#         self.logger = logging.getLogger('report.Sensor.{name}'.format(name=name))
#         self.t_data = -1
#         self.t_plot = -1
#         self.data = TrackedValues()
        
#         self.multipage = multipage
#         self.snapshot = snapshot
#         self.xlabel = xlabel
#         self.ylabel = ylabel
#         self.labels = labels
#         self.snap_path = os.path.join(self.rep.path, fname)
#         self.mp_path = os.path.join(self.rep.path, mp_fname.format(**locals()))

#         from matplotlib.backends.backend_pdf import PdfPages
#         if self.multipage:
#             self.pdf = PdfPages(self.mp_path)

#     def tick(self, t):
#         if self.mode == 'temporal':
#             if t - self.t_data >= self.data_interval:
#                 self.record_data(t)
        
#         if t - self.t_plot >= self.plot_interval:
#             self.update_plot(t)
            
#     def record_data(self, t):
#         data = np.array(self.get_func(self))
#         self.logger.debug('recording data of shape {0}'.format(data.shape))
#         self.data.store(t, data)
#         self.t_data = t
        
#     def start_plot(self, t, occasion=""):
#         pp.figure()
#         if not self.description:
#             pp.title(" ".join([self.name, occasion]))
#         else:
#             pp.title(self.description)

#     def end_plot(self, t, occasion=""):
#         pp.xlabel(self.xlabel)
#         pp.ylabel(self.ylabel)
#         pp.legend(loc='lower right')
#         if self.snapshot:
#             path = cska.ensure_path(self.snap_path.format(**locals()))
#             self.logger.debug("saving snapshot in '{0}'".format(path) )
#             pp.savefig(path)
        
#         if self.multipage:
#             self.logger.debug("adding page to '{0}'".format(self.mp_path) )
#             self.pdf.savefig()

#         pp.close()
#         self.t_plot = t
        
#     def do_plot(self, t, occasion=""):
#         if self.mode == 'temporal':
#             from itertools import izip_longest
#             #from adjustText import adjust_text
#             times, cols = self.data.read()
#             #print times, cols, label
#             if not len(times):
#                 return
            
#             lines = []
#             for label, row in izip_longest(self.labels, cols.T, fillvalue="none" ):
#                 lines.extend(self.plot_func(times, row, label=label))
            
#             #print self.name
#             t_trig = np.array(sorted(self.rep.triggers.keys()))
#             y_min = cols.min(axis=1)
#             #y_max = cols.max(axis=0)
            
#             yt = np.interp(t_trig, times, y_min)
#             labels = [self.rep.triggers[x] for x in t_trig]
#             repel_labels(t_trig, yt, labels)

#         elif self.mode == 'scatter':
#             if self.setup_func:
#                 x, y, title, label, data_labels = self.setup_func(self)
#             else:
#                 title = "{0}mer R-value scatter plot".format(self.opt.k)
#                 x, y = self.get_func(self)
#                 corr, pval = pearsonr(x,y)
#                 label = u"{0} R={1:.3f} ({2})".format(self.labels[0], corr, pval_str(pval))
#                 data_labels = self.opt.mdl.parameters.param_name

#             pp.title(title)
#             self.scatter_func(x, y, label=label, data_labels=data_labels)
#             self.logger.info("{self.name} scatter plot".format(**locals()) )
        
#     def update_plot(self, t, occasion="snapshot"):
#         self.start_plot(t, occasion=occasion)
#         self.do_plot(t, occasion=occasion)
#         self.end_plot(t, occasion=occasion)

#     def close(self):
#         if self.multipage:
#             self.pdf.close()
    
        
# class OptReporting(object):
#     def __init__(self, opt, path='./', track=[], report_interval=200, comp=None, triggers=[], ref=None):
#         self.opt = opt
#         self.path = path
#         if not os.path.exists(path):
#             os.makedirs(path)
        
#         self.logger = logging.getLogger('report.OptReporting')
#         self.conc_labels = ['{0:.2f} nM'.format(conc) for conc in self.opt.rbp_conc]
#         self.triggers = {}
#         self.trigger_filter = set(triggers)
#         self.ref = ref
#         self.sensors = []
        
#         if ref and len(ref.seqs):
#             self.sensors.extend(self.add_sensor_ref())

#         # populate with sensors
#         for name in track:
#             adder = getattr(self, "add_sensor_{0}".format(name))
#             sensors = adder()
#             self.sensors.extend(sensors)

#     def close(self):
#         self.logger.info('broadcasting close() to {0} sensors'.format(len(self.sensors)) )
#         # broadcast close to all attached sensors
#         for sensor in self.sensors:
#             sensor.close()

#     def tick(self, t):
#         self.logger.debug('broadcasting tick() to {0} sensors'.format(len(self.sensors)) )
#         t0 = time.time()
#         # broadcast to all attached sensors
#         for sensor in self.sensors:
#             sensor.tick(t)
#         t1 = time.time()
#         self.logger.debug('tick() completed in {0:.2f}ms'.format(1000. * (t1-t0)) )

#     def trigger_plots(self, t, occasion="trigger", mode="scatter"):
#         self.logger.debug("received trigger '{occasion}' at time {t} for {mode}-sensors".format(**locals()) )
#         self.triggers[t] = occasion
#         t0 = time.time()
#         if occasion in self.trigger_filter:
#             self.logger.info('trigger {0} is filtered! skipping plot updates'.format(occasion) )
#         else:
#             for s in self.sensors:
#                 #s.update_plot(t, occasion=occasion)
#                 if s.mode == mode:
#                     s.update_plot(t, occasion=occasion)

#         t1 = time.time()
#         self.logger.debug('trigger_plots("{0}") completed in {1:.2f} ms'.format(occasion, 1000. * (t1-t0)) )

#     def set_opt(self, opt):
#         self.logger.debug('broadcasting set_opt() to {0} sensors'.format(len(self.sensors)) )
#         # broadcast to all attached sensors
#         for sensor in self.sensors:
#             sensor.opt = opt
        
#     def add_sensor_correlation(self):
#         sensor = Sensor(
#             self, "kmer_correlation", 
#             get_func = lambda this : this.opt.correlation(), 
#             ylabel=r"log(R-value) correlation coefficient", 
#             labels=self.conc_labels,
#         )
#         return [sensor,]
    
#     def add_sensor_betas(self):
#         sensor = Sensor(
#             self, "beta",
#             get_func = lambda this : this.opt.current.params[this.opt.nA:],
#             plot_func = pp.semilogy,
#             ylabel=r"estimated sample background ($\beta$)",
#             labels=self.conc_labels
#         )
#         return [sensor,]
        
#     def add_sensor_errors(self):
#         sensor = Sensor(
#             self, "errors",
#             get_func = lambda this : this.opt.global_errors(this.opt.current.R),
#             plot_func = pp.semilogy,
#             ylabel=r"kmer R-value mean squared errors",
#             labels=self.conc_labels + ['all']
#         )
#         return [sensor,]
        
#     def add_sensor_R_values(self):
#         sensors = []
#         for i, label in enumerate(self.conc_labels):
#             # add one sensor per experiment
#             sensor = Sensor(
#                 self, "R_values_{0}".format(label),
#                 get_func = lambda this, i=i : (np.log2(this.opt.R_obs[i]), np.log2(this.opt.current.R[i])),
#                 ylabel=r"predicted kmer enrichment $\log_2(R)$",
#                 xlabel=r"observed kmer enrichment $\log_2(R)$",
#                 mode='scatter',
#                 scatter_func = density_scatter_plot,
#                 labels=[label,],
#                 fname="{self.name}/{self.opt.input_reads.rbp_name}_{occasion}_{self.opt.k}mers_{self.name}_{t}.pdf",
#                 plot_interval=1000,
#                 multipage=True# 100
#             )
#             sensors.append(sensor)
#         return sensors

#     def add_sensor_ref(self):
#         def setup_func(this):
#             x = np.log10(this.rep.ref.observed_affinities)
#             y = np.log10(this.rep.ref.predict_affinities(this.opt.mdl))

#             R, ppval = pearsonr(x,y)
#             rho, pval = spearmanr(x,y)

#             title = "comparison to literature values"
#             label = r"log-affinity R={R:.3f} (P < {ppval:.3e}) $\rho$={rho:.3f} (P < {pval:.3e}) ".format(**locals())
#             data_labels = this.rep.ref.seqs
#             print x, y, title, label, data_labels
#             return x, y, title, label, data_labels

#         sensor = Sensor(
#             self, "reference",
#             setup_func = setup_func,
#             ylabel=r"modeled affinity $-\log_{10}(K_d)$",
#             xlabel=r"literature affinity $-\log_{10}(K_d)$",
#             mode='scatter',
#             scatter_func = density_scatter_plot,
#             labels=["reference",],
#             fname="{self.name}/{self.opt.input_reads.rbp_name}_{occasion}_{self.opt.k}mers_{self.name}_{t}.pdf",
#             plot_interval=1000,
#             multipage=True# 100
#         )
#         return [sensor,]
        
#         #self.comp = comp
#         #if comp and not track:
#             #track = sorted(comp.uniq_kmers)

#         #self.tracked_kmers = [t for t in track if len(t)== self.opt.k]
#         #import cska.cyska
#         #self.tracked_indices = [cyska.seq_to_index(t) for t in self.tracked_kmers]
#         #self.tracked = set(self.tracked_indices)
#         #self.tracked_history = defaultdict(list)
#         #self.tracked_updated = defaultdict(list)
#         #for param_i in self.tracked_indices:
#             #self.tracked_history[param_i].append( (self.opt.t, self.opt.current.params[param_i]) )

#         #self.report_interval = report_interval
#         #self.last_report = 0
#         #self.logger = logging.getLogger('OptReporting')
#         #self.betas = TrackedValues()
#         ##self.affinities = TrackedValues()
    
#     #def tick(self, t):
#         #self.betas.store(t, self.opt.current.params[self.opt.nA:])
#         ##self.affinities.store(t, self.opt.current.params[:self.opt.nA])
        
#         #if t > self.last_report + self.report_interval:
#             #self.plot_errors()
#             #self.plot_correlations()
#             #self.plot_betas()
#             ##self.plot_tracked_kmer_histories()
#             ##self.plot_affinity_history()
            
#             #self.plot_R_value_agreement()
#             #self.plot_known_comparison()
            
#             #for param_i in self.tracked_indices:
#                 #self.plot_sweep(param_i)
#             #self.last_report = t
        
#         #for param_i in self.tracked_indices:
#             #self.tracked_history[param_i].append( (t, self.opt.current.params[param_i]) )
#             #if param_i == self.opt.sched.last_param_update:
#                 #self.tracked_updated[param_i].append(t)

#     def plot_affinity_history(self, n=10):
#         pp.figure()
#         t, aff_matrix = self.affinities.read()
#         last = aff_matrix[-1]
#         top_kmer_ind = last.argsort()[::-1][:n]
#         top_kmers = self.opt.kmers[top_kmer_ind]

#         for kmer, aff in zip(top_kmers, aff_matrix.T[top_kmer_ind] ):
#             pp.semilogy(t, aff, label=kmer)
        
#         pp.xlabel('time step')
#         pp.ylabel('affinity [1/nM]')
#         pp.legend(loc='lower right')
#         pp.savefig(os.path.join(self.path,'affinity_history.pdf'))
#         pp.close()
        
        
#     def plot_tracked_kmer_histories(self):
#         self.logger.info("rendering tracked kmer affinity history plots")
#         pp.figure()
#         for i, param_i in enumerate(self.tracked_indices):
#             kmer = self.tracked_kmers[i]
#             t, param = np.array(self.tracked_history[param_i]).T
            
#             pp.semilogy(t, param, label=kmer)
#             known = self.opt.known_params[param_i]
#             if np.isfinite(known):
#                 pp.axhline(known, label='{0} reference'.format(kmer))

#             #for t_update in self.tracked_updated[param_i]:
#                 #pp.axvline(t_update)
                
#         pp.xlabel('optimization step')
#         pp.ylabel('affinity [1/nM]')
#         pp.legend(loc='lower right')
#         pp.savefig(os.path.join(self.path, "tracked_kmers.pdf"))
#         pp.close()

#     def plot_known_comparison(self):
#         if not self.comp:
#             return
        
#         x = self.comp.observed_affinities
#         y = self.comp.expected_affinities
#         R = np.corrcoef(np.log(x), np.log(y))[0][1]
#         pp.figure()
#         pp.loglog(x, y, '.r', label="R={:.3f}".format(R))
#         pp.xlabel("observed/known affinity [1/nM]")
#         pp.ylabel("expected from fit [1/nM]")
#         pp.legend(loc='lower right')
#         pp.savefig(os.path.join(self.path, "known_affinity_comparison.pdf"))
#         pp.close()
        
        
#     def plot_gradient_descent(self, n_top=100):
#         I = self.opt.R_obs.argsort()[::-1][:n_top]
        
#         x = np.arange(len(I))
#         pp.figure()
#         t = self.opt.t
#         pp.title("gradient-descent at step {0}".format(t))
        
#         plot = pp.semilogy
#         plot(x, self.opt.known_invkd[I], 'k', label='known affinities')
#         plot(x, self.opt.prev_invkd[I], '.b', label='prev. delta')
#         plot(x, self.opt.trial_invkd[I], '.r', label='last delta')
#         pp.xlim(-1, len(x))
#         self.descent_pdf.savefig()
#         pp.savefig(os.path.join(self.path, "descent_t{0}.pdf".format(t)) )

        
#     def plot_jacobi(self):
        
#         for conc, jac in zip(self.opt.rbp_conc, self.opt.jacobi_new):
#             pp.figure()
#             pp.title("Jacobi matrix @{1}nM at t={0}".format(self.opt.t, conc))
#             pp.imshow(np.arcsinh(jac), cmap='hot')
#             pp.colorbar(label='arcsinh(jacobi matrix)')
#             pp.savefig('jacobi_{1}nM_t{0}.pdf'.format(self.opt.t, conc))

#         pp.show()
#         pp.close()
        
#     def plot_R_value_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
#         #to_mark_i = [cyska.seq_to_index(x) for x in to_mark]
        
#         self.logger.info('rendering R-value agreement plot')
#         kmer_i = self.opt.sched.last_param_update
#         if kmer_i == None:
#             kmer = "none"
#         else:
#             kmer = self.opt.mdl.parameters.param_name[kmer_i]
        
#         pp.figure()
#         pp.title('R-value fit after step {0}'.format(self.opt.t) )
#         R_a = self.opt.R_obs
#         R_b = self.opt.current.R
#         for i,rbp_conc in reversed(list(enumerate(self.opt.rbp_conc))):
#             corr = np.corrcoef(np.log(R_a[i]), np.log(R_b[i]))[0][1]
#             patches = pp.loglog(R_a[i], R_b[i], 'o', markeredgecolor='none', markersize=3, alpha=.75, label="P={0:.2f}nM (R={1:.3f})".format(rbp_conc, corr) )

#         if kmer_i < self.opt.nA and kmer_i != None:
#             pp.loglog(R_a[:, kmer_i], R_b[:, kmer_i], 'o', markersize=6, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

#         #m = min(R_a.min(), R_b.min())
#         #M = max(R_a.max(), R_b.max())
#         m = R_a.min() * .75 # always use experiment as reference
#         M = R_a.max() * 1.25

#         pp.plot([m,M],[m,M], '--k', zorder=np.inf)
#         pp.xlim(m,M)
#         pp.ylim(m,M)
#         pp.xlabel(r'{0} [R-value]'.format("observed/simulated") )
#         pp.ylabel(r'{0} [R-value]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
#         pp.legend(loc='upper left')
#         pp.tight_layout()
#         self.R_pdf.savefig()
#         pp.savefig(os.path.join(self.path, "predicted_vs_obs_R_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
#         pp.close()
        
#     #def plot_invkd_agreement(self, to_mark = ['UUUUU','UUUUG', 'UUUUC', 'UUUGU', 'AUUUU', 'CUUUU', 'GUUUU', 'AAUUU', 'UCUUU']):
#         ##to_mark_i = [cyska.seq_to_index(x) for x in to_mark]
        
#         #kmer_i = self.opt.sched.last_param_update
#         #kmer = self.opt.mdl.param_name[kmer_i]
        
        
#         #pp.figure()
#         #pp.title('affinity agreement after step {0}'.format(self.opt.t) )
#         #A_a = self.opt.known_invkd
#         #A_b = self.opt.trial_invkd
#         #x = A_a
#         #y = A_b
        
#         #max_error_conc = np.fabs(self.opt.kmer_error_new).argmax(axis=0)
#         #for i,rbp_conc in enumerate(self.opt.rbp_conc):
#             #ind = max_error_conc == i
#             ##print ind.shape, ind, x[ind]
#             #patches = pp.loglog(x[ind], y[ind], 'o', markeredgecolor='none', markersize=5, label="max error at P={0:.2f}nM".format(rbp_conc) )

#         #corr = np.corrcoef(np.log(A_a), np.log(A_b))[0][1]
#         #pp.loglog(x[kmer_i], self.opt.prev_invkd[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='gray', label="previous values" )
#         #pp.loglog(A_a[kmer_i], A_b[kmer_i], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', label="updated {0}".format(kmer) )

#         #m = min(A_a.min(), A_b.min())
#         #M = max(A_a.max(), A_b.max())
#         #pp.plot([m,M],[m,M], '--k', zorder=np.inf)
#         #pp.xlabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("observed/simulated") )
#         #pp.ylabel(r'{0} $\frac{{1}}{{K_d}}$ [$\frac{{1}}{{nM}}$]'.format("predicted after {0} steps of optimization".format(self.opt.t)) )
#         #pp.legend(loc='lower right')
#         ##pp.xlim(1e-1,1e2)
#         ##pp.ylim(1e-1,1e2)
#         #pp.tight_layout()
#         #self.invkd_pdf.savefig()
#         #pp.savefig(os.path.join(self.path, "predicted_vs_obs_invkd_updated_{0}_t{1}.pdf".format(kmer, self.opt.t) ))
#         #pp.close()
     
# class EnrichmentBarPlot(object):
#     def __init__(self, rbns_comparison):
#         self.rbns_comparison = rbns_comparison
        
#     def make_plot(self, k, fname=None, dest="./", fmt='svg', Z_cut=2, figsize=(7,5) ):
#         import matplotlib
#         #matplotlib.use(fmt)
#         import matplotlib.pyplot as pp
#         import numpy as np
#         if not fname:
#             fname = "r_values.{self.rbns_comparison.pd_reads.name}.{k}mers".format(self=self, k=k)
        
#         R, R_err = self.rbns_comparison.R_values(k)
#         kmers = np.array(list(cyska.yield_kmers(k)))

#         I = R.argsort()
#         R = R[I]
#         R_err = R_err[I]
#         kmers = kmers[I]
        
#         Z = (R - np.mean(R))/ np.std(R)
#         enriched_i = (Z > Z_cut).argmax()
#         depleted_i = (Z < -Z_cut).argmin()
        
        
#         pp.figure(figsize=figsize)

#         n = len(R)
#         n_enriched = len(R) - enriched_i
#         n_ns = enriched_i
#         #x_ns = np.linspace(0, 1., num=n_ns)
#         x_ns = np.arange(n_ns)
#         x_enr = np.arange(n_ns,n)
#         pp.fill_between(x_ns, 1, R[:enriched_i], color="0.75", label=None)
#         pp.fill_between(x_enr, 1, R[enriched_i:], color='k', label="Z-score > {0}".format(Z_cut))
        
#         #xlabel = 
#         for x, y, mer in zip(x_enr, R[enriched_i:], kmers[enriched_i:]):
#             #pp.text(x, r, mer, fontdict=dict(size=6), withdash=True)
#             pp.annotate(mer, xy=(x, y), xytext=(x-.05*n, y), arrowprops=dict(facecolor='black', arrowstyle="->, head_length = .2, head_width = .2"), horizontalalignment='right', verticalalignment='center', fontsize=6)
        
#         pp.ylabel('{0}mer enrichment [R-value]'.format(k) )
#         pp.xlabel('rank')
#         pp.axhline(1, color='k', linewidth=.5)
#         pp.xlim(0,n)
#         pp.gca().set_yscale('log')
#         pp.legend(loc='upper left')

#         #x_enriched = np.linspace(0, 1,n_enriched) + 1.1
#         #print x_enriched.shape, R[enriched_i:].shape
#         #pp.bar(x_enriched, R[enriched_i:], width = .05, color='k')
        
        
#         path = os.path.join(dest, "{0}.{1}".format(fname, fmt) )
#         pp.savefig(path)


class Container(object):
    pass


class RunReport(object):
    def __init__(self, path):
        self.path = path

    def load_descent(self, fname):
        data = []
        for line in file(os.path.join(self.path, fname)):
            if line.startswith("#"):
                continue
            row = line.split('\t')
            data.append([float(col) for col in row])

        data = np.array(data).T
        n_samples = (data.shape[0] - 5) / 2

        A0 = data[0]
        errors = data[3:3+n_samples]
        correlations = data[3+n_samples:3+2*n_samples]
        nfev, step = data[-2:]

        print data.shape, errors.shape
        descent = Container()
        # descent.history = history
        descent.ls_nfev = nfev
        descent.ls_step = step
        descent.history = []
        for err, corr in zip(errors.T, correlations.T):
            state = Container()
            print err, corr
            state.sample_errors = err
            state.correlations = (corr, 0)
            descent.history.append(state)
        

        fparams = os.path.join(self.path, os.path.dirname(fname), 'parameters.tsv')
        from cska.params import ModelParametrization
        descent.params = ModelParametrization.load(fparams, n_samples)

        return descent 


class GradientDescentReport(object):
    def __init__(self, fname, path='.', comp=None):
        import shelve
        self.comp = comp
        self.shelve = shelve.open(fname, flag='r')
        self.logger = logging.getLogger('plot.GradientDescentReport')
        self.t = np.arange(self.find_max_t())
        self.rbp_conc = self.shelve['rbp_conc']
        if (self.rbp_conc == np.round(self.rbp_conc)).all():
            self.rbp_conc = np.array(self.rbp_conc, dtype=int)

        self.R_exp = self.shelve['R_exp']
        self.logR0 = np.log2(self.R_exp)
        self.n_samples = self.R_exp.shape[0]
        self.k_mer = int(np.log2(self.R_exp.shape[1])/2.)
        self.path = path

    def find_max_t(self):
        t = -1
        while self.shelve.has_key("params_t{}".format(t+1)):
            t += 1
        return t

    def get(self, name, t):
        # TODO: allow adding more than one history and bisect on t, 
        # so that we read quasi-consecutively
        if t == -1:
            t = self.t[-1]
        return self.shelve["{0}_t{1}".format(name, t)]

    def read_sample_errors(self):
        errors = np.array([self.get('stats', t).errors for t in self.t])
        return errors

    def read_correlations(self):
        pearsonR = np.array([self.get('stats', t).pearsonR for t in self.t])
        pearsonP = np.array([self.get('stats', t).pearsonP for t in self.t])
        return pearsonR, pearsonP

    def read_linesearch(self):
        return np.array([self.get('linesearch', t) for t in self.t]).T


    def plot_report(self):
        pp.figure(figsize=(6,12))

        pp.subplot(311)
        errors = (self.read_sample_errors()**2).mean(axis=2)
        m_err = errors.mean(axis=1)
        pp.semilogy(m_err, 'k-', label='sample mean')
        for i, err in enumerate(errors.T):
            pp.semilogy(err, label='{0} nM'.format(self.rbp_conc[i]))

        pp.legend(loc='upper right')
        pp.ylabel("mean squared R-value error")

        pp.subplot(312)
        corr, pval = self.read_correlations()
        for i, c in enumerate(corr.T):
            pp.plot(c, label='{0} nM'.format(self.rbp_conc[i]))

        pp.legend(loc='upper right')
        pp.ylabel("R-value correlation")

        pp.subplot(313)
        nfev, step = self.read_linesearch()
        pp.semilogy(nfev, label='no. function evaluations during line-search')
        pp.legend(loc='upper right')
        pp.semilogy(step, label='step size')
        pp.xlabel('time step')
        pp.legend(loc='upper right')

        pp.tight_layout()
        pp.savefig(os.path.join(self.path,"descent_report.pdf"))
        pp.close()


    def plot_scatter(self, t=-1):
        if t == -1:
            t = self.t[-1]

        stats = self.get("stats", t)
        logRt = np.log2(self.get("R", t))
        for i in range(self.n_samples):
            pp.figure()
            title = "{0}mer R-value scatter plot".format(self.k_mer)
            pp.title(title)

            x = self.logR0[i]
            y = logRt[i]
            label = u"{0} nM R={1:.3f} ({2})".format(self.rbp_conc[i], stats.pearsonR[i], pval_str(stats.pearsonP[i]))
            # data_labels = self.opt.mdl.parameters.param_name
            density_scatter_plot(x, y, label=label, tick_exp=2)
            pp.legend(loc='upper left', frameon=False)
            pp.xlabel("observed {}-mer enrichment".format(self.k_mer))
            pp.ylabel("predicted {}-mer enrichment".format(self.k_mer))
            pp.savefig(os.path.join(self.path,"scatter_{0}mers_{1}nM_t{2}.pdf".format(self.k_mer, self.rbp_conc[i], t)), dpi=300)
            pp.close()


    def plot_literature(self, debug=True, t=-1):
        if self.comp is None:
            return

        if t == -1:
            t = self.t[-1]

        x = self.comp.observed_Kd
        if not len(x):
            return 

        x_err = self.comp.observed_Kd_err
        y = 1/self.comp.predict_affinities_from_paramset(self.get('params', t))
        lfc = np.log2(y/x)
        I = lfc.argsort()

        print x
        print y
        from scipy.stats import pearsonr, spearmanr
        rho, p_spearman = spearmanr(np.log(x), np.log(y))
        R, p_pearson = pearsonr(np.log(x), np.log(y))
        psstr = pval_str(p_spearman)
        ppstr = pval_str(p_pearson)
        label="$R={R:.2f}$ ($P < {p_pearson:.2e}$)\n$\\rho={rho:.2f}$ ($P < {p_spearman:.2e}$)".format(**locals())

        # self.results.info("R={R:.3f} {ppstr} rho={rho:.3f} {psstr}".format(**locals()))
        if debug:
            print u">>> R={R} {ppstr}".format(**locals())
            print u">>> rho={rho} {psstr}".format(**locals())
            print "seq\tknown\tpredict\tlog-ratio"
            for _x, _y, seq in zip(x[I], y[I], self.comp.seqs[I]):
                print seq, '\t', _x, '\t', _y, '\t', np.log2(_y/_x)

        m = min(x.min(), y.min())
        M = max(x.max(), y.max())
        
        import seaborn as sns
        with sns.axes_style("ticks", sns_style):
            matplotlib.rc('xtick.major', width = .1)
            matplotlib.rc('ytick.major', width = .1)

            pp.figure(figsize=(6,6))
            pp.title("comparison to {} literature affinities".format(self.comp.n))
            pp.errorbar(x, y, xerr=x_err, fmt='.', ecolor='k', elinewidth=.5, capsize=3, capthick=.5, label=label)
            pp.loglog([m,M],[m,M], 'k-', linewidth=.5)

            ax = pp.gca()
            ax.set_xscale("log", nonposx='clip')
            ax.set_yscale("log", nonposy='clip')

            pp.legend(loc='upper left', shadow=False, fancybox=False)
            pp.ylabel(r"predicted {} $K_d$ [nM]".format(self.comp.rbp_name))
            pp.xlabel(r"measured {} $K_d$ [nM]".format(self.comp.rbp_data))
            pp.tight_layout()
            sns.despine(trim=False)

            pp.savefig(os.path.join(self.path,"literature_comparison_t{}.pdf".format(t)))
            pp.close()


    def plot_A0_fit(self, t=-1):
        state = self.descent.history[t]
        if t == -1:
            t = self.descent.t

        if not hasattr(state, "_A0_data"):
            return

        a0 = state._A0_data.a0
        rerr = state._A0_data.rerr
        asem = getattr(state._A0_data, "asem", None)
        rcorr = state._A0_data.rcorr

        import matplotlib.pyplot as plt
        plt.figure()
        plot = plt.loglog
        plt.subplot(311)

        plot(a0, rerr, '.b', label='MSE')
        plot(a0, rerr, '-b')
        plt.legend(loc='upper left')

        plt.subplot(312)
        if not asem is None:
            plot(a0, asem, '.r', label='SEM')
            plot(a0, asem, '-r')
            plt.legend(loc='upper left')

        plt.subplot(313)
        plt.plot(a0, rcorr, '.k', label='best correlation')
        plt.plot(a0, rcorr, '-k')
        plt.legend(loc='upper left')

        a_opt = a0[rerr.argmin()]

        plt.tight_layout()
        plt.savefig(os.path.join(self.path,"A0_fit_{0}mers_t{1}.pdf".format(self.descent.model.k, t)))
        plt.close()

    def plot_psam(self, psam, title):
        pp.pcolor(psam.T, cmap='bwr', vmin=-1, vmax=+1)
        pp.xlabel(title)
        pp.ylabel("base")
        pp.yticks(np.arange(0.5,4.5,1), ['A','C','G','U'])
        pp.colorbar(label='weight', shrink=.5, orientation='horizontal')
        
    def plot_gradients(self, psam_correct, local_grad, grad):
        print descent
        pp.figure()
        pp.subplot(131)
        self.plot_psam(unity_matrix(psam_correct - self.psam),'actual delta')
        pp.subplot(132)
        self.plot_psam(unity_matrix(local_grad),'local gradient')
        pp.subplot(133)
        self.plot_psam(unity_matrix(grad),'RMSprop')

    def plot_param_hist(self):
        from matplotlib.colors import LogNorm
        A0 = np.array([state.params.A0 for state in self.descent.history])
        a = np.array([state.params.psam_vec[1:] for state in self.descent.history])
        betas = np.array([state.params.betas for state in self.descent.history])

        pp.figure(figsize=(6,10))
        pp.subplot(311)
        pp.plot(A0, label=r'$A_0$')
        pp.legend(loc='upper right')
        pp.ylabel("affinity [1/nM]")

        pp.subplot(312)
        pp.imshow(a.T, interpolation='nearest', cmap='inferno', norm=LogNorm(vmin=1e-6, vmax=1), aspect='auto')
        
        t = []
        for i in range(self.descent.params.k):
            for n in 'ACGU':
                t.append('{0}{1}'.format(n,i+1))
        l = len(t)
        pp.yticks(np.arange(l), t)

        t = [1e-5, 1e-3, 1e-1]
        pp.colorbar(orientation='horizontal', shrink=.5, ticks=t, label="PSAM values")
        pp.ylabel("matrix elements")

        pp.subplot(313)
        for i,b in enumerate(betas.T):
            pp.semilogy(b, label='beta{0}'.format(i))

        pp.legend(loc='upper right')
        pp.ylabel("background estimate")
        pp.xlabel("optimization step")
        pp.tight_layout()

        pp.savefig(os.path.join(self.path,"descent_params_{0}mer.pdf".format(self.descent.params.k)))
        pp.close()


class FootprintCalibrationReport(object):
    def __init__(self, fparams, out_path='.'):
        """
        fparams is path to calibrated.tsv params file
        expects database 'history' in same folder to retrieve
        intermediate results
        """ 
        from cska.params import ModelSetParams
        import shelve
        self.out_path = out_path
        self.logger = logging.getLogger('plot.FootprintCalibrationReport')
        dbfile = os.path.join(os.path.dirname(fparams), 'history')
        self.shelve = shelve.open(dbfile, flag='r')
        self.rbp_conc = self.shelve['rbp_conc']
        
        self.params = ModelSetParams.load(fparams, len(self.rbp_conc))
        self.motifs = [par.as_PSAM().consensus for par in self.params]
        if (self.rbp_conc == np.round(self.rbp_conc)).all():
            self.rbp_conc = np.array(self.rbp_conc, dtype=int)

    def get_profile_data(self, motif, k, s):
        opt = self.shelve["{motif}_{k}_{s}".format(**locals())]
        punp_predict, punp_a_one, res, res_a_one = self.shelve["{motif}_opt_profile_{k}_{s}".format(**locals())]
        punp_input = self.shelve["{motif}_punp_profiles".format(**locals())]
        punp_naive = self.shelve["{motif}_naive_profiles".format(**locals())]

        return res, res_a_one, opt, punp_input, punp_naive, punp_predict, punp_a_one

    def read_sample_errors(self):
        errors = np.array([self.get('stats', t).errors for t in self.t])
        return errors

    def plot_profile(self, motif, acc_k, acc_shift):
        res, res_a_one, opt, punp_input, punp_naive, punp_expect, punp_a_one = self.get_profile_data(motif, acc_k, acc_shift)
        err0 = np.sum((punp_naive - punp_input[1:])**2)

        import seaborn as sns
        import matplotlib.pyplot as plt
        # pwm = self.params.as_PSAM()
        pad = (punp_input.shape[1] - len(motif)) / 2
        x = np.arange(-pad, len(motif) + pad )

        plt.figure()
        # if acc_k:
        #     plt.title("acc_k = {acc_k} acc_shift = {acc_shift}".format(**locals()))
        # else:
        #     plt.title("expectation w/o acc. footprint".format(**locals()))

        colors = sns.color_palette("husl", 8)
        plt.plot(x, punp_input[0], '-k', label='input')
        for i, (obs, conc, color) in enumerate(zip(punp_input[1:], self.rbp_conc, colors)):
            plt.plot(x, obs, '-', color=color, label="{} nM".format(conc))

        import matplotlib.patches as patches
        ymin, ymax = plt.gca().get_ylim()
        height = ymax - ymin
        h = height * .02
        rect = patches.Rectangle(
            (acc_shift, ymin + h), 
            acc_k, h,
            linewidth=1,
            edgecolor='r',
            facecolor='r',
            label='footprint'
        )
        plt.gca().add_patch(rect)

        for i, (pred, one, conc, color) in enumerate(zip(punp_expect, punp_a_one, self.rbp_conc, colors)):
            lbl = "RNAfold (a=1) prediction err={rerr:.1f}%".format(rerr = 100. * res_a_one.fun/err0)
            plt.plot(x, one, ':', color=color, label=lbl if i == 0 else None)
            
            lbl = 'optimized (a={res.x[0]:.2f}) prediction err={rerr:.1f}%'.format(res=res, rerr = 100. * res.fun/err0)
            plt.plot(x, pred, '--', color=color, label=lbl if i == 0 else None)

        cons = motif
        plt.xticks(x, [str(p) for p in range(-pad,0)] + list(cons) + [str(p) for p in range(1, pad+1)])
        plt.axvline( - .5, color='k', linewidth=.5, linestyle='dashed', zorder=-1000)
        plt.axvline(len(motif) - .5, color='k', linewidth=.5, linestyle='dashed', zorder=-1000)

        plt.legend(
            bbox_to_anchor=(0., 1.02, 1., .202), 
            loc=3, ncol=2, mode="expand", borderaxespad=0.,
            frameon=False
        )


        plt.ylabel(r"$P_{unpaired}$ (motif-weighted)")
        plt.xlabel('pos. rel to motif (consensus) [nt]')

        fname = os.path.join(self.out_path, '{motif}_{acc_k}_{acc_shift}.pdf'.format(**locals()))
        plt.tight_layout()
        sns.despine(trim=True)
        sparse_y(plt.gca())
        self.logger.debug("saving plot: '{}'".format(fname))
        plt.savefig(fname)
        plt.close()


    def matrix_plots(self, results):
        self.logger.debug("matrix plot")
        import seaborn as sns
        import matplotlib.pyplot as plt
        err, acc_k, acc_shift, a, A0 = np.array(results).T
        k_base = int(acc_k.min())
        n_k = int(acc_k.max()) - k_base + 1
        shift = int(np.fabs(acc_shift).max())
        n_shift = shift * 2 + 1 
        mid_shift = shift

        mat_a = np.zeros((n_k, n_shift), dtype=float) + np.NaN
        mat_err = np.zeros((n_k, n_shift), dtype=float) + np.NaN
        for err, acc_k, acc_shift, a, A0 in results:
            mat_a[acc_k - k_base, acc_shift + mid_shift] = a
            mat_err[acc_k - k_base, acc_shift + mid_shift] = self.err0/err

        fig = plt.figure(figsize=(6,6))
        # fig.suptitle("accessibility footprint analysis")
        plt.subplot(211)
        plt.pcolor(mat_err, cmap="viridis")
        plt.colorbar(label=r'fold error reduction', fraction=.05)

        plt.ylabel("size [nt]")
        plt.xlabel("shift [nt]")

        plt.xticks(np.arange(n_shift)+.5, [str(s) for s in range(-shift, shift+1)])
        plt.yticks(np.arange(n_k)+.5, [str(k) for k in range(k_base, k_base + n_k)])
        plt.ylim(3, k_base+n_k)

        plt.subplot(212)
        plt.pcolor(mat_a, cmap="inferno")
        plt.colorbar(label=r'accessibility scaling', fraction=.05)

        plt.ylabel("size [nt]")
        plt.xlabel("shift [nt]")

        plt.xticks(np.arange(n_shift)+.5, [str(s) for s in range(-shift, shift+1)])
        plt.yticks(np.arange(n_k)+.5, [str(k) for k in range(k_base, k_base + n_k)])
        plt.ylim(3, k_base+n_k)

        plt.tight_layout()
        plt.savefig(os.path.join(self.path, '{self.consensus}_footprint.pdf'.format(self=self)))


if __name__ == "__main__":
    # rep = RunReport('/scratch/data/RBNS/MBNL1/cska/1M')
    # descent = rep.load_descent('opt_nostruct/descent.tsv')
    # print descent.params
    # print descent.history
    grep = GradientDescentReport('cska/recent/opt_nostruct/history')
    grep.plot_report()
    grep.plot_scatter(t=0)
    grep.plot_scatter(t=-1)

    # N = 4**6
    # x = np.array(np.random.random(N))
    # y = np.array(x + np.random.random(N) * .1)
    # mers = np.array([str(i) for i in y])
    # logging.basicConfig(level=logging.DEBUG)
    # logging.getLogger('matplotlib').setLevel(logging.INFO)
    # density_scatter_plot(x,y, data_labels=mers)
    # pp.show()
