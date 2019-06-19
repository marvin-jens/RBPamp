import re, glob, sys, os
import numpy as np
import shelve
import cska.report
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

class Results(object):
    def __init__(self, **kw):
        self._keys = set()
        self.add_results(**kw)
        
    def add_results(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
            self._keys.add(k)

    def _tostr(self, prefix=''):
        def fmt(k):
            x = getattr(self, k)
            if type(x) == Results:
                return x._tostr(prefix=k + '__')
            else:
                return "{0}{1}\t{2}".format(prefix, k, x)

        parts = [fmt(k) for k in sorted(self._keys)]
        return "\n".join(parts)

    def __str__(self):
        return self._tostr()

# res = Results(bla = True, nested = Results(blup = 2, bleh="meep"))
# print res
# 1/0

def get_descent(fname, err_thresh=.05):
    # print fname
    sname = os.path.join(os.path.dirname(fname), "history")
    shelf = shelve.open(sname, flag='r')
    try:
        lines = [l for l in file(fname).readlines() if not l.startswith("#")]
    except:
        logging.warning('file "{}" not found!'.format(fname))
        return Results()

    values = [line.rstrip().split('\t') for line in lines]
    # sometimes old runs were resumed later and have unequal column numbers. In that case drop the extra columns
    lens = set()
    for row in values:
        lens.add(len(row))
    l = np.array(list(lens)).min()
    values = [row[:l] for row in values]

    data = np.array(values, dtype=float)
    l0 = data[0]
    lf = data[-1]
    if l0[-1] == 0:
        n = (len(l0) - 5) / 2 # we have nfev and step output
    else:
        n = (len(l0) - 3) / 2

    res = Results(
        n_samples = n,
        rbp_conc = shelf['rbp_conc'],
        params = shelf['params_t0'],
        n_PSAM = len(shelf['params_t0'].param_set),
        best_corr = lf[3+n:3+2*n].max(),
        corr_initial = l0[3+n:3+2*n].max(),
        err_final = lf[2],
        err_samples = lf[3:3+n],
        err_initial = l0[2],
        Kd = 1./lf[1],
        n_steps = lf[0],
    )
    aff, err = data.T[1:3]
    Kd = 1./aff
    # select states of the descent with error within 5% of minimum
    I = np.fabs(err - err.min() ) < err_thresh * err.min()
    res.add_results(
        Kd_max = Kd[I].max(),
        Kd_min = Kd[I].min(),
        Kd_var = (Kd[I].max() - Kd[I].min()) / res.Kd
    )
    err_dict = {}
    for conc, err in zip(res.rbp_conc, res.err_samples):
        err_dict[conc] = err

    res.add_results(err_drop = res.err_initial/ res.err_final)
    res.add_results(err_dict = err_dict)
    res.add_results(err_perc = 100. * res.err_final / res.err_initial)
    res.add_results(corr_inc = res.best_corr - res.corr_initial)
    # print res
    return res

def get_footprint(fname):
    lines = [l for l in file(fname).readlines() if not l.startswith("acc_k")]
    data = np.array([line.split('\t') for line in lines], dtype=float)
    error = data.T[4]
    i = error.argmin()
    row = data[i]

    res = Results(
        err_drop = error.max() / error.min(),
        acc_k = row[0], 
        acc_shift = row[1], 
        acc_scale = row[2], 
        A0 = row[3], 
        err_min = row[4]
    )
    return res

def get_params(fname):
    lines = [l for l in file(fname).readlines() if not l.startswith("#")]
    head = lines[0].split(' ')
    mat = np.array([l.rstrip().split('\t') for l in lines[1:]])
    psam = np.array(mat[:,:-1], dtype=float)
    disc = (psam.max(axis=1) / psam.sum(axis=1) - .25 ) / .75
    c_str = []
    n_disc = 0
    for d, x in zip(disc, mat.T[-1]):
        if d < .5:
            c_str.append(x.lower())
        else:
            c_str.append(x.upper())
            n_disc += 1

    res = Results(
        w = int(head[2].split('=')[1]),
        consensus = "".join(c_str),
        avg_discrimination = disc.mean(),
        specific_bases = n_disc,
    )

    return res

def extract_error_corr(path):
    d = {}
    results = {}
    for fname in glob.glob(path):
        rbp = fname.split('/')[-3]
        sys.stderr.write(fname+'\n')
        print rbp
        try:
            res = Results(rbp=rbp)
            res.add_results(nostruct = get_descent(os.path.join(fname, "opt_nostruct/descent.tsv")))
            res.add_results(drop_initial = np.round(100. * (res.nostruct.err_drop - 1)) )
            # res.add_results(fp = get_footprint(os.path.join(fname, "footprint/footprints.tsv")))
            res.add_results(full = get_descent(os.path.join(fname, "opt_full/descent.tsv")))
        except IndexError:
            sys.stderr.write("error parsing data for {} \n".format(rbp))
            continue
        
        # res.add_results(params = get_params(os.path.join(fname, "opt_full/parameters.tsv")))
        # if res.Kd_stable and res.full_panel and res.good_fit:
        #     print res.rbp, res.nostruct.Kd
        d[rbp] = (res.nostruct.err_final, res.nostruct.best_corr)
        results[rbp] = res
    return d, results

def extract_results(path):
    stable = 0
    for fname in glob.glob(path):
        rbp = fname.split('/')[1]
        # sys.stderr.write(fname+'\n')
        # print rbp
        try:
            res = Results(rbp=rbp)
            res.add_results(nostruct = get_descent(os.path.join(fname, "opt_nostruct/descent.tsv")))
            res.add_results(full = get_descent(os.path.join(fname, "opt_full/descent.tsv")))
            res.add_results(drop_initial = np.round(100. * (res.nostruct.err_drop - 1)) )
            res.add_results(drop_full = np.round(100. * (res.nostruct.err_final / res.full.err_final - 1)) )
        except IndexError:
            sys.stderr.write("error parsing data for {} \n".format(rbp))
            continue
        
        # res.add_results(fp = get_footprint(os.path.join(fname, "footprint/footprints.tsv")))
        # res.add_results(params = get_params(os.path.join(fname, "opt_full/parameters.tsv")))
        res.add_results(
            Kd_stable = (res.nostruct.Kd_var < 1.),# and (res.full.Kd_var < 1.),
            full_panel = res.nostruct.n_samples > 2,
            # good_fit = res.nostruct.best_corr > .85,
            good_fit = res.full.best_corr > .85,
            inc_corr = res.full.best_corr > res.nostruct.best_corr,
            dec_err = res.nostruct.err_final - res.full.err_final
        )
        # if res.Kd_stable and res.full_panel and res.good_fit:
        #     print res.rbp, res.nostruct.Kd

        yield res

    sys.stderr.write("n_stable={}\n".format(stable))

        # print rbp
        # 
        # if not lines:
        # 	continue
        # cols0 = lines[0].split('\t')
        # cols1 = lines[-1].split('\t')
        # n = (len(cols0) - 3)/2
        # #print cols1
        # #print n
        # rerr = float(cols1[2]) / float(cols0[2])
        # ferr = float(cols1[2])
        # steps = int(cols1[0])
        # 
        
        # yield rbp, rerr, ferr, best_corr, steps


    
pattern = "RBNS/*/cska/multi_10M_2"
# pattern = "RBNS/*/cska/mparams1M_samples"
pattern = "RBNS/*/cska/recent"
pattern = "RBNS/*/cska/CI"

def load_or_make(pattern, base = "/home/mjens/engaging/", redo=False):
    import cPickle as pickle
    pf = "{key}.pkl".format(key=pattern.__hash__())
    if os.path.exists(pf) and not redo:
        res = pickle.load(file(pf, 'rb'))
    else:
        res = extract_error_corr(base + pattern)
        pickle.dump(res, open(pf, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    return res
    

# d_std, res_std = extract_error_corr("RBNS/*/cska/std")
# d_xsrbp, res_xsrbp = extract_error_corr("RBNS/*/cska/xsrbp")
# d_linocc, res_linocc = extract_error_corr("RBNS/*/cska/linocc")
# d_dumb, res_dumb = extract_error_corr("RBNS/*/cska/dumb")
# d_s, res_s = extract_error_corr("RBNS/*/cska/single")
# d_o, res_o = extract_error_corr("RBNS/*/cska/oneconc")


# d_std, res_std = load_or_make("RBNS/*/cska/std.72")
d_std, res_std = load_or_make("RBNS/*/cska/CI", redo=False)
d_xsrbp, res_xsrbp = load_or_make("RBNS/*/cska/xsrbp")
# d_xsrbp, res_xsrbp = load_or_make("RBNS/*/cska/std")
d_linocc, res_linocc = load_or_make("RBNS/*/cska/linocc")
d_dumb, res_dumb = load_or_make("RBNS/*/cska/dumb")
d_s, res_s = load_or_make("RBNS/*/cska/single")
d_o, res_o = load_or_make("RBNS/*/cska/oneconc")



from cska import dominguez_rbps as dom_rbps
rbps = sorted(d_std.keys())
rbps = np.array(dom_rbps)
print len(rbps), "RBPs are being considered"

err_std, corr_std = np.array([d_std[rbp] for rbp in rbps]).T
err_xs, corr_xs = np.array([d_xsrbp[rbp] for rbp in rbps]).T
err_lo, corr_lo = np.array([d_linocc[rbp] for rbp in rbps]).T
err_d, corr_d = np.array([d_dumb[rbp] for rbp in rbps]).T
err_s, corr_s = np.array([d_s[rbp] for rbp in rbps]).T
err_o, corr_o = np.array([d_o[rbp] for rbp in rbps]).T

n_psams = np.array([res_std[rbp].nostruct.n_PSAM for rbp in rbps])
multi_psam_rbps = rbps[n_psams > 1]
n_conc = np.array([len(res_std[rbp].nostruct.rbp_conc) for rbp in rbps])
# print n_psams
# print np.array(rbps)[n_psams > 1]

err_nostruct = np.array([res_std[rbp].nostruct.err_final for rbp in rbps])
err_full = np.array([res_std[rbp].full.err_final for rbp in rbps])

corr_nostruct = np.array([res_std[rbp].nostruct.best_corr for rbp in rbps])
corr_full = np.array([res_std[rbp].full.best_corr for rbp in rbps])


params_full = [res_std[rbp].full.params for rbp in rbps]
gc = []
scale = []
for rbp, params in zip(rbps, params_full):
    for par in params.param_set:
        psam = par.as_PSAM()
        print "\t".join([str(o) for o in [rbp, psam.consensus_ul, psam.fraction_GC, par.acc_scale, par.acc_k, par.acc_shift]])
        if par.acc_k > 3 and par.acc_scale > 0:
            gc.append(psam.fraction_GC)
            scale.append(par.acc_scale)


plt.figure(figsize=(2,2))
x = np.array(gc)
y = np.array(scale)
print y

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print slope, intercept, r_value, p_value, std_err
plt.plot(x, scale, '.')
plt.plot(x, (x*slope + intercept), '-r')
plt.xlabel("PSAM G+C content")
plt.ylabel("folding energy scale")
sns.despine()
plt.tight_layout()
plt.savefig("GC_scale.pdf")
sys.exit(0)

## impact of secondary structure
labels_s = ['PSAMs + structure', 'PSAMs only']
colors_s = ['r', 'k']

plt.figure(figsize=(2, 2))
_min = np.inf
_max = -np.inf
regressions = []

x = np.arange(len(err_full))
I = np.argsort(err_full)
for i, err in enumerate([err_full, err_nostruct]):
    plt.semilogy(x, err[I], '.', color=colors_s[i], label=labels_s[i], alpha=1., markersize=1)

    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log2(err_std), np.log2(err))
    # regressions.append( (slope, intercept) )
    _min = min(_min, err.min())
    _max = max(_max, err.max())

plt.legend(loc='lower center', ncol=1)
plt.xlabel("RBP index")
plt.ylabel("model error")
sns.despine()
plt.tight_layout()
plt.savefig("model_errors_struct.pdf")
plt.close()


plt.figure(figsize=(2, 2))
_min = np.inf
_max = -np.inf
regressions = []
import scipy.stats
x = np.arange(len(corr_full))
I = np.argsort(corr_full)
labels_s = ['PSAMs + structure', 'PSAMs only']
colors_s = ['r', 'k']
for i, corr in enumerate([corr_full, corr_nostruct]):
    plt.plot(x, corr[I], '.', color=colors_s[i], label=labels_s[i], alpha=1., markersize=1)

    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log2(err_std), np.log2(err))
    # regressions.append( (slope, intercept) )
    _min = min(_min, err.min())
    _max = max(_max, err.max())

plt.legend(loc='lower center', ncol=1)
plt.xlabel("RBP index")
plt.ylabel("6-mer correlation")
sns.despine()
plt.tight_layout()
plt.savefig("model_corrs_struct.pdf")
plt.close()


def n_PSAM_plot(n_psams):
    multi = {}
    for line in file("/home/mjens/git/cska/rRBNS/domains.txt"):
        rbp, domains = line.split('\t')
        nd = domains.count(',')
        if nd > 0:
            multi[rbp] = True
        # else:
        #     print rbp, nd, domains.rstrip()
    
    mask = np.array([multi.get(rbp, False) for rbp in rbps])
    for rbp, x, m in zip(rbps, n_psams, mask):
        if not multi.get(rbp, False):
            print "single-domain w multiple PSAMS", rbp, x

    nm = np.bincount(n_psams[mask])[1:]
    ns = np.bincount(n_psams[~mask])[1:]

    from scipy.stats import mannwhitneyu
    print "multi domain median PSAMs", np.median(n_psams[mask])
    print "single domain median PSAMS", np.median(n_psams[~mask])
    print mannwhitneyu(n_psams[mask], n_psams[~mask])

    print "multi", nm
    print "single", ns
    # vm = nm / float(nm.sum())
    # vs = ns / float(ns.sum())

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(len(nm)))

    plt.figure(figsize=(3, 2))
    plt.bar(np.arange(len(ns)), ns, width=0.4, color=outer_colors[0], label="single-domain")
    plt.bar(np.arange(len(nm))+.45, nm, width=0.4, color=outer_colors[2], label="multi-domain")

    plt.xlabel("PSAMs per RBP")
    plt.ylabel("count")
    plt.tight_layout()
    sns.despine()
    plt.xticks(np.arange(5), ["1","2","3","4","5"])
    # plt.gca().set(aspect="equal")
    plt.savefig("n_PSAMs_bar.pdf")
    plt.close()

n_PSAM_plot(n_psams)
# print n_conc
# print np.array(rbps)[n_conc == 1]

err_ratios = [err_xs / err_std, err_lo / err_std, err_d / err_std, (err_s / err_std)[n_psams > 1], err_o / err_std]
corr_ratios = [corr_xs / corr_std, corr_lo / corr_std, corr_d / corr_std, (corr_s / corr_std)[n_psams > 1], corr_o / corr_std]

labels = ["mass-action", "excess RBP", "linear occ.", "excess RBP +\nlinear occ.", "single PSAM"] #, "single\nconcentration"]
flatui = ["k", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
flatui = ["k", "#9b59b6", "#3498db", "#e74c3c", "#95a5a6", "#34495e", "#2ecc71"]
colors = ['#e75621', '#27335d', '#f5a601', '#a8c784', '#639bbe']
colors = flatui
# import seaborn as sns
# colors = sns.color_palette(flatui)(np.linspace(0, 1, len(labels)))
# print "colors", colors
symbols = ['o', '^', 's', '*', '.']

plt.figure(figsize=(3, 3))
# extract the sample errors after optimization for exacty the sample being used in the oneconc runs
single_conc = [res_o[rbp].nostruct.rbp_conc[0] for rbp in rbps]
es = np.array([res_std[rbp].nostruct.err_dict[conc] for (conc, rbp) in zip(single_conc, rbps)])
eo = np.array([res_o[rbp].nostruct.err_dict[conc] for (conc, rbp) in zip(single_conc, rbps)])

plt.plot(es[n_conc > 1], eo[n_conc > 1], symbols[4], color=colors[4], label="error of best sample after optimization")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")

_min = min(es.min(), eo.min())
_max = min(es.max(), eo.max())

_min *= 1 - np.sign(_min) * .05
_max *= 1 + np.sign(_max) * .05

plt.plot([_min, _max], [_min, _max], '--', color='gray', linewidth=.5)
plt.xlim(_min, _max)
plt.ylim(_min, _max)
plt.legend(loc='best')
plt.xlabel("best sample alone")
plt.ylabel("all samples together")
plt.tight_layout()
sns.despine()
plt.savefig("oneconc_vs_all.pdf")
plt.close()

for j in range(len(labels) - 1):
    print "top proteins that benefit from", labels[j + 1]
    for i in err_ratios[j].argsort()[:5]:
        r = err_ratios[j][i]
        if j == 3:
            rbp = multi_psam_rbps[i]
        else:
            rbp = rbps[i]
        if r < 1:
            print rbp, "err_ratio", r, "corr_ratio", corr_ratios[j][i]


plt.figure(figsize=(4,2))
plt.subplot(121)
# print len(labels), err_ratios.shape, corr_ratios.shape
lerr = [np.log2(r) for r in err_ratios][:-1]
lcorr = [np.log2(r) for r in corr_ratios][:-1]
bplot = plt.boxplot(
    lerr,
    notch=True,  # notch shape
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
    labels=labels[1:],  # will be used to label x-ticks
)
plt.ylabel("rel. error [log2]")
plt.axhline(0, color='gray', linewidth=.5, linestyle='dashed')
plt.yticks([-2, -1,0,1,2,3], ["0.25", "0.5", "1", "2", "4", "8"])

plt.subplot(122)
bp2 = plt.boxplot(lcorr,
    notch=True,  # notch shape
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
    labels=labels[1:],  # will be used to label x-ticks
)
plt.ylabel("rel. correlation [log2]")
plt.axhline(0, color='gray', linewidth=.5, linestyle='dashed')
plt.yticks([-3, -2, -1,0,1,2], ["0.125","0.25", "0.5", "1", "2", "4"])
plt.xticks(rotation=90)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'orange']
for bplot in (bplot, bp2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig("model_performance.pdf")
plt.close()

plt.figure(figsize=(2, 2))
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")
# plt.gca().set_aspect(1.0)
_min = np.inf
_max = -np.inf
regressions = []
import scipy.stats
x = np.arange(len(err_std))
I = np.argsort(err_std)
for i, err in enumerate([err_std, err_xs, err_lo, err_d, err_s]):
    plt.semilogy(x, err[I], '.', color=colors[i], label=labels[i], alpha=1., markersize=1)

    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log2(err_std), np.log2(err))
    # regressions.append( (slope, intercept) )
    _min = min(_min, err.min())
    _max = max(_max, err.max())

w = np.fabs(_max - _min)
print "before", _min, _max, w

_min *= 1 - np.sign(_min) * .2
_max *= 1 + np.sign(_max) * .2

w = np.fabs(_max - _min)
print "after", _min, _max, w

# plt.plot([_min, _max], [_min, _max], '--', color='gray', )

# print "regressions", regressions
# for (slope, intercept), color in zip(regressions, colors):
#     x = np.linspace(np.log2(_min), np.log2(_max), 2)
#     plt.plot(2**x, 2**(x*slope + intercept), '-', color=color, linewidth=1)

# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()
# plt.xlim(min(xmin, ymin), max(xmax, ymax))
# plt.ylim(min(xmin, ymin), max(xmax, ymax))
# plt.ylim(_min, _max)
plt.legend(loc='upper center', ncol=2)
plt.xlabel("RBP index")
plt.ylabel("model error")
sns.despine()
plt.tight_layout()
plt.savefig("model_errors.pdf")
plt.close()

plt.figure(figsize=(2,2))
_min = np.inf
_max = -np.inf
x = np.arange(len(corr_std))
I = np.argsort(corr_std)
for i, corr in enumerate([corr_std, corr_xs, corr_lo, corr_d, corr_s]):
# for i, err in enumerate([err_std, err_xs, err_lo, err_d, err_s]):
    plt.plot(x, corr[I], '.', color=colors[i], label=labels[i], alpha=1., markersize=1)    
    # plt.plot(corr_std, corr, symbols[i], color=colors[i], label=labels[i])
    _min = min(_min, corr.min())
    _max = max(_max, corr.max())

_min *= 1 - np.sign(_min) * .1
_max *= 1 + np.sign(_max) * .1

_min = 0
_max = 1
# plt.plot([_min, _max], [_min, _max], '--', color='gray', linewidth=.5)
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")

# plt.xlim(_min, _max)
# plt.ylim(_min, _max)
plt.legend(loc='best')
plt.xlabel("RBP index")
plt.ylabel("6-mer correlation")
plt.tight_layout()
sns.despine()
plt.savefig("model_corrs.pdf")
plt.close()
# 	#print "\t".join([rbp,str(1./float(score))])
# 	print "\t".join([rbp, str(rerr), str(ferr), str(best_corr), str(steps)])

