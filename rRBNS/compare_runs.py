# -*- coding: future_fstrings -*-
from __future__ import print_function
import re, glob, sys, os
import numpy as np
import shelve
import logging
import cska.report
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import scipy.stats
import cska
import logging
logging.basicConfig(level=logging.INFO)

# formatting for box-plots
bpkw = dict(
    medianprops=dict(color='red'),
    boxprops=dict(linewidth=.5,),
    whiskerprops=dict(linewidth=.5,),
    capprops=dict(linewidth=.5,),
    flierprops=dict(marker='.', markerfacecolor='k', markersize=3),
    notch=False,  # notch shape
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
)

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
    
    def get(self, k, default=None):
        if k in self._keys:
            return getattr(self, k)
        
        return default

    def __str__(self):
        return self._tostr()

# res = Results(bla = True, nested = Results(blup = 2, bleh="meep"))
# print res
# 1/0

def get_descent(fname, err_thresh=.05):
    # print "get_descent", fname
    sname = os.path.join(os.path.dirname(fname), "history")
    try:
        shelf = shelve.open(sname, flag='r')
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
        corr_samples = lf[3+n:3+2*n],
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
    corr_dict = {}
    for conc, err, corr in zip(res.rbp_conc, res.err_samples, res.corr_samples):
        err_dict[conc] = err
        corr_dict[conc] = corr

    res.add_results(err_drop = res.err_initial/ res.err_final)
    res.add_results(err_dict = err_dict)
    res.add_results(corr_dict = corr_dict)
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


def n_PSAM_plot(nm, ns, nd, fname="n_PSAMs_bar.pdf"):
    print("multi", nm)
    print("single", ns)
    print("KHDRBS2+3 (likely dimers)", nd)
    # vm = nm / float(nm.sum())
    # vs = ns / float(ns.sum())

    cmap = plt.get_cmap("tab20")
    outer_colors = cmap(np.arange(len(nm)))

    plt.figure(figsize=(2, 2))

    # n = np.bincount(n_psams)
    # plt.bar(np.arange(len(n)), n, width=0.8, color=outer_colors[1])
    plt.bar(np.arange(len(ns)), ns, width=0.3, color=outer_colors[1], label="1 RBD")
    plt.bar(np.arange(len(nm))+.3, nm, width=0.3, color=outer_colors[0], label="2+ RBDs")
    plt.bar(np.arange(len(nd))+.6, nd, width=0.3, color=outer_colors[2], label="dimers")

    # cs = np.cumsum(ns) / float(ns.sum())
    # cm = np.cumsum(nm) / float(nm.sum())
    # cd = np.cumsum(nd) / float(nd.sum())

    # plt.plot(cs, "-", color=outer_colors[1], label="1 RBD")
    # plt.plot(cm, "-", color=outer_colors[0], label="2 RBDs")
    # plt.plot(cd, "-", color=outer_colors[2], label="3+ RBDs")
    plt.ylabel("cumulative fraction")

    plt.legend(loc='upper left')
    plt.ylabel("count")
    plt.xlabel("PSAMs per RBP")
    sns.despine()
    plt.tight_layout()
    plt.xticks(np.arange(5), ["1","2","3","4","5"])
    # plt.gca().set(aspect="equal")
    plt.savefig(fname)
    plt.close()



def n_PSAM_significance(pattern, rbps, variant, plot=False):
    from cska.params import ModelSetParams
    # setup
    multi = {}
    dom_counts = {}
    valid = {}
    for line in file("/home/mjens/git/cska/rRBNS/domains.txt"):
        rbp, domains = line.split('\t')
        doms = domains.rstrip().split(',')
        
        dom_counts[rbp] = len(doms)
        if dom_counts[rbp] > 1:
            multi[rbp] = True
        else:
            multi[rbp] = False

        if 'other' in domains or rbp.startswith('KHDRBS'):
            valid[rbp] = False
        else:
            valid[rbp] = True
    
    # load PSAM count for valid rbps
    n_psams = {}
    valid_rbps = []
    for rbp in rbps:
        if valid[rbp]:
            valid_rbps.append(rbp)

        fname = pattern.format(rbp=rbp, variant=variant)
        params = ModelSetParams.load(fname, 1)
        n_psams[rbp] = len(params.param_set)
        # if rbp == 'MSI1':
        #     print params
        #     print "<<<", len(params.param_set)
        

    n_multi = np.array([n_psams[rbp] for rbp in valid_rbps if multi[rbp] == True])
    n_single = np.array([n_psams[rbp] for rbp in valid_rbps if dom_counts[rbp] == 1])
    # n_multi = np.array([n_psams[rbp] for rbp in valid_rbps if dom_counts[rbp] == 2])
    n_dimer = np.array([n_psams[rbp] for rbp in ['KHDRBS2', 'KHDRBS3']])
    # n_dimer = np.array([n_psams[rbp] for rbp in valid_rbps if dom_counts[rbp] > 2])

    def stars(p):
        if p < .01:
            return "**"
        elif p < .05:
            return "* "
        else:
            return "ns"

    # print pattern
    from scipy.stats import mannwhitneyu, ttest_ind
    # print "multi domain mean PSAMs", n_multi.mean()
    # print "single domain mean PSAMS", n_single.mean()
    mwu = mannwhitneyu(n_multi, n_single)
    tt = ttest_ind(n_multi, n_single)
    # tt = mannwhitneyu(n_dimer, n_single) # abuse for 3+ domains

    ratio = n_multi.mean() / n_single.mean()

    mwu_p = mwu.pvalue
    tt_p = tt.pvalue

    mwu_s = stars(mwu_p)
    tt_s = stars(tt_p)
    print(f"{variant:30s} multi/single={ratio:.3f} p_MWU={mwu_p:.3e} {mwu_s}  p_ttest={tt_p:.3e} {tt_s} ")
    # print "MWU", mwu
    # print "t-test", tt

    nm = np.bincount(n_multi)[1:]
    ns = np.bincount(n_single)[1:]
    nd = np.bincount(n_dimer)[1:]

    if (mwu_p < .05 and tt_p < .05) or plot:
        n_PSAM_plot(nm, ns, nd, fname=f"n_PSAM_{variant}.pdf")
        # for rbp in rbps:
        #     print "{} -> n_dom={} n_psam={}".format(rbp, dom_counts[rbp], n_psams[rbp])

        # print "n_multi", n_multi
        # print "n_single", n_single
    return mwu, tt, nm, ns, ns


def extract_error_corr(path):
    d = {}
    results = {}
    errors = []

    for fname in glob.glob(path):
        rbp = fname.split('/')[-3]
        sys.stderr.write(fname+'\n')
        # print rbp
        try:
            res = Results(rbp=rbp)
            res.add_results(nostruct = get_descent(os.path.join(fname, "opt_nostruct/descent.tsv")))
            res.add_results(drop_initial = np.round(100. * (res.nostruct.err_drop - 1)) )
            # res.add_results(fp = get_footprint(os.path.join(fname, "footprint/footprints.tsv")))
            res.add_results(full = get_descent(os.path.join(fname, "opt_struct/descent.tsv")))
        except (IndexError, AttributeError, ValueError):
            sys.stderr.write("error parsing data for {} \n".format(rbp))
            errors.append(rbp)
            continue
        
        # res.add_results(params = get_params(os.path.join(fname, "opt_full/parameters.tsv")))
        # if res.Kd_stable and res.full_panel and res.good_fit:
        #     print res.rbp, res.nostruct.Kd
        d[rbp] = (res.nostruct.err_final, res.nostruct.best_corr)
        results[rbp] = res
    
    if errors:
        print("Errors occurred with the following rbps", errors)

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
            res.add_results(full = get_descent(os.path.join(fname, "opt_struct/descent.tsv")))
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

def load_fp_calibrated_params(pattern, rbps, variant, plot=False):
    from cska.params import ModelSetParams
    
    # load PSAM count for valid rbps
    params = {}
    for rbp in rbps:
        fname = pattern.format(rbp=rbp, variant=variant)
        params[rbp] = ModelSetParams.load(fname, 1)

    return params

def load_or_make(pattern, base = "/home/mjens/engaging/", redo=False):
    import cPickle as pickle
    pf = "{key:x}.pkl".format(key=pattern.__hash__())
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

def compare_runs(runs, rbps):
    rkeys = sorted(runs.keys())
    mcorrs = []
    merrs = []
    all_errs = []
    all_corrs = []
    for run in rkeys:
        print(">>", run)
        errs, corrs = np.array([runs[run][rbp] for rbp in rbps]).T
        merrs.append(np.mean(np.log10(errs)))
        mcorrs.append(np.mean(corrs))
    
        all_errs.append(errs)
        all_corrs.append(corrs)

        eperc = np.percentile(np.log10(errs), [5, 25, 50, 75, 95])
        cperc = np.percentile(corrs, [5, 25, 50, 75, 95])
        print("  log10 error quartiles", np.round(eperc, 2))
        print("  correlation quartiles", np.round(cperc, 2))
        # mean corr {:.3f}".format(run, merrs[-1], mcorrs[-1]) 

    rkeys = np.array(rkeys)
    merrs = np.array(merrs)
    mcorrs = np.array(mcorrs)

    print(">> most variable RBPs")
    all_errs = np.array(all_errs)
    all_corrs = np.array(all_corrs)
    vc = np.std(all_corrs, axis=0)
    I = vc.argsort()[::-1]
    for i in I[:15]:
        print(rbps[i], np.round(vc[i], 2), np.round(all_corrs[:, i], 3), np.round(all_errs[:, i], 3))


    ie = merrs.argmin()
    ic = mcorrs.argmax()

    print("-> run with lowest error {}, highest corr {}".format(rkeys[ie], rkeys[ic]))



def GC_acc_scale_plot():
    params_full = [res_std[rbp].full.params for rbp in rbps]
    gc = []
    scale = []
    for rbp, params in zip(rbps, params_full):
        for par in params.param_set:
            psam = par.as_PSAM()
            print("\t".join([str(o) for o in [rbp, psam.consensus_ul, psam.fraction_GC, par.acc_scale, par.acc_k, par.acc_shift]]))
            if par.acc_k > 3 and par.acc_scale > 0:
                gc.append(psam.fraction_GC)
                scale.append(par.acc_scale)


    plt.figure(figsize=(2,2))
    x = np.array(gc)
    y = np.array(scale)
    # print y

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    # print slope, intercept, r_value, p_value, std_err
    plt.plot(x, scale, '.')
    plt.plot(x, (x*slope + intercept), '-r')
    plt.xlabel("PSAM G+C content")
    plt.ylabel("folding energy scale")
    sns.despine()
    plt.tight_layout()
    plt.savefig("GC_scale.pdf")
    plt.close()
    

def add_significance(data, labels, y, x0=1, ref=0, name="differences", ax=None):
    from scipy.stats import mannwhitneyu, ttest_1samp
    from cska.report import pval_stars

    print(name)
    err_stars = ['']
    for e, label in zip(data[1:], labels[1:]):
        if ref is not None:
            m = mannwhitneyu(data[ref], e)
        else:
            m = ttest_1samp(e, 0)

        print(label, m)
        err_stars.append(pval_stars(m.pvalue))

    if ax == None:
        ax = plt.gca()
    print(err_stars)
    for i, stars in enumerate(err_stars):
        ax.text(i + x0, y, stars, horizontalalignment='center', verticalalignment='center')


def plot_and_regress(x, y, ax=None, logy=False, **kw):
    if ax is None:
        ax = plt.gca()
    from scipy.stats import linregress
    if logy:
        y = np.log10(y)
    ax.plot(x, y, '.', **kw)
    a, b, r, p, stderr = linregress(x, y)
    print(f"slope={a:.3f} intercept={b:.3f} rvalue={r:.3f} pvalue={p:.3e}")
    ax.plot(x, a*x+b, '-', color='maroon', label=f'r={r:.2f} P < {p:.2e}')
    if logy:
        set_ylogticks(ax)

def set_ylogticks(ax):
    yl = np.array([.05, 0.1, .2, .4, .6])
    y = np.log10(yl)
    ax.set_yticks(y)
    ax.set_yticklabels(yl)

    
class ModelComparisons(object):
    def __init__(self, variant_dict, rbps=cska.dominguez_rbps, labels=None):
        self.logger = logging.getLogger('ModelComparisons')
        self.rbps = np.array(rbps)
        self.variants = variant_dict.keys()
        assert "std" in variant_dict # baseline ref

        self.labels=labels
        for name, (d, res) in variant_dict.items():
            setattr(self, f'd_{name}', d)
            setattr(self, f'res_{name}', res)

            err, corr = np.array([d[rbp] for rbp in self.rbps]).T
            setattr(self, f'err_{name}', err)
            setattr(self, f'corr_{name}', corr)

        self.n_psams = np.array([self.res_std[rbp].nostruct.n_PSAM for rbp in self.rbps])
        self.n_conc = np.array([len(self.res_std[rbp].nostruct.rbp_conc) for rbp in self.rbps])
        self.err_std_fp = np.array([self.res_std[rbp].full.get("err_final", np.NaN) for rbp in self.rbps])
        self.corr_std_fp = np.array([self.res_std[rbp].full.get("best_corr", np.NaN) for rbp in self.rbps])
        self.load_domains()

        self.rbp_domain = np.array([self.dom_type[rbp] for rbp in self.rbps])

    def load_domains(self, fname='domains.txt'):

        self.dom_type = {}
        self.domains = {}

        def _domain(dom):
            if type(dom) == float:
                return 'NA'
            from collections import defaultdict
            ds = defaultdict(int)
            for d in dom.split(','):
                ds[d] += 1
            # tbl = np.array(sorted([(v,k) for k,v in ds.items()]))[::-1]
            if len(ds.keys()) == 1:
                return ds.keys()[0]
                # return "{} {}".format(ds.values()[0], ds.keys()[0]) # only one domain type
            else:
                return "mixed"

        for line in file(fname):
            rbp, doms = line.rstrip().split('\t')
            self.domains[rbp] = doms.split(',')
            self.dom_type[rbp] = _domain(doms)

    def left_out_sample_data(self, min_corr=.1):
        assert "single" in self.variants
        assert "std_eval" in self.variants
        assert "single_eval" in self.variants

        comparable_rbps = []
        # lratios = []
        mean_errs = []
        mean_corrs = []
        for rbp in self.rbps:
            used = set(self.res_std[rbp].nostruct.rbp_conc)
            best = set(self.res_single[rbp].nostruct.rbp_conc)
            left = set(self.res_std_eval[rbp].nostruct.rbp_conc) - (used | best)
            
            all_conc = list(self.res_std_eval[rbp].nostruct.rbp_conc)

            if left:
                sample_conc = sorted(left)
                sample_idx = [all_conc.index(c) for c in sorted(left)]
                errs_full = np.array([self.res_std_eval[rbp].nostruct.err_dict[c] for c in sample_conc])
                errs_single = np.array([self.res_single_eval[rbp].nostruct.err_dict[c] for c in sample_conc])

                corrs_full = np.array([self.res_std_eval[rbp].nostruct.corr_dict[c] for c in sample_conc])
                corrs_single = np.array([self.res_single_eval[rbp].nostruct.corr_dict[c] for c in sample_conc])

                # drop junk samples that don't correlate at all
                keep = (corrs_single > min_corr) | (corrs_full > min_corr)
                if keep.sum() < 1:
                    continue

                corrs_full = corrs_full[keep]
                corrs_single = corrs_single[keep]
                errs_full = errs_full[keep]
                errs_single = errs_single[keep]

                # lratio = np.log2(errs_single/errs_full)
                mean_errs.append( (errs_full.mean(), errs_single.mean()) )
                mean_corrs.append( (corrs_full.mean(), corrs_single.mean()) )

                # print(f"{rbp} : {sample_idx} conc {sample_conc} errs_full ={errs_full:.3e} errs_single={errs_single:.3e} lratio={lratio:.3e}")
                comparable_rbps.append(rbp)
                # lratios.append(lratio)

        # for rbp in sorted(comparable_rbps.keys()):
        #     sample_idx = comparable_rbps[rbp]
        #     print(f"{rbp} : {sample_idx}")
        comparable_rbps = np.array(comparable_rbps)
        

        return comparable_rbps, np.array(mean_errs), np.array(mean_corrs)

    def delta_barplot(self, ax, y0, y1, name='delta', func = lambda y0, y1 : y0 - y1, ax_test=None, xticks=[10, 30, 50, 70]):
        delta = func(y0, y1)
        I = delta.argsort()
        N = len(delta)

        x = np.arange(N)
        ax.bar(x, delta[I], width=1., linewidth=0, color=np.where(delta[I] > 0, 'k', 'r'))
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        # statistical testing.
        gt = (y1 > y0).sum()
        le = (y1 <= y0).sum()

        from scipy.stats import binom_test, mannwhitneyu, ttest_ind, ttest_1samp
        bt = binom_test(gt, N)
        mwu = mannwhitneyu(y0, y1)
        tt = ttest_ind(y0, y1)
        tt1 = ttest_1samp(delta, 0)

        print(f"{name} gt={gt} le={le} N={N}")
        print(f"binomial test P-value for '{name}' {bt}")
        print(f"MWU P-value for '{name}' {mwu}")
        print(f"ttest P-value for '{name}' {tt}")
        print(f"ttest P-value for delta mean != 0 '{name}' {tt1}")

        ax.axhline(0, color='k', linewidth=.5)
        if ax_test:
            ax_test.bar(
                [0, 1],
                [gt, le],
                color=['r', 'k']
            )
            ax_test.set_ylabel("# RBPs")
            ax_test.set_xticks([0, 1])
            ax_test.set_xlim(-1, 8)
            ax_test.set_xticklabels(["lower", "higher"], rotation=90)

        return mwu, tt, tt1, bt


    # def left_out_single_conc_plot(self):

    #     I = full.argsort()
    #     x = np.arange(len(I))
    #     plt.figure(figsize=(3, 3))
    #     plt.gca().set_yscale('log')
    #     plt.plot(x, full[I], '.k')
    #     plt.plot(x, single[I], '^r')
    #     plt.xlabel('RBP index')
    #     plt.ylabel("model error of left-out samples")
    #     plt.tight_layout()
    #     plt.savefig('left_out.pdf')
    #     print("single > full", (single > full).sum(), "single <= full", (single <= full).sum(), "N=", len(I))
    #     from scipy.stats import binom_test
    #     print("binomial test P-value", binom_test((single > full).sum(), len(I)))
    #     plt.close()

    def single_multi_scatter(self):
        plt.figure(figsize=(3, 3))
        minerr = .5*min(err_s.min(), err_std.min())
        maxerr = 2*max(err_s.max(), err_std.max())
        
        colors = ['cyan', 'k', 'orange', 'tomato', 'red', 'violet']
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')

        plt.legend(loc='upper left')
        lfc = np.log2(err_s / err_std)
        for i in np.argsort(lfc):
            if lfc[i] > - 0.2:
                break
            print("error lower in single PSAM", rbps[i], err_s[i], '<', err_std[i], 'n_psam', n_psams[i])

        plt.plot([minerr, maxerr], [minerr, maxerr], '--k', linewidth=.5)
        # plt.colorbar()
        plt.xlabel("single PSAM")
        plt.ylabel("PSAM set")
        plt.xlim(minerr, maxerr)
        plt.ylim(minerr, maxerr)
        sns.despine()
        plt.tight_layout()
        plt.savefig("PSAM_set.pdf")
        plt.close()

    def single_multi_PSAMS(self, corr_ymin=None, corr_ymax=None):
        assert "single" in self.variants
        fig, ((ax_err, ax_ebp), (ax_corr, ax_cbp)) = plt.subplots(2, 2, gridspec_kw=dict(width_ratios=[1, 1]), figsize=(3.5, 2.5), sharex='col')
        
        ## per RBP errors as dots/triangles. Upper left panel
        emulti = self.err_std[self.n_psams > 1]
        esingle = self.err_single[self.n_psams > 1]

        self.delta_barplot(
            ax_err, 
            emulti, 
            esingle,
            name = "multi PSAM error",
            func = lambda y0, y1 : np.log2(y0/y1),
            # ax_test=ax_es
        )

        # I = emulti.argsort()
        
        # ax_err.set_yscale('log')
        # sa = ax_err.plot(esingle[I], '^r', markersize=2, markeredgewidth=0) #, color='blue', markersize=6, alpha=.5, ))
        # ma = ax_err.plot(emulti[I], '.k', markersize=3, markeredgewidth=0) #, color='blue', markersize=6, alpha=.5))
        # ax_err.legend((ma[0], sa[0]), ("multiple PSAMs", "single PSAM"), loc='lower right')
        ax_err.set_ylabel('model error')

        ## per RBP CHANGE in error. Upper right panel
        nrange = range(1, self.n_psams.max()+1)
        domains = ['RRM', 'KH', 'ZNF', 'mixed', 'other']
        n_psam_masks = [self.n_psams == n for n in nrange]
        domain_masks = [self.rbp_domain == dom for dom in domains]

        labels = [f'{n}' for n in nrange]
        lerr_multi = [np.log2(np.array(self.err_std[mask])) for mask in n_psam_masks]
        lerr_single = [np.log2(np.array(self.err_single[mask])) for mask in n_psam_masks]

        deltas=[m-s for m, s in zip(lerr_multi, lerr_single)]
        sns.swarmplot(
            data=deltas,
            ax =ax_ebp,
            size=1.5,
        )
        ax_ebp.plot(np.arange(len(deltas)), [np.mean(d) for d in deltas], '_', color='red', markersize=10, solid_capstyle='round')
        add_significance(deltas, labels, 0.2, x0=0, ref=None, ax=ax_ebp, name='log2 error ratio single vs. multi PSAM')
        ax_ebp.set_ylabel(u'Δ($\log_2$ model error)')

        ## per RBP correlations as dots/triangles. Lower left panel
        cmulti = self.corr_std[self.n_psams > 1]
        csingle = self.corr_single[self.n_psams > 1]

        self.delta_barplot(
            ax_corr, 
            cmulti, 
            csingle,
            name = "multi PSAM correlation",
            # func = lambda y0, y1 : np.log2(y0/y1), 
            # ax_test=ax_es
        )

        # I = cmulti.argsort()
        
        # sa = ax_corr.plot(csingle[I],'^r',  markersize=2, markeredgewidth=0) #, color='blue', markersize=6, alpha=.5, markeredgewidth=0))
        # ma = ax_corr.plot(cmulti[I],'.k', markersize=3, markeredgewidth=0) #, color='blue', markersize=6, alpha=.5, markeredgewidth=0))

        ax_corr.set_ylabel('6-mer correlation')
        if corr_ymax:
            ax_corr.set_ylim(corr_ymin, corr_ymax)
        # ax_corr.set_xticklabels(rbps[I], rotation=90)
        ax_corr.set_xlabel('RBP index')

        ## per RBP CHANGE in corr. Lower right panel
        corr_multi = [np.array(self.corr_std[mask]) for mask in n_psam_masks]
        corr_single = [np.array(self.corr_single[mask]) for mask in n_psam_masks]

        deltas=[m-s for m, s in zip(corr_multi, corr_single)]
        from scipy.stats import ttest_1samp
        print("multiple_PSAM effects as function of no. of PSAMS")
        for i, d in enumerate(deltas):
            tt = ttest_1samp(d, 0)
            print(f"{i+1} PSAM(s) -> {tt}")

        ## by DOMAIN
        corr_multi = [np.array(self.corr_std[mask]) for mask in domain_masks]
        corr_single = [np.array(self.corr_single[mask]) for mask in domain_masks]

        ddeltas=[m-s for m, s in zip(corr_multi, corr_single)]
        from scipy.stats import ttest_1samp
        print("multiple_PSAM effects as function of domain type")
        for i, d in enumerate(ddeltas):
            tt = ttest_1samp(d, 0)
            print(f"{domains[i]} PSAM(s) -> {tt}")
        
        sns.set_palette(sns.color_palette("GnBu_d"))
        sns.swarmplot(
            data=deltas,
            ax=ax_cbp,
            size=1.5,
        )
        ax_cbp.plot(np.arange(len(deltas)), [np.mean(d) for d in deltas], '_', color='red', markersize=10)
        add_significance(deltas, labels, .3, x0=0, ref=None, ax=ax_cbp, name='correlation difference single vs. multi PSAM')
        ax_cbp.set_xlabel('# PSAMs')
        ax_cbp.set_xticklabels("12345")
        ax_ebp.axhline(0, color='k', linewidth=.5)
        ax_cbp.axhline(0, color='k', linewidth=.5)
        # ax_cbp.set_xlabel('RNA-binding domain')
        # ax_cbp.set_xticklabels(domains)
        ax_cbp.set_ylabel(u'Δ(correlation)')
        # ax_cbp.set_ylim(-.05, .4)
        # ax_err.set_yticks([1e-3, 1e-2,1e-1])
        # ax_err.set_yticklabels([0.001, 0.01, 0.1])
        # ax_corr.set_yticks([.6,.7,.8,.9,1.])
        # ax_corr.set_yticklabels([.6,.7,.8,.9,1.])

        plt.tight_layout()
        plt.savefig('PSAM_set_grouped.pdf')
        plt.close()

    def oneconc(self, corr_ymin=None, corr_ymax=None):
        assert "oneconc" in self.variants
        fig, ((ax_err, ax_es), (ax_corr, ax_cs)) = plt.subplots(
            2, 2, 
            gridspec_kw=dict(width_ratios=[1, 1]), 
            figsize=(3.5, 2.5), 
            sharex='col'
        )

        comparable_rbps, mean_errs, mean_corrs = self.left_out_sample_data()
        assert len(comparable_rbps) == len(mean_errs) == len(mean_corrs)

        full_err, single_err = mean_errs.T
        self.delta_barplot(
            ax_err, 
            full_err, 
            single_err,
            name = "left-out model error",
            func = lambda y0, y1 : np.log2(y0/y1), 
            ax_test=ax_es,
            xticks=[10, 30, 50]
        )
        ax_err.set_xlabel('RBP index')
        ax_err.set_ylabel("left-out\nmodel error log2 ratio")
        ax_es.set_ylabel("# RBPs")
        ax_es.set_xticks([0, 1])
        ax_es.set_xticklabels(["single > multi", "single <= multi"])


        full_corr, single_corr = mean_corrs.T
        self.delta_barplot(
            ax_corr, 
            full_corr, 
            single_corr,
            name = "left-out correlation",
            ax_test=ax_cs,
            xticks=[10, 30, 50]
        )
        ax_corr.set_xlabel('RBP index')
        ax_corr.set_ylabel(u"Δ(correlation)")
        ax_corr.set_yticks([0,0.3,0.6])
        ax_corr.set_yticklabels([0,0.3,0.6])
        if corr_ymax:
            ax_corr.set_ylim(corr_ymin, corr_ymax)

        ax_cs.set_ylabel("# RBPs")
        ax_cs.set_xticks([0, 1])
        ax_cs.set_xticklabels(["lower", "higher"], rotation=90)

        plt.tight_layout()
        sns.despine()
        plt.savefig("oneconc_vs_all.pdf")
        plt.close()


    def variant_plot(
        self, 
        models=["std", "xsrbp", "linocc", "dumb", ], 
        labels = ["mass-action", "excess RBP", "linear occ.", "excess RBP +\nlinear occ."], 
        symbols = ['.', '^', '.', '*', '.'],
        plot_kw = [{'zorder' : 3000},{},{},{},{}],
        corr_ymin=None, corr_ymax=None
        ):

        for m in models:
            assert m in self.variants

        # for j in range(len(labels) - 1):
        #     print "top proteins that benefit from", labels[j + 1]
        #     for i in err_ratios[j].argsort()[:5]:
        #         r = err_ratios[j][i]
        #         if j == 3:
        #             rbp = multi_psam_rbps[i]
        #         else:
        #             rbp = rbps[i]
        #         if r < 1:
        #             print rbp, "err_ratio", r, "corr_ratio", corr_ratios[j][i]

        # mass action, xsRBP, linocc, xs+lin, single PSAM
        colors = ['black', '#f98e23', '#1f77b4', '#c44802', 'teal']
        # symbols = ['.', '^', 'v', 's']

        for variant in ['xsrbp', 'linocc', 'dumb']:
            fig, ((ax_err, ax_es), (ax_corr, ax_cs)) = plt.subplots(
                2, 2, 
                gridspec_kw=dict(width_ratios=[1, 1]), 
                figsize=(3.5, 2.5), 
                sharex='col'
            )
            self.delta_barplot(
                ax_err, 
                self.err_std, 
                getattr(self, f"err_{variant}"),
                name = variant,
                func = lambda y0, y1 : np.log2(y0/y1),
                ax_test=ax_es
            )
            self.delta_barplot(
                ax_corr, 
                self.corr_std, 
                getattr(self, f"corr_{variant}"),
                name = variant,
                ax_test=ax_cs
            )
            ax_err.set_xlabel('RBP index')
            ax_err.set_ylabel("model error log2 ratio")

            ax_corr.set_xlabel('RBP index')
            ax_corr.set_ylabel(u"Δ(correlation)")
            if corr_ymax:
                ax_corr.set_ylim(corr_ymin, corr_ymax)


            # ax_corr.set_yticks([0,0.3,0.6])
            # ax_corr.set_yticklabels([0,0.3,0.6])

            sns.despine()
            plt.tight_layout()

            plt.savefig(f"{variant}_impact.pdf")

        return

        # fig, ((ax_err, ax_ebp), (ax_corr, ax_cbp)) = plt.subplots(
        #     2, 2, 
        #     gridspec_kw=dict(width_ratios=[3, 2]),
        #     figsize=(3., 2.5),
        #     sharex='col'
        # )

        # lerr = np.log10(np.array([self.err_std, self.err_xsrbp, self.err_linocc, self.err_dumb])) #
        # bplot = ax_ebp.boxplot(
        #     list(lerr),
        #     labels=labels,  # will be used to label x-ticks
        #     **bpkw
        # )
        # add_significance(lerr, labels, y=-.5, name="model error differences", ax=ax_ebp)
        # ax_ebp.set_xticks([])
        # # ax_ebp.set_ylabel("model error")
        
        # y = np.linspace(-3, -1, 3)
        # ax_ebp.set_yticks(y)
        # # ax_ebp.set_ylim(-3, 0)
        # ax_ebp.set_yticklabels(10.0**y)

        # corr = np.array([self.corr_std, self.corr_xsrbp, self.corr_linocc, self.corr_dumb])
        # bp2 = ax_cbp.boxplot(
        #     list(corr),
        #     labels=labels,  # will be used to label x-ticks
        #     **bpkw
        # )
        # add_significance(corr, labels, y=1., name="correlation differences", ax=ax_cbp)
        # ax_cbp.set_xticks([])
        # # ax_cbp.set_ylabel("6-mer correlation")
        # # plt.axhline(0, color='gray', linewidth=.5, linestyle='dashed')
        # ax_cbp.set_yticks([.6, .8, 1.])
        # ax_cbp.set_yticklabels(["0.6", "0.8", '1'])

        # # fill with colors
        # for bplot in (bplot, bp2):
        #     for patch, color in zip(bplot['boxes'], colors):
        #         patch.set_facecolor(color)

        # regressions = []
        # import scipy.stats
        # x = np.arange(len(self.err_std))
        # I = np.argsort(self.err_std)
        # for i, err in enumerate([self.err_std, self.err_xsrbp, self.err_linocc, self.err_dumb]):
        #     ax_err.semilogy(x, err[I], symbols[i], color=colors[i], label=labels[i], alpha=1., markersize=3, markeredgewidth=0, **plot_kw[i])

        # ax_err.set_xticks([10,30,50,70])
        # ax_err.set_xticklabels([10,30,50,70])
        # ax_err.set_yticks(10**y)
        # ax_err.set_yticklabels(10.0**y)

        # ax_err.set_xlabel("RBP index")
        # ax_err.set_ylabel("model error")

        # x = np.arange(len(self.corr_std))
        # I = np.argsort(self.corr_std)
        # for i, corr in enumerate([self.corr_std, self.corr_xsrbp, self.corr_linocc, self.corr_dumb]):
        #     ax_corr.plot(x, corr[I], symbols[i], color=colors[i], label=labels[i], markersize=3, markeredgewidth=0, **plot_kw[i])

        # ax_corr.legend(loc='best')
        # ax_corr.set_xlabel("RBP index")
        # ax_corr.set_ylabel("6-mer correlation")
        # plt.tight_layout()
        # sns.despine()
        # plt.savefig("model_comparison.pdf")
        # plt.close()

    def mdl_comp_struct_plot(self, corr_ymin=None, corr_ymax=None):
        fig, ((ax_err, ax_es), (ax_corr, ax_cs), ) = plt.subplots(
            2, 2, 
            gridspec_kw=dict(width_ratios=[1, 1]), 
            figsize=(3.5, 2.5), 
            sharex='col'
        )
        labels = ['PSAMs only', 'PSAMs + footprint']

        self.delta_barplot(
            ax_err, 
            self.err_std_fp,
            self.err_std, 
            name = "footprint",
            func = lambda y0, y1 : np.log2(y0/y1),
            ax_test=ax_es
        )
        self.delta_barplot(
            ax_corr, 
            self.corr_std_fp,
            self.corr_std, 
            name = "footprint",
            ax_test=ax_cs
        )
        ax_err.set_xlabel('RBP index')
        ax_err.set_ylabel("model error log2 ratio")

        ax_corr.set_xlabel('RBP index')
        ax_corr.set_ylabel(u"Δ(correlation)")
        if corr_ymax:
            ax_corr.set_ylim(corr_ymin, corr_ymax)


        # # plotting the one-marker-per-RBP panels
        # x = np.arange(len(self.err_std))
        # I = np.argsort(self.err_std)
        # ax_err.set_yscale('log')
        # ax_err.plot(x, self.err_std[I], '.k', label="PSAMs only", markersize=3, markeredgewidth=0)
        # ax_err.plot(x, self.err_std_fp[I], '^r', label="PSAMs + footprint", markersize=2, markeredgewidth=0)

        # x = np.arange(len(self.corr_std))
        # I = np.argsort(self.corr_std)
        # ax_corr.plot(x, self.corr_std[I], '.k', label="PSAMs only", markersize=3, markeredgewidth=0)
        # ax_corr.plot(x, self.corr_std_fp[I], '^r', label="PSAMs + footprint", markersize=2, markeredgewidth=0)

        # # # lerr = np.log10(np.array([self.err_std, self.err_std_fp])) #
        # # # bplot = ax_ebp.boxplot(
        # # #     list(lerr),
        # # #     # labels=labels,  # will be used to label x-ticks
        # # #     **bpkw
        # # # )
        # # # add_significance(lerr, labels, y=-.5, name="model error", ax=ax_ebp)

        # # # corr = [self.corr_std, self.corr_std_fp]
        # # # bp2 = ax_cbp.boxplot(
        # # #     list(corr),
        # # #     labels=labels,  # will be used to label x-ticks
        # # #     **bpkw
        # # # )
        # # # add_significance(corr, labels, y=1., name="6mer correlation", ax=ax_cbp)
        # # # # fill with colors
        # # # for bplot in (bplot, bp2):
        # # #     for patch, color in zip(bplot['boxes'], ['k', 'r']):
        # # #         patch.set_facecolor(color)
        # N = len(I)
        # efpgt = (self.err_std_fp > self.err_std).sum()
        # efple = N - efpgt

        # cfpgt = (self.corr_std_fp > self.corr_std).sum()
        # cfple = N - cfpgt
        # print(f"model error: PSAM+FP > PSAM={efpgt} PSAM+FP <= PSAM={efple} N={N}")
        # print(f"correlation: PSAM+FP > PSAM={cfpgt} PSAM+FP <= PSAM={cfple} N={N}")
        
        # from scipy.stats import binom_test
        # bte = binom_test(efpgt, N)
        # print("binomial test P-value for PSAM+FP vs PSAM error", bte)
        # ax_es.bar(
        #     [0, 1],
        #     [efpgt, efple],
        #     color=['r', 'k']
        # )
        # btc = binom_test(cfpgt, N)
        # print("binomial test P-value for PSAM+FP vs PSAM error", btc)
        # ax_cs.bar(
        #     [0, 1],
        #     [cfpgt, cfple],
        #     color=['r', 'k']
        # )

        # # legends and labels

        # # ax_err.legend(loc='lower center', ncol=1)
        # # ax_corr.legend(loc='lower center', ncol=1)
        # ax_err.set_xlabel("RBP index")
        # ax_err.set_ylabel('model error')
        # ax_err.set_yticks([1e-3, 1e-2,1e-1])
        # ax_err.set_yticklabels([0.001, 0.01, 0.1])
        # ax_corr.set_xticks([10,30,50,70])
        # ax_corr.set_xticklabels([10,30,50,70])
        # ax_corr.set_xlabel("RBP index")
        # ax_corr.set_ylabel('6-mer correlation')
        # # ax_cs.set_ylim(-.3, .1)
        # ax_corr.set_yticks([.6,.7,.8,.9,1.])
        # ax_corr.set_yticklabels([.6,.7,.8,.9,1.])

        # ax_es.set_ylabel("# RBPs")
        # ax_cs.set_ylabel("# RBPs")
        # ax_es.set_xticks([0, 1])
        # ax_cs.set_xticks([0, 1])
        # # ax_es.set_xticklabels(["PSAM+FP > PSAM", "PSAM+FP <= PSAM"], rotation=90)
        # ax_cs.set_xticklabels(["PSAM+FP >\nPSAM", "PSAM+FP <=\nPSAM"], rotation=90)


        # # ax_cbp.set_xticks([])
        # # ax_cbp.set_yticks([.6, .8, 1.])
        # # ax_cbp.set_yticklabels(["0.6", "0.8", '1'])

        # # y = np.linspace(-3, -1, 3)
        # # ax_ebp.set_xticks([])
        # # ax_ebp.set_yticks(y)
        # # ax_ebp.set_yticklabels(10.0**y)

        sns.despine()
        plt.tight_layout()
        plt.savefig("footprint_vs_seqonly.pdf")
        plt.close()

    def footprint_effect(self):
        fig, (ax_kd_change, ax_kd_vs_GC) = plt.subplots(
            2, 1, 
            # gridspec_kw=dict(width_ratios=[3, 1]), 
            figsize=(2.5, 2.5), 
            # sharex='col'
        )
        kd_no = np.array([self.res_std[rbp].nostruct.Kd for rbp in self.rbps])
        kd_struct = np.array([self.res_std[rbp].full.Kd for rbp in self.rbps])
        kd_lfc = np.log2(kd_struct/kd_no)
        ax_kd_change.hist(kd_lfc, bins=10)
        ax_kd_change.set_xlabel(r'$\log_2 \frac{K_d^{fp}}{K_d}$')
        ax_kd_change.set_ylabel('count')

        # statistics
        gc = np.array([self.params_GC[rbp].mean() for rbp in self.rbps])
        plot_and_regress(gc, kd_lfc)
        ax_kd_vs_GC.plot(gc, kd_lfc, '.')
        ax_kd_vs_GC.set_xlabel('mean GC preference')
        ax_kd_vs_GC.set_ylabel(r'$\log_2 \frac{K_d^{fp}}{K_d}$')

        sns.despine()
        plt.tight_layout()
        plt.savefig("footprint_Kd_effects.pdf")
        plt.close()

    def footprint_plot(self):
        pattern = "/home/mjens/engaging/RBNS/{rbp}/cska/{variant}/footprint/parameters.tsv"
        params_dict = load_fp_calibrated_params(pattern, self.rbps, "z4t75p01k99fix", plot=True)
        self.fp_cal_params = params_dict
        self.params_GC = {}

        rel_errors = []
        acc_scales = []
        acc_k = []
        acc_shift = []
        GC = []
        OV = []

        mean_rel_errors = []
        mean_acc_scales = []
        mean_GC = []
        min_overlap = []
        mean_overlap = []
        mean_acc_k = []

        n_all_improve = 0
        n_all_bad = 0
        n_overlap = 0
        motif_rbps = []
        for rbp in self.rbps:
            re = []
            sc = []
            gc = []
            ov = []
            k = []
            s = []
            print(rbp)
            for par in params_dict[rbp].param_set:
                re.append(par.rel_err)
                sc.append(par.acc_scale)
                gc.append(par.as_PSAM().fraction_GC)
                k.append(par.acc_k)
                s.append(par.acc_shift)
                overlap = min(par.acc_k + min(par.acc_shift, 0), min((par.k - par.acc_shift), par.acc_k))
                print(f"acc_k={par.acc_k}, acc_shift={par.acc_shift}, n={par.k} -> overlap={overlap}nt")
                ov.append(overlap/float(par.acc_k))

            re = np.array(re)
            sc = np.array(sc)
            gc = np.array(gc)
            ov = np.array(ov)
            k = np.array(k)
            s = np.array(s)

            rel_errors.append(re)
            acc_scales.append(sc)
            GC.append(gc)
            self.params_GC[rbp] = gc

            OV.append(ov)
            acc_k.append(k)
            acc_shift.append(s)

            motif_rbps.extend([rbp] * len(re))

            mean_rel_errors.append(re.mean())
            mean_acc_scales.append(sc.mean())
            mean_GC.append(gc.mean())
            min_overlap.append(ov.min())
            mean_overlap.append(ov.mean())
            mean_acc_k.append(k.mean())

            if (re < 1).all():
                n_all_improve += 1
            
            if (re >= 1).all():
                print("all bad", rbp)
                n_all_bad += 1
            
            if (ov >= .5).all():
                n_overlap += 1

        mean_rel_errors = np.array(mean_rel_errors)
        mean_acc_scales = np.array(mean_acc_scales)
        mean_GC = np.array(mean_GC)
        min_overlap = np.array(min_overlap)
        mean_overlap = np.array(mean_overlap)
        mean_acc_k = np.array(mean_acc_k)

        print(mean_rel_errors)
        print(mean_acc_scales)
        print(mean_GC)
        print(mean_acc_k)
        print(mean_overlap)

        all_rel_err = np.concatenate(rel_errors)
        all_acc_k = np.concatenate(acc_k)
        all_acc_s = np.concatenate(acc_shift)
        all_acc_a = np.concatenate(acc_scales)
        all_ov = np.concatenate(OV)
        all_GC = np.concatenate(GC)

        MIN_a = .05
        MIN_ov = .2
        M = (all_acc_a > MIN_a) & (all_ov > MIN_ov) # (all_rel_err < .8) &
        print(M.sum(), "motifs make the cut out of", len(M))
        motif_rbps = np.array(motif_rbps)
        from collections import defaultdict
        failcount = defaultdict(int)
        for r in motif_rbps[~M]:
            failcount[r] += 1
        
        print("RBPs with failed motifs", failcount)

        fig, ((ax_ov_hist, ax_hist_scale, ax_hist_err, ), (ax_k_shift, ax_k_scale, ax_err_scale), (ax31, ax32, ax_hist_k)) = plt.subplots(
            3, 3, 
            figsize=(6, 5),
            # gridspec_kw=dict(width_ratios=[1, 1]),
        )
        ax_hist_err.hist([all_rel_err[M], all_rel_err[~M]], bins=20, histtype='stepfilled', stacked=True)
        # ax_hist_err.hist(all_rel_err[~M], bins=20, histtype='stepfilled')
        ax_hist_err.set_xlabel(r'$P_{unpaired}$ error ratio')
        ax_hist_err.set_ylabel('# motifs')

        ax_hist_scale.hist([all_acc_a[M], all_acc_a[~M]], bins=20, histtype='stepfilled', stacked=True)
        ax_hist_scale.set_xlabel('FP scale')
        ax_hist_scale.axvline(MIN_a, color='r')
        ax_hist_scale.set_ylabel('# motifs')

        size_bins = np.arange(4,13) + .5
        # kcount = np.bincount(all_acc_k, minlength=14)
        ax_hist_k.hist([all_acc_k[M], all_acc_k[~M]], bins=size_bins, histtype='stepfilled', stacked=True)
        ax_hist_k.set_xlabel('FP size [nt]')
        ax_hist_k.set_xticks([6,8,10,12])
        ax_hist_k.set_ylabel('# motifs')


        print(f"RBPs with fp working {n_all_improve}, not working {n_all_bad}. overlap >= .5 for every motif {n_overlap}")

        I = mean_rel_errors.argsort()[::-1]
        for i in I:
            print(self.rbps[i], mean_acc_k[i], mean_acc_scales[i], mean_rel_errors[i], rel_errors[i], mean_overlap[i])


        m = (mean_rel_errors < .8) & (mean_acc_scales < .8) & (mean_overlap > .3)
        # ax_k_shift.plot(all_acc_k[M], all_ov[M], '.', alpha=.1)
        h = ax_k_shift.hist2d(all_acc_k[M], all_ov[M], bins=[size_bins, np.linspace(0,1,10)])
        plt.colorbar(h[3], ax=ax_k_shift, fraction=.1, shrink=.5, label='# motifs')
        ax_k_shift.set_xticks([6,8,10,12])
        ax_k_shift.set_xticklabels([6,8,10,12])
        ax_k_shift.set_xlabel('FP size [nt]')
        ax_k_shift.set_ylabel('FP-motif overlap')
        # ax_k_shift.colorbar()

        plot_and_regress(all_acc_k[M], all_acc_a[M], ax=ax_k_scale, alpha=.1, logy=True)
        # plot_and_regress(all_rel_err[M], all_acc_a[M], ax=ax_err_scale, logy=False)
        ax_k_scale.legend(loc='upper center')
        ax_k_scale.set_xlabel('FP size [nt]')
        ax_k_scale.set_xticks([6,8,10,12])
        ax_k_scale.set_ylabel('FP scale [log]')


        plot_and_regress(all_rel_err[M], all_acc_a[M], ax=ax_err_scale, logy=True)
        ax_err_scale.legend(loc='upper right')
        ax_err_scale.set_xlabel(r'$P_{unpaired}$ error ratio')
        ax_err_scale.set_ylabel('FP scale [log]')

        plot_and_regress(all_GC[M], all_rel_err[M], ax=ax31)
        ax31.legend()
        ax31.set_xlabel("GC")
        ax31.set_ylabel("rel_err")

        plot_and_regress(all_GC[M], all_acc_a[M], ax=ax32, logy=True)
        ax32.legend()
        ax32.set_xlabel("GC")
        ax32.set_ylabel('FP scale [log]')

        ax_ov_hist.hist([all_ov[M], all_ov[~M]], bins=10, histtype='stepfilled', stacked=True)
        ax_ov_hist.set_xlabel('FP-motif overlap')
        ax_ov_hist.set_ylabel('# motifs')
        ax_ov_hist.axvline(MIN_ov, color='r')

        fig.tight_layout()
        fig.savefig('fp_hists.pdf')
        plt.close()


from cska import dominguez_rbps as dom_rbps
# dom_rbps.pop(dom_rbps.index('HNRNPA0'))
rbps = np.array(dom_rbps)



i = (rbps == 'HNRNPA0').argmax()
rbps = list(rbps)
rbps.pop(i)
# print rbps
print(len(rbps), "RBPs are being considered")
pattern = "/home/mjens/engaging/RBNS/{rbp}/cska/{variant}/seed/initial.tsv"
# n_PSAM_significance(pattern, rbps, "z4t75p01k99fix", plot=True)
# sys.exit(0)
# # n_PSAM_significance(pattern, rbps, "sgd", plot=True)
# # n_PSAM_significance(pattern, rbps, "CI", plot=True)
# # n_PSAM_significance(pattern, rbps, "seed_z4_thresh_8", plot=True)
# # n_PSAM_significance(pattern, rbps, "seed_z5_thresh_9")
# # n_PSAM_significance(pattern, rbps, "seed_z5_thresh_82")
# # n_PSAM_significance(pattern, rbps, "seed_z5_thresh_84")
# # n_PSAM_significance(pattern, rbps, "seed_z6_thresh_9")

# # n_PSAM_significance(pattern, rbps, "seed_z5_thresh_85")
# # n_PSAM_significance(pattern, rbps, "seed_z5.5_thresh_85")
# # n_PSAM_significance(pattern, rbps, "seed_z6_thresh_85")
# # n_PSAM_significance(pattern, rbps, "seed_z5.25_thresh_85")
# # n_PSAM_significance(pattern, rbps, "seed_z5.75_thresh_85")
# # n_PSAM_significance(pattern, rbps, "seed_z5.5_thresh_84")
# # n_PSAM_significance(pattern, rbps, "seed_z5.5_thresh_86")

# # n_PSAM_significance(pattern, rbps, "seed_z5_thresh_85_m10")
# # n_PSAM_significance(pattern, rbps, "seed_z5.75_thresh_85_m10")

# # n_PSAM_significance(pattern, rbps, "sgd_CI_z4_thresh_75", plot=True)
# # n_PSAM_significance(pattern, rbps, "sgd_CI_z4_thresh_75_pseudo.01", plot=True)
# # n_PSAM_significance(pattern, rbps, "sgd_CI_z4_thresh_75_pseudo.01_nn", plot=True)
# # n_PSAM_significance(pattern, rbps, "sgd_CI_z4_thresh_75_pseudo.10", plot=True)


# redo8 = False
# rbase = "std.8"
# rbase = "std.8.sgd"
# rbase = "sgd"
rbase = "z4t75p01k99fix"
redo8 = False
# d_std, res_std = load_or_make("RBNS/*/cska/" + rbase, redo=redo8  )
# n_psams = np.array([res_std[rbp].nostruct.n_PSAM for rbp in rbps])

# # n_PSAM_plot(n_psams)

# d_s, res_s = load_or_make("RBNS/*/cska/" + rbase + ".1", redo=redo8  )
# d_o, res_o = load_or_make("RBNS/*/cska/" + rbase + ".s", redo=redo8  )
# d_xsrbp, res_xsrbp = load_or_make("RBNS/*/cska/" + rbase + ".xsrbp", redo=redo8)
# d_linocc, res_linocc = load_or_make("RBNS/*/cska/" + rbase + ".linocc", redo=redo8)
# d_dumb, res_dumb = load_or_make("RBNS/*/cska/" + rbase + ".dumb", redo=redo8)

MC = ModelComparisons(variant_dict=dict(
    std = load_or_make("RBNS/*/cska/" + rbase, redo=redo8),
    single = load_or_make("RBNS/*/cska/" + rbase + ".1", redo=redo8),
    oneconc = load_or_make("RBNS/*/cska/" + rbase + ".s", redo=redo8),
    xsrbp = load_or_make("RBNS/*/cska/" + rbase + ".xsrbp", redo=redo8),
    linocc = load_or_make("RBNS/*/cska/" + rbase + ".linocc", redo=redo8),
    dumb = load_or_make("RBNS/*/cska/" + rbase + ".dumb", redo=redo8),
    std_eval = load_or_make("RBNS/*/cska/" + rbase + "_eval", redo=redo8),
    single_eval = load_or_make("RBNS/*/cska/" + rbase + ".s_eval", redo=redo8),
), rbps=rbps)

for rbp, n in zip(MC.rbps, MC.n_psams):
    print(rbp, n)
# MC.left_out_single_conc_plot() ## This one is deprecated!

# cymin = -.1
# cymax = .9
cymin = None
cymax = None

MC.single_multi_PSAMS(corr_ymin=cymin, corr_ymax=cymax)
MC.oneconc(corr_ymin=cymin, corr_ymax=cymax)
MC.mdl_comp_struct_plot(corr_ymin=cymin, corr_ymax=cymax)
MC.variant_plot(corr_ymin=cymin, corr_ymax=cymax)
# MC.footprint_plot()
# MC.footprint_effect()


# d_std, res_std = load_or_make("RBNS/*/cska/std.72", redo=False  )
# d_7, res_7 = load_or_make("RBNS/*/cska/std.7", redo=False  )
# d_ci, res_ci = load_or_make("RBNS/*/cska/CI", redo=False)
# d_xsrbp, res_xsrbp = load_or_make("RBNS/*/cska/std")
# d_s, res_s = load_or_make("RBNS/*/cska/single")
# d_s, res_s = load_or_make("RBNS/*/cska/std.72.1")
# d_s7, res_s7 = load_or_make("RBNS/*/cska/std.7.1")
# d_o, res_o = load_or_make("RBNS/*/cska/std.8.s")

sys.exit(0)

d_s_eval, res_s_eval = load_or_make("RBNS/*/cska/" + rbase + ".s_eval", redo=redo8  )
d_std_eval, res_std_eval = load_or_make("RBNS/*/cska/" + rbase + "_eval", redo=redo8  )


# runs = {
#     # 'std.7' : d_7,
#     'sgd' : d_std,
#     'sgd.1' : d_s,
#     'sgd.s' : d_o,
#     'sgd.xsrbp' : d_xsrbp,
#     'sgd.linocc' : d_linocc,
#     # 'std.7.1' : d_s7,
#     # 'std.72' : d_std,
#     # 'std.72.1' : d_s,
#     'CI' : d_ci,
# }


# compare_runs(runs, rbps)
# sys.exit(0)

err_std, corr_std = np.array([d_std[rbp] for rbp in rbps]).T
err_s, corr_s = np.array([d_s[rbp] for rbp in rbps]).T
err_o, corr_o = np.array([d_o[rbp] for rbp in rbps]).T

err_xs, corr_xs = np.array([d_xsrbp[rbp] for rbp in rbps]).T
err_lo, corr_lo = np.array([d_linocc[rbp] for rbp in rbps]).T
err_d, corr_d = np.array([d_dumb[rbp] for rbp in rbps]).T

n_psams = np.array([res_std[rbp].nostruct.n_PSAM for rbp in rbps])
multi_psam_rbps = rbps[n_psams > 1]
n_conc = np.array([len(res_std[rbp].nostruct.rbp_conc) for rbp in rbps])
# print n_psams
# print np.array(rbps)[n_psams > 1]

err_nostruct = np.array([res_std[rbp].nostruct.err_final for rbp in rbps])
err_full = np.array([res_std[rbp].full.get("err_final", np.NaN) for rbp in rbps])

print("model errors after structure aware gradient descent", err_full)
corr_nostruct = np.array([res_std[rbp].nostruct.get("best_corr", np.NaN) for rbp in rbps])
corr_full = np.array([res_std[rbp].full.get("best_corr", np.NaN) for rbp in rbps])


# left_out_single_conc_plot()
# n_PSAM_plot(n_psams)
# GC_acc_scale_plot()
# mdl_comp_struct_plot()
# mdl_comparison_plot()

# 	#print "\t".join([rbp,str(1./float(score))])
# 	print "\t".join([rbp, str(rerr), str(ferr), str(best_corr), str(steps)])

