import os, sys
import numpy as np
import matplotlib.pyplot as pp
from scipy.stats import spearmanr, pearsonr
import scipy.stats
from cska.report import density_scatter_plot
import cska.ska_kmers as cyska
from byo.io import fasta_chunks
import os, sys, logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.INFO)


class MutualInformationScore(object):
    def __init__(self, X, Y,n=10, n_permut = 100):
        self.X = X
        self.Y = Y
        self.n = n
        self.n_permut = n_permut
        self.xbins, self.Xd, self.Xf = self.make_eq_bins(X, n=n)
        self.xbins, self.Yd, self.Yf = self.make_eq_bins(Y, n=n)
        
        self.joint = MutualInformationScore.joint_freq(self.Xd, self.Yd, n)
        # print self.joint.shape
        self.indep = np.outer(self.Xf, self.Yf)
        # print self.indep.shape
        self.MI = MutualInformationScore.mutual_information(self.joint, self.indep)

        self.MI_permut = []
        N = len(self.Xd)
        for i in xrange(n_permut):
            perm = np.random.permutation(N)
            xd = self.Xd[perm]
            joint = MutualInformationScore.joint_freq(xd, self.Yd, n)
            self.MI_permut.append(MutualInformationScore.mutual_information(joint, self.indep))
        
        self.MI_permut = np.array(self.MI_permut)
        
        # ad hoc p-value
        s = np.std(self.MI_permut)
        m = np.mean(self.MI_permut)
        self.z = (self.MI - m)/s
        self.p_value = scipy.stats.norm.sf(self.z)

    
    @staticmethod
    def mutual_information(joint, indep):
        return (joint * np.log2(joint/indep)).sum()

    @staticmethod
    def joint_freq(xd, yd, n):
        joint = np.ones((n, n), dtype=np.float32)
        for x,y in zip(xd, yd):
            joint[x,y] += 1

        joint /= float(joint.sum())
        return joint
    
    def heatmap_plot(self, fname):
        pp.figure()
        pp.pcolor(self.joint, cmap='viridis')
        pp.colorbar(orientation='horizontal')
        pp.savefig(fname)
        pp.close()

    def dist_plot(self, fname):
        pp.figure()
        pp.hist(self.MI_permut, lw=2, histtype='step', bins=self.n_permut/self.n)
        pp.axvline(self.MI)
        pp.savefig(fname)
        pp.close()

    def make_eq_bins(self, x, n=10):
        I = x.argsort()
        N = len(I)

        bp_i = np.linspace(0,N-1, num=n)

        bins = [x[I[int(bp)]] for bp in bp_i]
        d = np.digitize(x, bins, right=True) # discretized version
        n = np.bincount(d) + 1
        f = n / float(n.sum())
        # print bins, f

        return bins, d, f




class nsRBNSOligos(object):

    def __init__(self, fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT', xtalk_file='blast/results.out'):
        self._seqs_raw = []
        self._fa_ids_raw = []
        self._index = {}
        self._seqs = {}

        self.nt_freqs = []
        def GC_content(seq):
            gc = 0
            for s in seq:
                if s == 'G' or s == 'C':
                    gc += 1
            return gc / float(len(seq))

        self.entropy = []
        def nt_entropy(seq):
            S = seq.upper().replace('U','T')
            counts = np.array([S.count('A'), S.count('C'), S.count('G'), S.count('T')])
            f = counts / float(counts.sum())
            self.nt_freqs.append(f)

            return - np.where(f > 0, f * np.log2(f), 0).sum()

        self.l5 = len(adap5)
        self.l3 = len(adap3)

        self.adap5 = adap5
        self.adap3 = adap3
        self.fa_name = fa_name
        for i, (fa_id, seq) in enumerate(fasta_chunks(file(fa_name))):
            self._seqs_raw.append(seq)
            self._fa_ids_raw.append(fa_id)
            self._seqs[fa_id] = seq
            self._index[fa_id] = i
            self.entropy.append(nt_entropy(seq))

        self.N = len(self._seqs_raw)
        self.SEQ = [s[self.l5:-self.l3] for s in self._seqs_raw] 
        self.nt_freqs = np.array(self.nt_freqs)
        self.GC = self.nt_freqs[:,[1,2]].sum(axis=1)
        self.entropy = np.array(self.entropy)

        self.xtalk, self.xtalk_score = self.xtalk_from_blast(fname=xtalk_file)

    def xtalk_from_blast(self, fname='blast/results.out'):
        N = self.N
        xtalk = np.zeros( (N,N), dtype=np.float32)
        print "loading"

        for line in file(fname):
            parts = line.split('\t')
            query, subject, score = parts[:3]
            i = self._index[query]
            j = self._index[subject]

            xtalk[i,j] += float(score)

        rowsum = xtalk.sum(axis=1)
        ind = rowsum > 800
        xtalk = xtalk[ind,:][:,ind]
        # xtalk = xtalk[:100,:100]
        return xtalk, rowsum

    def xtalk_plot(self, fname='xtalk.pdf'):
        xtalk = self.xtalk
        print xtalk.shape
        print xtalk.min(), xtalk.max(), xtalk.argmax()
        print "plotting"
        pp.figure()
        pp.imshow(np.log10(xtalk), cmap='viridis', interpolation='none')
        pp.colorbar()
        pp.savefig(fname)
        pp.close()

    def get_SPA_model(self, k=7, rna_conc=1000., rbp_conc=[25.,125.,625.]):
        from cska.reads import RBNSReads
        import cska.spa

        path = os.path.dirname(self.fa_name)
        reads = RBNSReads.from_seqs(self.SEQ, fname=self.fa_name, rbp_name='nsRBNS', rna_conc=rna_conc, acc_storage_path='cska/acc', adap5=self.adap5, adap3=self.adap3)
        # reads.seqm
        mdl = cska.spa.SPAModel(reads, k, rbp_conc, n_subsample=0)
        return mdl



def phist(*argc, **kwargs):
    pp.hist(*argc, lw=2, bins=100, histtype='step', normed=True, cumulative=True, **kwargs)


class nsRBNSExperiment(object):
    def __init__(self, nsrbns, fcount_matrix, faffinities, name='nsRBNS', rbp_conc = [25.,125.,625.], pseudo=10, skip_xtalk=True, skip_low=True):
        self.name = name
        self.ns = nsrbns
        
        self.fcount_matrix = fcount_matrix
        self.indices, self.counts = self.load_counts(fcount_matrix, skip_xtalk=skip_xtalk, skip_low=skip_low)

        self.N = self.counts.sum(axis=1)
        self.scale = self.N[1:]/self.N[0]
        self.enr = (self.counts[1:,:] + pseudo) / (self.counts[0,:] + pseudo) * self.scale[:,np.newaxis]
        self.frac = self.counts / self.N[:,np.newaxis]
        self.f0 = self.frac[0]
        self.rbp_conc = rbp_conc

        self.faffinities = faffinities
        self.state = self.evaluate_SPA(faffinities, rbp_conc)
        self.betas = self.optimal_betas()
        
        self.enr_expect, self.pearson, self.ppval, self.spearman, self.spval = self.prediction()

    def load_counts(self, fname, skip_xtalk=True, skip_low=True):
        counts = []
        indices = []

        for line in file(fname):
            parts = line.split('\t')
            name = parts[0]
            if skip_xtalk and (self.ns.xtalk_score[self.ns._index[name]] > 0):
                # skip cross-talking oligos!
                continue

            n0 = float(parts[1]) # freq in input
            if skip_low and n0 < 100:
                continue

            indices.append(self.ns._index[name])
            counts.append(parts[1:])
            
            s = self.ns._seqs[name]
            if not self.ns._seqs_raw[self.ns._index[name]] == s:
                print ">>>MISMATCH", name
                print "CM ",s
                print "RAW", self.ns._seqs_raw[self.ns._index[name]]

            # if (not 'GCATG' in s) and (not 'GCACG' in s):
            #     motif.append(0)
            #     # print s
            # elif 'TGCATGT' in s or 'TGCATGC' in s:
            #     motif.append(10)
            # elif 'AGCATG' in s or 'CGCATG' in s or 'TGCATG' in s:
            #     motif.append(5)
            # else:
            #     motif.append(1)
        counts = np.array(counts, dtype=float).T
        indices = np.array(indices)

        return indices, counts


    def evaluate_SPA(self, fname, rbp_conc, k=7):
        letters = 'ACGT'
        def convert(bits):
            return "".join([letters[i] for i in bits])
        mdl = self.ns.get_SPA_model(k=k, rbp_conc = rbp_conc)        

        # this selects directly form the index_matrix, NOT seqm. So seqm is invalid after subsample!?
        mdl.new_subsample(indices=self.indices)
        if fname.endswith('rnacompete'):
            mdl.parameters.load_rbpbind(fname)
        else:
            mdl.parameters.load(fname)

        mdl.params[:mdl.nA]
        state = mdl.evaluate(mdl.params)

        return state

    def noaffinity_analysis(self, perc=10.):
        thresh = np.percentile(self.state.Z1, perc)
        I = self.state.Z1 < thresh

        from sklearn import linear_model
        from sklearn import preprocessing
        import pandas as pd
        from sklearn.metrics import mean_squared_error, r2_score

        acc = np.log(self.state.mdl.reads.acc_storage.get_raw(11).acc[self.indices].mean(axis=1))
        print acc.shape
        A,C,G,T = self.ns.nt_freqs[self.indices].T
        GC = C+G
        AT = A+T
        res = []
        GC_score = preprocessing.scale(GC/(1-GC))
        Z1 = preprocessing.scale(np.log(self.state.Z1))
        entropy = 2 - self.ns.entropy[self.indices]
        for conc, enr in zip(self.rbp_conc, self.enr):
            P = conc * .001
            n = 1.
            data = np.array([
                A, C, G, T, GC_score, GC_score**2,
                np.log(self.f0),
                entropy,
                # np.log((P*self.state.Z1)**n/(1 + (P*self.state.Z1)**n)),
                Z1,
                Z1**2,
                Z1**3,
                acc,
            ])
            data = preprocessing.scale(data.T)
            df = pd.DataFrame(data=data, columns = ['A','C','G','T','GC', 'GC2', 'f0', 'entropy','binding','bind2', 'bind3', 'mean_acc'])
            # df = pd.DataFrame(data=data, columns = ['A','C','G','T','GC', 'f0', 'entropy','binding',])

            # reg = linear_model.LinearRegression()
            reg = linear_model.RidgeCV(alphas=[.1,.3,.5,.75,1.])
            # reg.fit(df.iloc[I], y[I])
            y = np.log2(enr)
            reg.fit(df, y)
            # predict on full data
            y_pred = reg.predict(df)
            R, p_val = spearmanr(y,y_pred)

            pp.figure()
            # pp.plot(y, y_pred, '.', label='{conc:.1f}nM: rho={R:.3f} (P < {p_val:.2e})'.format(**locals()))
            density_scatter_plot(y, y_pred,x_ref=False, label='{conc:.1f}nM: rho={R:.3f} (P < {p_val:.2e})'.format(**locals()))
            pp.xlabel('log2 nsRBNS enrichment')
            pp.ylabel('linear model prediction')
            pp.legend(loc='upper center', facecolor='white')
            pp.savefig('{conc}_scatter.pdf'.format(**locals()))
            pp.close()
            
            res = y-y_pred
            # pp.plot(df['A'], res, '.', label='A')
            # pp.plot(df['C'], res, '.', label='C')
            # pp.plot(df['G'], res, '.', label='G')
            # pp.plot(df['T'], res, '.', label='T')
            # pp.plot(df['GC'], res, '.', label='log(GC)')
            # pp.plot(df['f0'], res, '.', label='log(f0)')
            # pp.plot(df['entropy'], res, '.', label='entropy')
            
            # pp.plot(df['binding'], res, '.', label='log(Z1)')
            print "coeff", reg.coef_
            print "intercept", reg.intercept_
            print "r2 on bg", r2_score(y[I], y_pred[I])
            print ">>>>>", conc, "r2 full ", r2_score(y, y_pred)
            print "alpha", reg.alpha_

            for col in df.columns:
                pp.figure()
                pp.title('{0} residuals'.format(col))
                density_scatter_plot(df[col], res, x_ref=False, label=col)
                pp.legend(loc='upper center', facecolor='white')
                pp.savefig('{col}_{conc}_residual.pdf'.format(**locals()))
                pp.close()

            # res.append() # keep the residuals


        # pp.show()




    def optimal_betas(self):
        betas = []
        for p, obs in zip(self.state.p_bound, self.enr):
            betas.append(self.optimize_beta(p, obs))
        
        return np.array(betas)

    def affinity_selection_plot(self, fname = 'affinity_selection.pdf'):
        ### Affinity selection plot
        pp.figure()
        Z = np.log10(self.state.Z1)
        # Z -= Z.mean()
        # Z /= Z.std()
        phist(Z, color='gray', label='input')
        for i, f in enumerate(self.counts[1:]):
            phist(Z, weights = f, label="{0}nM".format(self.rbp_conc[i]))

        pp.xlabel("natural sequence affinity [1/nM] (log10)")
        pp.ylabel("rel. cumulative frequency of pull-down")
        pp.legend(loc='upper left')
        pp.savefig(fname)
        pp.close()

    def GC_bias_plot(self, fname='GC.pdf'):
        ### GC bias plot
        pp.figure()
        GC = self.ns.GC[self.indices]
        phist(GC, weights=self.counts[0], color='gray', label="input")
        for i, c in enumerate(self.counts[1:]):
            phist(GC, weights=c, label="{0}nM".format(self.rbp_conc[i]))

        pp.xlabel("G,C content of natural sequence")
        pp.ylabel("rel. cumulative frequency")
        pp.legend(loc='upper left')
        pp.savefig(fname)
        pp.close()

    def entropy_plot(self, fname='entropy.pdf'):
        ### nt entropy bias plot
        pp.figure()
        ent = self.ns.entropy[self.indices]
        phist(ent, weights=self.counts[0], color='gray', label="input")
        for i, c in enumerate(self.counts[1:]):
            phist(ent, weights=c, label="{0}nM".format(self.rbp_conc[i]))

        pp.xlabel("nt entropy of natural sequence [bits]")
        pp.ylabel("rel. cumulative frequency")
        pp.legend(loc='upper left')
        pp.savefig(fname)
        pp.close()

    def predict_R(self, p, beta):
        expect = (p + beta) * self.f0 
        expect /= expect.sum()
        expect /= self.f0
        return expect

    
    def optimize_beta(self, p, obs):

        def to_opt(beta):
            expect = self.predict_R(p, beta)
            # R = np.corrcoef(np.log2(obs), np.log2(expect))[0][1]
            R, p_value = pearsonr(np.log2(obs), np.log2(expect))

            return (R - 1)**2
        
        from scipy.optimize import minimize_scalar

        opt = minimize_scalar(to_opt, bounds=[0,1.], method='bounded')
        # print opt
        return opt.x


    def prediction(self):
        pearson = []
        ppval = []
        spearman = []
        spval = []
        enr_expect = []

        for conc, p, obs, beta in zip(self.rbp_conc, self.state.p_bound, self.enr, self.betas):
            expect = self.predict_R(p, beta)
            enr_expect.append(expect)
            
            R, pp_value = pearsonr(np.log2(obs), np.log2(expect))
            rho, p_value = spearmanr(np.log2(obs), np.log2(expect))
            pearson.append(R)
            ppval.append(pp_value)
            spearman.append(rho)
            spval.append(p_value)

        return np.array(enr_expect), np.array(pearson), np.array(ppval), np.array(spearman), np.array(spval)

    def scatter_plot(self, fname='{self.fcount_matrix}_{conc:.2f}.pdf'):
        for conc, expect, obs, R, pp_value, rho, p_value in zip(self.rbp_conc, self.enr_expect, self.enr, self.pearson, self.ppval, self.spearman, self.spval):
            pp.figure()
            pp.loglog(obs, expect, '.', label=r"conc={conc:.1f} R={R:.2f} (P={pp_value:.2e}) $\rho=${rho:.2f} (P={p_value:.2e})".format(**locals()))
            pp.legend()
            pp.savefig(fname.format(**locals()))
            pp.close()

    def heatmap_plot(self, fname='heatmap.pdf'):
        Z = self.state.Z1
        I = (-Z).argsort()
        N = len(Z)

        pp.figure(figsize=(5,7))
        pp.subplot(311)
        pp.semilogy(Z[I], color='k', label='RBNS model prediction')
        pp.xlim(0,N)
        pp.ylabel('predicted affinity [1/nM]')
        pp.xlabel('natural sequence index')
        pp.legend(loc='upper right')
        
        import matplotlib.colors as colors
        pp.subplot(312)
        data = self.enr[:,I]
        vmin, vmax = np.percentile(data, [5.,95.])
        pp.pcolor(data, cmap='viridis', norm=colors.LogNorm(vmin=vmin, vmax=vmax) )
        pp.yticks(np.arange(len(self.rbp_conc))+.5, ["{0:.1f} nM".format(c) for c in self.rbp_conc])
        pp.xticks([],[])
        pp.xlim(0,N)
        pp.colorbar(orientation='horizontal', fraction=.05, label='observed nsRBNS enrichment')

        pp.subplot(313)
        # pp.semilogy(Z[I])
        pp.ylabel('observed mean log2 enrichment')
        
        import scipy.ndimage.filters
        for conc, R in zip(self.rbp_conc, self.enr):
            lR = np.log2(R[I])
            # avg, var = moving_average(lR)
            avg = scipy.ndimage.filters.gaussian_filter1d(lR, 50.)

            pp.plot(avg, label="nsRBNS {0:.1f} nM".format(conc))
        
        pp.xlim(0,N)
        pp.legend(loc='upper right')
        pp.tight_layout()
        pp.savefig(fname)
        pp.close()

    def error_analysis(self, track, name='source', fname='error_{name}.pdf'):
        # sort oligos by prediction error
        N = len(self.indices)
        all_lfc = np.log2(self.enr_expect/self.enr)
        II = []

        pp.figure(figsize=(5,7))
        pp.subplot(311)
        for conc, lfc in zip(self.rbp_conc, all_lfc):
            I = (-lfc).argsort()
            pp.plot(lfc[I], label='prediction error @{0:.1f}nM'.format(conc))
            II.append(I)

        II = np.array(II)
        pp.xlim(0,N)
        pp.ylabel(r'$\log_2 \frac{expect}{observed}$')
        pp.xlabel('natural sequence index')
        pp.legend(loc='upper right')
        
        import matplotlib.colors as colors
        pp.subplot(312)
        track = track[self.indices]
        data = np.array([track[I] for I in II])
        vmin, vmax = np.percentile(data, [5.,95.])

        pp.pcolor(data, cmap='viridis', vmin=vmin, vmax=vmax )
        pp.yticks(np.arange(len(self.rbp_conc))+.5, ["{0:.1f} nM".format(c) for c in self.rbp_conc])
        pp.xticks([],[])
        pp.xlim(0,N)

        from matplotlib import ticker
        cb = pp.colorbar(orientation='horizontal', fraction=.05, label=name)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.locator = tick_locator
        cb.update_ticks()

        pp.subplot(313)
        # pp.semilogy(Z[I])
        pp.ylabel('observed mean {name}'.format(name=name))
        
        import scipy.ndimage.filters
        for conc, R in zip(self.rbp_conc, data):
            avg = scipy.ndimage.filters.gaussian_filter1d(R, 50.)
            pp.plot(avg, label="nsRBNS {0:.1f} nM".format(conc))
        
        pp.xlim(0,N)
        pp.legend(loc='lower center')
        pp.tight_layout()
        pp.savefig(fname.format(**locals()))
        pp.close()

        


nsrbns = nsRBNSOligos(fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT', xtalk_file='blast/results.out')
# nsrbns = nsRBNSOligos(fa_name = '3utrOligoPool_final_T7.fa', adap5='GGGAGTTCTACAGTCCGACGATC', adap3='TGGAATTCTCGGGTGCCAAG', xtalk_file='blast/bridget_results.out')
exp = nsRBNSExperiment(nsrbns, sys.argv[1], sys.argv[2], skip_xtalk=True)
# exp.noaffinity_analysis(nsrbns.GC, 'GC content')
# exp.noaffinity_analysis(nsrbns.entropy, 'entropy')
exp.noaffinity_analysis()

for conc, obs, expect in zip(exp.rbp_conc, exp.enr, exp.enr_expect):
    mis = MutualInformationScore(obs, expect)
    # mis.heatmap_plot('MI_obs_predicted_{0:0f}nM.pdf'.format(conc))
    print conc, "observed vs expected", mis.MI, 'bits'
    print "null", np.mean(mis.MI_permut), np.std(mis.MI_permut)
    mis.dist_plot('MI_obs_predicted_{0:0f}nM.pdf'.format(conc))
    print mis.z, mis.p_value

nsrbns.xtalk_plot()
exp.affinity_selection_plot()
exp.GC_bias_plot()
exp.entropy_plot()
exp.scatter_plot()
exp.heatmap_plot()
print "analysis"
exp.error_analysis(nsrbns.entropy, name='entropy')
print "residual log error vs. entropy"
all_lfc = np.log2(exp.enr_expect/exp.enr)
for conc, lfc in zip(exp.rbp_conc, all_lfc):
    mis = MutualInformationScore(lfc, nsrbns.entropy[exp.indices])
    # mis.dist_plot('MI_entropy_lfc_{0:0f}nM.pdf'.format(conc))
    print conc,"nM", mis.MI, mis.z, mis.p_value

exp.error_analysis(nsrbns.GC, name='GC_content')
print "residual log error vs. GC content"
all_lfc = np.log2(exp.enr_expect/exp.enr)
for conc, lfc in zip(exp.rbp_conc, all_lfc):
    mis = MutualInformationScore(lfc, nsrbns.GC[exp.indices])
    # mis.dist_plot('MI_entropy_lfc_{0:0f}nM.pdf'.format(conc))
    print conc,"nM", mis.MI, mis.z, mis.p_value

exp.error_analysis(nsrbns.xtalk_score, name='xtalk')
print "residual log error vs. xtalk"
all_lfc = np.log2(exp.enr_expect/exp.enr)
for conc, lfc in zip(exp.rbp_conc, all_lfc):
    mis = MutualInformationScore(lfc, nsrbns.xtalk_score[exp.indices])
    # mis.dist_plot('MI_entropy_lfc_{0:0f}nM.pdf'.format(conc))
    print conc,"nM", mis.MI, mis.z, mis.p_value

pp.show()



# pp.figure()
# pp.loglog(frac[0], frac[1],'x')
# pp.loglog(frac[0], frac[2],'.')
# pp.loglog(frac[0], frac[3],'^')

# pp.figure()
# for conc, R in zip(rbp_conc, enr):
#     pp.hist(np.log10(R), bins=100, normed=True, cumulative=True)



# pp.figure()
# pp.hist(GC, bins=100)

# pp.figure()
# pp.title("GC content")
# x = np.log2(enr[1])
# pp.hist(x[GC < .4], bins=100, alpha=.6, normed=True)
# pp.hist(x[GC > .5], bins=100, alpha=.6, normed=True)
# pp.hist(x[GC > .6], bins=100, alpha=.6, normed=True)

# pp.figure()
# pp.title("motif content")

# def plot_dist(cond, color, label, bins =100):
#     y = x[cond]
#     m = np.median(y)
#     pp.hist(y, color=color, bins=bins, alpha=.6, normed=True, label=label)
#     pp.axvline(m, color=color)
    
# plot_dist(motif == 0, 'gray', 'no GCAYG')
# plot_dist((0 < motif) & (motif < 5), 'blue', 'GCATG')
# plot_dist((5 <= motif) & (motif < 10), 'yellow', 'HGCATG')
# plot_dist(motif == 10, 'red', 'TGCATGY')
# pp.legend()





# sys.exit(0)
# reads = RBNSReads(fname, format='fasta', rbp_name='nsRBNS', rna_conc=100., acc_storage_path='cska/acc')

def evaluate_RBPbind(seqs, rbp_conc, state):
    import copy
    import rbpbind
    rbpbind.init(T=22)
    rbpbind.set_kmer_invkd_lookup(state.A, k) # assign same affinities as used for SPA model
    p_bound = []
    Z1 = []

    for i,s in enumerate(np.array(SEQ)[indices]):
        z0 = rbpbind.compute_Z(s,0)
        z1 = rbpbind.compute_Z(s,1, update_seq=False)
        Z1.append(z1)
        
        pb = []
        for conc in rbp_conc:
            zc = rbpbind.compute_Z(s,conc, update_seq=False)
            pb.append(1. - np.exp(-(z0-zc)))
        
        p_bound.append(pb)
        print '\r {0}'.format(i),
    
    nstate = copy.copy(state)
    nstate.Z1 = np.exp(np.array(Z1))
    nstate.p_bound = np.array(p_bound).T

    return nstate

# state = evaluate_SPA(SEQ, rbp_conc)

# nstate = evaluate_RBPbind(SEQ, rbp_conc, state)

# pp.figure()

# for conc, prbp, pspa in zip(rbp_conc, nstate.p_bound, state.p_bound):
#     pp.loglog(prbp, pspa, '.', label='{0:.1f}'.format(conc))

# pp.legend(loc='upper left')
# pp.xlabel('RBPbind')
# pp.ylabel('SPA')
# pp.show()

# state = nstate


def detailed_analysis(i, flavor='detail'):
    j = indices[i]
    fa_id = _fa_ids_raw[j]
    s = adap5.lower() + np.array(SEQ)[indices][i] + adap3.lower()
    pp.figure()
    pp.title(fa_id)
    print i, fa_id
    print s

    # import rbpbind
    # rbpbind.init(T=22)
    # rbpbind.set_kmer_invkd_lookup(state.A, k)
    print state.mdl.subsample_acc.shape, len(adap5), len(adap3)
    start = len(adap5)-k+1
    # end = len(s) - len(adap3)+k-1
    acc = state.mdl.subsample_acc[i][start:-len(adap3)+k-1]
    ind = state.mdl.subsample_index_matrix[i]
    aff = state.A[ind]

    Z = aff * acc
    pos = Z.argmax()
    print s[start+pos:start+pos+k], "best hit", Z[pos], aff[pos], acc[pos]

    pp.semilogy(acc, color='gray', label=r'$\alpha$')
    pp.semilogy(aff, color='red', label=r'$\frac{1}{K_d}$ [1/nM]')


    # for conc in rbp_conc:
    #     # occ = rbpbind.occupancy_vector(s, conc) + 1e-6
    #     # pos = occ.argmax()
    #     # conc, occ.min(), occ.max(), pos, s[pos:pos+k]
    #     occ = (conc*Z) / ( 1 + conc*Z)
    #     pp.semilogy(occ,label=r'$\theta_i$ ({0:.1f}nM)'.format(conc))
    
    pp.legend()
    pp.xlabel('nt pos.')
    pp.ylabel('occupancy')
    pp.savefig('{0}_{1}.pdf'.format(flavor, i))
    pp.close()

    # sys.exit(0)





# pp.figure()
# a = enr[0]
# b = enr[1]
# pp.loglog(a[motif==0], b[motif==0],'.',color='gray')
# # pp.loglog(a[motif==1], b[motif==1],'r.')

# pp.loglog(a[GC < .4], b[GC < .4],'.',color='gray')
# pp.loglog(a[GC > .6], b[GC > .6],'r.')
