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

class nsRBNSOligos(object):

    def __init__(self, fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa', adap5 = 'GGGCCTTGACACCCGAGAATTCCA', adap3 = 'GATCGTCGGACTGTAGAACT'):
        self._seqs_raw = []
        self._fa_ids_raw = []
        self._index = {}
        self._seqs = {}

        self.GC = []
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
            self.GC.append(GC_content(seq))
            self.entropy.append(nt_entropy(seq))

        self.N = len(self._seqs_raw)
        self.SEQ = [s[self.l5:-self.l3] for s in self._seqs_raw] 
        self.GC = np.array(self.GC)
        self.entropy = np.array(self.entropy)

        self.xtalk, self.xtalk_score = self.xtalk_from_blast()

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
        ind = rowsum > 80
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
    def __init__(self, nsrbns, fcount_matrix, faffinities, name='nsRBNS', rbp_conc = [25.,125.,625.], pseudo=10, skip_xtalk=True):
        self.name = name
        self.ns = nsrbns
        
        self.fcount_matrix = fcount_matrix
        self.indices, self.counts = self.load_counts(fcount_matrix, skip_xtalk=skip_xtalk)

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

    def load_counts(self, fname, skip_xtalk=True):
        counts = []
        indices = []

        for line in file(fname):
            parts = line.split('\t')
            name = parts[0]
            if skip_xtalk and (self.ns.xtalk_score[self.ns._index[name]] > 0):
                # skip cross-talking oligos!
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
        print opt
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

nsrbns = nsRBNSOligos()
exp = nsRBNSExperiment(nsrbns, sys.argv[1], sys.argv[2], skip_xtalk=False)
# nsrbns.xtalk_plot()
# exp.affinity_selection_plot()
# exp.GC_bias_plot()
# exp.entropy_plot()
# exp.scatter_plot()
# exp.heatmap_plot()
exp.error_analysis(nsrbns.entropy, name='entropy')
exp.error_analysis(nsrbns.GC, name='GC_content')
exp.error_analysis(nsrbns.xtalk_score, name='xtalk')
pp.show()
