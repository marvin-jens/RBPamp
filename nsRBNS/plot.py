import os, sys
import numpy as np
import matplotlib.pyplot as pp
from scipy.stats import spearmanr, pearsonr

from byo.io import fasta_chunks

oligos = []
_seqs = {}
_index = {}
seqs = []
counts = []
motif = []
GC = []
indices = []
_seqs_raw = []
_fa_ids_raw = []

def GC_content(seq):
    gc = 0
    for s in seq:
        if s == 'G' or s == 'C':
            gc += 1
    return gc / float(len(seq))

adap5 = 'GGGCCTTGACACCCGAGAATTCCA'
adap3 = 'GATCGTCGGACTGTAGAACT'
l5 = len(adap5)
l3 = len(adap3)

fa_name = 'nsRBNS_oligos_taliaferro_et_al.fa'
for i, (fa_id, seq) in enumerate(fasta_chunks(file(fa_name))):
    _seqs[fa_id] = seq
    _seqs_raw.append(seq)
    _index[fa_id] = i
    _fa_ids_raw.append(fa_id)

SEQ = [s[l5:-l3] for s in _seqs_raw]

for line in file(sys.argv[1]):
# for line in file('msi1_matrix.csv'):
    parts = line.split('\t')
    name = parts[0]
    oligos.append(name)
    counts.append(parts[1:])
    s = _seqs[name]
    seqs.append(s)
    GC.append(GC_content(s))
    indices.append(_index[name])

    if not _seqs_raw[_index[name]] == s:
        print ">>>MISMATCH", name
        print "CM ",s
        print "RAW",_seqs_raw[_index[name]]
    # else:
    #     print ">>>MATCH", name

    if (not 'GCATG' in s) and (not 'GCACG' in s):
        motif.append(0)
        # print s
    elif 'TGCATGT' in s or 'TGCATGC' in s:
        motif.append(10)
    elif 'AGCATG' in s or 'CGCATG' in s or 'TGCATG' in s:
        motif.append(5)
    else:
        motif.append(1)

GC = np.array(GC)
counts = np.array(counts, dtype=float).T
f0 = counts[0] / float(counts[0].sum())
indices = np.array(indices)
oligos = np.array(oligos)

print indices.shape, "<- indices shape"
motif = np.array(motif)
print np.bincount(motif)
N = counts.sum(axis=1)
print N
scale = N[1:]/N[0]
print scale
enr = (counts[1:,:] + 10) / (counts[0,:] + 10) * scale[:,np.newaxis]
print enr
rbp_conc = [25.,125.,625.]

frac = counts / N[:,np.newaxis]

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

import os, sys, logging
logging.basicConfig(level=logging.DEBUG)

fname = fa_name
path = os.path.dirname(fname)

k = 7
rbp_conc = np.array([25., 125., 625.])
betas = [.00015, .00001, .000]

letters = 'ACGT'
def convert(bits):
    return "".join([letters[i] for i in bits])

# sys.exit(0)
# reads = RBNSReads(fname, format='fasta', rbp_name='nsRBNS', rna_conc=100., acc_storage_path='cska/acc')
def evaluate_SPA(seqs, rbp_conc):
    from cska.reads import RBNSReads
    import cska.spa
    global fname
    reads = RBNSReads.from_seqs(SEQ, fname=fname, rbp_name='nsRBNS', rna_conc=1000., acc_storage_path='cska/acc', adap5=adap5, adap3=adap3)
    reads.seqm

    mdl = cska.spa.SPAModel(reads, k, rbp_conc, n_subsample=0)
    # print indices
    # print oligos

    # this selects directly form the index_matrix, NOT seqm. So the initial assertions were bonkers
    mdl.new_subsample(indices=indices)
    fname = sys.argv[2]
    if fname.endswith('rnacompete'):
        mdl.parameters.load_rbpbind(fname)
    else:
        mdl.parameters.load(fname)

    mdl.params[:mdl.nA]
    state = mdl.evaluate(mdl.params)

    return state

def evaluate_RBPbind(seqs, rbp_conc, state):
    import copy
    import rbpbind
    rbpbind.init(T=22)
    rbpbind.set_kmer_invkd_lookup(state.A, k) # assign same affinities as used for SPA model
    p_bound = []
    Z1 = []

    for i,s in enumerate(SEQ):
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
    nstate.Z1 = np.array(Z1)
    nstate.p_bound = np.array(p_bound).T

    return nstate

state = evaluate_SPA(SEQ, rbp_conc)
state = evaluate_RBPbind(SEQ, rbp_conc, state)

def phist(*argc, **kwargs):
    pp.hist(*argc, lw=2, bins=100, histtype='step', normed=True, cumulative=True, **kwargs)

def affinity_selection_plot(state, counts):
    ### Affinity selection plot
    pp.figure()
    Z = np.log10(state.Z1)
    # Z -= Z.mean()
    # Z /= Z.std()
    phist(Z, color='gray', label='input')
    for i, f in enumerate(counts[1:]):
        phist(Z, weights = f, label="{0}nM".format(rbp_conc[i]))

    pp.xlabel("natural sequence affinity [1/nM] (log10)")
    pp.ylabel("rel. cumulative frequency")
    pp.legend(loc='upper left')
    pp.savefig('affinity_selection.pdf')

def GC_bias_plot(GC, counts):
    ### GC bias plot
    pp.figure()
    phist(GC, weights=counts[0], color='gray', label="input")
    for i, c in enumerate(counts[1:]):
        phist(GC, weights=c, label="{0}nM".format(rbp_conc[i]))

    pp.xlabel("G,C content of natural sequence")
    pp.ylabel("rel. cumulative frequency")
    pp.legend(loc='upper left')

def predict_R(p, beta):
    expect = (p + beta) * f0 
    expect /= expect.sum()
    expect /= f0
    return expect

def optimize_beta(p, obs):

    def to_opt(beta):
        expect = predict_R(p, beta)
        R = np.corrcoef(np.log2(obs), np.log2(expect))[0][1]
        # R, p_value = spearmanr(np.log2(obs), np.log2(expect))

        return (R - 1)**2
    
    from scipy.optimize import minimize_scalar

    opt = minimize_scalar(to_opt, bounds=[0,1.], method='bounded')
    print opt
    return opt.x


def scatter_plot(rbp_conc, state, enr):
    import scipy.stats
    from cska.report import density_scatter_plot
    import cska.ska_kmers as cyska
    pearson = []
    ppval = []
    spearman = []
    spval = []

    for conc, p, obs in zip(rbp_conc, state.p_bound, enr):
        beta = optimize_beta(p, obs)
        expect = predict_R(p, beta)

        R = np.corrcoef(np.log2(obs), np.log2(expect))[0][1]
        R, pp_value = pearsonr(np.log2(obs), np.log2(expect))
        rho, p_value = spearmanr(np.log2(obs), np.log2(expect))
        pearson.append(R)
        ppval.append(pp_value)
        spearman.append(rho)
        spval.append(p_value)

        print conc, R
        lfc = np.log2(expect/obs)
        pp.figure()
        # density_scatter_plot(obs, expect, label="conc={0:.1f} R={1:.3f}".format(conc, R))
        pp.loglog(obs, expect, '.', label=r"conc={conc:.1f} R={R:.2f} (P={pp_value:.2e}) $\rho=${rho:.2f} (P={p_value:.2e})".format(**locals()))
        # pp.loglog(obs[GC > .6], expect[GC > .6], '.', label='high GC')
        # pp.loglog(obs[GC < .4], expect[GC < .4], '.', label='low GC')
        # pp.loglog(obs[motif == 0], expect[motif == 0], '.', label='no motif')
        # pp.loglog(obs, state.Z1, '.', label="conc={0:.1f} R={1:.3f}".format(conc, R))
        # print "most overpredicted"
        # for i in lfc.argsort()[::-1][:10]:
        #     print lfc[i], i, indices[i], oligos[i], state.Z1[i], Z[i], obs[i]
        #     print seqs[i]
        #     aff = mdl.params[mdl.subsample_index_matrix[i]]
        #     acc = mdl.subsample_acc[i][mdl.openen.ofs - k + 1:]
        #     acc = acc[:len(aff)]
        #     z = aff*acc
        #     kmers = [cyska.index_to_seq(x, k) for x in mdl.subsample_index_matrix[i]]
        #     I = z.argsort()[::-1]
        #     print "highest affinity contributions to Z1"
        #     for j in I[:10]:
        #         print kmers[j], z[j], aff[j], acc[j]

        pp.legend()
        pp.savefig('{0}_{1:.2f}.pdf'.format(sys.argv[1], conc))
        pp.close()

    return pearson, ppval, spearman, spval

def heatmap_plot(rbp_conc, state, enr):
    Z = state.Z1
    I = (-Z).argsort()

    N = len(Z)
    def moving_average(data, w=100):
        m = np.ones(w)/float(w)
        avg = np.convolve(data, m, mode='valid')
        # var = (data - avg)**2 / w
        var = None

        return avg, var

    pp.figure(figsize=(5,7))
    pp.subplot(311)
    pp.semilogy(Z[I], color='k', label='RBNS model prediction')
    pp.xlim(0,N)
    pp.ylabel('predicted affinity [1/nM]')
    pp.xlabel('natural sequence index')
    pp.legend(loc='upper right')
    
    import matplotlib.colors as colors
    pp.subplot(312)
    data = enr[:,I]
    vmin, vmax = np.percentile(data, [5.,95.])
    pp.pcolor(data, cmap='viridis', norm=colors.LogNorm(vmin=vmin, vmax=vmax) )
    pp.yticks(np.arange(len(rbp_conc))+.5, ["{0:.1f} nM".format(c) for c in rbp_conc])
    pp.xticks([],[])
    pp.xlim(0,N)
    pp.colorbar(orientation='horizontal', fraction=.05, label='observed nsRBNS enrichment')

    pp.subplot(313)
    # pp.semilogy(Z[I])
    pp.ylabel('observed mean enrichment')
    
    import scipy.ndimage.filters
    for conc, R in zip(rbp_conc, enr):
        lR = np.log2(R[I])
        # avg, var = moving_average(lR)
        avg = scipy.ndimage.filters.gaussian_filter1d(lR, 50.)

        pp.plot(avg, label="nsRBNS {0:.1f} nM".format(conc))
    
    pp.xlim(0,N)
    pp.legend(loc='upper right')
    pp.tight_layout()
    pp.savefig('heatmap.pdf')




# pp.figure()
# a = enr[0]
# b = enr[1]
# pp.loglog(a[motif==0], b[motif==0],'.',color='gray')
# # pp.loglog(a[motif==1], b[motif==1],'r.')

# pp.loglog(a[GC < .4], b[GC < .4],'.',color='gray')
# pp.loglog(a[GC > .6], b[GC > .6],'r.')



affinity_selection_plot(state, counts)
GC_bias_plot(GC, counts)
print scatter_plot(rbp_conc, state, enr)
heatmap_plot(rbp_conc, state, enr)
pp.show()
