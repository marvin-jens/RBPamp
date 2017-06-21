import rbpbind as rbp
from cska.rbns_reads import RBNSReads
from cska.rbns_model import RBNSSimulator, RBNSGenerator
from cska.folding import RBNSOpenen, OpenenStorage, OpenenDiscretization

import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as pp
#pp.style.use('ggplot')
import logging
import os, sys, time
import numpy as np
import cska.ska_kmers
import scipy.stats
logging.basicConfig(level=logging.DEBUG)

#k = int(sys.argv[1])
temp = 22.
RT = (temp + 273.15) * 8.314459848/4.184E3 # RT in kcal/mol

sources = [
    ('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads',0),
    #('/scratch/data/RBNS/RBFOX2/RBFOX2_121.reads',121),
    #('/scratch/data/RBNS/RBFOX2/RBFOX2_365.reads',365),
]

reads = [RBNSReads(src, rbp_name='RBFOX2', rbp_conc=P, pseudo_count = 1) for src,P in sources]
storages = [OpenenStorage(r, '/scratch/data/RBNS/RBFOX2/ska_RBFOX2/openen/', disc_mode='gamma') for r in reads]

protein_conc = [.01, 40., 160., 3300.]
#protein_conc = [40., ]

k = 5
r = reads[0]
o = storages[0].get_discretized(k)
print "done loading"
sim = RBNSSimulator(r, o, k)
gen = RBNSGenerator(k,l=40, seed=47110815)
#gen.store_invKd('rbns_gen_sorted.txt')

print "computing single protein partition function model on input reads using correct energies"
p_bound_matrix, kmer_count_matrix, openen_kmer_bincount_matrix = sim.expected_kmer_counts(gen.kmer_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True)

def convergence_analysis_scatter(n_samples = [1000, 10000, 100000, 1000000]):
    print "convergence analysis"
    pp.figure(figsize=(6,4))
    pp.title("subsampling vs. accuracy")
    m = kmer_count_matrix[1].min()
    M = kmer_count_matrix[1].max()
    for n_sample in n_samples:
        # randomly select n_max reads
        #n = np.random.permutation(np.arange(sim.reads.N))[:n_sample]
        n = np.random.randint(sim.reads.N, size=n_sample)
        pb, sample_kmer_counts, sample_openen_bincounts = sim.expected_kmer_counts(gen.kmer_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, indices=n, n_max=n_sample)

        scale = r.N / float(n_sample)
        pp.loglog(kmer_count_matrix[1], sample_kmer_counts[1] * scale, 'o', label=str(n_sample))
        m = min(m, sample_kmer_counts[1].min())
        M = max(M, sample_kmer_counts[1].max())
        
    print m, M
    m += 1000
    pp.plot([m, M], [m, M], 'k--', label=None)
    pp.legend(loc='lower right')
    pp.xlabel("kmer-freq predicted on subsample")
    pp.ylabel("kmer-freq predicted on all reads".format(r.N))
    pp.tight_layout()
    pp.savefig('convergence_scatter.pdf')

def convergence_analysis_CV(n_samples = [1000, 10000, 100000, 1000000], n_rep=10):
    print "CV analysis"
    from scipy.stats import variation
    pp.figure(figsize=(6,4))
    pp.title("CV of pulldown kmer-abundance estimates")
    m = kmer_count_matrix[1].min()
    M = kmer_count_matrix[1].max()
    for n_sample in n_samples:
        scale = r.N / float(n_sample)
        scale = 1
        reps = []
        for i in range(n_rep):
            # randomly select n_max reads
            #n = np.random.permutation(np.arange(sim.reads.N))[:n_sample]
            n = np.random.randint(sim.reads.N, size=n_sample)
            pb, sample_kmer_counts, sample_openen_bincounts = sim.expected_kmer_counts(gen.kmer_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, indices=n, n_max=n_sample)
            reps.append(sample_kmer_counts[1] * scale)
        
        reps = np.array(reps)
        
        CV = variation(reps,axis=0)
        print "CV of highest affinity kmer", n_sample, CV[-1], reps[:,1023]
        y, x, patches = pp.hist(CV, bins=100, label=str(n_sample), histtype='stepfilled')
        #print x
    
    #m += 1
    #pp.plot([m, M], [m, M], 'k--', label=None)
    pp.gca().set_xscale('log')
    pp.legend(loc='upper right')
    pp.xlabel("Coefficient of Variation")
    pp.ylabel("frequency")
    pp.tight_layout()
    pp.savefig('convergence_CV.pdf')
    pp.show()
    
#convergence_analysis_scatter()
#convergence_analysis_CV()

f0 = r.kmer_frequencies(k)
f0 /= f0.sum()

freq_matrix = np.array(kmer_count_matrix, dtype=float)
freq_matrix /= freq_matrix.sum(axis=1)[:, np.newaxis]

R = freq_matrix / f0[np.newaxis:]
R_order = R.min(axis=0).argsort()[::-1]

def kmer_error(expected, observed, i):
    """This is signed and should have a root"""
    #err = np.mean((np.log(expected) - np.log(observed))**2)
    err = np.mean( expected[:,i] - observed[:,i])
    return err

def model_error(expected, observed):
    """This is unsigned and should have (at least) one minimum"""
    #err = np.mean((np.log(expected) - np.log(observed))**2)
    #err = np.mean( np.log2( (expected + 1e6) / (observed + 1e6) )**2)
    err = np.mean( ( (observed - expected) **2) )
    return err


def sweep(i, min_ikd, max_ikd, steps = 1000, local=True, n_max=10000):
   
    invkd = np.copy(trial_invkd)
    
    errors = []
    x = np.exp(np.linspace(np.log(min_ikd), np.log(max_ikd), steps))
    scale = sim.reads.N/ n_max
    for ikd in x:
        invkd[i] = ikd
        pb_trial, kmer_count_trial, openen_kmer_bincount_trial = sim.expected_kmer_counts(invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, n_max=n_max)
        
        #print kmer_count_trial[:,i].shape
        #print ikd, scale*kmer_count_trial[:,i], kmer_count_matrix[:,i]
        if local:
            errors.append(kmer_error(scale*kmer_count_trial, kmer_count_matrix, i) )
        else:
            errors.append(model_error(scale*kmer_count_trial, kmer_count_matrix) )

    return x, np.array(errors)
    
def plot_sweep_debug(i, min_invkd = 1e-12, max_invkd = 1e2):
    dGs, err = sweep(i, min_invkd, max_invkd, steps=1000, local=True)
    #dGs, err_global = sweep(i, -30., 30., steps=1000, local=False)
    best = dGs[np.fabs(err).argmin()]
    pp.title("greedy kmer-fit for {0}".format(kmers[i]) )
    pp.semilogx(dGs, err)
    #pp.plot(dGs, err_global)
    pp.axvline(gen.kmer_invkd[i], color='r', label="correct value")
    pp.axvline(best, color='b', label="root")
    pp.axhline(0, color='k', linestyle='dashed')
    pp.xlabel(r"$\frac{1}{K_d}$ [nM]")
    pp.ylabel(r"expected - observed")
    pp.legend(loc='upper left')
    pp.tight_layout()
    pp.show()

def plot_congruence(ref, before, after, title="congruence"):
    pp.figure()
    pp.title(title)
    pp.loglog(ref, before, 'xb')
    pp.loglog(ref, after, 'or')
    #pp.figure()
    #for obs, exp in zip(kmer_count_matrix, expected_counts):
        #pp.plot(obs, exp, '.')

    m = min(ref.min(), before.min(), after.min())
    M = max(ref.max(), before.max(), after.max())
    pp.loglog([m,M],[m,M], '-k', linestyle='dashed')
    pp.xlim(m,M)
    pp.ylim(m,M)
    
    
    pp.show()

def plot_R_value_agreement(new_expect, title='R-value agreement'):
    I_real = gen.kmer_invkd.argsort()
    pp.figure(figsize=(6,4))
    pp.title(title)
    x = np.arange(len(R[1][I_real]))
    pp.fill_between(x, 0, R[1][I_real], step = 'mid', label='pull-down')
    new_freq = new_expect / new_expect.sum(axis=1)[:,np.newaxis]
    new_R = new_freq[1] / f0
    pp.plot(new_R[I_real], '^', label='expect')
    pp.show()
    
def select_reads_with_kmer(kmer, n_max=0):
    presence = r.kmer_presence(kmer)
    subset = r.seqm[presence > 0]
    
    if not n_max:
        n_max = len(subset)

    return subset[:n_max], float(n_max)/r.N

def predict_kmer_counts(trial_invkd, seqm = None):
    print "predicting kmer counts on all reads"
    pb_trial, kmer_count_trial, openen_kmer_bincount_trial = sim.expected_kmer_counts(trial_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, seqm=seqm)
    print "done"
    return kmer_count_trial

def optimize(trial_invkd, i, local=True, min_invkd = 1e-12, max_invkd = 1e2, n_max=1000):
    
    # operate on a local copy!
    trial_invkd = np.copy(trial_invkd)
    scale = sim.reads.N/ n_max

    obs_counts = kmer_count_matrix#[:,i]
    #print "reference counts", obs_counts
    
    def predict(invkd):
        
        trial_invkd[i] = invkd #np.log(invKd*1e-9)
        pb_trial, kmer_count_trial, openen_kmer_bincount_trial = sim.expected_kmer_counts(trial_invkd, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, n_max=n_max)

        return kmer_count_trial
    
    def to_optimize_local(invkd):
        #invkd = args[0]
        
        trial_counts = predict(invkd)
        err = kmer_error(scale*trial_counts, obs_counts, i)
        #print invkd, gen.kmer_energies[i], err
        #print "invkd", invkd,  "err=", err # "expected counts", trial_counts[:,i],
        return err

    def to_optimize_global(invkd):
        #invkd = np.log(args)
        
        trial_counts = predict(invkd)
        err = model_error(scale*trial_counts, obs_counts)
        #print invkd, gen.kmer_energies[i], err
        print "invkd", invkd,  "err=", err # "expected counts", trial_counts[:,i],
        return err
    
    from scipy.optimize import minimize, brentq, minimize_scalar
    
    old_expect = predict(trial_invkd[i])
    if local:
        
        to_optimize = to_optimize_local
        left, right = to_optimize(min_invkd), to_optimize(max_invkd)
        if (left > 0) == (right > 0):
            print "no sign change! can't find root. reporting global minimum instead"
            #plot_sweep_debug(i)
            res = minimize_scalar(to_optimize_local, bounds=(min_invkd, max_invkd), method='bounded' )
            #res = minimize_scalar(to_optimize_global, bracket=(min_invkd, max_invkd), method='brent' )
            print res
            return optimize(trial_invkd, i, local=False, n_max=n_max)
            #invkd_opt = trial_invkd[i]
            #return False, invkd_opt, to_optimize(invkd_opt)
        
        #print "bounds", left, right
        
        invkd_opt, res = brentq(to_optimize, min_invkd, max_invkd, full_output=True)
        #print invkd_opt, res
        err = to_optimize(invkd_opt)
        #print "error", err
        return res.converged, invkd_opt, err, predict(invkd_opt), old_expect
    else:
        to_optimize = to_optimize_global
        #res = minimize(to_optimize_global, (1.), bounds=[(np.exp(-30.),np.exp(30.))], method='TNC')
        res = minimize_scalar(to_optimize_global, bounds=(min_invkd, max_invkd), method='bounded' )
        #res = minimize_scalar(to_optimize_global, bracket=(-30.,30. ), method='brent' )
        print res
        return res.success, res.x, res.fun, predict(res.x),old_expect


def gradient(I, trial_invkd, delta=0.0001, n_max=1000):
    E0 = np.copy(trial_invkd)
    grad = []
    
    # randomly select n_max reads
    n = np.random.permutation(np.arange(sim.reads.N))[:n_max]
    
    scale = sim.reads.N / float(n_max)
    
    for i in I:
        
        pb_trial, kmer_count0, openen_kmer_bincount_trial = sim.expected_kmer_counts(E0, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, indices=n)
        err0 = model_error(scale*kmer_count0, kmer_count_matrix)
        
        E0[i] += delta
        pb_trial, kmer_count1, openen_kmer_bincount_trial = sim.expected_kmer_counts(E0, protein_conc, _do_not_unpickle=True, _do_not_pickle=True, indices=n)
        err1 = model_error(scale*kmer_count1, kmer_count_matrix)
        grad.append( (err1 - err0)/delta)
        print i, grad[-1]
        
    return np.array(grad)

def local_optima(I, trial_invkd, n_max=1000):
    opt = np.copy(trial_invkd)
    for i in I:
        plot_sweep_debug(i)
        kmer = cska.ska_kmers.index_to_seq(i, k)
        success, ikd_opt, err, new_expect = optimize(trial_invkd, i, local=True, n_max=n_max)
        print "{3} real invkd={0:.2e}, current_invkd={1:.2e}, fitted_invkd={2:.2e}".format(corr_invkd[i], trial_invkd[i], ikd_opt, kmer)
        if success:
            print "accept"
            opt[i] = ikd_opt

    return opt
    
        
    
kmers = np.array(list(cska.ska_kmers.yield_kmers(5)))
corr_invkd = gen.kmer_invkd
trial_invkd = np.ones(corr_invkd.shape, dtype=np.float32)*1e-12

print "kmers with actual high affinity"
I_real = gen.kmer_energies.argsort()[:20]

for i in I_real:
    print cska.ska_kmers.index_to_seq(i, k), corr_invkd[i], gen.kmer_energies[i], R[:,i], freq_matrix[:,i] * 4**k

I = R_order[:50]



i = I[0]
n_max=10000
kmer = cska.ska_kmers.index_to_seq(i, k)
success, ikd_opt, err, new_expect, old_expect = optimize(trial_invkd, i, local=True, n_max=n_max)
plot_R_value_agreement(new_expect)

print "most enriched kmer is {0} with optimal inv_kd {1} ({2})".format( kmer, ikd_opt, corr_invkd[i] )
delta = new_expect - old_expect
scale = sim.reads.N / float(n_max)
mean_residual = np.mean(kmer_count_matrix / scale - new_expect, axis=0)

print "kmers with max residual error after correcting for", kmer
for i in mean_residual.argsort()[::-1][:10]:
    print cska.ska_kmers.index_to_seq(i, k), mean_residual[i]
    

print "real affinity kmers"
predict_freq_matrix = new_expect / new_expect.sum(axis=1)[:, np.newaxis]
for i in I_real:
    print cska.ska_kmers.index_to_seq(i, k), corr_invkd[i], "residual", mean_residual[i], "freq", freq_matrix[:,i] * 4**k, "expect", predict_freq_matrix[:,i] * 4**k
    

trial_invkd[i] = ikd_opt
i = mean_residual.argmax()
print "max residual impacted after last change", cska.ska_kmers.index_to_seq(i,k)
success, ikd_opt, err, new_expect, old_expect = optimize(trial_invkd, i, local=True, n_max=n_max)

trial_invkd[i] = ikd_opt
print "changed kmers"
delta = new_expect - old_expect
scale = sim.reads.N / float(n_max)
mean_residual = np.mean(kmer_count_matrix / scale - new_expect, axis=0)

print "max residual impacted after last change", cska.ska_kmers.index_to_seq(i,k)
success, ikd_opt, err, new_expect, old_expect = optimize(trial_invkd, i, local=True, n_max=n_max)

for i in delta.mean(axis=0).argsort()[::-1][:20]:
    print cska.ska_kmers.index_to_seq(i,k)
    print "delta", delta[:,i]
    print "expect", new_expect[:,i]
    print "observed", kmer_count_matrix[:,i] / scale
    print "residual", kmer_count_matrix[:,i] / scale - new_expect[:,i]
    print "mean", mean_residual[i]

sys.exit(0)

weights = R[:,I].mean(axis=0)
weights /= (weights.max())




print "computing local optima"
trial_invkd = local_optima(I, trial_invkd, seqm=None)
trial_invkd[I] *= weights

        
new_expect = predict_kmer_counts(trial_invkd)
mdl_err =  model_error(new_expect, kmer_count_matrix)

for t in range(200):
    print "computing gradient around current guess", t
    grad = np.zeros(trial_invkd.shape)
    grad[I] = gradient(I, trial_invkd, seqm=None, delta=.0001)
    for i in I:
        kmer = cska.ska_kmers.index_to_seq(i, k)
        print "{3} real invkd={0:.2e}, current_invkd={1:.2e}, gradient={2:.2e}".format(corr_invkd[i], trial_invkd[i], grad[i], kmer)
    
    plot_congruence(gen.kmer_energies, trial_invkd, trial_invkd - grad)
    trial_invkd -= grad

sys.exit(0)



for t in range(100):
    dG_updates = []
    trial_invkd_before = np.copy(trial_invkd)
    expect_before = np.copy(new_expect)

    print "cycle {0}, optimizing kmers {1} with weights {2}".format(t, I, weights)

    new_weights = []
    # select top-enriched kmers
    for j,(i, w) in enumerate(zip(I, weights)):
        kmer = cska.ska_kmers.index_to_seq(i, k)
        #seqm, f = select_reads_with_kmer(kmer, n_max=1000)
        seqm = None
        #print "selected kmer '{0}' with R-values '{1}'".format(kmer, R[:,i])
        
        #plot_sweep_debug(i)

        success, dG_opt, err, new_expect = optimize(trial_invkd, i, local=True, seqm=seqm )
        new_err =  model_error(new_expect, kmer_count_matrix)
        rel_change = (new_err - mdl_err)/mdl_err

        if rel_change > 0:
            print "predicted increase in error! flipping sign"
            dG_opt *= -1
            #new_weights.append(1.)
        #else:
        new_weights.append( np.fabs(rel_change))

        print "{4} (w={5}) real energy={0:.2f}, current_energy={1:.2f}, fitted energy={2:.2f}, reduction in model_error={3:.2e} %%".format(gen.kmer_energies[i], trial_invkd[i], dG_opt, 100*rel_change, kmer, w)
        dG_updates.append(dG_opt)
        #pp.figure()
        #pp.plot(kmer_count_matrix[:,i], new_expect[:,i], 'o')
        #pp.show()
        
        #trial_invkd[i] = dG_opt
        #plot_congruence(kmer_count_matrix, new_expect)
        #trial_invkd[i] = 0

    
    dG_updates = np.array(dG_updates)
    trial_invkd[I] = trial_invkd[I] * (1-weights) + dG_updates * weights
    new_expect = predict_kmer_counts(trial_invkd)
    mdl_err =  model_error(new_expect, kmer_count_matrix)
    
    #plot_congruence(kmer_count_matrix, new_expect_before, new_expect, "kmer counts")
    plot_congruence(gen.kmer_energies, trial_invkd_before, trial_invkd, "kmer energies, t={0}".format(t))

    weights = np.array(new_weights)
    weights /= (2*weights.max())

sys.exit(0)

        




def most_divergent_kmer_index(expected, observed, blocked = set()):

    delta = np.mean( R  * np.log2((observed + 1e-9) / (expected + 1e-9))**2, axis=0)
    #delta = np.mean((expected - observed), axis=0)

    candidates = []

    for worst in delta.argsort()[::-1]:
    #for worst in R_order:
        if R[:,worst].min() < 1.5:
            # skip non-enriched kmers
            continue

        if not worst in blocked:
            print "{0} d={1} R={7} trial_dG={5} real_dG={6} expect={2} observed={3} blocked={4}".format(kmers[worst], delta[worst], expected[:,worst], observed[:,worst], worst in blocked, trial_invkd[worst], gen.kmer_energies[worst], R[:,worst])
            candidates.append(worst)
        else:
            print "BLOCKED", kmers[worst]
            
        if len(candidates) >= 10:
            break

    return candidates

    #for worst in delta.argsort()[::-1]:
        #if not worst in blocked:
            #print "selected", worst, kmers[worst], delta[worst], "real energy", gen.kmer_energies[worst]
            #return worst
        #worst = delta.argmax()
    #
    #return worst


    
blocked = []
failed = []

expected_counts = predict_kmer_counts(trial_invkd)
mdl_err = model_error(expected_counts, kmer_count_matrix)


for r in range(100):
    print "optimization step", r
    if r:
        plot_congruence()
        
    if len(blocked) > 10:
        blocked.pop(0)

    if len(failed) > 10:
        failed = []
        # we ran into a lot of failure to optimize. Let's tweak the parameters we have modified so far.
        print "RE-OPTIMIZATION RUN"
        candidates = np.random.permutation((trial_invkd != 0).nonzero()[0])
        local = False
    else:
        candidates = most_divergent_kmer_index(expected_counts, kmer_count_matrix, set(blocked) | set(failed))
        local = True

    if failed:
        failed.pop(0)

    for i in candidates:
        print "selected", i, kmers[i], R[:,i], trial_invkd[i], "actual energy", gen.kmer_energies[i]
        #print "kmer counts", kmer_count_matrix[:,i]

        if i == 375:
            plot_sweep_debug(i)

        dG0 = trial_invkd[i]
        success, dG_opt, err = optimize(trial_invkd, i, local=local)
        
        if not success:
                print "did not converge! trying global optimization"
            ##plot_sweep_debug(i)
            #success, dG_opt, err = optimize(trial_invkd, i, local=False)
            #if not success:
                print "also failed. skipping"
                trial_invkd[i] = dG0/2.
                failed.append(i)
                continue

        trial_invkd[i] = dG_opt
        new_expect = predict_kmer_counts(trial_invkd)
        new_err =  model_error(new_expect, kmer_count_matrix)
        
        rel_change = (new_err - mdl_err)/mdl_err
        
        if new_err > (1.01) * mdl_err:
            print "??>> dG0", dG0, "dG_opt", dG_opt, "(real one =", gen.kmer_energies[i],")"
            print "??>> model error woukd chang by {0:.2f}%% residual error={1}".format( 100.* rel_change, new_err)
            print "kmer optimization did not decrease model error! skipping"
            #plot_sweep_debug(i)
            trial_invkd[i] = dG0
            failed.append(i)
            #break
            continue
        
        # keep kmer optimization changes
        blocked.append(i)
        expected_counts = new_expect
        mdl_err = new_err
        print "!!>> dG0", dG0, "dG_opt", dG_opt, "(real one =", gen.kmer_energies[i],")"
        #print ">> reference counts", kmer_count_matrix[:,i]
        #print ">> expected counts", expected_counts[:,i]
        print "!!>> model error changed by {0:.2f}%% residual error={1}".format( 100.* rel_change, new_err)
        

#print "kmer counts", kmer_count_trial[:,i]


