from cska.rbns_model import *
from cska.rbns_reads import *
from cska.folding import *
import logging
logging.basicConfig(level=logging.DEBUG)
seqs = [
    'AAAAAAAAA',
    'CCCCCCCCC',
    'GGGGGGGGG',
    'TTTTTTTTT',
    'AAAATATAA',
    'AACAAGACT',
    'ACAGCAACG',
    'ACCAGGATT',
    'CCACCGCCT',
    'CACAGATCC',
    'CTCGCGCAG',
    'GGAGGCGGT',
    'TTATTCTTG',
    'TAATACTAG',
    'TCATCGTCT',
    'TGATGCTTT',
    'TTGATTATC',
    'TTTTTGGGA',
]
reads = RBNSReads.from_seqs(seqs, pseudo_count = 1e-6)
openen = RBNSOpenen.from_array(reads, 3, np.zeros(reads.seqm.shape, dtype=np.uint8), disc = OpenenDiscretization(3, 40) )

rbp_conc = [1., 40., 100.]
mdl = SPAModel(reads, openen, 3, rbp_conc, n_subsample=0, sub_replace=False)
#invkd = np.ones(4**3, dtype=np.float32)
#betas = np.array([0,0,0], dtype=np.float32)
#params = np.concatenate( (invkd, betas) )
params = np.zeros(4**3 + len(rbp_conc), dtype=np.float32)
params[0:4**3] += 1e-9
params[0] = .1
params[-3:] = [0.1, 0.05, 0.1]
state = mdl.evaluate(params, keep=True, do_jacobi=True)
#print state.p_bound
##print state.pd_freq / state.pd_sum[:, np.newaxis]
##print mdl.f0
##print state.pd_freq / state.pd_sum[:, np.newaxis]
##print state.pd_freq / state.pd_sum[:, np.newaxis] / mdl.f0[np.newaxis,:]
##print state.R_values
for conc, R in zip(rbp_conc, state.R_values):
    print conc, "highest R-values"
    for i in R.argsort()[::-1][:10]:
        m = 1./mdl.f0[i]
        print cska.ska_kmers.index_to_seq(i, 3), R[i], "max=", m, "scaled=", R[i]/m * 100
    
jac = mdl.state.jacobi_matrix
print jac.shape
print jac[:,0,:]
#print mdl.state.p_bound[:,0]
#print mdl.state.pi_kmer[:,0]
