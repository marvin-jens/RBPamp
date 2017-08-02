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
    'CTGAAGTGT',
    'TGCGTCTAC',
    'GAAGAGACT',
    'CGAGTTGTA',
]

k = 3
L = 40
reads = RBNSReads.from_seqs(seqs, pseudo_count = 1e-6)
openen = RBNSOpenen.from_array(reads, k, np.zeros(reads.seqm.shape, dtype=np.uint8), disc = OpenenDiscretization(k, L) )

rbp_conc = [1., 40., 100.] #, 100.]
#rbp_conc = [1.,]
mdl = SPAModel(reads, openen, k, rbp_conc, n_subsample=0, sub_replace=False)
params = np.zeros(4**k + len(rbp_conc), dtype=np.float32)
params[0:4**k] += 1e-9
params[0] = 1.
params[1] = .1
params[10] = .01

#params[-3:] = [0.1, 0.05, 0.1]
#params[-2:] = [0.01, 0.01]
#params[-1:] = [0.01,]
ref = mdl.evaluate(params, keep=True, do_jacobi=True)

#print "reference model gives R-values"
#print ref.R
    
#for conc, R in zip(rbp_conc, ref.R):
    #print conc, "highest R-values"
    #for i in R.argsort()[::-1][:10]:
        #m = 1./mdl.f0[i]
        #print cska.ska_kmers.index_to_seq(i, k), R[i], "max=", m, "scaled=", R[i]/m * 100

    #print conc, "lowest R-values"
    #for i in R.argsort()[::][:10]:
        #m = 1./mdl.f0[i]
        #print cska.ska_kmers.index_to_seq(i, k), R[i], "max=", m, "scaled=", R[i]/m * 100

import matplotlib.pyplot as pp    
opt = ModelOptimization(reads, openen, k, ref.R, R_err=[], known_params = params, rbp_conc=rbp_conc)

## place on optimum and test gradient
#opt.current = opt.mdl.evaluate(params, keep=True, do_jacobi=True)

#emp_grad = opt.current.sum_square_emp_gradients(opt.R_obs, delta = 1e-6, f = 1e-5)
#grad = opt.current.sum_square_gradients(opt.R_obs)

#pp.figure()
#pp.plot(grad[0], '.', label='analytic')
#pp.plot(emp_grad[0], 'x', label='empirical')
#pp.legend()
#pp.ylabel('d_err/d_param')
#pp.xlabel('model parameters')
#pp.show()
#pp.close()

#sys.exit(0)
#print opt.estimate_background()
try:
    while not opt.converged():
        opt.step_kmer()
        opt.step_gradient()
        
except KeyboardInterrupt:
    pass

print "converged/interrupted after {0} steps.".format(opt.t)
opt.print_summary()

pp.figure()
pp.loglog(params, params, 'x', label='reference')
pp.loglog(params, opt.current.params, '.', label='fit')

pp.figure()
pp.semilogy(opt.errors)
pp.ylabel('global optimization error')

pp.show()



#jac = mdl.state.jacobi_matrix
#print jac.shape
#print jac[:,0,:]
##print mdl.state.p_bound[:,0]
##print mdl.state.pi_kmer[:,0]
