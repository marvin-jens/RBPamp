import numpy as np
import logging
import time
from scipy.optimize import minimize_scalar

class SelfConsistency(object):
    def __init__(self, Z1, rna_conc, bins=0):
        self.logger = logging.getLogger("model.SelfConsistency")
        self.Z1 = Z1[Z1 > 0]
        # print self.Z1.shape, self.Z1.min(), self.Z1.max()
        self.N = len(self.Z1)
        self.rna_conc = rna_conc
        self.logger.debug('rna_conc={0:.3e}'.format(self.rna_conc))
        
        if bins:
            # logarithmic binning
            lZ = np.log(self.Z1)
            self.counts, bins = np.histogram(lZ, bins=bins)
            self.bins = np.exp(bins)
            # midpoint integration
            self.x = (self.bins[1:] + self.bins[:-1])/2.
            
            # switch over to fast approximation
            self.free_rbp = self.fast_free_rbp

    def fast_free_rbp(self, rbp_total, Z_scale=1.):
        # TODO: use the binned version. Compare accuracy!
        t0 = time.time()
        def to_optimize(p_free):
            Z = p_free * self.x * Z_scale
            p = Z / (Z + 1.)
            
            rbp_bound = ((p * self.rna_conc)*self.counts).sum() / self.N
            return ((rbp_total - rbp_bound) - p_free)**2
            
        res = minimize_scalar(to_optimize, bounds = (0, rbp_total), method='Bounded')
        t1 = time.time()
        perc = 100. * res.x / rbp_total
        complex = rbp_total - res.x
        self.logger.debug("total RBP={0:.1f} free={1:.1f} ({2:.2f}%) complex={4:.2e} nM in {3:.2f} ms".format(rbp_total, res.x, perc, 1000*(t1-t0), complex) )
        
        return res.x
        
    def free_rbp(self, rbp_total, Z_scale=1.):
    
        t0 = time.time()
        def to_optimize(p_free):
            Z = p_free * self.Z1 * Z_scale
            p = Z / (Z + 1.)
            
            rbp_bound = (p * self.rna_conc).sum() / self.N
            
            return ((rbp_total - rbp_bound) - p_free)**2
            
        res = minimize_scalar(to_optimize, bounds = (0, rbp_total), method='Bounded')
        t1 = time.time()
        perc = 100. * res.x / rbp_total
        self.logger.debug("total={0:.1f} free={1:.1f} ({2:.2f}%) in {3:.2f} ms".format(rbp_total, res.x, perc, 1000*(t1-t0)) )
        
        return res.x
        
    def free_rbp_vector(self, rbp_total, Z_scale=1.):
        free = np.array([self.free_rbp(rbp, Z_scale=Z_scale) for rbp in rbp_total])
        return free
    
    #def _spa_free_protein(self, Z1, rbp_conc):
        #rbp_free = [self.self_consistent_free_rbp(Z1, total) for total in rbp_conc]
        #return np.array(rbp_free, dtype= np.float32)
    
if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger('model').setLevel(logging.DEBUG)
    Z1 = np.load('/scratch/data/RBNS/RBFOX3/new/opt/5mers/5mer_Z1_.npy')
    sc = SelfConsistency(Z1, 1000., bins=10000)
    print 5, "->", sc.free_rbp(5.)
    print 5, "->", sc.fast_free_rbp(5.)
