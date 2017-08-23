import numpy as np
import scipy
import logging
import time
import sys
import os
#from collections import defaultdict
#from scipy.optimize import minimize, brentq, minimize_scalar

#import cska.ska_kmers 

#from cska.rbns_reads import RBNSReads

#from cska.caching import CachedBase, cached, pickled

def Kd_to_kcal(K,temp=22):
    RT = (temp + 273.15) * 8.314459848# RT in Joules/mol
    kcal = 4.184E3 # kcal in Joules
    # Kd in nM to E in kcal/mol
    return np.log(K/1E9)*RT/kcal


def kcal_to_Kd(E,temp=22):
    RT = (temp + 273.15) * 8.314459848# RT in Joules/mol
    kcal = 4.184E3 # kcal in Joules
    #print "1./RT",kcal/RT
    # kcal/mol to Kd in nM
    return np.exp(E*kcal/RT)*1e9

class AffinityDistribution(object):
    def __init__(self, invkd, T=22):
        self.RT = (T + 273.15) * 8.314459848 / 4.184E3 # RT in kcal/mol
        self.invkd = invkd
        
    @classmethod
    def singleton(cls, best = 'GCATG', best_kd=1., bg_kd=1e6, T = 22):
        k = len(best)
        invkd = np.ones(4**k, dtype=np.float32) / bg_kd

        best_i = cska.ska_kmers.seq_to_index(best)
        invkd[best_i] = 1./best_kd
        
        return cls(invkd, T=T)
    
    @classmethod
    def doublet(cls, a = 'GCATG', b='GCACG', a_kd=1., b_kd=20., bg_kd=1e6, T=22):
        k = len(best)
        invkd = np.ones(4**k, dtype=np.float32) / bg_kd

        invkd[cska.ska_kmers.seq_to_index(a)] = 1./a_kd
        invkd[cska.ska_kmers.seq_to_index(b)] = 1./b_kd
        
        return cls(invkd, T=T)
 
