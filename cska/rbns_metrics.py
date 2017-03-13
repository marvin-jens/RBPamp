__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import sys
import itertools
import numpy as np
import copy
import time
import os
import logging
import collections
import cska.ska_kmers
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pp


class RBNSMetrics(object):            
    def run_ROC(self):
        
        self.logger.info("computing receiver-operator-characteristic for {0} samples".format(len(self.reads) -1 ) )
        def make_plot(k, order_by="ska_weights", name="SKA"):
            
            pp.figure()
            pp.title('discrimination of {self.rbp_name} pd/input by {name}'.format(**locals()))

            for reads in self.reads[1:]:
                res = self.runs[ (k, reads.rbp_conc) ]

                order = getattr(res, order_by).argsort()[::-1]

                recall_pd = res.pd_reads.recall(order)
                recall_in = res.in_reads.recall(order)
                                
                x = np.array([0,] + list(recall_in.cumsum()))
                y = np.array([0,] + list(recall_pd.cumsum()))
                AUC = np.trapz(y, x)
                pp.step(x, y, where='post', label='{0}mersfloat(in_N) @{1}nM (AUC={2:.3f})'.format(k, reads.rbp_conc, AUC))
                self.AUCs[ (k, reads.rbp_conc) ] = (AUC, name, k, reads.rbp_conc)

            pp.plot([0,1.],[0,1.], color='gray', linestyle = 'dashed')
            
            pp.xlabel('fraction of input explained')
            pp.ylabel('fraction of pulldown explained')
            pp.legend(loc='lower right')
            pp.savefig("{self.out_path}/{self.rbp_name}.{k}mer.{name}.ROC.pdf".format(**locals()))

        for k in self.k_range:
            make_plot(k, order_by="ska_weights", name="SKA")
            make_plot(k, order_by="R_values", name="R")
            make_plot(k, order_by="f_ratios", name="f_ratios")
        
        # TODO: find rank at wich the ROC curve slope drops below 1. 
        # This is where reads with the kmer are no longer more abundant in pd than input
        # (f-value ratio is 1)!
        self.best_auc, self.best_method, self.best_k, self.best_rbp_conc = sorted(self.AUCs.values())[-1]
        print "BEST", self.best_auc, self.best_method, self.best_k, self.best_rbp_conc


    def compare_k(self, z_cut=2):
        import matplotlib.pyplot as pp
        pp.figure()
        k_range = sorted(self.run.results.keys())
        hits_at_k = []
        
        def spread(x, data, min_w=1.1, max_n=20):
            n = len(x)
            lo, hi = x.min(), x.max()
            step = (hi-lo)/float(n-1)
            center = (hi - lo)/2
            
            if n > max_n:
                I = x.argsort()
                data = list(np.array(data)[I])
                x = list(np.array(x)[I])

                data = data[:max_n/2] + ['...'] + data[-max_n/2:]
                x = np.array(x[:max_n/2] + [center,] + x[-max_n/2:])
                
                n = len(x)
                lo, hi = x.min(), x.max()
                step = (hi-lo)/float(n-1)
                center = (hi - lo)/2

            if step < min_w:
                w = min_w * (n-1)
                lo = center - w/2
                hi = lo + w
                
                step = (hi-lo)/float(n-1)

            return np.arange(lo, hi+step, step), data

        def noise(x, amp=.3):
            return x + np.random.random(len(x))*amp - amp/2.
        
        def displace(x, amp=.3):
            n = len(x)
            lo, hi = x.min()- amp/2, x.max()+amp/2
            step = (hi-lo)/float(n-1)
            
            return np.arange(lo, hi+step, step)[:n]
            
        for k in k_range:
            res = self.run.results[k]
            top_i = (res.z_scores_ska > z_cut).nonzero()[0]
            n = len(top_i)
            
            kmers = [cska.ska_kmers.index_to_seq(i,k) for i in top_i]
            scores = res.ska_weights[top_i]
            errors = res.ska_weights_err[top_i]
            
            x = displace((np.ones(len(scores)) * k), amp=.4)
            #print x, len(x), len(scores)
            pp.errorbar(x, scores, yerr=errors, fmt='o', color='r', alpha=.5)
            
            y, kmers = spread(scores, kmers)
            for mer, score, _y in zip(kmers, scores, y):
                pp.gca().text(k - .5, _y, mer, fontsize=6)

            hits_at_k.append( (kmers, scores, errors) )
        pp.savefig(os.path.join(self.run.out_path,"k_comparison.pdf"))

    

            
        
    def get_cooccurrence_tensor(self, k, n_max=20, resume=True):
        import pickle

        tname = "{self.out_path}/cooccurrence_tensor_{k}mers.pkl".format(**locals())
        if os.path.exists(tname):
            self.logger.debug("loading from '{tname}'".format(tname=tname) )
            kmer_list, indices, best_rbp_conc, pd_tensor, in_tensor, R_values, pd_N, in_N = pickle.load(file(tname,'rb'))

        else:
            kmer_list, indices, best_rbp_conc = self.select_significant_kmers(k, n_max=n_max)
            self.logger.info("cooccurrence tensor analysis for k={k} rbp_conc={best_rbp_conc}nM kmers='{kmer_list}'".format(**locals()) )
            
            #kmer_list = sorted(kmer_list) # TODO: re-order by layer correlation
            best_run = self.runs[(k, best_rbp_conc)]

            assert len(kmer_list) == len(indices)
            pd_freqs = best_run.pd_reads.kmer_counts(k)[indices]
            in_freqs = best_run.in_reads.kmer_counts(k)[indices]
            pd_N = best_run.pd_reads.N * (best_run.pd_reads.L - k + 1)
            in_N = best_run.in_reads.N * (best_run.in_reads.L - k + 1)
            

            pd_tensor = best_run.pd_reads.kmer_cooccurrence_distance_tensor(kmer_list)
            in_tensor = best_run.in_reads.kmer_cooccurrence_distance_tensor(kmer_list)
            R_values = best_run.R_values[indices]
            self.logger.debug("saving to '{tname}'".format(tname=tname) )
            pickle.dump( (kmer_list, indices, best_rbp_conc, pd_tensor, in_tensor, R_values, pd_N, in_N), file(tname,'wb'), protocol=pickle.HIGHEST_PROTOCOL )
        
        
        return kmer_list, indices, best_rbp_conc, pd_tensor, in_tensor, R_values, pd_N, in_N

        
    def cooccurrence_tensor_analysis(self, kmer_list, indices, best_rbp_conc, pd_tensor, in_tensor, R_values, pd_N, in_N, pseudo_count=20.):
      
        #print pd_tensor
        n,m,l = pd_tensor.shape
        print n,m,l
        print kmer_list
        k = len(kmer_list[0])
        #print "expected co-occurrences"
        #print in_freqs.shape
        
        exp_in = R_values[:, np.newaxis, np.newaxis] * R_values[np.newaxis, :, np.newaxis] * (np.ones(l) )[np.newaxis, np.newaxis,:]
        #print exp_in.shape
        #exp_in[:,:,0] = 0
        
        #print exp_in
        print "ratio of input tensor to expected"
        
        scale = pd_N / float(in_N)
        print "scale", scale, pd_N, in_N
        enr_tensor = np.log2( (pd_tensor+1) / (in_tensor*exp_in + 1) )
        
        #for s in range(1,l-5):
        for s in range(1,k*2):
            pd_layer = pd_tensor[:,:,s]
            in_layer = in_tensor[:,:,s]
            exp_layer = exp_in[:,:,s]
            enr_layer = enr_tensor[:,:,s]
            #print "input"
            #print in_layer
            #print "expected (independent)"
            #print exp_layer * in_layer
            #print "observed (pull down)"
            #print pd_layer
            #print "log2 ratios"
            #print enr_layer
            ordered = enr_layer.argsort(axis=None)[::-1]
            #print "most co-occuring at spacing",s
            #for n in ordered[:10]:
                ##print n
                #m1,m2 = np.unravel_index(n, enr_layer.shape)
                ##print m1,m2
                #print kmer_list[m1]
                #print " "*(s-1), kmer_list[m2], enr_layer[m1,m2]

            
            pp.figure()
            pp.title("k={0} spacing={1}".format(k, s))
            z = enr_layer.T
            print z.argmax(axis=0)
            import scipy.cluster.hierarchy
            Z = scipy.cluster.hierarchy.linkage(z, method='single')
            print Z[n-2]
            
            y, x = np.mgrid[slice(0, n),slice(0, n)]
            # symmetric, dynamic range of colorbar
            dr = np.fabs(z).max()
            pp.pcolor(x,y,z, cmap=pp.get_cmap('seismic'), vmin=-dr, vmax=dr)

            pp.yticks(np.arange(n)+.5, kmer_list) 
            pp.xticks(np.arange(n)+.5, kmer_list, rotation=90) 
            pp.xlabel("first kmer")
            pp.ylabel("second kmer")
            #pp.xlim(0,)
            cbar = pp.colorbar(orientation="horizontal", fraction=0.1, shrink=0.75, label=r"$\log_2( \frac{pd}{in} )$")
            cbar.ax.tick_params(labelsize=8)
        
            pp.savefig("{self.out_path}/{k}_{s}.heatmap.pdf".format(**locals()))
            
        enr_tensor = np.log2((pd_tensor + pseudo_count) / (in_tensor + pseudo_count))
        
        # length dependence
        #import matplotlib.pyplot as pp
        #return
    
        #pp.figure()
        #pp.title("k={0}".format(k)))
        #pp.plot(enr_tensor.max(axis=0).max(axis=0))
        #pp.show()
        
        #from mayavi import mlab
        #x_max, y_max, z_max = enr_tensor.shape
        
        #x, y, z = np.meshgrid(np.arange(x_max), np.arange(y_max), np.arange(z_max))
        #mlab.points3d(x, y, z, enr_tensor, transparent=True, mode='sphere', colormap='hot')
        
        #for i,mer in enumerate(kmer_list):
            #mlab.text3d(i,0,0, mer, orient_to_camera=False, orientation = ( 0, -2, -90), scale=.5)
            #mlab.text3d(0,i,0, mer, orient_to_camera=False, orientation = ( -2, 0, 190), scale=.5)
        #mlab.xlabel('first kmer')
        #mlab.ylabel('second kmer')
        #mlab.zlabel('spacing')
        #mlab.colorbar()
        #mlab.show()

        
    def find_interactors(self, k, n_top=2, k_flank_max=4):
        
        for core in self.select_significant_kmers(k, n_top):
            for k_int in range(1, k_flank_max+1):
                t0 = time.time()
                screen = PairInteractionScreen(self.runs[(k, self.best_rbp_conc)], core, k_int)
                t1 = time.time()
                self.logger.debug("interaction analysis of {0} with {1}mers took {2:.3f}s".format(core, k_int, (t1-t0)) )
                
                fplot = os.path.join(self.out_path, "{0}_interacting_with_{1}mers.pdf".format(core, k_int) )
                screen.make_plot(fplot)
    

class PairInteractionScreen(object):
    def __init__(self, res, core, k_int, pseudo=10.):
        self.res = res
        self.core = core
        self.k_int = k_int
        self.k_core = len(core)
        
        #print "scanning flanking {0}-mers".format(k_int)
        pd_matrix, pd_mask = self.res.pd_reads.kmer_flank_profiles(core, k_int)
        in_matrix, in_mask = self.res.in_reads.kmer_flank_profiles(core, k_int)

        self.core_density_pd = pd_mask.sum(axis=0)
        self.core_density_bg = in_mask.sum(axis=0)
        
        pd_matrix = np.array(pd_matrix, dtype=np.float32) + pseudo
        in_matrix = np.array(in_matrix, dtype=np.float32) + pseudo

        obsv = pd_matrix / pd_matrix.sum(axis=0)[np.newaxis,:]
        bgnd = in_matrix / in_matrix.sum(axis=0)[np.newaxis,:]

        # Kullback-Leibler (KL) divergence terms
        self._KL = (obsv * np.log2(obsv / bgnd) )
        # and per-position KL
        self.KL = self._KL.sum(axis=0)
        self.log_ratios = np.log2(obsv / bgnd)

        # mask positions overlapping with the core motif
        self.l = self.res.pd_reads.L - self.k_core
        self.KL[self.l-self.k_int+1:self.l+self.k_core] = 0
        self.log_ratios[:,self.l-self.k_int+1:self.l+self.k_core] = 0
        #print "mask",self.l-self.k_int+1,self.l+self.k_core 
        #print "log_ratios after masking", self.log_ratios[:,self.l-self.k_int+1:self.l+self.k_core]

    def top_interactors(self, n_top=10):
        #TODO: cook a set of candidate interacting kmers from significance for now let's just take top 10
        top_i = self._KL.max(axis=1).argsort()[::-1][:n_top]
        # and sort alphabetically to ensure reproducibility across successive runs
        top_i = sorted(top_i)
        top_kmers = [cska.ska_kmers.index_to_seq(i, self.k_int) for i in top_i]
        #print "top interacting kmer candidate list", top_kmers
        return top_i, top_kmers
    
    def make_plot(self, fname, n_top=10):
        import matplotlib as mp
        mp.rcParams['font.family'] = 'Arial'
        mp.rcParams['font.size'] = 8
        mp.rcParams['font.sans-serif'] = 'Arial'
        mp.rcParams['legend.fontsize'] = 'small'
        mp.rcParams['legend.frameon'] = False
        #mp.rcParams['axes.labelsize'] = 8
        
        import matplotlib.pyplot as pp
        fig = pp.figure()
        fig.subplots_adjust(hspace=0.5)
        
        pp.title("{self.res.pd_reads.rbp_name}@{self.res.pd_reads.rbp_conc}nM {self.core} interacting with {self.k_int}-mers".format(self=self))
        pp.subplot(311)
        pp.gca().set_title("density of {0} core".format(self.core.upper()))
        pp.plot(self.core_density_pd, drawstyle='steps-mid', label="pd")
        pp.plot(self.core_density_bg, drawstyle='steps-mid', label="in")
        pp.gca().locator_params(axis='y',nbins=3)
        pp.gca().locator_params(axis='x',nbins=10)

        pp.legend(loc='upper left')
        pp.xlabel("read start pos [nt]")
        pp.ylabel("frequency")
        
        pp.subplot(312)
        pp.title("Kullback-Leibler divergence of flanking kmer composition")
        x = np.arange(len(self.KL)) - len(self.KL)/2 +1.
        pp.plot(x,self.KL, drawstyle='steps-mid', label="{0}mers around {1}".format(self.k_int, self.k_core) )
        pp.xlim(-self.l-.5,self.l+.5)
        pp.xlabel("rel. {0}-mer start pos [nt]".format(self.k_int))
        pp.ylabel("KL [bits]")
     
        pp.subplot(313)
        pp.gca().set_title("enriched {0}-mers".format(self.k_int))
        top_i, top_kmers = self.top_interactors(n_top = n_top)
        z = self.log_ratios[top_i[::-1],:]
        y, x = np.mgrid[slice(0, len(top_i)+1),slice(-(self.l+.5), +self.l+1)]
        
        # symmetric, dynamic range of colorbar
        dr = np.fabs(z).max()
        pp.pcolor(x,y,z, cmap=pp.get_cmap('seismic'), vmin=-dr, vmax=dr)

        pp.yticks(np.arange(len(top_i))+.5, top_kmers[::-1]) # reverse order of kmers so pcolor is not upside down
        pp.xlim(-self.l-.5,self.l+.5)
        cbar = pp.colorbar(orientation="horizontal", fraction=0.1, shrink=0.75, label=r"$\log_2( \frac{pd}{in} )$")
        cbar.ax.tick_params(labelsize=8)
        pp.savefig(fname)
        pp.close()
        
