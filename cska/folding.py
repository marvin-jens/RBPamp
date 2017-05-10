#!/usr/bin/env python
import sys
import os
import logging
import copy
import time
import numpy as np
import cPickle as pickle
from subprocess import PIPE, Popen
from multiprocessing import Process, Event, JoinableQueue as Queue
import multiprocessing
from Queue import Empty
from collections import defaultdict
from cska.caching import CachedBase, cached, pickled

logger = logging.getLogger("cska.folding")

class OpenenHistCollection(object):
    def __init__(self, name="openen.hist", path=".", min_en=0., max_en=20., n_bins=100, n_chunk=100000):
        
        self.path = path
        self.name = name
        self.n_chunk = n_chunk

        step = (max_en-min_en)/n_bins
        self.bins = np.arange(min_en, max_en+step, step)
        self.bins[-1] = np.inf

        #self.kmer_raw = defaultdict(lambda : defaultdict(list))
        self.kmer_binned = {}

        self.n_raw = 0
        self.logger = logging.getLogger("OpenenHistCollection({0})".format(self.name) )
        self.pickles_completed = {}
        self.digest_completed = {}
    
    def add_binned(self, k , binned_data):
        if not k in self.kmer_binned:
            self.kmer_binned[k] = {}
            
        for kmer, counts in binned_data.items():
            if not kmer in self.kmer_binned[k]:
                self.kmer_binned[k][kmer] = counts
            else:
                self.kmer_binned[k][kmer] += counts
        
    def store_pickle(self, suffix="", skip=False, sync=True):
            
        for k in sorted(self.kmer_binned.keys()):
            fname = os.path.join(self.path, "{self.name}{suffix}.{k}mers.pkl".format(**locals()) )
            self.logger.debug("store_pickle('{0}')".format(fname) )
            pickle.dump( (self.bins, k, self.kmer_binned[k]), file(fname, 'wb') )

    @classmethod
    def from_pickle(cls, k, name="openen.hist", path="./", suffix=""):
        OC = cls(name=name, path=path)
        fname = os.path.join(path, "{name}{suffix}.{k}mers.pkl".format(**locals()) )
        OC.load_pickle(fname)
        return OC
        
    def load_pickle(self, path):
        self.logger.debug("load_pickle('{0}')".format(path) )
        self.bins, k, kmer_binned = pickle.load(file(path, 'rb') )
        #print kmer_binned.keys()[0]
        # HACK, CLUDGE, WORKAROUND, HOTFIX, REMOVE!!!!
        self.kmer_binned[k+1] = kmer_binned # TODO: clean up OpenenHistCollection creation to fix this bug!
   
    def __getitem__(self, kmer):
        k = len(kmer)
        return self.kmer_binned[k][kmer]

   
    def occ(self, kmer, P, k_bare, disable=False, temp=22.):
        """
        mid-point integration of the binding equation over the empirical 
        open-energy distribution.
        """
        k = len(kmer)
        counts = self.kmer_binned[k][kmer]

        RT = (temp + 273.15) * 8.314459848/4.184E3# RT in kcal/mol
        #U = (self.bins[:-1] + self.bins[1:]) / 2.
        U = self.bins[:-1]
        acc = np.exp(-U/RT)

        Z = np.trapz(counts, U)
        fU = counts / Z
        
        integrand = fU * P/ (P + k_bare / acc)
        return np.trapz(integrand, U)


    def k_bare_from_occ(self, kmer, P, occ_est, min_k = 1e-9, max_k=1e6, temp=22.):
        """
        Given the observed open-energy distrubtion for the kmer, find the 
        Kd_bare that best predicts the observed (or estimated) occupancy at 
        the given concentration.
        Uses `brentq` root finding from scipy.optimize
        """
        k = len(kmer)
        counts = self.kmer_binned[k][kmer]

        RT = (temp + 273.15) * 8.314459848/4.184E3# RT in kcal/mol
        #U = (self.bins[:-1] + self.bins[1:]) / 2.
        U = self.bins[:-1]
        acc = np.exp(-U/RT)

        Z = np.trapz(counts, U)
        fU = counts / Z
        
        def err(k_bare):
            integrand = fU * P / (P + k_bare / acc)
            predict = np.trapz(integrand, U)
            #print k_bare, predict, occ_est
            return occ_est - predict
        
        #print kmer, P, occ_est, "err", err(min_k), err(max_k)
        from scipy.optimize import minimize, newton, brentq
        fit = brentq(err, min_k, max_k)
        #print "optimum kd", fit
        
        return fit
    
    def test_k_bare_from_occ(self, kmer, P, **kwargs):
        occ_est = np.arange(1e-4, 1, 1e-4)
        k_fit = np.array([self.k_bare_from_occ(kmer, P, o, **kwargs) for o in occ_est])
        
        import matplotlib.pyplot as pp
        pp.figure()
        pp.loglog(occ_est, k_fit)
        pp.loglog(occ_est, P/occ_est - P, linestyle='dashed', color='k')
        pp.xlabel(r'$\theta$')
        pp.ylabel(r'$K_d$')
        pp.savefig('occ_test.pdf')
        
        return occ_est, k_fit


class ViennaOpenen(object):
    def __init__(self, k_min=3, k_max=8, temp=22., adap5="gggaguucuacaguccgacgauc", adap3="uggaauucucgggugucaagg", vienna_bin="RNAplfold_cska", L=84, skip_adap=True, **kwargs):
        cmd=[vienna_bin, "-O", "-u {0}".format(k_max), "-W {0}".format(L), "-L {0}".format(L), "-T {0}".format(temp)]
        self.cmd = " ".join(cmd)
        self.p = Popen(
            cmd, 
            stdin=PIPE, 
            stdout=PIPE, 
            bufsize=1, 
            close_fds=True
        )
        
        self.k_indices = np.arange(k_min, k_max+1)
        
        self.first = 0
        self.last = L
        
        if skip_adap:
            self.first = len(adap5)
            self.last = L - len(adap3)

        self.l_insert = self.last - self.first
        self.krange = np.arange(k_min, k_max+1)
        self.k_min = k_min
        self.k_max = k_max
        
        self.adap5 = adap5
        self.adap3 = adap3
        
        self.n_total = 0
        self.logger = logging.getLogger('ViennaOpenen')

    def process_sequences(self, seq_src):
        n = 0
        for seq in seq_src:
            S = self.adap5 + seq.rstrip() + self.adap3
            l = len(S)
            #print "folding", S, len(S)
            self.p.stdin.write("{0}\n".format(S) )
            
            data = [ np.zeros(self.l_insert - k + 1, dtype=np.float32) for k in self.krange ]
            for i in range(l+2):
                j = i - 1
                
                line = self.p.stdout.readline()
                if j <= self.first:
                    continue
                
                if j >= self.last:
                    continue
                
                cols = line.split('\t')
                for k in self.krange:
                    if j >= k:
                        data[k - self.k_min][j-k-self.first] = float(cols[k])

            yield self.krange, data
            n += 1

        #self.logger.debug('folded {0} sequences'.format(n))
        self.n_total += n
        

    def close(self):
        self.p.stdin.close()
        self.p.stdin.close()
        ex = self.p.wait()
        self.logger.debug('close(): {0} exited with code {1} after folding {2} sequences'.format(self.cmd, ex, self.n_total) )
    

class OpenenStorage(object):
    def __init__(self, path='./', name="openen", bins=None, dtype=np.float32):
        self.path = path
        self.name = name
        self.bins = bins
        self.dtype = dtype
        
        self.k_sinks = {}
        self.k_bins = {}
        self.logger = logging.getLogger('OpenenStorage')
        self.n_sets = 0
        
    def get_or_create(self, k):
        if self.bins:
            fmt = "bins-{0}".format(self.dtype.__name__)
        else:
            fmt = "raw-{0}".format(self.dtype.__name__)

        if not k in self.k_sinks:
            fname = os.path.join(self.path, "{0}.{1}.{2}.bin".format(self.name,k,fmt) )
            self.k_sinks[k] = file(fname,'wb')
            self.logger.info("created '{0}'".format(fname))
    
        return self.k_sinks[k]
    
    def store(self, k, vec):
        if self.bins:
            vec = np.array(np.digitize(vec, self.bins)-1, dtype=self.dtype)

        self.get_or_create(k).write(vec.tobytes())

    def store_set(self, krange, data):
        for k, vec in zip(krange, data):
            self.store(k, vec)
        self.n_sets += 1
        
    def close(self):
        for sink in self.k_sinks.values():
            sink.close()
        self.logger.info("closed all files after writing {0} data sets".format(self.n_sets) )


class OpenenDiscretization(object):
    """
    To save space and time, open-energy values are discretized. This class offers the tools
    to determine optimal bins, discretize floating point raw values, and convert 
    (approximately) back.
    """
    
    # parameters of the gamma distribution that best approximate the k-mer open energy
    # distribution observed for a sample of real RBNS input reads of length L. 
    # key is (L, k)
    opt_gamma_params = {
        (40,3) : (1.0628281681967706, -6.8214607159662128e-05, 1.2402078509734689),
        (40,4) : (1.2674929471836962, -0.0020650480683494271, 1.2929616570525289),
        (40,5) : (1.4888004568584394, -0.0081636662031228657, 1.3237981203329996),
        (40,6) : (1.7285627206360989, -0.021580429627936448, 1.3376752736147013),
        (40,7) : (2.0032560511697439, -0.046740618049563296, 1.3323232531823281),
        (40,8) : (2.2847020288586801, -0.086073860842223043, 1.3242768850690814),
    }

    def __init__(self, k, L, dtype):
        self.n = 2**(dtype().nbytes*8) # highest number of bins encodable by dtype
        self.k = k
        self.L = L
        self.dtype = dtype
        
        step = 1./self.n
        q = np.arange(0,1.+step,step) # n+1 "percentiles"
    
        import scipy.stats
        params = OpenenDiscretization.opt_gamma_params[(L, k)]
        
        # compute optimal bin boundaries
        self.bins = scipy.stats.gamma.ppf(q, *params)
        
        # compute openen values that optimally represent each bin
        q_x = q[:-1] + 0.5*step
        self.x = scipy.stats.gamma.ppf(q_x, *params)

    @staticmethod
    def from_filename(fname):
        import re
        M = re.search(r'discretized_gamma_(?P<L>\d+)_(?P<k>\d+)_(?P<dtype>\w+)', fname)
        d = M.groupdict()
        L = int(d['L'])
        k = int(d['k'])
        dtype_name = d['dtype']
        dtype = getattr(np, dtype_name)
        
        return OpenenDiscretization(k, L, dtype)
        
    def to_filename(self):
        return "discretized_gamma_{0}_{1}_{2}".format(self.L, self.k, self.dtype.__name__)
        
    def discretize(self, data):
        bins = self.bins
        return np.digitize(data, bins) - 1

        
class RBNSOpenen(CachedBase):
    """
    Analogous to RBNSReads, which holds the raw sequences, instances of this class hold 
    open-energies for all kmer start-positions inside the raw sequences.
    """
    def __init__(self, fname, rbns_reads, k, oem=[]):

        CachedBase.__init__(self)

        self.fname = fname
        self.rbns_reads = rbns_reads
        self.k = k
        
        self.logger = logging.getLogger('RBNSOpenen({self.fname} k={self.k})'.format(self=self))
        if len(oem):
            self.cache_preload("__cached_oem", oem)
            N, L = oem.shape
            self.cache_preload("__cached_N", N)
            self.cache_preload("__cached_L", L)
        else:
            self.is_subsample = False
        
        self.discretized = ("discretized" in self.fname)
        
        if self.discretized:
            # recovering discretization scheme from file-name
            self.disc = OpenenDiscretization.from_filename(fname)
            self.dtype = self.disc.dtype
        else:
            # we have the raw floating point values
            self.disc = None
            self.dtype = np.float32

        self.logger.info("initialized")
    @property
    def cache_key(self):
        return "{self.fname}".format(self=self)

    @property
    @cached
    @pickled
    def N(self):
        N, L = self.oem.shape
        return N

    @property
    @cached
    @pickled
    def L(self):
        N, L = self.oem.shape
        return L
   
    @property
    @cached
    def oem(self):
        """
        load and keep all open-energies in memory (optionally discretized)
        """
        L = self.rbns_reads.L - self.k + 1
        oem = np.fromfile(self.fname, dtype=self.dtype)
        if self.rbns_reads.n_max:
            oem = oem[:L*self.rbns_reads.n_max]

        N = len(oem) / L
        
        oem = np.reshape(oem, (N,L) )
        #print oem.shape
        return oem
    
    def discretize(self, disc=None):
        if not disc:
            disc = OpenenDiscretization(self.k, self.L+self.k-1, np.uint8)

        d_oem = disc.discretize(self.oem)
        path, fname = os.path.split(self.fname)
        dname = os.path.join(path, "{0}.{1}".format(disc.to_filename(), fname) )

        print dname
        doe = RBNSOpenen(
            dname,
            self.rbns_reads,
            self.k,
            oem = d_oem
        )
        
        return doe
    
    def integrate(self, integrand, bin_weights):
        assert self.discretized
        return np.trapz(integrand * bin_weights, self.disc.x)

    def store(self):
        self.oem.tofile(self.fname)


# Here come a couple of functions that allow parallel folding using the multiprocessing 
# module and RNAplfold
def queue_iter(queue, stop_item = None):
    """
    Small generator/wrapper around multiprocessing.Queue allowing simple
    for-loop semantics: 
    
        for item in queue_iter(queue):
            ...

    """
    while True:
        item = queue.get()
        if item == stop_item:
            # signals end->exit
            break
        else:
            yield item


def seq_dispatcher(src, queue, chunk_size=100, max_depth=50, throttle_sleep=1., **kwargs):

    logger = logging.getLogger('seq_dispatcher')
    chunk = []
    n_chunk = 0
    n_seqs = 0
    for read in src:
        chunk.append( read )
        n_seqs += 1
        if len(chunk) >= chunk_size:
            # avoid overloading the queue
            while queue.qsize() > max_depth:
                #logger.debug('qsize > {0} -> sleeping for {1} second'.format(max_depth, throttle_sleep) )
                time.sleep(throttle_sleep)

            queue.put( (n_chunk, chunk) )
            n_chunk += 1
            chunk = []

    if chunk:
        queue.put( (n_chunk, chunk) )
        n_chunk += 1

    logger.info('{0} sequences dispatched in {1} chunks. Closing down.'.format(n_seqs, n_chunk) )

def fold_worker(seq_queue, data_queue, **vienna_kwargs):

    vienna = ViennaOpenen(**vienna_kwargs)
    for n_block, block in queue_iter(seq_queue):
        # received a chunk of sequences. Fold them en-bloc
        results = list(vienna.process_sequences(block))
        
        # and return results
        data_queue.put( (n_block, results) )
        
    # cleaning up
    vienna.close()


def result_collector(storage, res_queue):
    import heapq
    heap = []
    n_chunk_needed = 0
    t0 = time.time()
    t1 = t0
    n_rec = 0

    logger = logging.getLogger('result_collector')
    for n_chunk, results in queue_iter(res_queue):
        heapq.heappush(heap, (n_chunk, results) )
        while(heap and (heap[0][0] == n_chunk_needed)):
            n_chunk, results = heapq.heappop(heap)
            for krange, data in results:
                storage.store_set(krange, data)
                n_rec += 1
        
            n_chunk_needed += 1

        t2 = time.time()
        if t2-t1 > 10:
            dT = t2 - t0
            logger.debug("processed {0} records in {1:.0f} seconds (average {2:.3f} records/second)".format(n_rec, dT, n_rec/dT) )
            t1 = t2
        
    # by the time None pops from the queue, all chunks 
    # should have been processed!
    assert len(heap) == 0

    # close all open files and make sure stuff is on disk
    storage.close()
    dT = time.time() - t0
    logger.debug("finished processing {0} records in {1:.0f} seconds (average {2:.3f} records/second)".format(n_rec, dT, n_rec/dT) )
    
def parallel_fold(src, storage, n_parallel=8, **kwargs):
    
    import multiprocessing
    seq_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    
    dispatcher = multiprocessing.Process(
        target = seq_dispatcher, 
        name='seq_dispatcher', 
        args=(src, seq_queue), 
        kwargs=kwargs 
    )
    dispatcher.daemon = True
    dispatcher.start()
    
    workers = []
    for n in range(n_parallel):
        worker = multiprocessing.Process(
            target = fold_worker, 
            name='fold_worker_{0}'.format(n), 
            args=(seq_queue, res_queue), 
            kwargs=kwargs
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    collector = multiprocessing.Process(
        target = result_collector,
        name = 'result_collector',
        args = (storage, res_queue)
    )
    collector.daemon = True
    collector.start()
    
    # wait until all sequences have been thrown onto seq_queue
    dispatcher.join()
    # signal all fold-workers to finish
    for n in range(n_parallel):
        seq_queue.put(None)

    for worker in workers:
        # make sure all results are on res_queue
        worker.join()
    
    # signal the collector to stop
    res_queue.put(None)
    collector.join()
   
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    #src = file('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads') #.readlines()[:30051]
    #storage = OpenenStorage(path='tmp')
    #parallel_fold(src, storage, n_parallel=10)
    #sys.exit(1)

    #test_memory_consumption()

    from cska.rbns_reads import RBNSReads
    reads = RBNSReads('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads', n_max=100000)
    openen = RBNSOpenen('tmp/openen.7.raw-float32.bin', reads, 7)
    
    dopenen = openen.discretize()
    print dopenen.oem[:10]
    
    sys.exit(0)
    
    k8 = np.fromfile('openen.8.raw-float32.bin', dtype=np.float32)
    k7 = np.fromfile('openen.7.raw-float32.bin', dtype=np.float32)
    k6 = np.fromfile('openen.6.raw-float32.bin', dtype=np.float32)
    k5 = np.fromfile('openen.5.raw-float32.bin', dtype=np.float32)
    k4 = np.fromfile('openen.4.raw-float32.bin', dtype=np.float32)
    k3 = np.fromfile('openen.3.raw-float32.bin', dtype=np.float32)
    
    import matplotlib.pyplot as pp
    
    L=40
    k=7
    disc = OpenenDiscretization(7,40,np.uint8)
    print disc.bins
    mid = 0.5*(disc.bins[1:] + disc.bins[:-1])
    pp.loglog(mid, disc.x)
    pp.show()
    sys.exit(1)
    
    

    #dist = scipy.stats.beta # gamma
    dist = scipy.stats.gamma
    data = k8
    
    #params = dist.fit(data)
    #print params
    bins = gamma_bins_k(8,dtype=np.uint8)
    print "low bins",bins[:10]
    print "low data",sorted(data)[:10]
    print "low data->bins", np.digitize(sorted(data)[:10], bins) -1
    dig = np.digitize(data, bins) -1
    
    print dig.min(), dig.max()
    print bins, np.bincount(dig)
    
    mid = 0.5*(bins[1:] + bins[:-1]) # mid-points
    print len(bins)
    x = np.arange(0, data.max(), .01)
    #pp.hist(data, bins=bins, normed=True)
    #pp.plot(x, dist.pdf(x, *params))
    pp.loglog(data, mid[dig],'ob')
    
    RMSD = np.sqrt(np.mean((mid[dig] - data)**2))
    print "RMSD",RMSD
    pp.show()
    sys.exit(1)
    
    print make_bins(5)
    src = file('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads')
    store = OpenenStorage()
    vienna = ViennaOpenen()

    import time
    t0 = time.time()
    for n, krange, data in vienna.process_sequences(src):
    #for n, krange, data in vienna_openen(src, L=64):
        store.store_set(krange, data)
        if n and not n % 1000:
            t1 = time.time()
            print "{0:.2f} seqs/second".format(1000./(t1-t0))
            t0 = t1
    
    #oa = OpenenHistCollection(name=sys.argv[1])
    
    #tm = ThreadManager(n_threads=4)
    #tm.process_reads(sys.stdin, oa)
    
    #print oa['TGCATGT']
             
