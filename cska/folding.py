#!/usr/bin/env python
import sys
import os
import logging
import copy
import time
import numpy as np
import cPickle as pickle
from subprocess import PIPE, Popen
#from threading  import Thread, Event
from multiprocessing import Process, Event, JoinableQueue as Queue
import multiprocessing
from Queue import Empty
from collections import defaultdict
from cska.caching import CachedBase, cached, pickled

logger = logging.getLogger("cska.folding")

def getsizeof(obj):
    size = sys.getsizeof(obj)
    if hasattr(obj, "items"):
        #print "dict", type(obj)
        for o in obj.items():
            size += getsizeof(o)
    
    elif hasattr(obj, "__iter__"):
        #print "iterable", type(obj)
        for o in obj:
            size += getsizeof(o)
    elif hasattr(obj, "nbytes"):
        size += obj.nbytes
        
    return size


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

class RNAplfoldChunk(object):
    min_k = 3
    max_k = 8
    start = 0
    end = 1E6

    def __init__(self):
        self.n = 0
        self.openen = defaultdict(lambda : defaultdict(list))
        
    def set_header(self, head):
        self.seq = head[1:].rstrip().upper().replace('U','T')
        
    def add_line(self, line):
        self.n += 1
        if self.n < self.min_k:
            return

        if self.n < self.start:
            return

        if self.n >= self.end:
            return
        
        for k in range(self.min_k, min(self.max_k+1, self.n+1)):
            kmer = self.seq[self.n-k:self.n+1]
            col = line.split('\t')[k]
            self.openen[k][kmer].append(float(col) )

    def get_dict(self, k):
        return self.openen[k]
        
        
def plfold_chunks(src):
    chunk = RNAplfoldChunk()
    n = 0
    for line in src:
        if line.startswith('>'):
            if chunk.n: 
                n += 1
                yield chunk
                
            chunk = RNAplfoldChunk()
            chunk.set_header(line)
        elif line.startswith('#'): continue
        elif line.startswith(' #'): continue
        else: 
            chunk.add_line(line)
    
    if chunk.n: 
        n += 1
        yield chunk
        
    logger.info('plfold_chunks: no more data. exiting after {0} chunks...'.format(n))
    

def RNAplfold_dispatch_reads(src, processes, adap5, adap3, interrupt_event, n_max=0):
    N = len(processes)
    t0 = time.time()
    n = 0
    for i,line in enumerate(src):
        #print "dispatch", i
        read = line.rstrip()
        if 'N' in read:
            continue
        
        n += 1
        # round-robin
        proc = processes[i % N]
        seq = adap5 + read + adap3
        
        # use entire sequence as FASTA header to aid parsing threads
        proc.stdin.write(">{0}\n{0}\n".format(seq)) 
        if interrupt_event.is_set():
            break

        if n_max and n >= n_max:
            break
        
        if n and not n % 1000:
            t = time.time()

            ident = multiprocessing.current_process().name
            rps = 1000. / (t-t0)
            logger.debug("subprocess {1} dispatched {0}k reads. current rate is {2:.2f} reads/second".format(n/1000, ident, rps) )
            
            t0 = t
            
    for proc in processes:
        proc.stdin.flush()

    logger.info('ending dispatch after {0} reads'.format(n))

class Binned(dict):
    pass


def RNAplfold_output_handler(out, queue, bins, n_chunk=1000):
    
    cache = defaultdict(lambda : defaultdict(list))
    n = 0
    n_total = 0
    def digest():
        binned = Binned()
        binned.n = n
        
        t0 = time.time()
        for k, data in cache.items():
            kmer_binned = {}
            for kmer, openen in data.items():
                counts, b = np.histogram(np.array(openen), bins=bins)
                kmer_binned[kmer] = counts
            
            binned[k] = kmer_binned
            
            
        ident = multiprocessing.current_process().name
        logger.debug("subprocess {1} binned cached openen data of {2} chunks in {0:.2f}ms".format(1000.*(time.time()-t0), ident, binned.n) )
        
        return binned
    
    for i,chunk in enumerate(plfold_chunks(iter(out.readline, b''))):
        
        #if not chunk.openen:
            #logger.debug("skipping empty chunk")
            #continue
        
        # aggregate kmer openen data
        n += 1
        n_total += 1
        for k in chunk.openen.keys():
            for kmer, openen in chunk.get_dict(k).items():
                cache[k][kmer].extend(openen)
        
        if i and not (i % n_chunk):
            # time to bin the data
            queue.put(digest())
            
            # release memory
            cache = defaultdict(lambda : defaultdict(list))
            n = 0

    logger.debug("RNAplfold_output_handler exiting after {0} chunks processed {1} still in queue".format(n_total, n))
    # flush out results
    if n:
        queue.put(digest())
        
    # release memory
    cache = defaultdict(lambda : defaultdict(list))
    
    out.close()
    queue.join()



class ThreadManager(object):
    def __init__(self, n_threads=8):
        #self.name = name
        self.n_threads = n_threads
        self.logger = logging.getLogger("ThreadManager({0})".format(n_threads))

    def process_reads(self, src, ohc, min_k=3, max_k=8, vienna_bin="RNAplfold_cska", adap5="gggaguucuacaguccgacgauc", adap3="uggaauucucgggugucaagg", l=84, temp=22, n_max=0, pickle_interval=20*60, qjam=100):
        cmd=[vienna_bin, "-O", "-u {0}".format(max_k), "-W {0}".format(l), "-L {0}".format(l), "-T {0}".format(temp)]
        self.processes = []
        self.queues = []
        self.parse_threads = []
        self.q = Queue()

        
        interrupt_event = Event()

        # Start the RNAplfold instances
        for n in range(self.n_threads):
            p = Popen(
                cmd, 
                stdin=PIPE, 
                stdout=PIPE, 
                bufsize=0, 
                close_fds=True
            )
            self.processes.append(p)
        
        # prepare the FASTA line round-robin dispatcher thread
        self.dispatch_thread = Process(
            target=RNAplfold_dispatch_reads, 
            args=(src, self.processes, adap5, adap3, interrupt_event),
            kwargs=dict(n_max = n_max)
        )
        self.dispatch_thread.daemon = True # thread dies with the program

        # prepare the output parsing threads
        RNAplfoldChunk.min_k = min_k
        RNAplfoldChunk.max_k = max_k
        
        ## skip the adapters
        #RNAplfoldChunk.start = len(adap5)
        #RNAplfoldChunk.end = l - len(adap3)
        
        for n in range(self.n_threads):
            t = Process(
                target=RNAplfold_output_handler,
                args=(self.processes[n].stdout, self.q, ohc.bins)
            )
            #(out, queue, bins, throttle=100, n_chunk=10000):
            t.daemon = True # thread dies with the program
            
            # will block until there is RNAplfold output or pipes closed
            t.start()
            self.parse_threads.append(t)
        
        # start the dispatch:
        self.dispatch_thread.start()
        
        def pending():
            done = True
            for t in self.parse_threads:
                if t.is_alive():
                    done = False
            return not done

        n = 0
        shutdown = False
        t0 = time.time()
        T0 = t0
        while pending():
            t = time.time()
            elapsed = t - t0

            # pickle current data every now and then
            if elapsed > pickle_interval and self.q.qsize() < qjam and n:
                self.logger.debug("{0:.1f}sec since last pickle. Storing current data derived from {1:.3f} k reads".format(elapsed, n/1e3))
                ohc.store_pickle(suffix="_temp")
                self.logger.debug("queue has grown to {0} entries during pickle".format(self.q.qsize()))
                t0 = t

            # are we done?
            if not self.dispatch_thread.is_alive() and not shutdown:
                self.logger.info('dispatcher exited. Closing stdin pipes')
                # signal to RNAplfold to exit
                for proc in self.processes:
                    proc.stdin.flush()
                    proc.stdin.close()
                    
                time.sleep(5)
                self.logger.info('terminating workers')
                for proc in self.processes:
                    if proc.poll() == None:
                        proc.terminate()
                
                shutdown = True

            try:
                binned = self.q.get(True, 5.)
                
                if self.q.qsize() > qjam:
                    self.logger.debug("queue depth {0}".format(self.q.qsize()))
                    
                self.logger.debug("adding binned data from {0} reads".format(binned.n))
                for k, kmer_binned in binned.items():
                    ohc.add_binned(k, kmer_binned)
                self.q.task_done()
                n += binned.n

            except Empty:
                pass

        elapsed = time.time() - T0
        self.logger.debug("finished processing {0} reads after {1:.1f}sec. Storing data".format(n, elapsed))
        ohc.store_pickle()
                
        return ohc

def test_memory_consumption():
    oa = OpenenHistCollection(name="random_test")
    print "testing kmers"
    for chunk in random_plfold_chunks(8,N=1E7):
        oa.add_chunk(chunk)

def random_plfold_chunks(k, N=0, L=84):
    import cska.ska_kmers
    n = 0
    while True:
        chunk = RNAplfoldChunk()
        l = L-k+1
        openen = [np.random.random(size=l) * 20]
        kmers = [cska.ska_kmers.index_to_seq(i,k) for i in np.random.randint(0,4**k,size=l)]
        chunk.openen[k] = zip(kmers, openen)
        
        yield chunk
        n += 1
        if N and n >=N:
            break
        



class RNAplfoldOutput(object):
    min_k = 3
    max_k = 8
    start = 0
    end = 1E6

    def __init__(self):
        self.n = 0
        self.openen = defaultdict(lambda : defaultdict(list))
        
    def set_header(self, head):
        self.seq = head[1:].rstrip().upper().replace('U','T')
        
    def add_line(self, line):
        self.n += 1
        if self.n < self.min_k:
            return

        if self.n < self.start:
            return

        if self.n >= self.end:
            return
        
        for k in range(self.min_k, min(self.max_k+1, self.n+1)):
            kmer = self.seq[self.n-k:self.n+1]
            col = line.split('\t')[k]
            self.openen[k][kmer].append(float(col) )

    def get_dict(self, k):
        return self.openen[k]
        
        
def binned(src):
    out = RNAplfoldOutput()
    n = 0
    for line in src:
        if line.startswith('>'):
            if out.n: 
                n += 1
                yield out
                
            out = RNAplfoldOutput()
            out.set_header(line)
        elif line.startswith('#'): continue
        elif line.startswith(' #'): continue
        else: 
            out.add_line(line)
    
    if out.n: 
        n += 1
        yield out
        
    logger.info('binned(): no more data. exiting after {0} outs...'.format(n))


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

        self.logger.debug('folded {0} sequences'.format(n))
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



### TODO: Wrap up in a class that works together with OpenenStorage
def gamma_bins(params,n=256):
    """
    Create histogram bins equally spanning the percentiles of a
    gamma distribution with the given parameters.
    Provided the data are (approximately) from this distribution, 
    this binning would assign equal data points to each bin and thus 
    reduce the loss of precision inherent in the binning to a minimum.
    """
    step = 1./n
    q = np.arange(0,1.+step,step) # n+1 "percentiles"
    return scipy.stats.gamma.ppf(q, *params)
        
def gamma_bins_k(k,dtype=np.uint8):
    """
    Get optimal bins for open energies of k-mers using predetermined 
    parameters for the gamma distribution that best approximate the 
    open energy distribution.
    """
    n = 2**(dtype().nbytes*8) # highest number of bins encodable by dtype
    params = opt_k_openen_gamma_params[k]
    return gamma_bins(params, n=n)

opt_k_openen_gamma_params = {
    # for 40mer inserts
    3 : (1.0628281681967706, -6.8214607159662128e-05, 1.2402078509734689),
    4 : (1.2674929471836962, -0.0020650480683494271, 1.2929616570525289),
    5 : (1.4888004568584394, -0.0081636662031228657, 1.3237981203329996),
    6 : (1.7285627206360989, -0.021580429627936448, 1.3376752736147013),
    7 : (2.0032560511697439, -0.046740618049563296, 1.3323232531823281),
    8 : (2.2847020288586801, -0.086073860842223043, 1.3242768850690814),
}


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


def seq_dispatcher(src, queue, chunk_size=50, max_depth=20, throttle_sleep=1., **kwargs):

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
                logger.debug('qsize > {0} -> sleeping for {1} second'.format(max_depth, throttle_sleep) )
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
    
    for n_chunk, results in queue_iter(res_queue):
        heapq.heappush(heap, (n_chunk, results) )
        while(heap and (heap[0][0] == n_chunk_needed)):
            n_chunk, results = heapq.heappop(heap)
            for krange, data in results:
                storage.store_set(krange, data)
        
            n_chunk_needed += 1

    # by the time None pops from the queue, all chunks 
    # should have been processed!
    assert len(heap) == 0

    # close all open files and make sure stuff is on disk
    storage.close()
    
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

    src = file('/scratch/data/RBNS/RBFOX2/RBFOX2_input.reads').readlines()[:351]
    storage = OpenenStorage(path='tmp')
    parallel_fold(src, storage, n_parallel=8)
    sys.exit(1)

    #test_memory_consumption()

    k8 = np.fromfile('openen.8.raw-float32.bin', dtype=np.float32)
    k7 = np.fromfile('openen.7.raw-float32.bin', dtype=np.float32)
    k6 = np.fromfile('openen.6.raw-float32.bin', dtype=np.float32)
    k5 = np.fromfile('openen.5.raw-float32.bin', dtype=np.float32)
    k4 = np.fromfile('openen.4.raw-float32.bin', dtype=np.float32)
    k3 = np.fromfile('openen.3.raw-float32.bin', dtype=np.float32)
    
    
    import scipy.stats
    import matplotlib.pyplot as pp

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
             
