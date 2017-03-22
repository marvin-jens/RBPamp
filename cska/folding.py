#!/usr/bin/env python
import sys
import os
import logging
import numpy as np
import cPickle as pickle
from subprocess import PIPE, Popen
from threading  import Thread, Event
from Queue import Queue, Empty
from collections import defaultdict
from cska.caching import CachedBase, cached, pickled


class OpenenHistCollection(object):
    def __init__(self, name="openen.hist", path=".", min_en=0., max_en=20., n_bins=100, n_chunk=10000):
        
        self.path = path
        self.name = name
        self.n_chunk = n_chunk

        step = (max_en-min_en)/n_bins
        self.bins = np.arange(min_en, max_en+step, step)
        self.bins[-1] = np.inf

        self.kmer_raw = defaultdict(lambda : defaultdict(list))
        self.kmer_binned = defaultdict(dict)

        self.n_raw = 0
        self.logger = logging.getLogger("OpenenHistCollection({0})".format(self.name) )
    
    def add_chunk(self, chunk):
        for k, data in enumerate(chunk.openen):
            #if not data:
                #return
            for kmer, openen in data:
                self.kmer_raw[k][kmer].append(openen)
        self.n_raw += 1
        
        if self.n_raw and (self.n_raw % self.n_chunk == 0):
            self.digest(min_data=10)
            self.store_pickle(suffix="_temp")

    def store_pickle(self, suffix=""):
        print "tadaa"
        for k in sorted(self.kmer_binned.keys()):
            fname = os.path.join(self.path, "{self.name}{suffix}.{k}mers.pkl".format(**locals()) )
            self.logger.debug("store_pickle('{0}')".format(fname) )
            print "store_pickle('{0}')".format(fname)
            pickle.dump( (self.bins, k, self.kmer_binned[k]), file(fname, 'wb') )
                                 
    def load_pickle(self, path):
        self.logger.debug("load_pickle('{0}')".format(path) )
        self.bins, k, kmer_binned = pickle.load(file(path, 'rb') )
        self.kmer_binned[k] = kmer_binned
    
    def digest(self, min_data=0):
        self.logger.debug("digest() at n_raw={0}".format(self.n_raw) )
        for k in sorted(self.kmer_raw.keys()):
            self.logger.debug("digesting {1} {0}mers".format(k, len(self.kmer_raw[k])) )
            for kmer, data in self.kmer_raw[k].items():
                if len(data) > min_data:
                    binned, bins = np.histogram(np.array(data), bins=self.bins)
                    if not kmer in self.kmer_binned:
                        self.kmer_binned[k][kmer] = binned
                    else:
                        self.kmer_binned[k][kmer] += binned
                    
                    # release memory
                    self.kmer_raw[k][kmer] = []
        
        return self.kmer_binned

    def __getitem__(self, kmer):
        k = len(kmer)
        return self.kmer_binned[k][kmer]

   

class RNAplfoldChunk(object):
    min_k = 3
    max_k = 8
    start = 0
    end = 1E6

    def __init__(self):
        self.n = 0
        self.openen = [[] for k in range(0, self.max_k+1)]
        
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
            self.openen[k].append( (kmer, float(col) ) )


def plfold_chunks(src, chunk_type=RNAplfoldChunk):
    chunk = RNAplfoldChunk()
    for line in src:
        if line.startswith('>'):
            yield chunk
            chunk = RNAplfoldChunk()
            chunk.set_header(line)
        elif line.startswith('#'): continue
        elif line.startswith(' #'): continue
        else: 
            chunk.add_line(line)
            
    yield chunk
    


class ThreadManager(object):
    def __init__(self, n_threads=8):
        #self.name = name
        self.n_threads = n_threads

    def process_reads(self, src, ohc, min_k=3, max_k=8, vienna_bin="RNAplfold_cska", adap5="gggaguucuacaguccgacgauc", adap3="uggaauucucgggugucaagg", l=84, temp=22, n_max=0, chunk_type=RNAplfoldChunk):
        cmd=[vienna_bin, "-O", "-u {0}".format(max_k), "-W {0}".format(l), "-L {0}".format(l), "-T {0}".format(temp)]
        self.processes = []
        self.queues = []
        self.parse_threads = []
        self.q = Queue()

        def dispatch_reads(processes, interrupt_event):
            n = len(processes)
            for i,line in enumerate(src):
                #print "dispatch", i
                read = line.rstrip()
                if 'N' in read:
                    continue
                proc = processes[i % n]
                seq = adap5 + read + adap3
                
                # use entire sequence as FASTA header to aid parsing threads
                proc.stdin.write(">{0}\n{0}\n".format(seq)) 
                if interrupt_event.is_set():
                    break

                if n_max and i >= n_max:
                    break
                
            # signal to RNAplfold that all data are written
            for proc in processes:
                proc.stdin.close()
        
        interrupt_event = Event()
        # prepare the FASTA line round-robin dispatcher thread
        self.dispatch_thread = Thread(target=dispatch_reads, args=(self.processes, interrupt_event) )
        self.dispatch_thread.daemon = True # thread dies with the program

        # prepare the output parsing threads
        chunk_type.min_k = min_k
        chunk_type.max_k = max_k
        
        ## skip the adapters
        #chunk_type.start = len(adap5)
        #chunk_type.end = l - len(adap3)
        
        def enqueue_output(out, queue):
            for i,chunk in enumerate(plfold_chunks(iter(out.readline, b''), chunk_type)):
                #print ".", i
                queue.put(chunk)
            
            out.close()
            queue.join()
        
        for n in range(self.n_threads):
            p = Popen(cmd, stdin=PIPE, stdout=PIPE, bufsize=1, close_fds=True)
            self.processes.append(p)
            t = Thread(target=enqueue_output, args=(p.stdout, self.q))
            t.daemon = True # thread dies with the program
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

        try:
            while pending():
                #print "reading", self.q.qsize()
                try:
                    chunk = self.q.get(True, 1.)
                    #print chunk.openen
                    ohc.add_chunk(chunk)
                    self.q.task_done()
                except Empty:
                    #print "empty"
                    pass

        except KeyboardInterrupt:
            # signal the dispatcher to stop and exit as soon as 
            # everything in the pipes is processed
            interrupt_event.set()

            while pending():
                #print "reading", self.q.qsize()
                try:
                    chunk = self.q.get(True, 1.)
                    #print chunk.openen
                    ohc.add_chunk(chunk)
                    self.q.task_done()
                except Empty:
                    #print "empty"
                    pass

                
        return ohc


     
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    oa = OpenenHistCollection(name=sys.argv[1])
    
    tm = ThreadManager()
    tm.process_reads(sys.stdin, oa)
    
    print oa['TGCATGT']
             
