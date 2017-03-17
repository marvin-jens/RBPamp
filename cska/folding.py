#!/usr/bin/env python
import sys
import logging
import numpy as np
import cPickle as pickle
from subprocess import PIPE, Popen
from threading  import Thread
from Queue import Queue, Empty
from collections import defaultdict
from cska.caching import CachedBase, cached, pickled
from itertools import *

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


class OpenenHistCollection(CachedBase):
    def __init__(self, name="openen.hist", min_en=0., max_en=20., n_bins=100, n_chunk=1000):
        
        CachedBase.__init__(self)
        
        self.name = name
        step = (max_en-min_en)/n_bins
        self.bins = np.arange(min_en, max_en+step, step)
        self.bins[-1] = np.inf
        self.n_chunk = n_chunk
        self.kmer_raw = defaultdict(list)
        self.kmer_binned = {}
        self.n_raw = 0
        self.logger = logging.getLogger("OpenenHistCollection({0})".format(self.name) )
    
    @property
    def cache_key(self):
        return self.name
    
    def add_chunk(self, chunk):
        for kmer, openen in chunk.openen:
            self.kmer_raw[kmer].append(openen)
        self.n_raw += 1
        
        if self.n_raw and (self.n_raw % self.n_chunk == 0):
            self.digest(min_data=10, _do_not_unpickle=True, _do_not_cache=True)
            #self.n_raw = 0
    
    @cached
    @pickled
    def digest(self, min_data=0):
        self.logger.debug("digest() at n_raw={0}".format(self.n_raw) )
        for kmer, data in self.kmer_raw.items():
            if len(data) > min_data:
                binned, bins = np.histogram(np.array(data), bins=self.bins)
                if not kmer in self.kmer_binned:
                    self.kmer_binned[kmer] = binned
                else:
                    self.kmer_binned[kmer] += binned
                
                # release memory
                self.kmer_raw[kmer] = []
        
        return self.kmer_binned

    @cached
    def __getitem__(self, kmer):
        return self.digest()[kmer]

    def aggregate_from_file(self, src):
        self.logger.debug("aggregating data from {0}".format(src) )
        for chunk in plfold_chunks(src):
            self.add_chunk(chunk)
            
        return self.digest()

   
class RNAplfoldChunk(object):
    def __init__(self):
        self.n = 0
        self.openen = []
        
    def set_header(self, head):
        kmers, pos = head[1:].split(' | p=')
        self.kmers = kmers.split(',')
        self.k = len(self.kmers[0])
        self.kmer_pos = dict([(int(p) + self.k, kmer) for p, kmer in zip(pos.split(','), self.kmers)])
        
    def add_line(self, line):
        self.n += 1
        if self.n in self.kmer_pos:
            kmer = self.kmer_pos[self.n]
            col = line.split('\t')[self.k]
            self.openen.append( (kmer, float(col) ) )


def plfold_chunks(src):
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
    

def enqueue_output(out, queue):
    for i,chunk in enumerate(plfold_chunks(iter(out.readline, b''))):
        #print ".", i
        queue.put(chunk)
    
    out.close()
    queue.join()
    #print "quit"

def dispatch_fasta(src, processes):
    n = len(processes)
    for i,rec in enumerate(grouper(src, 2, "")):
        #print "dispatch", i
        proc = processes[i % n]
        proc.stdin.write(rec[0])
        proc.stdin.write(rec[1])

    # signal to RNAplfold that all data are written
    for proc in processes:
        proc.stdin.close()


class ThreadManager(object):
    def __init__(self, n_threads=8):
        #self.name = name
        self.n_threads = n_threads


    def process_FASTA(self, src, ohc, cmd=["/home/mjens/git/rsek/ViennaRNA-2.2.5/src/bin/RNAplfold", "-O", "-u 7", "-W 84", "-L 84", "-T 22"]):
        self.processes = []
        self.queues = []
        self.parse_threads = []
        self.q = Queue()
        
        # prepare the FASTA line round-robin dispatcher thread
        self.dispatch_thread = Thread(target=dispatch_fasta, args=(src, self.processes))
        self.dispatch_thread.daemon = True # thread dies with the program

        # prepare the output parsing threads
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
    oa = OpenenHistCollection()
    #oa.aggregate_from_file(sys.stdin)
    oa._do_not_unpickle = True
    
    tm = ThreadManager()
    tm.process_FASTA(sys.stdin, oa)
    
    print oa['TGCATGT']
             
