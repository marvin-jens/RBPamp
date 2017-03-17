#!/usr/bin/env python
import sys
import numpy as np
from collections import defaultdict
import cPickle as pickle
from cska.caching import CachedBase, cached, pickled
import logging

class OpenenHistCollection(CachedBase):
    def __init__(self, name="openen.hist", min_en=0., max_en=20., n_bins=100, n_chunk=100):
        
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
            self.n_raw = 0
    
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
        chunk = RNAplfoldChunk()
        for line in src:
            if line.startswith('>'):
                self.add_chunk(chunk)
                chunk = RNAplfoldChunk()
                chunk.set_header(line)
            elif line.startswith('#'): continue
            elif line.startswith(' #'): continue
            else: 
                chunk.add_line(line)
                
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    oa = OpenenHistCollection()
    oa.aggregate_from_file(sys.stdin)
    print oa['TGCATGT']
             
