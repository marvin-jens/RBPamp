#!/usr/bin/env python
import sys
import numpy as np
from collections import defaultdict

class OpenenAggregator(object):
    def __init__(self, min_en=0., max_en=20., n_bins=100, n_chunk=10000):
        step = (max_en-min_en)/n_bins
        self.bins = np.arange(min_en, max_en+step, step)
        self.bins[-1] = np.inf
        self.n_chunk = n_chunk
        self.kmer_raw = defaultdict(list)
        self.kmer_binned = defaultdict(lambda : np.zeros(n_bins, dtype=int))
        self.n_raw = 0

    def add_chunk(self, chunk):
        for kmer, openen in chunk.openen:
            self.kmer_raw[kmer].append(openen)
        self.n_raw += 1
        
        if self.n_raw >= self.n_chunk:
            self.process(min_data=100)
            self.n_raw = 0
    
    def process(self, min_data=0):
        for kmer, data in self.kmer_raw.items():
            if len(data) > min_data:
                binned, bins = np.histogram(np.array(data), bins=self.bins)
                self.kmer_binned[kmer] += binned
                self.kmer_raw[kmer] = []
        
        print self.kmer_binned['GCATG']
        return self.kmer_binned

    def aggregate_from_file(self, src):
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
        return self.process()
        
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



oa = OpenenAggregator()
print oa.aggregate_from_file(sys.stdin)['GCATG']
