#!/usr/bin/env python
import sys
import os
import numpy as np
import cska.ska_kmers as cyska
import logging

class ReferenceComparison(object):
    def __init__(self, opt, ref_file):
        self.opt = opt
        self.sequences = []
        self.kmer_sets = []
        self.names = []
        self.seqs = []
        self.affinities = []
        self.affinity_errs = []
        self.uniq_kmers = set()
        
        self.logger = logging.getLogger("ReferenceComparison")
        import cska.ska_kmers
        for line in file(ref_file):
            if line.startswith("#"):
                continue

            if not line.strip():
                continue
            
            parts = line.rstrip().split('\t')
            if len(parts) < 5:
                continue

            rbp, name, seq, kd, kd_err = parts[:5]
            if not rbp == self.opt.reads.rbp_name:
                continue
            
            kmers = self.split_kmers(seq)
            if not kmers:
                # can not predict affinity for sequence with non-canonical bases
                continue

            self.uniq_kmers |= set(kmers)
            self.seqs.append(seq)
            self.kmer_sets.append(np.array([cska.ska_kmers.seq_to_index(mer) for mer in kmers]))
            
            self.logger.debug("{seq} {kmers}".format(**locals()) )

            a = 1./float(kd)
            self.affinities.append(a)
            self.affinity_errs.append(a**2 * float(kd_err))
            self.names.append(name)
            
        self.observed_affinities = np.array(self.affinities)
        self.observed_affinity_errors = np.array(self.affinity_errs)
        
        self.logger.info("read {0} reference affinities".format(len(self.seqs)) )

    def split_kmers(self, seq):
        kmers = []
        k = self.opt.k
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer.count('A') + kmer.count('C') + kmer.count('G') + kmer.count('U') < k:
                # discovered non-canonical nucleotide
                continue
            kmers.append(kmer)
            
        return kmers

    @property
    def expected_affinities(self):
        a = []
        for s in self.kmer_sets:
            a.append(self.opt.current.params[s].sum())

        return np.array(a)
