#!/usr/bin/env python
import sys
import os
import numpy as np
import cska.cyska as cyska
import logging

class RefComparison(object):
    def __init__(self, rbp_name, ref_file=""):
        self.rbp_name = rbp_name
        self.sequences = []
        self.kmer_sets = []
        self.names = []
        self.seqs = []
        self.affinities = []
        self.affinity_errs = []
        
        self.logger = logging.getLogger("report.ReferenceComparison")
        import cska.cyska
        if not ref_file:
            ref_file = os.path.join(os.path.dirname(__file__),"../known_kds.csv")

        for line in file(ref_file):
            if line.startswith("#"):
                continue

            if not line.strip():
                continue
            
            parts = line.rstrip().split('\t')
            if len(parts) < 5:
                continue

            rbp, name, seq, kd, kd_err = parts[:5]
            if not rbp == rbp_name:
                continue
            
            if self.noncanonical(seq):
                # can not predict affinity for sequence with non-canonical bases
                continue

            self.seqs.append(seq)           
            self.logger.debug("{seq}".format(**locals()) )

            a = 1./float(kd)
            self.affinities.append(a)
            self.affinity_errs.append(a**2 * float(kd_err))
            self.names.append(name)
            
        self.observed_affinities = np.array(self.affinities)
        self.observed_affinity_errors = np.array(self.affinity_errs)
        self.seqs = np.array(self.seqs)
        self.logger.info("found {0} reference affinities for {1}".format(len(self.seqs), rbp_name) )

    def noncanonical(self, seq):
        S = seq.upper()
        return S.count('A') + S.count('C') + S.count('G') + S.count('T') + S.count('U') < len(S)

    def split_kmers(self, seq, k):
        kmers = []
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer.count('A') + kmer.count('C') + kmer.count('G') + kmer.count('U') < k:
                # discovered non-canonical nucleotide
                continue
            kmers.append(cyska.seq_to_index(kmer))
            
        return kmers

    def predict_affinities(self, mdl):
        a = []
        if hasattr(mdl, "parameters"):
            aff = mdl.parameters.affinities
        else:
            aff = mdl.affinities

        for seq in self.seqs:
            I = np.array(self.split_kmers(seq, mdl.k))
            a.append(aff[I].sum())

        return np.array(a)
