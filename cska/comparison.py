#!/usr/bin/env python
import sys
import os
import numpy as np
import cska.cyska as cyska
import logging

class RefComparison(object):
    def __init__(self, rbp_name, ref_file=""):
        self.rbp_name = rbp_name
        self.rbp_data = rbp_name
        self.sequences = []
        self.kmer_sets = []
        self.names = []
        self.seqs = []
        self.affinities = []
        self.Kd = []
        self.Kd_err = []
        self.affinity_errs = []
        
        self.logger = logging.getLogger("report.ReferenceComparison")
        import cska.cyska
        if not ref_file:
            ref_file = os.path.join(os.path.dirname(__file__),"../known_kds.csv")

        for line in file(ref_file):
            if line.startswith("# alias"):
                name, alias = line.rstrip().split(" ")[2:]
                if name == rbp_name:
                    rbp_name = alias
                    self.rbp_data = alias
                continue

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

            self.Kd.append(kd)
            self.Kd_err.append(kd_err)
            a = 1./float(kd)
            self.affinities.append(a)
            self.affinity_errs.append(a**2 * float(kd_err))
            self.names.append(name)
            
        self.observed_Kd = np.array(self.Kd, dtype=float)
        self.observed_Kd_err = np.array(self.Kd_err, dtype=float)
        self.observed_affinities = np.array(self.affinities)
        self.observed_affinity_errors = np.array(self.affinity_errs)
        self.seqs = np.array(self.seqs)
        self.n = len(self.seqs)
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

    def __len__(self):
        return len(self.seqs)
    
    def predict_affinities(self, mdl):
        a = []
        # if hasattr(mdl, "parameters"):
        #     aff = mdl.parameters.affinities
        # else:
        #     aff = mdl.affinities

        from cska.seed import Alignment
        A = Alignment()
        A.matrix = mdl.params.psam_matrix

        for seq in self.seqs:
            l = len(seq)
            # if l >= mdl.k_mdl:
            #     # the seq is longer than our motifs/model
            #     I = np.array(self.split_kmers(seq, mdl.k_mdl))
            #     a.append(aff[I].sum())
            # else:
            #     # the seq is shorter than our motifs/model
            ofs, score = A.align(seq, multiply=True, min_overlap=7, end_weight=False)
            print seq, ofs, score
            a.append(mdl.params.A0 * score)

        a = np.array(a)
        print a.min(), a.max(), a.mean()
        return a

if __name__ == "__main__":
    from cska.pwm import PSAM
    from cska.reads import RBNSReads
    from cska.partfunc import PartFuncModel
    from cska.gradient import ModelParametrization, GradientDescent
    from cska import auto_detect
    from cska.analysis import read_kmer_matrix

    run_folder = "/scratch/data/RBNS/RBFOX3/cska/recent/"
    k_R = 6
    rbp_name, read_files, rbp_conc = auto_detect(os.path.join(run_folder,"../../"))
    rbp_conc2, R0, R0_err = read_kmer_matrix(os.path.join(run_folder,"metrics/{rbp_name}.R_value.{k_R}mer.tsv".format(**locals())))

    print rbp_conc, rbp_conc2
    print R0.shape

    reads = RBNSReads(read_files[0], temp=4, rbp_conc=0, rbp_name=rbp_name, n_max=10000)
    from glob import glob
    pwm_file = list(glob(os.path.join(run_folder, 'meanfield/mean_field_*mer_PSAM.tsv')))[0]
    motif = PSAM.load(pwm_file)
    print motif
    params = ModelParametrization(motif.n, len(rbp_conc2), psam=motif.psam, A0= motif.A0)
    # need: params
    # TODO: alias support in known_kds.csv
    print params
    mdl = PartFuncModel(reads, params, R0, rbp_conc=rbp_conc)
    descent = GradientDescent(mdl, params)

    from cska.comparison import RefComparison
    from cska.report import GradientDescentReport, LiteratureComparisonReport
    ref = RefComparison(rbp_name, ref_file="")
    lrep = LiteratureComparisonReport(descent, ref, path='.')

    lrep.plot_scatter()
    