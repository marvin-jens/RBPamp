import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp
import logging
import os

from itertools import izip_longest
import cska.ska_kmers as cyska
from cska.ska_kmers import yield_kmers
import cska
from cska.caching import CachedBase, cached, pickled

class Alignment(object):
    def __init__(self, seqs=[], weights=[]):
        self.matrix = []
        for s,w in izip_longest(seqs, weights, fillvalue=1.):
            ofs, score = self.align(s)

        self.seqs = []
        self.ofs = []
        self.weights = []

    def align(self, seq):
        bits = cyska.seq_to_bits(seq)
        l = len(seq)
        n = len(self.matrix)
        if not len(self.matrix):
            return 0, 1 # offset, alignment score
        else:
            scores = []
            ofs_range = range(-l+1,n)
            # print seq
            for ofs in ofs_range:
                m_start = max(0, ofs)
                m_end = min(n,ofs+l)
                n_cols = m_end - m_start

                s_start = max(-ofs, 0)
                s_end = s_start + n_cols
                score = 0
                for i in range(n_cols):
                    if bits[i+s_start] > 3:
                        continue # skip gaps
                    score += self.matrix[i+m_start, bits[i+s_start]]
                
                scores.append(score)
        
                # print ofs, s_start,":",s_end, seq[s_start:s_end], m_start,":", m_end, self.matrix[m_start:m_end], "->", score
            x = np.array(scores).argmax()
            return ofs_range[x], scores[x]


    def blend(self, seq, ofs, weight):
        self.seqs.append(seq)
        self.weights.append(weight)
        if ofs < 0:
            self.ofs = [o - ofs for o in self.ofs]
            matrix = np.zeros((len(self.matrix)-ofs,4))
            matrix[-ofs:] =  self.matrix[:]
            self.matrix = matrix
            ofs = 0
        
        d = ofs + len(seq) - len(self.matrix)
        if d > 0:
            matrix = np.zeros((len(self.matrix)+d,4))
            if len(self.matrix):
                matrix[:len(self.matrix)] =  self.matrix[:]
            self.matrix = matrix
        
        self.ofs.append(ofs)

        bits = cyska.seq_to_bits(seq)
        l = len(seq)
        for i in range(l):
            if bits[i] > 3:
                continue # skip gaps
            self.matrix[i+ofs, bits[i]] += weight

    def add(self, seq, weight=1.):
        ofs, score = self.align(seq)
        self.blend(seq, ofs, weight)

        return score

    @property
    def score(self):
        return self.matrix.max(axis=0).mean()
    
    @property
    def max_score(self):
        return self.matrix.max(axis=1).sum()

    @property
    def wlen(self):
        colw = self.matrix.max(axis=1) / self.matrix.max()
        return colw.sum()

    def __str__(self):
        buf = []
        for s, o, w in zip(self.seqs, self.ofs, self.weights):
            spacer = " "*o
            buf.append("{w:3.3e}  {spacer}{s}".format(**locals()))

        perc = 100. * self.score / self.max_score
        buf.append("average max. column score {0:.2f} of {1:.2f} ({2:.2f}%)".format(self.score, self.max_score, perc))
        return "\n".join(buf)

    def save_logo(self, fname):
        from cska.pwm import weblogo_save
        weblogo_save(self.matrix, fname)

    def to_PSAM(self, keep_weight=1, n_max=0):
        frac = self.matrix.sum(axis=1) 
        F = self.matrix.sum()
        n = len(self.matrix)

        best = {n : (1 ,0 ,n)}
        for i in range(n):
            for j in range(i, n+1):
                
                f = frac[i:j].sum()/F
                l = j-i
                if l in best:
                    if f < best[l][0]:
                        continue

                best[l] = (f, i, j)

        # print best
        bylength = sorted(best)
        for l in bylength:
            f,i,j = best[l]
            if f >= keep_weight:
                break

        if not n_max or j-i <= n_max:
            m = self.matrix[i:j] + 1
        else:
            if n_max <= n:
                f, i, j = best[n_max]
                m = self.matrix[i:j] + 1
            else:
                # need to pad
                s = n_max - n
                m = self.matrix
                left = s/2
                right = s - left
                if left:
                    m = np.concatenate((np.zeros((left,4), m)))
                if right:
                    m = np.concatenate((m, np.zeros((right,4))))

        psam = m / m.max(axis=1)[:,np.newaxis]
        # print m
        # print psam
        A0 = m.max(axis=1).sum()
        from cska.pwm import PSAM
        return PSAM(psam, A0=A0)
        

class DependentKmerAnalysis(CachedBase):
    def __init__(self, rbns, km=4):
        self.rbns = rbns
        self.km = km
        self.linear = Alignment()
        self.A = Alignment()
        self.B = Alignment()
        self.parts = [self.A, self.B]
        self.partscores = [0, 0]
        self.logger = logging.getLogger("seed.DependentKmerAnalysis")
        
        CachedBase.__init__(self)

        profs = []
        joints = []
        
        for reads in rbns.reads:
            self.logger.debug("collecting joint kmer frequencies for {reads.name}".format(reads=reads))
            joint = reads.joint_kmer_freq_distance_profile(km)
            joints.append(joint)
            prof = reads.kmer_mutual_information_profile(km)
            profs.append(prof)
            # free some memory!
            # reads.cache_flush("__cached_get_index_matrix")
            # reads.cache_flush("__cached_seqm")
            reads.cache_flush()
        
        self.profs = np.array(profs)
        self.joints = np.array(joints)
        #self.best_sample = np.unravel_index(self.joints.argmax(), self.joints.shape)[0]
        self.best_sample = self.profs.max(axis=1)[1:].argmax() + 1
        #print "best_sample_candidates", self.best_sample, len(rbns.reads)
        #print self.joints.max(axis=3).max(axis=2).max(axis=1)
        #print self.profs.max(axis=1)
        self.logger.debug("best_sample = {0}".format(rbns.reads[self.best_sample].name) )

    @property
    def cache_key(self):
        return "{self.rbns.cache_key}.km={self.km}".format(self=self)    

    # @pickled
    def build_matrices(self, thresh=.7):
        j0 = self.joints[0]

        joint = self.joints[self.best_sample]
        reads = self.rbns.reads[self.best_sample]

        S_lin = 0
        S_A = 0
        S_B = 0
        kmers = list(cyska.yield_kmers(self.km))
        self.spaced_score = np.zeros(18,dtype=np.float32)

        for d in range(18):
            # print reads.name, d
            self.logger.debug("build_matrices(d={0})".format(d))
            
            jR = np.log2(joint / j0)
            jRm = jR.max()
            I = jR[:,:,d].flatten().argsort()[::-1]

            for n in I:
                i, j = np.unravel_index(n, joint.shape[:2])
                if jR[i,j,d] <= jRm * thresh:
                    break
                
                #print "most-co-enriched mers at d=", d, kmers[i], kmers[j], jR[i,j,d], jRm
                merge = kmers[i] + "-" * d + kmers[j]
                score = jR[i,j,d]
                
                s_lin = self.linear.add(merge, score)
                s_A = self.A.add(kmers[i], score)
                s_B = self.B.add(kmers[j], score)

                S_lin += s_lin
                S_A += s_A
                S_B += s_B
                self.spaced_score[d] += s_A + s_B

                # # autodetect order of sub-motifs
                # Z = np.array([p.max_score for p in self.parts])
                # sA = np.array([p.align(kmers[i])[1] for p in self.parts]) / Z
                # sB = np.array([p.align(kmers[j])[1] for p in self.parts]) / Z

                # iA = sA.argmax()
                # iB = sB.argmax()


                # if iB != iA + 1:
                #     print "weird scores"
                #     print kmers[i], sA
                #     print kmers[j], sB
        self.lin_score = S_lin
        self.A_score = S_A
        self.B_score = S_B
        self.logger.debug("build_matrices() done.")

    # @pickled
    def linear_PSAM_seed(self, keep_weight=.9, n_max=7):
        # find compact representation of linear motif
        self.logger.debug("building linear PSAM with max width={0}".format(n_max))
        psam_lin = self.linear.to_PSAM(keep_weight=keep_weight, n_max=n_max)
        return psam_lin

    # @pickled
    def bipartite_PSAM_seeds(self):
        self.logger.debug("building bipartite PSAMs")
        # find compact representations of sub-motifs
        pA = self.A.to_PSAM(keep_weight=.9)
        pB = self.B.to_PSAM(keep_weight=.9)
        k = max(pA.n, pB.n)

        # use same k for both of them
        psam_A = self.A.to_PSAM(n_max=k)
        psam_B = self.B.to_PSAM(n_max=k)
        return psam_A, psam_B

    # @pickled
    def bipartite_PSAM_spacings(self, sample=0, psam_A=None, psam_B=None):
        
        if psam_A == None or psam_B == None:
            psam_A, psam_B = self.bipartite_PSAM_seeds()

        self.logger.debug("computing bipartite PSAM spacing cross-correlations")
        from copy import copy
        psam_A = copy(psam_A)
        psam_B = copy(psam_B)

        psam_A.A0 = 1
        psam_B.A0 = 1

        aff_A = psam_A.affinities
        aff_B = psam_B.affinities

        if not sample:
            sample = self.best_sample

        ctrl = self.rbns.reads[0]
        reads = self.rbns.reads[sample]

        Z_A = aff_A[ctrl.get_index_matrix(psam_A.n)]
        Z_B = aff_B[ctrl.get_index_matrix(psam_B.n)]
        xctrl = cyska.xcorr_Z(Z_A, Z_B, k1 = psam_A.n, k2 = psam_B.n) / (Z_A.sum() + Z_B.sum())

        Z_A = aff_A[reads.get_index_matrix(psam_A.n)]
        Z_B = aff_B[reads.get_index_matrix(psam_B.n)]
        xcorr = cyska.xcorr_Z(Z_A, Z_B, k1 = psam_A.n, k2 = psam_B.n) / (Z_A.sum() + Z_B.sum())

        return np.log2(xcorr/xctrl)


    def interaction_plot(self):
        self.logger.debug("generating interaction plot")
        ctrl = self.rbns.reads[0]
        reads = self.rbns.reads[self.best_sample]
        
        psam_lin = self.linear_PSAM_seed()
        # psam_lin.save_logo('lin_psam.eps')
        # print psam_lin

        psam_A, psam_B = self.bipartite_PSAM_seeds()
        psam_A.save_logo('A_psam.eps')
        psam_B.save_logo('B_psam.eps')
        # print psam_A
        # print psam_B

        spacing_w = self.bipartite_PSAM_spacings()
        L = len(spacing_w)
        x = np.arange(L) - L/2

        pp.figure()
        pp.title('{0} -> {1}'.format(psam_A.consensus, psam_B.consensus))
        pp.plot(x, spacing_w, '-.', linestyle='steps-mid', label=self.rbns.reads[self.best_sample].name)
        # pp.plot(x, xctrl, '-.', linestyle='steps-mid', label='{0} -> {1}'.format(psam_A.consensus, psam_B.consensus))
        pp.xlabel("distance [nt]")
        pp.ylabel("cross affinity log2-enrichment")
        pp.axvline(psam_A.n)
        pp.legend()

        pp.show()

    @property
    def linear_motif_score(self):
        ls = self.lin_score / self.linear.wlen
        ABs = (self.A_score + self.B_score) / (self.A.wlen + self.B.wlen)

        return ls / ABs

        # pp.figure()
        # pp.title(rbp_name)
        # for prof,reads in zip(profs[1:], rbns.reads[1:]):
        #     pp.plot(prof/profs[0], '.-', label=reads.name)
        
        # pp.legend()
        # pp.xlabel("{0}mer separation".format(km))
        # pp.ylabel("MI ratio to input")
        # pp.show()
        # sys.exit(0)

class SeedRefinement(object):
    def __init__(self, rbns, km=4, keep_weight=.9, max_linear_k=11):
        self.rbns = rbns
        self.logger = logging.getLogger("opt.SeedRefinement({0})".format(km))
        self.km = km
        self.analysis = DependentKmerAnalysis(self.rbns, km=km)
        self.analysis.build_matrices()
        self.psam_lin = self.analysis.linear_PSAM_seed(keep_weight=keep_weight, n_max=max_linear_k)
        self.logger.info("linear_motif score={0:.2f} for {1}mer {2}".format(self.analysis.linear_motif_score, self.psam_lin.n, self.psam_lin.consensus))
        self.psam_A, self.psam_B = self.analysis.bipartite_PSAM_seeds()
        self.linear_k = self.psam_lin.n
        self.bipart_k = self.psam_A.n
        
        if self.analysis.linear_motif_score < .9:
            self.logger.info("bipartite motifs are potentially a better match for this RBP")
            self.spacings = self.analysis.bipartite_PSAM_spacings(psam_A=self.psam_A, psam_B=self.psam_B)
            L = len(self.spacings)
            self.dist_cost = self.spacings[L/2:]
            self.logger.debug("bipartite spacing weights: {0}".format(self.dist_cost))

        self.store_logos()

    def distance_xcorr_plot(self, fname="xcorr.pdf"):
        self.logger.debug("generating xcorr plot")

        ctrl = self.rbns.reads[0]
        reads = self.rbns.reads[self.analysis.best_sample]
        
        spacing_w = self.analysis.bipartite_PSAM_spacings(psam_A = self.psam_A, psam_B = self.psam_B)
        L = len(spacing_w)
        x = np.arange(L) - L/2

        pp.figure(figsize=(4,3))
        pp.title('{0} -> {1} linear_motif_score={2:.3f}'.format(self.psam_A.consensus, self.psam_B.consensus, self.analysis.linear_motif_score))
        pp.plot(x[L/2:], spacing_w[L/2:], '-.', linestyle='steps-mid', label=self.rbns.reads[self.analysis.best_sample].name)
        # pp.plot(x, xctrl, '-.', linestyle='steps-mid', label='{0} -> {1}'.format(psam_A.consensus, psam_B.consensus))
        pp.xlabel("distance [nt]")
        pp.ylabel("cross affinity log2-enrichment")
        pp.axvline(self.psam_A.n)
        pp.legend()
        pp.tight_layout()
        pp.savefig(fname)
        pp.close()

    def store_logos(self):
        self.logger.debug("generating sequence logos")
        path = cska.ensure_path(os.path.join(self.rbns.out_path,'seed/'))
        rbp_name = self.rbns.reads[0].rbp_name

        self.psam_lin.save_logo(os.path.join(path, '{0}_linear.eps'.format(rbp_name)))
        self.psam_A.save_logo(os.path.join(path, '{0}_motif_A.eps'.format(rbp_name)))
        self.psam_B.save_logo(os.path.join(path, '{0}_motif_B.eps'.format(rbp_name)))
        # self.distance_xcorr_plot(fname = os.path.join(path, '{0}_motif_xcorr.pdf'.format(rbp_name)))


    def linear_seed_params(self, A0=1., aff0=1e-6):
        psam = self.psam_lin
        psam.A0 = A0

        return psam.kmer_affinity_table(aff0-aff0)

    # def optimize(self, eps=1e-3, A0=1.):

    #     from cska.optimize import ModelOptimization
    #     # free some memory
    #     self.opt.input_reads.cache_flush('__cached_get_index_matrix')
    #     self.opt.input_reads.acc_storage.cache_flush('__cached_get_raw')

    #     # create new optimizer and model
    #     new_opt = ModelOptimization(k, self.opt.rbns_analysis,
    #         mdl_params = params,
    #         t0 = self.opt.t,
    #         reporter = self.opt.reporter,
    #         kmer_opt_global = not self.opt.param_local_fit,
    #     )
    #     new_opt.errors = self.opt.errors + new_opt.errors
    #     new_opt.correlations = self.opt.correlations
    #     new_opt.rel_improvements = self.opt.rel_improvements

    #     # some plumbing to make reports/plots contiguous
    #     self.opt.reporter.set_opt(new_opt)
    #     self.opt.reporter.tick(0)
    #     self.opt.reporter.trigger_plots(self.opt.t, occasion="init")

    #     self.opt = new_opt
    #     self.opt.step_scale(min_scale=.01, max_scale=1000.)
    #     self.opt.reporter.trigger_plots(self.opt.t, occasion="scale")

    # def store_params(self):
    #     self.opt.mdl.parameters.store(cska.ensure_path(os.path.join(self.opt.out_path, "affinity/"))

if __name__ == "__main__":

    # A = Alignment()
    # A.add('GCAUG', 2.)
    # A.add('GCACG', .4)
    # A.add('UGCAU', 2.)

    # A.add('UGC-UG', 3.)

    # # print A.matrix
    # print A
    # A.save_logo("bla.eps")

    from cska.analysis import RBNSAnalysis
    from cska.reads import RBNSReads
    from cska import auto_detect

    rbp_name, reads_files, rbp_concentrations = auto_detect('.')

    rbns = RBNSAnalysis(
        rbp_name = rbp_name,
        out_path = 'cska',
        ska_runner = None,
    )
    
    for fname, rbp_conc in zip(reads_files, rbp_concentrations):
        reads = RBNSReads(
            fname, 
            rbp_conc=rbp_conc,
            rbp_name = rbp_name,
            n_max=10000000,
            pseudo_count=10, 
            rna_conc = 1000.,
            temp = 4,
            n_subsamples = 0,
            acc_storage_path = 'acc',
        )
        rbns.add_reads(reads)

    DK = DependentKmerAnalysis(rbns, km=3)    
    DK.build_matrices()

    # DK.linear.save_logo("linear.eps")
    # DK.A.save_logo("A.eps")
    # DK.B.save_logo("B.eps")
    # print "linear alignment"
    # print DK.linear
    # print DK.linear.matrix
    ls = DK.lin_score / DK.linear.wlen
    ABs = (DK.A_score + DK.B_score) / (DK.A.wlen + DK.B.wlen)
    print "A effective length", DK.A.wlen, "score", DK.A_score, "score-density", DK.A_score/DK.A.wlen
    print "B effective length", DK.B.wlen, "score", DK.B_score, "score-density", DK.B_score/DK.B.wlen
    print "combined", DK.A.wlen + DK.B.wlen, "score", DK.A_score + DK.B_score, "score-density", (DK.A_score + DK.B_score)/(DK.A.wlen + DK.B.wlen)

    print "linear eff length", DK.linear.wlen, "score", DK.lin_score, "score-density", DK.lin_score/DK.linear.wlen
    print "linear_motif score", DK.linear_motif_score

    DK.interaction_plot()


    print "A"
    psam = DK.A.to_PSAM()
    for mer, aff in zip(*psam.kmer_affinities):
        print mer, aff

    print "B"
    psam = DK.B.to_PSAM()
    kmers, aff = psam.kmer_affinities
    for mer, a in zip(kmers, aff):
        print mer, a



    # # TESTING mutual information
    # import matplotlib.pyplot as pp
    # from cska.ska_kmers import yield_kmers
    # km = 4
    # kmers = list(yield_kmers(km))
    # profs = []
    # joints = []
    # for reads in rbns.reads:
    #     joint = reads.joint_kmer_freq_distance_profile(km)
    #     joints.append(joint)
    #     prof = reads.kmer_mutual_information_profile(km)
    #     profs.append(prof)
    
    # j0 = joints[0]
    # for joint, reads in zip(joints[1:], rbns.reads):
    #     for d in range(18):
    #         print reads.name, d
            
    #         # print joint[:,:,d]
    #         # pp.figure()
    #         # pp.pcolormesh(joint[:,:,d])
    #         # pp.show()
    #         jR = np.log2(joint / j0)
    #         I = jR[:,:,d].flatten().argsort()[::-1]
    #         # print I
    #         for n in I[:10]:
    #             i, j = np.unravel_index(n, joint.shape[:2])
    #             # print n, i, j
    #             print "most-co-enriched 3mers at d=", d, kmers[i], kmers[j], jR[i,j,d]

    # pp.figure()
    # pp.title(rbp_name)
    # for prof,reads in zip(profs[1:], rbns.reads[1:]):
    #     pp.plot(prof/profs[0], '.-', label=reads.name)
    
    # pp.legend()
    # pp.xlabel("{0}mer separation".format(km))
    # pp.ylabel("MI ratio to input")
    # pp.show()
    # sys.exit(0)
    
