import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp
import logging
import os

from itertools import izip_longest
import cska.cyska as cyska
from cyska import yield_kmers
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

    def align(self, seq, normalize=False, multiply=False, contain=False, end_weight=False, min_overlap=1, core_k=None, core_start=None, debug=False):
        # TODO: handle core_k and core_start 
        bits = cyska.seq_to_bits(seq)
        l = len(seq)
        n = len(self.matrix)
        if not len(self.matrix):
            return 0, 1 # offset, alignment score
        else:
            scores = []
            if contain:
                assert n > l
                d = n - l
                ofs_range = range(-d, d+1)
            else:
                ofs_range = range(-l + min_overlap, n + 1 - min_overlap)
            # print seq
            if end_weight:
                func = np.mean
            else:
                func = np.min

            for ofs in ofs_range:
                m_start = max(0, ofs)
                m_end = min(n,ofs+l)
                
                if multiply:
                    start_avg = 1.
                    if m_start:
                        start_avg = func(self.matrix[:m_start], axis=1).prod()
                    
                    end_avg = 1.
                    if m_end < n:
                        end_avg = func(self.matrix[m_end:], axis=1).prod()
                else:
                    start_avg = 0
                    if m_start:
                        start_avg = func(self.matrix[:m_start], axis=1).sum()

                    end_avg = 1.
                    if m_end < n:
                        end_avg = func(self.matrix[m_end:], axis=1).sum()


                n_cols = m_end - m_start

                s_start = max(-ofs, 0)
                s_end = s_start + n_cols
                col_scores = []
                # if end_weight:
                #     score = start_avg*end_avg if multiply else start_avg + end_avg
                # else:
                #     score = 1 if multiply else 0
                score = start_avg*end_avg if multiply else start_avg + end_avg

                for i in range(n_cols):
                    if bits[i+s_start] > 3:
                        continue # skip gaps
                    
                    S = self.matrix[i+m_start, bits[i+s_start]]
                    col_scores.append(S)
                    score = score * S if multiply else score + S
                
                scores.append(score)
                if debug:
                    print ofs, s_start,":",s_end, seq[s_start:s_end], m_start,":", m_end, col_scores, "->", score

            x = np.array(scores).argmax()
            S = scores[x]
            if normalize:
                S /= self.max_score

            return ofs_range[x], S


    def blend(self, seq, ofs, weight, normalize=False):
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
        
        if normalize:
            self.matrix /= self.max_score

    def add(self, seq, weight=1.):
        ofs, score = self.align(seq)
        self.blend(seq, ofs, weight)

        return score

    @property
    def score(self):
        return self.matrix.max(axis=0).mean()
    
    @property
    def max_score(self):
        if len(self.matrix):
            return self.matrix.max(axis=1).sum()
        else:
            return 1.

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

    def to_PSAM(self, keep_weight=1, n_max=0, pseudo=1, col_scale=True, A0=None):
        # print self
        # print "to PSAM"
        # print self.matrix

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

        bylength = sorted(best.keys())
        def find_best():
            for l in bylength:
                if n_max and l > n_max:
                    # we exhausted all motifs of allowed length
                    break
                
                f,i,j = best[l]
                if f >= keep_weight:
                    # found the shortest motif that satisfies keep_weight
                    return f, i, j 

            # no allowed length satisfies keep_weight cutoff.
            # select the shortest motif that is as good as the longest allowed motif
            f_cut = best[n_max][0]
            for l in bylength:
                f,i,j = best[l]
                if f >= f_cut:
                    # found the shortest motif that satisfies keep_weight
                    return f, i, j 

        f, i, j = find_best()

        m = self.matrix[i:j] + pseudo
        if col_scale:
            # add pseudo-scores to columns 
            # with fewer observations/lower score
            M = m.max(axis=1)
            # print "maxima along positions", M
            # 0 for col with highest score, 
            # approaching 1 for lowest
            inc = 1 - M / M.max() 
            m += inc[:,np.newaxis]

        psam = m / m.max(axis=1)[:,np.newaxis]
        # A0 = m.max(axis=1).sum()
        from cska.pwm import PSAM
        max_weight = np.array(self.weights).max()
        if A0 is None:
            A0 = max_weight

        P = PSAM(psam, A0=A0)
        P._n_seqs = len(self.seqs)
        P._max_weight = max_weight
        return P
        

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
        
        self.logger.info("performing seed analysis")
        for reads in rbns.reads:
            self.logger.debug("collecting joint kmer frequencies for {reads.name}".format(reads=reads))
            joint = reads.joint_kmer_freq_distance_profile(km)
            joints.append(joint)
            prof = reads.kmer_mutual_information_profile(km)
            profs.append(prof)
            # free some memory!
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
        n_pairs = 0
        for d in range(18):
            # print reads.name, d
            self.logger.debug("build_matrices(d={0})".format(d))
            
            jR = np.log2(joint / j0)
            jRm = jR.max()
            I = jR[:,:,d].flatten().argsort()[::-1]

            for n in I:
                i, j = np.unravel_index(n, joint.shape[:2])
                if (jR[i,j,d] <= jRm * thresh):
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
                n_pairs += 1
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
        self.logger.debug("build_matrices() done. Aligned {0} kmer pairs".format(n_pairs))
        # print self.linear

    def motifs_from_R(self, k, keep_weight=.99, n_max=11, thresh = .7, z_cut=4, min_mer=.05, q_ns=.05): # UNDO HERE!!!
        from cska.seed import Alignment
        import cska.cyska as cyska

        alns = []
        R, R_err = self.rbns.R_value_matrix(k)
        R = R.mean(axis=0)
        Rns = np.quantile(R, q_ns)
        print "non-specific quantile", Rns
        R_err = R_err.mean(axis=0)

        R = R + Rns * ( (R - 1)/ (1 - Rns)) # corrected R-value, see suppl. methods

        I = R.argsort()[::-1]
        R_sort = R[I]
        z = (R_sort - R_sort.mean())/R_sort.std()
        # print R_sort[:10]
        # print "z-scores", z.min(), z.max(), z.mean()
        i_cut = (z < z_cut).argmax()
        # print i_cut
        R_cut = max(1, R[I[i_cut]])
        # print i_cut, "R_values above z-cut", R_cut
        
        r0 = R[I[0]]
        # print "maxR", r0
        n = 0
        kmer_set = []

        for i in I[:100]:
            kmer = cyska.index_to_seq(i, k)
            n += 1
            r = R[i] 
            # print i, kmer, r, "+/-", R_err[i], R_cut
            if r - R_err[i] < R_cut:
                break
            
            kmer_set.append( (kmer, r) )
        
        n_enriched = len(kmer_set)
        n_min = int(min_mer * n_enriched)

        kmer, r = kmer_set.pop(0)
        print "STARTING from", kmer, r
        aln = Alignment()
        aln.blend(kmer, 0, r, normalize=False)
        alns.append(aln)

        while kmer_set:
            # align all remaining enriched kmers to all motifs
            scores = []
            ofs = []
            print "re-aligning"
            for kmer, r in kmer_set:
                o, s = np.array([aln.align(kmer, normalize=True) for aln in alns]).T
                ofs.append(o)
                scores.append(s)

            scores = np.array(scores)
            ofs = np.array(ofs)

            if (scores < thresh).all():
                kmer, r = kmer_set[0]
                print "starting NEW MOTIF", kmer, r, scores[0]
                kmer_set.pop(0)
                aln = Alignment()
                aln.blend(kmer, 0, r, normalize=False)
                alns.append(aln)
            else:
                # find best aligning kmer and add to best matching motif
                mer_scores = scores.max(axis=1)
                best_i = mer_scores.argmax()
                # print "best matching kmer is", kmer_set[best_i]

                kmer, r = kmer_set.pop(best_i)
                j = scores[best_i].argmax()
                s = scores[best_i, j]
                o = ofs[best_i, j]

                alns[j].blend(kmer, int(o), r, normalize=False)
                cons = alns[j].to_PSAM(pseudo=0).consensus
                print "blended", kmer, r, "with", cons, scores[best_i], "ofs=", ofs[best_i]
                # if cons == 'AUAGCAU':
                #     print alns[j].matrix
                #     print alns[j].align(kmer, normalize=True, debug=True)
    

        print "done assembling {0} motifs from {1} kmers with z > {2}".format(len(alns), n, z_cut)
        return [aln.to_PSAM(pseudo=0, keep_weight=keep_weight, n_max=n_max) for aln in alns if len(aln.seqs) >= n_min]




    # @property
    def topR_PSAM_seed(self, k, keep_weight=.9, n_max=7, thresh = .6, z_cut=4):
        """
        Assemble a seed motif from the most enriched k-mers
        """
        from cska.seed import Alignment
        import cska.cyska as cyska

        aln = Alignment()
        R, R_err = self.rbns.R_value_matrix(k)
        R = R.mean(axis=0)
        R_err = R_err.mean(axis=0)

        I = R.argsort()[::-1]
        R_sort = R[I]
        z = (R_sort - R_sort.mean())/R_sort.std()
        # print R_sort[:10]
        # print "z-scores", z.min(), z.max(), z.mean()
        i_cut = (z < z_cut).argmax()
        # print i_cut
        R_cut = max(1, R[I[i_cut]])
        # print i_cut, "R_values above z-cut", R_cut
        
        r0 = R[I[0]]
        # print "maxR", r0
        for i in I[:100]:
            kmer = cyska.index_to_seq(i, k)
            r = R[i]
            # print i, kmer, r, "+/-", R_err[i], R_cut
            if r - R_err[i] < R_cut:
                break

            o, s = aln.align(kmer, normalize=True)
            if s < thresh:
                # print "skipping", kmer, r, o, s
                continue
            else:
                # print "blending", i, kmer, r, o, s
                aln.blend(kmer, o, r/r0, normalize=False)
                # print aln.matrix

        # print aln
        return aln.to_PSAM(pseudo=0)
        
        # psam = aln.to_PSAM(n_max = n_max, pseudo=0)
        # # before building a gradient, need to align to matrices and pad with zero columns
        # grad = self.params.copy()
        # grad.psam_matrix = psam.psam
        # grad.A0 = 1.
        # grad.betas[:] = 0
        # return grad

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
            # self.spacings = self.analysis.bipartite_PSAM_spacings(psam_A=self.psam_A, psam_B=self.psam_B)
            # L = len(self.spacings)
            # self.dist_cost = self.spacings[L/2:]
            # self.logger.debug("bipartite spacing weights: {0}".format(self.dist_cost))
        
        self.store_logos()

    def seeded_params(self, n_samples, **kwargs):
        from cska.params import ModelParametrization
        return ModelParametrization.from_PSAM(self.psam_lin, n_samples=n_samples, **kwargs)

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


    # def linear_seed_params(self, A0=1., aff0=1e-6):
    #     psam = self.psam_lin
    #     psam.A0 = A0

    #     return psam.kmer_affinity_table(aff0-aff0)

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

    import logging
    logging.basicConfig(level=logging.DEBUG)
    A = Alignment()
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
            n_max=0,
            pseudo_count=10, 
            rna_conc = 1000.,
            temp = 4,
            n_subsamples = 10,
            acc_storage_path = 'acc',
        )
        rbns.add_reads(reads)

    DK = DependentKmerAnalysis(rbns, km=3)    
    print "6mer R-value derived linear logo"
    kmer_seed = DK.topR_PSAM_seed(6, n_max=9)
    print kmer_seed
    kmer_seed.save_logo('kmer_seed.eps')

    DK.build_matrices()
    print "assembled linear logo"
    asm_seed = DK.linear_PSAM_seed(n_max=9)
    asm_seed.save_logo("asm_seed.eps")
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


    # print "A"
    # psam = DK.A.to_PSAM()
    # for mer, aff in zip(*psam.kmer_affinities):
    #     print mer, aff

    # print "B"
    # psam = DK.B.to_PSAM()
    # kmers, aff = psam.kmer_affinities
    # for mer, a in zip(kmers, aff):
    #     print mer, a



    # # TESTING mutual information
    # import matplotlib.pyplot as pp
    # from cyska import yield_kmers
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
    
