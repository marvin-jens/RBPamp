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

        self.ext_cost = np.ones(125)
        self.ext_cost[:9] = 0.01
        self.ext_cost[9:18] = [.01, .02, .03, .04, .05, .06, .07, .08, .1, ]
        # print self.ext_cost
        self.seqs = []
        self.ofs = []
        self.weights = []

    def align(self, seq, normalize=False, multiply=False, contain=False, end_weight=False, min_overlap=4, core_k=None, core_start=None, debug=False):
        # TODO: handle core_k and core_start 
        bits = cyska.seq_to_bits(seq)
        l = len(seq)
        n = len(self.matrix)
        if not len(self.matrix):
            return 0, 1  # offset, alignment score
        else:
            scores = []
            Ms = self.max_score(k=len(seq))
            if contain:
                assert n > l
                d = n - l
                ofs_range = range(-d, d+1)
            else:
                ofs_range = range(-l + min_overlap, n + 1 - min_overlap)
            # print seq
            if end_weight == True:
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
                    if m_start and not end_weight is None:
                        start_avg = func(self.matrix[:m_start], axis=1).sum()

                    end_avg = 0.
                    if m_end < n and not end_weight is None:
                        end_avg = func(self.matrix[m_end:], axis=1).sum()

                ext_n = max(0, -ofs) + max(0, (ofs + l) - n) # number of columns that would be added to matrix
                ext_cost = 0
                for i in range(n, n+ext_n):
                    ext_cost += self.ext_cost[i]

                ms0 = self.matrix[m_start:m_end].max(axis=1).sum()
                ms = (ms0 + Ms) / 2. # favor alignments that overlap the highest weight region
                # ms = Ms
                n_cols = m_end - m_start

                s_start = max(-ofs, 0)
                s_end = s_start + n_cols
                col_scores = []
                # if end_weight:
                #     score = start_avg*end_avg if multiply else start_avg + end_avg
                # else:
                #     score = 1 if multiply else 0
                score = start_avg*end_avg if multiply else start_avg + end_avg
                # if seq == "acuuacc":
                #     print "score before matching part", score

                for i in range(n_cols):
                    if bits[i+s_start] > 3:
                        continue # skip gaps
                    
                    S = self.matrix[i+m_start, bits[i+s_start]]
                    col_scores.append(S)
                    score = score * S if multiply else score + S
                
                # score += score/n_cols * .1 * abs(ofs)
                score0 = score
                if normalize:
                    score /= ms
                    score0 /= ms0

                score -= ext_cost
                score0 -= ext_cost                
                # score += .05 * abs(ofs)

                scores.append( (score, score0) )
                if debug:
                    print ofs, s_start,":",s_end, seq[s_start:s_end], m_start,":", m_end, col_scores, "->", score, "ext_n", ext_n, "ext_cost", ext_cost, "ms", ms
                    print "max_score", self.matrix[m_start:m_end].max(axis=1)

            x = np.array(scores).T[0].argmax()
            S = scores[x][0]

            return ofs_range[x], S


    def blend(self, seq, ofs, weight, normalize=False):
        self.seqs.append(seq)
        self.weights.append(weight)
        if ofs < 0:
            self.ofs = [o - ofs for o in self.ofs]
            matrix = np.zeros((len(self.matrix)-ofs,4))
            matrix[-ofs:] = self.matrix[:]
            self.matrix = matrix
            ofs = 0
        
        d = ofs + len(seq) - len(self.matrix)
        if d > 0:
            matrix = np.zeros((len(self.matrix)+d,4))
            if len(self.matrix):
                matrix[:len(self.matrix)] = self.matrix[:]
            self.matrix = matrix
        
        self.ofs.append(ofs)

        bits = cyska.seq_to_bits(seq)
        l = len(seq)
        m = self.matrix.max()
        if m == 0:
            m = np.inf

        for i in range(l):
            if bits[i] > 3:
                continue # skip gaps
            self.matrix[i+ofs, bits[i]] += weight #min(max(weight, self.matrix[i+ofs, bits[i]]), m)
        
        if normalize:
            self.matrix /= self.max_score(k=len(seq))

    def add(self, seq, weight=1.):
        ofs, score = self.align(seq)
        self.blend(seq, ofs, weight)

        return score

    @property
    def score(self):
        return self.matrix.max(axis=0).mean()
    
    def max_score(self, k=7):
        if len(self.matrix):
            ma = self.matrix.max(axis=1)
            slices = np.array([ma[i:i+k].sum() for i in range(len(self.matrix)-k+1)])
            return slices.max()
        else:
            return 1.

    @property
    def max_weight(self):
        return np.array(self.weights).max()

    @property
    def wlen(self):
        colw = self.matrix.max(axis=1) / self.matrix.max()
        return colw.sum()

    def __str__(self):
        buf = []
        for s, o, w in zip(self.seqs, self.ofs, self.weights):
            spacer = " "*o
            buf.append("{w:3.3e}  {spacer}{s}".format(**locals()))

        ms = self.max_score(k=len(self.matrix))
        perc = 100. * self.score / ms
        buf.append("average max. column score {0:.2f} of {1:.2f} ({2:.2f}%)".format(self.score, ms, perc))
        return "\n".join(buf)

    def save_logo(self, fname):
        from cska.pwm import weblogo_save
        weblogo_save(self.matrix, fname)

    def to_PSAM(self, keep_weight=1., n_max=0, pseudo=1, col_scale=True, A0=None):
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

        def find_best():
            bylength = sorted(best.keys())
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
        
        if A0 is None:
            A0 = self.max_weight

        P = PSAM(psam, A0=A0)
        P._n_seqs = len(self.seqs)
        P._max_weight = self.max_weight
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
        self.logger = logging.getLogger("seed.SeedRefinement({0})".format(km))
        self.km = km
        import shelve
        self.shelf = shelve.open(os.path.join(cska.ensure_path(os.path.join(self.rbns.out_path,'seed/')), 'history'), 'c')
        # self.analysis = DependentKmerAnalysis(self.rbns, km=km)
        # self.analysis.build_matrices()
        # self.psam_lin = self.analysis.linear_PSAM_seed(keep_weight=keep_weight, n_max=max_linear_k)
        # self.logger.info("linear_motif score={0:.2f} for {1}mer {2}".format(self.analysis.linear_motif_score, self.psam_lin.n, self.psam_lin.consensus))
        # self.psam_A, self.psam_B = self.analysis.bipartite_PSAM_seeds()
        # self.linear_k = self.psam_lin.n
        # self.bipart_k = self.psam_A.n
        
        # if self.analysis.linear_motif_score < .9:
        #     self.logger.info("bipartite motifs are potentially a better match for this RBP")
            # self.spacings = self.analysis.bipartite_PSAM_spacings(psam_A=self.psam_A, psam_B=self.psam_B)
            # L = len(self.spacings)
            # self.dist_cost = self.spacings[L/2:]
            # self.logger.debug("bipartite spacing weights: {0}".format(self.dist_cost))
        
        # self.store_logos()

    def seeded_params(self, n_samples, **kwargs):
        from cska.params import ModelParametrization
        return ModelParametrization.from_PSAM(self.psam_lin, n_samples=n_samples, **kwargs)

    def primer_analysis(self, k=7):
        from cska.seed import Alignment
        import cska.cyska as cyska

        dG = np.fromfile(
            os.path.join(
                os.path.dirname(__file__), '../adapters/7mer_adap3.dG'
            ),
            sep='\n'
        )
        print dG
        low_dG = np.percentile(dG, 50)
        mask = (dG <= low_dG)
        print low_dG, len(mask)
        g = dG[mask]

        alns = []
        R, R_err = self.rbns.R_value_matrix(k)
        from scipy.stats import spearmanr, pearsonr
        for j, r in enumerate(R):
            print "sample", j
            r_dG = np.log2(r[mask])
            print pearsonr(g, r_dG)
            print spearmanr(g, r_dG)

            import cska.report
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(g, r_dG, '.')
            plt.xlabel('dG')
            plt.ylabel('log2 R')
            plt.savefig('r_dG_{}.pdf'.format(j))
            plt.close()

        # R = R.mean(axis=0)
        # Rns = np.percentile(R, q_ns)
        # # print "non-specific quantile", Rns
        # R_err = R_err.mean(axis=0)


    def motifs_from_R(self, k=7, z_cut=4, n_min=5, q_ns=5., **kwargs): # UNDO HERE!!!
        from cska.seed import Alignment
        import cska.cyska as cyska

        alns = []
        R, R_err = self.rbns.R_value_matrix(k)
        R = R.mean(axis=0)
        Rns = np.percentile(R, q_ns)
        # print "non-specific quantile", Rns
        R_err = R_err.mean(axis=0)

        self.shelf['R0'] = R
        self.shelf['Rns'] = Rns

        R = R + Rns * ( (R - 1)/ (1 - Rns)) # corrected R-value, see suppl. methods
        self.shelf['R'] = R

        I = R.argsort()[::-1]
        self.shelf['I'] = I
        R_sort = R[I]
        z = (R_sort - R_sort.mean())/R_sort.std()
        self.shelf['z'] = z
        # print R_sort[:10]
        # print "z-scores", z.min(), z.max(), z.mean()
        i_cut = (z < z_cut).argmax()
        self.shelf['i_cut'] = i_cut
        self.shelf['i_ns'] = (R_sort > Rns).argmax() - 1
        # print i_cut
        R_cut = max(1, R[I[i_cut]])
        # print i_cut, "R_values above z-cut", R_cut

        r0 = R[I[0]]
        # print "maxR", r0
        n = 0
        kmer_set = []
        enriched = []

        for i in I:
            kmer = cyska.index_to_seq(i, k)
            n += 1
            r = R[i] 
            rerr = R_err[i]
            # self.logger.debug( "{i}, {kmer}, {r}, +/- {rerr}, {R_cut}".format(**locals()))
            if r - rerr <= R_cut and len(kmer_set) > n_min:
                break
            
            kmer_set.append( (kmer, r) )
            enriched.append( (r, kmer) )
        
        self.shelf['kmer_set'] = kmer_set
        n_enriched = len(kmer_set)
        self.logger.debug("seeding PSAMs from {0} significantly enriched {1}-mers".format(n_enriched, k))

        pb = PSAMBuilder(enriched, **kwargs)
        psams = pb.aggregate(n_min=n_min, **kwargs)

        maxlen = max([psam.n for psam in psams])
        motifs = ",".join([p.consensus_ul for p in psams])
        self.logger.info(
            "done assembling {0} motifs of width {1} from {2} kmers (at least {5} per motif) with z > {3}: {4}".format(
                len(psams), maxlen, len(kmer_set), z_cut, motifs, n_min
            )
        )
        
        self.shelf['psams'] = psams
        self.shelf['width'] = psams[0].n
        self.shelf['n_psam'] = len(psams)

        return psams


        # def make_psam(aln, **kwargs):
        #     return aln.to_PSAM(
        #         pseudo=0, 
        #         keep_weight=keep_weight, 
        #         A0=aln.max_weight/r0 * A0,
        #         **kwargs
        #     )

        # def get_motifs(alns):
        #     psams = [make_psam(a) for a in alns]
        #     return ",".join([p.consensus_ul for p in psams])

        # kmer, r = kmer_set.pop(0)
        # # print "STARTING from", kmer, r
        # aln = Alignment()
        # aln.blend(kmer, 0, r, normalize=False)
        # alns.append(aln)
        # self.logger.debug("starting first motif with {0} R_est={1:.1f}".format(kmer, r))

        # def update_scores(alns, kmer_set):
        #     scores = []
        #     ofs = []
        #     # print "re-aligning"
        #     for kmer, r in kmer_set:
        #         o, s = np.array([aln.align(kmer, normalize=True) for aln in alns]).T
        #         ofs.append(o)
        #         scores.append(s)

        #     scores = np.array(scores)
        #     ofs = np.array(ofs)
            
        #     return scores, ofs

        # def start_new(alns, kmer_set):
        #     kmer, r = kmer_set[0]
        #     current_motifs = get_motifs(alns)
        #     self.logger.debug("{0} R_est={1:.1f} does not match existing motifs ({2}). Seeding new motif".format(kmer, r, current_motifs))
        #     best_i = scores.max(axis=1).argmax()
        #     self.logger.debug("scores {0}:{1}, highest scores in set for {2}:{3} thresh={4}".format(kmer, scores[0], kmer_set[best_i][0], scores[best_i], thresh))
        #     # print "starting NEW MOTIF", kmer, r, scores[0]
        #     kmer_set.pop(0)
        #     aln = Alignment()
        #     aln.blend(kmer, 0, r, normalize=False)
        #     alns.append(aln)

        #     return alns, kmer_set

        # def blend_best(alns, kmer_set, scores, ofs):
        #     # find best aligning kmer and add to best matching motif
        #     mer_scores = scores.max(axis=1)
        #     best_i = mer_scores.argmax()
        #     # print "best matching kmer is", kmer_set[best_i]

        #     kmer, r = kmer_set.pop(best_i)
        #     j = scores[best_i].argmax()
        #     s = scores[best_i, j]
        #     o = ofs[best_i, j]

        #     alns[j].blend(kmer, int(o), r, normalize=False)
        #     # cons = alns[j].to_PSAM(pseudo=0).consensus
        #     # print "blended", kmer, r, "with", cons, scores[best_i], "ofs=", ofs[best_i]
        #     # if cons == 'AUAGCAU':
        #     #     print alns[j].matrix
        #     #     print alns[j].align(kmer, normalize=True, debug=True)
        #     return alns, kmer_set

        # while kmer_set:
        #     # align all remaining enriched kmers to all motifs
        #     scores, ofs = update_scores(alns, kmer_set)
        #     if (scores < thresh).all() and len(alns) < m_max:
        #         start_new(alns, kmer_set)
        #     else:
        #         blend_best(alns, kmer_set, scores, ofs)

        # keep = []
        # drop = []
        # orphan_set = []
        # for aln in alns:
        #     if len(aln.seqs) >= n_min:
        #         keep.append(aln)
        #     else:
        #         drop.append(aln)
        #         ks = [ (kmer, w*r0) for kmer, w in zip(aln.seqs, aln.weights)]
        #         orphan_set.extend(ks)

        # orphan_set = sorted(orphan_set, key = lambda x : x[1], reverse=True)
        # print "need to drop {} motifs with {} kmers".format(len(drop), len(orphan_set))
        # print "re-distributing kmers of weakest motfs", orphan_set
        # while orphan_set:
        #     # align all remaining enriched kmers to all motifs
        #     scores, ofs = update_scores(keep, orphan_set)
        #     blend_best(keep, orphan_set, scores, ofs)

        # psams = [make_psam(aln, n_max=n_max) for aln in keep]
        # maxlen = max([psam.n for psam in psams])
        # motifs = ",".join([p.consensus_ul for p in psams])
        # self.logger.info("done assembling {0} motifs of width {1} from {2} kmers (at least {5} per motif) with z > {3}: {4}".format(len(psams), maxlen, n, z_cut, motifs, n_min))
        # w = np.array([p.n for p in psams])
        # wm = w.max()

        # # second pass -> pad motifs to equal size
        # [p.pad_to_size(wm) for p in psams]
        # self.shelf['width'] = wm
        # self.shelf['n_psam'] = len(psams)
        # self.shelf['psams'] = psams
        # return psams


    def seeded_multi_params(self, n_samples, max_motifs=4, k_seed=7, thresh=.7, **kwargs):
        from cska.params import ModelSetParams, ModelParametrization
        params = []

        for i, psam in enumerate(self.motifs_from_R(k=k_seed, m_max=max_motifs, thresh=thresh, **kwargs)):
            params.append(ModelParametrization.from_PSAM(psam, n_samples=n_samples, **kwargs))

        param_set = ModelSetParams(params, sort=True)
        self.store_logos(param_set)
        return param_set


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

    def store_logos(self, params=None):
        self.logger.debug("generating sequence logos")
        path = cska.ensure_path(os.path.join(self.rbns.out_path,'seed/'))
        rbp_name = self.rbns.reads[0].rbp_name

        if not params is None:
            fname = os.path.join(path, 'seeded_{}.svg'.format(rbp_name))
            params.save_logos(fname, title="{} seeded PSAMs".format(rbp_name))
        else:
            self.psam_lin.save_logo(os.path.join(path, '{0}_linear.svg'.format(rbp_name)))
            self.psam_A.save_logo(os.path.join(path, '{0}_motif_A.svg'.format(rbp_name)))
            self.psam_B.save_logo(os.path.join(path, '{0}_motif_B.svg'.format(rbp_name)))
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

from cska.pwm import PSAM, project_column
class PSAMBuilder(object):
    def __init__(self, enriched, init=True, contaminants=[], keep_weight=.95, n_max=11, m_max=5, thresh=.72, n_min=5, A0=0.01, debug=False, **kwargs):
        self.logger = logging.getLogger("opt.seed.PSAMBuilder")
        self.debug = debug
        self.keep_weight = keep_weight
        self.n_max = n_max
        self.thresh = thresh
        self.A0 = A0
        self.alns = []
        for seq in contaminants:
            ca = Alignment()
            ca.blend(seq, 0, 1, normalize=False)
            self.alns.append(ca)

        self.n_contaminants = len(contaminants)

        self.enriched = enriched
        self.n_kmers = len(enriched)
        # self.P = [PSAM.from_kmer(mer, A0=R).matrix for R, mer in sorted(enriched, reverse=True)]
        # self.disc = [self.discrimination(p).sum() for p in self.P]
        # self.maxR = [p.max() for p in self.P]
        self.r0 = np.array([r for r, kmer in self.enriched]).max()
        # self.ext_cost = np.array([np.Inf, 0., 0., 0., 0., 0., 0., 0.0, 0.1, .15, .2, .3, 1.5, 2., np.inf, np.inf, np.inf, np.inf, np.inf])
        
        # n = len(self.P)
        # self.scores = np.zeros( (n, n) ) - np.Inf
        # self.shifts = np.zeros( (n, n) ) + np.NaN
        # self.merged = np.empty( (n, n), dtype=object )
        
        # self.align_debug = False
        # if init:
        #     self.fill_tables()
        # self.align_debug = True #False

    def make_psam(self, aln, **kwargs):
        return aln.to_PSAM(
            pseudo=0, 
            keep_weight=self.keep_weight, 
            A0=aln.max_weight * self.A0,
            **kwargs
        )

    def get_motifs(self):
        return ",".join([a.to_PSAM().consensus_ul for a in self.alns])

    def update_scores(self, alns, enriched, prev_scores=None, prev_ofs=None, col=None):
        
        scores = []
        ofs = []
        if col is None:
            # print "re-aligning"
            for r, kmer in enriched:
                o, s = np.array([aln.align(kmer, normalize=True, end_weight=None) for aln in alns]).T
                ofs.append(o)
                scores.append(s)
        else:
            scores = prev_scores
            ofs = prev_ofs

            for i,(r, kmer) in enumerate(enriched):
                o, s = alns[col].align(kmer, normalize=True, end_weight=None)
                ofs[i, col] = o
                scores[i, col] = s

        scores = np.array(scores)
        ofs = np.array(ofs)
        # print "shapes", scores.shape, ofs.shape
        if self.debug:
            print "UPDATE", col
            for j, i in enumerate(scores.max(axis=1).argsort()[::-1]):
                # print "j,i", j, i, len(enriched), len(alns)
                r, kmer = enriched[i]
                al = alns[scores[i].argmax()]
                print kmer, np.round(r/self.r0, 2), scores[i], "->", al.to_PSAM().consensus_ul, ofs[i]
                if j > 2:
                    break

        return scores, ofs

    def start_new(self, scores, ofs):
        r, kmer = self.enriched[0]
        # print "START NEW FROM", kmer
        current_motifs = self.get_motifs()
        self.logger.debug("{0} R_est={1:.1f} does not match existing motifs ({2}). Seeding new motif".format(kmer, r, current_motifs))
        best_i = scores.max(axis=1).argmax()
        self.logger.debug("scores {0}:{1}, highest scores in set for {2}:{3} thresh={4}".format(kmer, scores[0], self.enriched[best_i][1], scores[best_i], self.thresh))
        # print "starting NEW MOTIF", kmer, r, scores[0]
        self.enriched.pop(0)
        aln = Alignment()
        aln.blend(kmer, 0, r, normalize=False)
        self.alns.append(aln)

        return 0

    def blend_best(self, alns, enriched, scores, ofs):
        # find best aligning kmer and add to best matching motif
        mer_scores = scores.max(axis=1)
        best_i = mer_scores.argmax()
        # print "best matching kmer is", best_i, enriched[best_i]

        r, kmer = enriched.pop(best_i)
        j = scores[best_i].argmax()
        s = scores[best_i, j]
        o = ofs[best_i, j]

        self.logger.debug("BEST ALIGNMENT out of {} is {} + {} shift={}".format(self.get_motifs(), alns[j].to_PSAM().consensus_ul, kmer, s))
        if kmer == "augcacg":
            alns[j].align(kmer, normalize=True, debug=True, end_weight=None)

        if j >= self.n_contaminants:
            alns[j].blend(kmer, int(o), r, normalize=False)
        else:
            self.logger.debug("dropping contaminant-matching kmer {}".format(kmer))
        # cons = alns[j].to_PSAM(pseudo=0).consensus
        # print "blended", kmer, r, "with", cons, scores[best_i], "ofs=", ofs[best_i]
        # if cons == 'AUAGCAU':
        #     print alns[j].matrix
        #     print alns[j].align(kmer, normalize=True, debug=True)
        return best_i, j

    def fill_tables(self):
        n = len(self.P)
        
        for i in range(n):
            for j in range(i):
                score, shift, N = self.align(self.P[i], self.P[j], debug=self.align_debug)
                self.scores[i, j] = score
                self.shifts[i, j] = shift 
                self.merged[i, j] = N

        # self.scores += self.scores.T

    def discrimination(self, P):
        D = (P.max(axis=1) / P.sum(axis=1) - 1./4.) / 0.75
        w = (P.max(axis=1) / P.sum(axis=1).max()) **0
        # print "disc weights", w
        return D*w

    def mean_discrimination(self):
        d = np.array(self.disc)
        r = np.array(self.maxR)

        return (d*r).sum() / r.sum() # maxR-weighted mean

    def align(self, P1, P2, max_shift=5, ws=-0.1, debug=False):
        if len(P2) > len(P1):
            P1, P2 = P2, P1
        
        d1 = self.discrimination(P1)
        d2 = self.discrimination(P2)

        D1 = d1.sum()
        D2 = d2.sum()
        if debug:
            print "aligning", self.consensus(P1)
            print P1
            print "D1", d1, D1
            print "with", self.consensus(P2)
            print P2
            print "D2", d2, D2

        l1 = len(P1)
        l2 = len(P2)
        A1 = P1.max()
        A2 = P2.max()
        A = max(A1, A2)
        # the first is longer or same length
        shifts = range(- max_shift, max_shift + 1)
        if debug:
            print "shifts", max_shift, shifts
        # shifts = [-1] # DEBUG!
        normed1 = P1 / P1.max(axis=1)[:, np.newaxis]
        normed2 = P2 / P2.max(axis=1)[:, np.newaxis]

        scores = []
        for s in shifts:
            s1 = max(s,0)
            e1 = min(s+l2, l1)
            s2 = max(-s, 0)
            e2 = min(l2, s2+l2)

            M1 = P1[s1:e1]
            M2 = P2[s2:e2]


            N = np.zeros((max(s1 + l2, s2+l1), 4))
            # print "len N", len(N), "s2+l1", s2+l1, "s1+l2", s1+l2
            N[s2:s2+l1] += P1
            N[s1:s1+l2] += P2
            
            
            w = N.sum(axis=1)
            w /= w.sum()
            normed = N / N.max(axis=1)[:, np.newaxis]
            div = \
                (np.fabs(normed[s2:s2+l1] - normed1).sum(axis=1) * w[s2:s2+l1]).sum() + \
                (np.fabs(normed[s1:s1+l2] - normed2).sum(axis=1) * w[s1:s1+l2]).sum()

            dN = self.discrimination(N)
            DN = dN.sum()
            # print "DN", dN, DN

            # how much discrimination is in the overlap?
            al1 = dN[s1:e1].sum() / D1
            al2 = dN[s2:e2].sum() / D2

            Ln = len(N)
            L1 = len(P1)
            L2 = len(P2)

            # Ln = 1
            # L1 = 1
            # L2 = 1
            # df = DN/Ln / (A1*D1/L1 + A2*D2/L2) * (A1+A2)
            # df = DN / (A1*D1 + A2*D2) * (A1+A2)
            df = 2 * DN / (D1 + D2)
            
            ss = 0
            for x in range(l1, len(N)):
                ss -= self.ext_cost[x]

            # ss = - self.ext_cost[len(N)] * (len(N) - l1)
            # if debug:
            #     # print "s={s} l1={l1} l2={l2} M1={s1}:{e1} M2={s2}:{e2}".format(**locals())
            #     print "s={s} dN={dN} (fraction of mean={df})".format(**locals())
            # if debug:
            #     print "overlap buffer 1"
            #     print N
            # N[s2:s2+e1] /= (A2 + A1) # weighted mean

            # amax - N.max(axis=1)
            # N /= amax[:, np.newaxis]

            # print "overlap buffer NORMED"
            # print N
            N1 = N[s2+s1:s2+s1+len(M1)]
            N2 = N[s2+s1:s2+s1+len(M2)]
            # if debug:
            #     print "N1"
            #     print N1
            #     print "M1"
            #     print M1
            #     print "N2"
            #     print N2
            #     print "M2"
            #     print M2
            # dM1 = self.discrimination(M1) 
            # dM2 = self.discrimination(M2) 
            # x1 = np.fabs((dM1 - self.discrimination(N1)).sum()) / dM1.sum() # fraction of discrimination lost in overlapping region
            # x2 = np.fabs((dM2 - self.discrimination(N2)).sum()) / dM2.sum()

            # ss = ws * ((unal1 * A1) + (unal2 * A2)) / (A1 + A2)
            # ss = ws * (unal1  + unal2)/ 2.
            # score = (d1 * A1 + d2* A2)/(A1 + A2) + ss
            # score = (x1 + x2) / 2. + ss
            # keep = (A1 * al1 + A2 * al2)/ (A1 + A2)
            # keep = al1 * al2
            # score = df * keep + ss
            score = (1./(div + 1.) + ss)
            # print score
            # r1 = np.fabs(N1/(A1 + A2) - M1/A1).sum() / (M1.sum() / A1)
            # r2 = np.fabs(N2/(A1 + A2) - M2/A2).sum() / (M2.sum() / A2)
            # score = (r1 * A1 + r2* A2)/(A1 + A2) + ss
            # print "score", score
            if debug:
                print s, '->', np.round(score, 2), "div", np.round(div, 3), "ss", ss
                # print s, '->', np.round(score,2), "df", np.round(df,2), "keep", np.round(keep, 3), "ss", ss, 'dN', np.round(dN.sum(),3), 'al1', al1, 'al2', al2 #d1, "M2", d2, "shift", ss, "unal1", unal1, "unal2", unal2, 
            
            # N *= (A2 + A1) / A
            scores.append( (score, s, N) )
        
        best = sorted(scores, reverse=True)[0]
        if debug:
            print ">> best alignment <<", np.round(best[0], 5)
            s = int(best[1])
            si = max(-s, 0)
            sj = max(s, 0)

            print " "*si, self.consensus(P1)
            print " "*sj, self.consensus(P2)

        return best

    def consensus(self, mat):
        return "".join([project_column(col) for col in mat])

    def find_match(self):
        # tilt = .001 * np.log(self.maxR)
        scores = np.array(self.scores)
        # scores += tilt[:, np.newaxis]
        # scores += tilt[np.newaxis, :]
        i, j = np.unravel_index(scores.argmax(), scores.shape)
        return i, j, self.scores[i, j], self.shifts[i, j], self.merged[i, j]

    def drop(self, x, fill=-np.inf):
        self.P.pop(x)
        self.disc.pop(x)
        self.maxR.pop(x)
        for arr in [self.scores, self.shifts, self.merged]:
            arr[x:-1, :] = arr[x+1:, :]
            arr[:, x:-1] = arr[:, x+1:]
            arr[-1, :] = fill
            arr[:, -1] = fill

    def add(self, new):
        n = len(self.P)
        self.P.append(new)
        self.disc.append(self.discrimination(new).sum())
        self.maxR.append(new.max())

        for j in range(n):
            score, shift, N = self.align(self.P[n], self.P[j], debug=self.align_debug)
            self.scores[n, j] = score
            self.shifts[n, j] = shift 
            self.merged[n, j] = N

    def aggregate(self, n_valid=5):
        score_steps = []
        mean_disc = []
        mean_score = []
        valid_sets = []

        while len(self.P) > 1:
            print "PSAMs", len(self.P)
            i, j, score, shift, new = self.find_match()
            
            order = np.array(self.maxR).argsort()[::-1]
            cons = np.array([self.consensus(p) for p in self.P])
            for h, x in enumerate(order):
                if x == i or x ==j:
                    m = '*'
                else:
                    m = ' '
                
                y = self.scores[x,:].argmax()
                z = self.scores[:,x].argmax()


                if self.scores[x, y] > self.scores[z, x]:
                    N = self.merged[x, y]
                    score = self.scores[x, y]
                    other = cons[y]
                else:
                    N = self.merged[z, x]
                    score = self.scores[z, x]
                    other = cons[z]
                    y = z
                    
                nc = self.consensus(N) if N is not None else 'none'
                print m, cons[x], np.round(self.maxR[x], 2), "+", other, '->', nc, 'score=', score

                if len(self.P) == 20 and x == i:
                #     print "<<< SHOULD"
                #     self.align(self.P[order[h]], self.P[order[h+1]], debug=True)
                    print "<<< IS", i, j, shift, self.consensus(self.P[i]), self.consensus(self.P[j])
                    self.align(self.P[i], self.P[j], debug=True)

            score_steps.append(score)
            mean_disc.append(self.mean_discrimination())

            S = np.array(self.scores).flatten()
            mean_score.append(np.mean(S[np.isfinite(S)]))

            # i, j, score, shift, new = self.find_match()
            s = int(shift)
            si = max(-s, 0)
            sj = max(s, 0)

            seq1 = self.consensus(self.P[i])
            seq2 = self.consensus(self.P[j])
            if len(seq2) > len(seq1):
                seq1, seq2 = seq2, seq1
            
            print " "*(si+2), seq1
            print " "*(sj+2), seq2
            print "->", self.consensus(new)

            self.drop(i)
            self.drop(j)
            self.add(new)

            n = len(self.P)
            if n <= n_valid:
                # print "score matrix"
                # print self.scores[:n, :n]
                
                from cska.params import ModelParametrization, ModelSetParams
                from cska.pwm import PSAM
                psams = [PSAM(np.round(p, 0), A0=p.max()) for p in self.P]
                params = [ModelParametrization.from_PSAM(ps, n_samples=1) for ps in psams]
                param_set = ModelSetParams(params, sort=True)
                param_set.save_logos("seed_{}left.pdf".format(len(self.P)))

                valid_sets.append(param_set)

        mean_disc.append(np.array(self.disc).mean())
        print "score history", score_steps
        print "mean disc.", mean_disc
        print "mean score", mean_score
        
        import cska.report
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(score_steps, label="score", linestyle='steps')
        plt.plot(mean_disc, label="avg. discrimination", linestyle='steps')
        plt.plot(mean_score, label="mean score", linestyle='steps')
        plt.legend()
        plt.savefig('scores.pdf')

    def aggregate(self, n_min=5, m_max=5, **kwargs):
        r, kmer = self.enriched.pop(0)
        # self.logger.debug("STARTING from", kmer, r
        aln = Alignment()
        aln.blend(kmer, 0, r, normalize=False)
        self.alns = [aln, ]
        self.logger.debug("starting first motif with {0} R_est={1:.1f}".format(kmer, r))

        prev_scores = None
        prev_ofs = None
        col = None
        while self.enriched:
            self.logger.debug("{} kmers left".format(len(self.enriched)))
            # align all remaining enriched kmers to all motifs
            scores, ofs = self.update_scores(self.alns, self.enriched, prev_scores=prev_scores, prev_ofs=prev_ofs, col=col)
            if (scores < self.thresh).all() and (len(self.alns) - self.n_contaminants) < m_max:
                i = self.start_new(scores, ofs)
                n, m = scores.shape
                prev_scores = np.zeros( (n-1, m+1), dtype=float)
                prev_ofs = np.zeros( (n-1, m+1), dtype=int)
                prev_scores[:i, :-1] = scores[:i, :]
                prev_scores[i:, :-1] = scores[i+1:, :]
                prev_ofs[:i, :-1] = ofs[:i, :]
                prev_ofs[i:, :-1] = ofs[i+1:, :]
                col = m

            else:
                i, j = self.blend_best(self.alns, self.enriched, scores, ofs)
                # print "drop scores for kmer", i
                n, m = scores.shape
                prev_scores = np.zeros( (n-1, m), dtype=float)
                prev_ofs = np.zeros( (n-1, m), dtype=int)
                prev_scores[:i, :] = scores[:i, :]
                prev_scores[i:, :] = scores[i+1:, :]
                prev_ofs[:i, :] = ofs[:i, :]
                prev_ofs[i:, :] = ofs[i+1:, :]
                col = j

        keep = []
        drop = []
        orphan_set = []
        for aln in self.alns[self.n_contaminants:]:
            if len(aln.seqs) >= n_min:
                keep.append(aln)
            else:
                drop.append(aln)
                ks = [ (w*self.r0, kmer) for kmer, w in zip(aln.seqs, aln.weights)]
                orphan_set.extend(ks)

        orphan_set = sorted(orphan_set, key = lambda x : x[1], reverse=True)
        print "need to drop {} motifs with {} kmers".format(len(drop), len(orphan_set))
        print "re-distributing kmers of weakest motfs", orphan_set
        while orphan_set:
            # align all remaining enriched kmers to all motifs
            scores, ofs = self.update_scores(keep, orphan_set)
            self.blend_best(keep, orphan_set, scores, ofs)

        psams = [self.make_psam(aln, n_max=self.n_max) for aln in keep]
        w = np.array([p.n for p in psams])
        wm = w.max()

        # second pass -> pad motifs to equal size
        [p.pad_to_size(wm) for p in psams]
        return psams


if __name__ == "__main__":

    test_data = [
        (100, 'UGCAUGC'),
        (100, 'GCAUGCA'),
        (90, 'UGCAUGU'),
        (80, 'GCAUGCA'),
        (79, 'GCAUGCU'),
        (78, 'GCAUGCC'),
        (77, 'GCAUGCG'),
        (70, 'GCAUGUA'),
        (69, 'GCAUGUU'),
        (68, 'GCAUGUC'),
        (67, 'GCAUGUG'),
        (85, 'AGCAUGU'),
        (75, 'CGCAUGU'),
        (55, 'GGCAUGU'),
        (79, 'AGCAUGC'),
        (78, 'CGCAUGC'),
        (58, 'GGCAUGC'),
        (30, 'UGCACGC'),
        (25, 'UGCACGU'),
        (15, 'UGCACGA'),
        (15, 'UGCACGG'),
        (90, 'GCAAUGC'), # test, secondary motif
        (85, 'GCAAUGU'),
        (88, 'UGCAAUG'),
        (75, 'AGCAAUG'),
    ]

    test_data_msi = [
        (9, 'UAGUUAG'),
        (6, 'UAGAUAG'),
        (5.6, 'UUAGUUA'),
        (5.2, 'UAGUUUA'), # <- 3
        (5, 'AUAGUUA'),
        (4.7, 'UAGGUAG'),
        (4.3, 'UAGCUAG'),
        (4.5, 'AGUUAGU'),
        (4.1, 'UUAGUUU'), # <- 8
        (4.2, 'AGUUUAG'), # <- 9
        (4.0, 'UUUAGUU'),
    ]
    pb = PSAMBuilder(test_data_msi, init=False)
    print "done building tables"
    pb.align(pb.P[3], pb.P[8], debug=True)
    pb.align(pb.P[3], pb.P[0], debug=True)
    sys.exit(0)

    pb.aggregate()


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
    #         for n in I[:10]:        data_colors = plt.get_cmap("YlOrBr")(np.linspace(.3, 1, len(labels)-1))

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
    
