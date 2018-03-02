import numpy as np
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as pp

from itertools import izip_longest
import cska.ska_kmers as cyska
from cska.ska_kmers import yield_kmers

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

    def to_PSAM(self):
        m = self.matrix + 1
        psam = m / m.max(axis=1)[:,np.newaxis]
        # print m
        # print psam
        A0 = self.matrix.max(axis=1).sum()
        from cska.pwm import PSAM
        return PSAM(psam, A0=A0)
        

class DependentKmerAnalysis(object):
    def __init__(self, rbns, km=4):
        self.rbns = rbns
        self.km = km
        self.linear = Alignment()
        self.A = Alignment()
        self.B = Alignment()
        self.parts = [self.A, self.B]
        self.partscores = [0, 0]
        
        profs = []
        joints = []
        for reads in rbns.reads:
            joint = reads.joint_kmer_freq_distance_profile(km)
            joints.append(joint)
            prof = reads.kmer_mutual_information_profile(km)
            profs.append(prof)
        
        self.profs = np.array(profs)
        self.joints = np.array(joints)
        self.best_sample = np.unravel_index(self.joints.argmax(), self.joints.shape)[0]
        print "best_sample", self.best_sample


    def build_matrices(self, thresh=.75):
        j0 = self.joints[0]

        joint = self.joints[self.best_sample]
        reads = self.rbns.reads[self.best_sample]
        S_lin = 0
        S_A = 0
        S_B = 0
        kmers = list(cyska.yield_kmers(self.km))
        for d in range(18):
            print reads.name, d
            
            jR = np.log2(joint / j0)
            jRm = jR.max()
            I = jR[:,:,d].flatten().argsort()[::-1]

            for n in I:
                i, j = np.unravel_index(n, joint.shape[:2])
                if jR[i,j,d] < jRm * thresh:
                    break
                
                # print "most-co-enriched mers at d=", d, kmers[i], kmers[j], jR[i,j,d], jRm
                merge = kmers[i] + "-" * d + kmers[j]
                score = jR[i,j,d]
                
                s_lin = self.linear.add(merge, score)
                s_A = self.A.add(kmers[i], score)
                s_B = self.B.add(kmers[j], score)

                S_lin += s_lin
                S_A += s_A
                S_B += s_B

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

    def interaction_plot(self):
        reads = self.rbns.reads[self.best_sample]
        psam_A = self.A.to_PSAM()
        psam_B = self.B.to_PSAM()

        aff_A = psam_A.affinities
        aff_B = psam_B.affinities

        Z_A = aff_A[reads.get_index_matrix(psam_A.n)]
        Z_B = aff_B[reads.get_index_matrix(psam_B.n)]

        # TODO: implement multiplication/accumulation
        # cyska.interaction(Z_A, Z_B, k1 = psam_A.n, k2 = psam_B.n, d_max=15)

        pp.plot(Z_A.mean(axis=0))
        pp.plot(Z_B.mean(axis=0))

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
            n_max=1000000,
            pseudo_count=10, 
            rna_conc = 1000.,
            temp = 4,
            n_subsamples = 0,
            acc_storage_path = 'acc',
        )
        rbns.add_reads(reads)

    DK = DependentKmerAnalysis(rbns)    
    DK.build_matrices()

    DK.linear.save_logo("linear.eps")
    DK.A.save_logo("A.eps")
    DK.B.save_logo("B.eps")

    DK.interaction_plot()
    print "linear alignment"
    print DK.linear
    print DK.linear.matrix

    print "left"
    print DK.A
    print DK.A.matrix

    print "right"
    print DK.B
    print DK.B.matrix

    print "wlen", DK.linear.wlen, DK.A.wlen, DK.B.wlen
    print DK.linear.score / DK.linear.wlen
    print DK.A.score / DK.A.wlen
    print DK.B.score / DK.B.wlen
    print DK.linear_motif_score

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
    