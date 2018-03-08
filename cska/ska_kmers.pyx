__license__ = "MIT"
__version__ = "0.9.8"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

"""
A fast Cython implementation of the "Streaming K-mer Assignment" 
algorithm initially described in Lambert et al. 2014 (PMID: 24837674)
"""

from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np
cimport cython
cimport openmp

from libc.math cimport exp, log
from libc.stdlib cimport abort, malloc, free

ctypedef np.uint8_t UINT8_t
ctypedef np.uint16_t UINT16_t
ctypedef np.uint32_t UINT32_t
ctypedef np.uint64_t UINT64_t
ctypedef np.int32_t INT32_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

# maps ASCII values of A,C,G,T (U) to correct bits
cdef UINT8_t letter_to_bits[256]
for i in range(256):
    letter_to_bits[i] = 255

letter_to_bits[ord('a')] = 0 
letter_to_bits[ord('c')] = 1 
letter_to_bits[ord('g')] = 2
letter_to_bits[ord('t')] = 3
letter_to_bits[ord('u')] = 3

letter_to_bits[ord('A')] = 0 
letter_to_bits[ord('C')] = 1 
letter_to_bits[ord('G')] = 2
letter_to_bits[ord('T')] = 3
letter_to_bits[ord('U')] = 3

cdef UINT8_t bits_to_letters[4]
bits_to_letters[:] = [ord('A'), ord('C'), ord('G'), ord('U')]
    
def yield_kmers(k):
    import itertools
    """
    An iterater to all kmers of length k in alphabetical order
    """
    bases = 'ACGU'
    for kmer in itertools.product(bases, repeat=k):
        yield ''.join(kmer)
                                
cdef UINT64_t rand_state[2]
cdef UINT64_t RAND_MAX = 2**64 - 1
cdef FLOAT32_t FRAND_MAX = RAND_MAX

cdef inline UINT64_t randint():
    """
    Cython version of xorshift128plus by Vigna, Sebastiano 
    https://arxiv.org/abs/1404.0390
    """
    cdef UINT64_t x = rand_state[0]
    cdef UINT64_t y = rand_state[1]
    rand_state[0] = y
    x ^= x << 23
    rand_state[1] = x ^ y ^ (x >> 17) ^ (y >> 26)
    return rand_state[1] + y


cdef inline FLOAT32_t rand():
    cdef FLOAT32_t x = randint()
    return x / FRAND_MAX

def rand_seed(UINT64_t seed, burn=1000):
    cdef UINT64_t rnd
    
    rand_state[0] = seed
    rand_state[1] = (~ seed) << 3
    
    for i in range(burn):
        rnd = randint()


def fast_randint(int N, max=RAND_MAX):
    cdef np.ndarray[UINT64_t] rnd = np.empty(N, dtype=np.uint64)
    cdef int i
    
    for i in range(N):
        rnd[i] = randint() % max

    return rnd

def fast_rand(int N):
    cdef np.ndarray[FLOAT32_t] rnd = np.empty(N, dtype=np.float32)
    cdef int i
    
    for i in range(N):
        rnd[i] = rand()

    return rnd

def random():
    return rand()
    
# default initialization
import time
rand_seed(int(1000*time.time()) + 11)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
cdef inline UINT8_t rand_choice_uint8(UINT64_t [:] cum, int ofs, int n):
    cdef UINT64_t rnd = randint()
    cdef UINT8_t x = 0

    for x in range(n-1):
        if rnd < cum[ofs+x]:
            return x

    return n-1

def generate_random_sequence_matrix(UINT32_t l, UINT32_t N):
    cdef np.ndarray[UINT8_t, ndim=2] seqm_ = np.empty((N, l), dtype=np.uint8)
    cdef UINT8_t [:, :] seqm = seqm_ # MemoryView
    
    cdef UINT64_t i, j
    
    for j in range(N):
        for i in range(l):
            seqm[j,i] = randint() & 3 # use lower 2 bits
            
    return seqm_

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def generate_random_sequence_matrix_dinuc(UINT32_t l, UINT32_t N, np.ndarray[FLOAT32_t] nt_freqs, np.ndarray[FLOAT32_t, ndim=2] di_freqs):
    cdef np.ndarray[UINT8_t, ndim=2] seqm_ = np.empty((N, l), dtype=np.uint8)
    cdef UINT8_t [:, :] seqm = seqm_ # MemoryView
    
    cdef UINT64_t i, j, nuc
    cdef UINT64_t [:] cum_nt
    cdef UINT64_t [:] cum_di
    
    cum_nt = np.array(nt_freqs.cumsum() * FRAND_MAX, dtype=np.uint64)
    cum_di = np.array(di_freqs.cumsum(axis=1) * FRAND_MAX, dtype=np.uint64).flatten()

    #print "nt", cum_nt
    #print "di", cum_di
    
    for j in range(N):
        nuc = rand_choice_uint8(cum_nt, 0, 4)
        seqm[j,0] = nuc
        for i in range(1,l):
            nuc = rand_choice_uint8(cum_di, nuc << 2, 4)
            seqm[j,i] = nuc
            
    return seqm_




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)    
def simulate_rbns_reads(
        UINT32_t L, 
        UINT32_t N,
        UINT64_t k,
        np.ndarray[FLOAT32_t] nt_freqs, 
        np.ndarray[FLOAT32_t, ndim=2] di_freqs,
        np.ndarray[FLOAT32_t] scaled_kmer_energies_, # already in units of RT
        FLOAT32_t P, # protein concentration in nM
        FLOAT32_t p_ns, # prob. for non-specific binding
    ):

    cdef np.ndarray[UINT8_t, ndim=2] seqm_ = np.empty((N, L), dtype=np.uint8)

    cdef UINT8_t [:, :] seqm = seqm_ # MemoryView
    cdef UINT64_t [:] cum_nt
    cdef UINT64_t [:] cum_di
    cdef FLOAT32_t [:] kmer_energies =  scaled_kmer_energies_
    
    cum_nt = np.array(nt_freqs.cumsum() * FRAND_MAX, dtype=np.uint64)
    cum_di = np.array(di_freqs.cumsum(axis=1) * FRAND_MAX, dtype=np.uint64).flatten()

    cdef FLOAT32_t mu = np.log(P*1e-9)
    cdef FLOAT32_t [:] boltzmann_weights = np.exp(-scaled_kmer_energies_ + mu)

    
    cdef UINT8_t [:] seq_bits
    cdef FLOAT64_t Z, w, p_bound, p_obs
    cdef UINT64_t i, j, nuc, index, n_bound=0, n_simulated=0, l=L-k+1
    cdef UINT64_t MAX_INDEX = 4**k - 1, ofs
    cdef UINT8_t s
    
    #P *= 1e-9 # convert from nano Molars to Molars

    j = 0
    while j < N:
        # generate a random read with dinuc frequencies
        nuc = rand_choice_uint8(cum_nt, 0, 4)
        seqm[j,0] = nuc
        for i in range(1,L):
            nuc = rand_choice_uint8(cum_di, nuc << 2, 4)
            #nuc = rand_choice_uint8(cum_nt, 0, 4)
            seqm[j,i] = nuc
        n_simulated += 1
        
        # partition function for binding of a single protein
        Z = 0
        #ofs = j+l
        # compute index of first k-1 mer by bit-shifts
        index = 0
        for i in range(k-1):
            index += seqm[j, i] << 2 * (k - i - 2)

        # iterate over all k-mers in the read
        for i in range(k-1, L):
            # get next "letter"
            s = seqm[j, i]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            
            # Boltzmann weight for binding here
            Z += boltzmann_weights[index]
            #print "weight", exp(-kmer_energies[index]), "index", index, "energy", kmer_energies[index]

        #Z *= P # times protein concentration
        
        #print p_bound
        # do we observe this read?
        p_bound = Z / (Z + 1.)
        p_obs = 1 - (1-p_bound)*(1-p_ns)

        #if (seqm_[j,:] == 3).all(): #polyU
            #print "UUUUUU Z={0} p_bound={1} p_obs={2}".format(Z, p_bound, p_obs) 

        #print "Z={0} p_obs={1}".format(Z, p_obs)
        if rand() < p_obs:
            # was pulled down
            j += 1
    
    return seqm_, float(N)/n_simulated





def weighted_kmer_shifts(UINT64_t index, UINT64_t k, UINT64_t L, UINT64_t x, FLOAT32_t [:] kfreqs):
    """
    Used to compute the overlap matrix.
    """
    
    cdef UINT64_t i = 0, j = 0, s = 0
    cdef UINT64_t N = 4**x
    cdef UINT64_t l = L - k + 1
    
    cdef UINT64_t OV_MAX_INDEX, MAX_INDEX = (4**k) -1 
    
    cdef FLOAT32_t w

    cdef np.ndarray[UINT64_t] sindices_  = np.zeros(2*N, dtype=np.uint64)
    cdef np.ndarray[FLOAT32_t] sweights_ = np.zeros(2*N, dtype=np.float32)
    
    cdef UINT64_t [:] sindices = sindices_
    cdef FLOAT32_t [:] sweights = sweights_
    
    
    i = 0
    w = 1. * (l - 2*x) / l 
    OV_MAX_INDEX = 4**(k-x) - 1
    
    # shifted to the right, overlaps on the left with index
    ov = ( index << (2 * x) ) & MAX_INDEX
    for j in range(4**x): # all possible extensions
        s = j | ov
        sindices[i] = s
        sweights[i] = w * kfreqs[j]
        i += 1

    # shifted to the left, overlaps on the right with index
    ov = ( index >> (2 * x) ) & OV_MAX_INDEX
    for j in range(4**x): # all possible extensions
        s = (j << (2*(k-x)) ) | ov
        sindices[i] = s
        sweights[i] = w * kfreqs[j]
        i += 1

    return sweights_, sindices_
        
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)    
def write_seqm(np.ndarray[UINT8_t, ndim=2] seqm_, f):
    cdef UINT64_t N = seqm_.shape[0], l = seqm_.shape[1], i, j
    
    cdef UINT8_t [:, :] seqm = seqm_
    cdef np.ndarray[UINT8_t] seq_ = np.zeros(l+1, dtype=np.uint8)
    cdef UINT8_t [:] seq = seq_
    seq[l] = '\n'
    
    for j in range(N):
        # convert seq entries back to string
        for i in range(l):
            seq[i] = bits_to_letters[ seqm[j, i] ]

        f.write(seq_)
    
    

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def seq_to_bits(unsigned char *seq):
    cdef UINT32_t x, L
    cdef UINT8_t n
    
    L = len(seq)
    cdef np.ndarray[UINT8_t] _res = np.zeros(L, dtype=np.uint8)
    cdef UINT8_t [::1] res = _res
    
    for x in range(L):
        n = seq[x]
        res[x] = letter_to_bits[n]

    return _res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
cdef inline UINT64_t kbits_to_index(UINT8_t[:] kbits, UINT32_t k) nogil:
    cdef UINT64_t i, index = 0
    
    for i in range(k):
        index += kbits[i] << 2 * (k - i - 1)
    
    return index


def seq_to_index(seq):
    k = len(seq)
    bits = seq_to_bits(seq)
    return kbits_to_index(bits, k)

def index_to_seq(index, k):
    nucs = ['a','c','g','u']
    seq = []
    for i in range(k):
        j = index >> ((k-i-1) * 2)
        seq.append(nucs[j & 3])

    return "".join(seq)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def read_raw_seqs_chunked(src, str pre="", str post="", UINT32_t n_max=0, UINT32_t n_skip=0, int chunklines=1000000):
    cdef unsigned char* l
    cdef UINT64_t i, N=0, n=0, n0=0, L=0
    cdef UINT8_t x=0
    cdef UINT32_t chunkbytes = 0
    cdef list chunks = list()
    cdef bytes line
    cdef bytes _pre = <bytes>pre
    cdef bytes _post = <bytes>post
    cdef np.ndarray[UINT8_t] _buf
    cdef UINT8_t [::1] buf # fast memoryview into current buffer

    for line in src:
        if n_skip and N <= n_skip:
            continue

        #if 'N' in line:
            #continue

        line = line.rstrip() # remove trailing new-line characters

        #if pre or post:
            #line = _pre + line + _post
        
        if not L:
            L = len(line)
            chunkbytes = chunklines*L
        
        if N % chunklines == 0:
            if N: chunks.append(_buf)
            _buf = np.empty(L*chunklines, dtype=np.uint8)
            buf = _buf # initialize the view
            n = 0

        l = line # extract raw string content
        n0 = n
        for i in range(0,L):
            x = letter_to_bits[l[i]]
            if x > 3:
                # non-ACGT character!
                n = n0
                break
            
            buf[n] = x
            n += 1

        if n > n0:
            # we have actually read a sequence!
            N += 1
            
        if N >= n_max + n_skip and n_max:
            break

    chunks.append(_buf[:n])
    cat = np.concatenate(chunks)

    return cat.reshape((N,L))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def seq_set_kmer_count(np.ndarray[UINT8_t, ndim=2] seq_matrix, UINT64_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1

    # store k-mer counts here
    _counts = np.zeros(4**k, dtype = np.uint32)
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT32_t [::1] counts = _counts

    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j
    
    with nogil:
        for j in range(N):
            seq_bits = _seq_matrix[j*L:(j+1)*L]
            # compute index of first k-mer by bit-shifts
            index = kbits_to_index(seq_bits, k) 
            # count first k-mer
            counts[index] += 1
            # iterate over remaining k-mers
            for i in range(0, L-k):
                # get next "letter"
                s = seq_bits[i+k]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                # count
                counts[index] += 1
            
    return _counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def seq_set_kmer_count_matrix(UINT8_t [:,:] seq_matrix, UINT64_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1
    cdef int N = len(seq_matrix.base)
    cdef int L = len(seq_matrix.base[0])

    # store k-mer counts here
    cdef UINT32_t [:,:] counts = np.zeros((N, 4**k), dtype = np.uint32)

    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j
    
    with nogil:
        for j in range(N):
            index = 0
            for i in range(L):
                # get next "letter"
                s = seq_matrix[j, i]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                if i >= k-1:
                    # count
                    counts[j, index] += 1
            
    return counts.base


#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_crosstalk_matrix(UINT32_t [:,:] im1, UINT32_t [:,:] im2, UINT64_t k1, UINT64_t k2):

    assert k1 <= k2
    cdef int N = im1.base.shape[0]
    assert im2.base.shape[0] == N
    cdef int L1 = im1.base.shape[1]
    cdef int L2 = im2.base.shape[1]

    # store k-mer counts here
    cdef FLOAT32_t [:,:] overlaps = np.zeros((4**k1, 4**k2), dtype = np.float32)

    # helper variables to tell cython the types
    cdef UINT64_t index1, index2, i, j, l, ofs
    
    # because we start inside the 5' adapter,
    # k2-mer indices start earlier in the sequence if k2 > k1.
    ofs = k2 - k1
    
    with nogil:
        for j in range(N):
            for i in range(L1):
                index1 = im1[j,i]
                for l in range(max(0, i - k1 + ofs), min(L2, i + k1 + ofs)):
                    index2 = im2[j,l]
                    overlaps[index1, index2] += 1
                    
    return overlaps.base



#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
#@cython.overflowcheck(False)
#def seq_set_kmer_flag(UINT8_t [:,:] seq_matrix, UINT64_t k, UINT64_t kmer_index):
    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1

    #cdef UINT64_t N = seq_matrix.base.shape[0]
    #cdef UINT64_t L = seq_matrix.base.shape[1]
    #cdef UINT64_t l = L - k + 1

    ## store k-mer counts here
    #cdef UINT8_t [::1] flags = np.zeros(N, dtype = np.uint8)

    ## helper variables to tell cython the types
    #cdef UINT8_t s
    #cdef UINT64_t index, i, j
    
    #with nogil:
        #for j in range(N):
            #index = 0
            #for i in range(k-1):
                #index += seq_matrix[j, i] << 2 * (k - i - 2)
            
            #for i in range(0, l):
                ## get next "letter"
                #s = seq_matrix[j, i+k-1]
                
                ## compute next index from previous by shift + next letter
                #index = ((index << 2) | s ) & MAX_INDEX
                
                ## count
                #if index == kmer_index:
                    #flags[index] += 1
            
    #return flags.base


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
#@cython.overflowcheck(False)
#def eval_energy_model_on_seqs(UINT8_t [:,:] seq_matrix, UINT8_t [:,:] openen_matrix, FLOAT32_t [:] acc_lookup, FLOAT32_t [:] kmer_invkd, np.ndarray[FLOAT32_t] protein_conc, UINT64_t k, int n_max=0, FLOAT32_t E_ns=0, int do_jacobi=False, int do_openen=False, int openen_ofs=0):
    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT64_t MAX_INDEX = 4**k - 1
    #cdef UINT64_t N = seq_matrix.base.shape[0]
    #if n_max:
        #N = min(N, n_max)

    #cdef UINT64_t L = seq_matrix.base.shape[1]
    #cdef UINT64_t l = L - k + 1

    #cdef int n_P = len(protein_conc)
    #cdef n_threads = 8

    ## store predicted binding probabilty for each sequence here
    #cdef FLOAT32_t [:,:] p_bound = np.zeros((n_P, N), dtype = np.float32)
    
    ## store predicted k-mer counts here, for each protein concentration
    #cdef FLOAT32_t [:,:,:] counts = np.zeros((n_threads, n_P, 4**k), dtype = np.float32)

    ## store predicted k-mer *Jacobi matrix* here, for each protein concentration
    #cdef FLOAT32_t [:,:,:,:] jacobi = np.zeros((n_threads, n_P, 4**k, 4**k), dtype = np.float32)

    ## store predicted openen counts here, for each protein concentration
    #cdef FLOAT32_t [:,:,:] openen_bin_counts = np.zeros((n_P, 4**k, 256), dtype = np.float32)
    
    ## indices and openens are thread-local buffers. As cython does not support 
    ## this keyword yet, we add an n_threads dimension
    
    ## record indices of all kmers occurring in one sequence
    #cdef UINT32_t [:,:] indices = np.zeros( (n_threads,l) , dtype=np.uint32)

    ## record openen bins of all kmers occurring in one sequence
    #cdef UINT8_t [:,:] openens = np.zeros( (n_threads, l), dtype=np.uint8)
    
    #cdef int thread_num
    
    ## helper variables to tell cython the types
    #cdef UINT8_t s, o
    #cdef UINT64_t index, i=0, j=0, m=0, n=0
    #cdef FLOAT64_t w=0, pb=0, g=0, jac = 0 # Boltzmann weight, Prob(seq is bound), gradient
    #cdef FLOAT64_t Z1, Z1_m   # Single protein partition functions

    
    #if n_max:
        #N = n_max

    #with nogil, parallel(num_threads=8):
        ##for j in prange(N):
        #for j in prange(N, schedule='guided'):
            #thread_num = openmp.omp_get_thread_num()

            ## prepare index from first k-1 positions
            #index = 0
            #for i in range(k-1):
                #index = index + seq_matrix[j, i] << 2 * (k - i - 2)

            ## zero out partition functions
            ##for i in xrange(n_P):
                ###Z1[i] = 0
                ##Z1_thread[i] = 0
            #Z1 = 0
    
            ## iterate over all k-mers, always adding next base to index
            #for i in range(0, l):
                ## get next "letter"
                #s = seq_matrix[j, i+k-1]
                
                ## compute next index from previous by shift + next letter
                #index = ((index << 2) | s ) & MAX_INDEX
                
                ## record index
                ##indices[i] =  index
                #indices[thread_num, i] = index
                
                ## binding energy = sequence dep. + unfolding energy (binned) + non-specific binding
                #o = openen_matrix[j, i+openen_ofs]
                #openens[thread_num, i] = o
                
                ##grad_thread[i] = acc_lookup[o]
                #w = kmer_invkd[index] * acc_lookup[o]
                
                #Z1 = Z1 + w
                ### Add Boltzmann weights
                ##for m in xrange(0, n_P):
                    ###Z1[m] += protein_conc[m] * w
                    ##Z1_thread[m] += protein_conc[m] * w
            
            ## update expected frequencies in pull-down
            #for m in range(0, n_P):
                #Z1_m = Z1 * protein_conc[m]
                #pb = Z1_m / (1. + Z1_m)
                ##pb = Z1_thread[m] / (1. + Z1_thread[m])
                
                
                ## for gradient/jacobi computation
                #g = protein_conc[m] * (pb - pb*pb) / Z1_m
                
                ## record each encountered kmer
                #for i in range(0, l):
                    ##counts[m, indices[i]] += pb
                    ##openen_bin_counts[m, indices[i], openens[i]] += pb

                    #counts[thread_num, m, indices[thread_num, i]] += pb
                    #if do_jacobi:
                        #jac = g * acc_lookup[openens[thread_num, i]]
                        #index = indices[thread_num, i]
                        #for n in range(0, l): 
                            ## i is \delta A_i, n is \delta \pi_n
                            #jacobi[thread_num, m, indices[thread_num, n], index] += jac

                    #if do_openen:
                        #openen_bin_counts[m, indices[thread_num, i], openens[thread_num, i]] += pb
            
                #p_bound[m,j] = pb
            
    #return p_bound.base, counts.base.sum(axis=0), openen_bin_counts.base, jacobi.base.sum(axis=0)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def seq_matrix_to_index_matrix(UINT8_t [:,:] seq_matrix, UINT64_t k, UINT8_t [:] adap5, UINT8_t [:] adap3):
    """
    Converts a matrix of nucleotide values (0..3 instead of ACGT) into
    a matrix of kmer index values. (AAA -> 0, AAC -> 1, ..., TTT -> 63).
    """
    #assert k <= 8 # for k > 8 indices do not fit into UINT16 anymore!

    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef UINT64_t l = L - k + 1
    cdef UINT64_t Lt = L + k - 1

    # store k-mer indices here
    cdef UINT32_t [:,:] indices = np.zeros( (N, Lt), dtype=np.uint32)

    # helper variables to tell cython the types
    cdef UINT64_t i, j
    cdef UINT32_t index, s, a5_index

    # largest kmer index 
    cdef UINT16_t MAX_INDEX = 4**k - 1

    a5_index = 0
    for i in range(k-1):
        s = adap5[i]
        a5_index = ((a5_index << 2) | s ) & MAX_INDEX
        
    with nogil, parallel(num_threads=8):
        for j in prange(N):
            index = 0 # make thread-local
            index = a5_index # first kmer overlaps (k-1) with 5'adapter
            for i in range(0, L):
                # get next "letter"
                s = seq_matrix[j, i]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                indices[j, i] = index
            
            for i in range(k-1): # last kmers read into 3'adapter
                s = adap3[i]
                index = ((index << 2) | s ) & MAX_INDEX
                indices[j, L+i] = index
                
    return indices.base

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def index_matrix_rows_with_kmer(UINT32_t [:,:] index_matrix, UINT64_t k, UINT32_t kmer_index, int n_threads=8):
    """
    searches for sequences that contain the desired kmer at least once.
    returns a vector with row-indices into the index-matrix with hits.
    Used to speed up updating of the thermodynamic model by restricting
    updates to the sequences that actually change their contribution.
    """
    #assert k <= 8 # for k > 8 indices do not fit into UINT16 anymore!

    cdef UINT64_t N = len(index_matrix)
    cdef UINT64_t l = len(index_matrix[0])

    # store row indices here.
    cdef UINT32_t [:,:] row_indices = np.zeros((n_threads, N), dtype=np.uint32)
    cdef UINT32_t [:] n_hits = np.zeros(n_threads, dtype=np.uint32)
    
    # helper variables to tell cython the types
    cdef UINT64_t i, j, t
    cdef UINT32_t index, s

    with nogil, parallel(num_threads=8):
        for j in prange(N):
            t = openmp.omp_get_thread_num()
            for i in range(0, l):
                if index_matrix[j,i] == kmer_index:
                    row_indices[t, n_hits[t]] = j
                    n_hits[t] += 1
                    break

    # merge results
    res = [row_indices.base[t,:n_hits[t]] for t in range(n_threads)]
    return np.concatenate(res)


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
#@cython.overflowcheck(False)
#def eval_energy_model_on_index_matrix(UINT32_t [:,:] index_matrix, UINT8_t [:,:] openen_matrix, FLOAT32_t [:] acc_lookup, FLOAT32_t [:] kmer_invkd, np.ndarray[FLOAT32_t] protein_conc, UINT64_t k, int n_max=0, FLOAT32_t E_ns=0, int do_jacobi=False, int do_openen=False):
    #assert k <= 8 # must fit into UINT16 kmer-indices!
    ## largest index in array of DNA/RNA k-mer counts
    #cdef UINT32_t MAX_INDEX = 4**k - 1
    #cdef UINT64_t N = index_matrix.base.shape[0]
    #if n_max:
        #N = min(N, n_max)

    #cdef UINT64_t l = index_matrix.base.shape[1]

    #cdef int n_P = len(protein_conc)
    #cdef n_threads = 8

    ## store predicted binding probabilty for each sequence here
    #cdef FLOAT32_t [:,:] p_bound = np.zeros((n_P, N), dtype = np.float32)
    
    ## store predicted k-mer counts here, for each protein concentration
    #cdef FLOAT32_t [:,:,:] counts = np.zeros((n_threads, n_P, 4**k), dtype = np.float32)

    ## store predicted k-mer *Jacobi matrix* here, for each protein concentration
    #cdef FLOAT32_t [:,:,:,:] jacobi = np.zeros((n_threads, n_P, 4**k, 4**k), dtype = np.float32)

    ## store predicted openen counts here, for each protein concentration
    #cdef FLOAT32_t [:,:,:] openen_bin_counts = np.zeros((n_P, 4**k, 256), dtype = np.float32)
    
    ## indices and openens are thread-local buffers. As cython does not support 
    ## this keyword yet, we add an n_threads dimension
    
    #cdef int thread_num
    
    ## helper variables to tell cython the types
    #cdef UINT8_t o
    #cdef UINT64_t i=0, j=0, m=0, n=0
    #cdef UINT32_t index
    #cdef FLOAT64_t w=0, pb=0, g=0, jac = 0 # Boltzmann weight, Prob(seq is bound), gradient
    #cdef FLOAT64_t Z1, Z1_m   # Single protein partition functions

    
    #if n_max:
        #N = n_max

    #with nogil, parallel(num_threads=8):
        ##for j in prange(N):
        #for j in prange(N, schedule='guided'):
            #thread_num = openmp.omp_get_thread_num()

            #Z1 = 0
            ## iterate over all k-mers, always adding next base to index
            #for i in range(0, l):
                #index = index_matrix[j,i]
                #o = openen_matrix[j, i]

                #w = kmer_invkd[index] * acc_lookup[o]
                #Z1 = Z1 + w
            
            ## update expected frequencies in pull-down
            #for m in range(0, n_P):
                #Z1_m = Z1 * protein_conc[m]
                #pb = Z1_m / (1. + Z1_m)
                ##pb = Z1_thread[m] / (1. + Z1_thread[m])
                
                
                ## for gradient/jacobi computation
                #g = protein_conc[m] * (pb - pb*pb) / Z1_m
                
                ## record each encountered kmer
                #for i in range(0, l):
                    ##counts[m, indices[i]] += pb
                    ##openen_bin_counts[m, indices[i], openens[i]] += pb

                    #index = index_matrix[j, i]
                    #o = openen_matrix[j, i]
                    #counts[thread_num, m, index] += pb
                    #if do_jacobi:
                        #jac = g * acc_lookup[o]
                        #for n in range(0, l): 
                            ## i is \delta A_i, n is \delta \pi_n
                            #jacobi[thread_num, m, index_matrix[j, n], index] += jac

                    #if do_openen:
                        #openen_bin_counts[m, index, o] += pb
            
                #p_bound[m,j] = pb
            
    #return p_bound.base, counts.base.sum(axis=0), openen_bin_counts.base, jacobi.base.sum(axis=0)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def SPA_partition_function(UINT32_t [:,:] index_matrix, UINT8_t [:,:] openen_matrix, FLOAT32_t [:] acc_lookup, FLOAT32_t [:] kmer_invkd, UINT64_t k, int n_max=0, int openen_ofs=0):
    #assert k <= 8 # must fit into UINT16 kmer-indices!

    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]
    
    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:] Z = np.empty(N, dtype=np.float32)
    
    # helper variables to tell cython the types
    cdef UINT8_t o=0
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index=0
    cdef FLOAT64_t w=0
    cdef FLOAT64_t Z1=0 # Single protein partition function

    if n_max:
        N = min(N, n_max)

    with nogil, parallel():
        for j in prange(N, schedule='guided'):
            Z1 = 0
            # iterate over all k-mers, always adding next base to index
            for i in range(0, l):
                # assigned variables are thread-local
                index = index_matrix[j, i]
                o = openen_matrix[j, i + openen_ofs]
                w = kmer_invkd[index] * acc_lookup[o]
                Z1 = Z1 + w
            
            Z[j] = Z1

    return Z.base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def SPA_partition_function_raw(UINT32_t [:,:] index_matrix, FLOAT32_t [:,:] acc_matrix, FLOAT32_t [:] kmer_invkd, UINT64_t k, int n_max=0, int openen_ofs=0):
    assert k <= 16 # must fit into UINT32 kmer-indices!

    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]
    
    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:] Z = np.empty(N, dtype=np.float32)
    
    # helper variables to tell cython the types
    cdef FLOAT32_t a=0
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index=0
    cdef FLOAT32_t w=0
    cdef FLOAT64_t Z1=0 # Single protein partition function

    if n_max:
        N = min(N, n_max)

    with nogil, parallel():
        for j in prange(N, schedule='guided'):
            Z1 = 0
            # iterate over all k-mers
            for i in range(0, l):
                # assigned variables are thread-local
                index = index_matrix[j, i]
                a = acc_matrix[j, i + openen_ofs]
                w = kmer_invkd[index] * a
                Z1 = Z1 + w
            
            Z[j] = Z1

    return Z.base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def SPA_bipartite_partition_function_raw(
    UINT32_t [:,:] index_matrix, 
    FLOAT32_t [:,:] acc_matrix, 
    FLOAT32_t [:] aff_A, 
    FLOAT32_t [:] aff_B, 
    FLOAT32_t [:] dist_cost, 
    UINT64_t k, 
    int n_max=0, 
    int openen_ofs=0
    ):

    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t L = index_matrix.base.shape[1]
    cdef UINT64_t d_max = len(dist_cost.base)

    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:] Z = np.empty(N, dtype=np.float32)
    
    # helper variables to tell cython the types
    cdef FLOAT32_t a=0
    cdef int i=0, j=0, m=0, d=0
    cdef UINT32_t index=0
    cdef FLOAT32_t w_A=0, w_B=0
    cdef FLOAT32_t [:] Z_A = np.zeros(L, dtype=np.float32) # Single protein partition function terms
    cdef FLOAT32_t [:] Z_B = np.zeros(L, dtype=np.float32) # Single protein partition function terms

    cdef FLOAT32_t Z1 = 0

    if n_max:
        N = min(N, n_max)

    with nogil, parallel():
        for j in prange(N, schedule='guided'):
            Z1 = 0 # make thread-local
            # iterate over all k-mers and fill in single motif partition functions
            for i in range(L):
                # assigned variables are thread-local
                index = index_matrix[j, i]
                a = acc_matrix[j, i + openen_ofs]
                w_A = aff_A[index] * a
                w_B = aff_B[index] * a
                Z_A[i] = w_A
                # Z_B[i] = w_B
                
                Z1 = Z1 + w_A + w_B # add single motif contributions

                # scan "backwards" to add bi-partite contributions
                # w_B fixed, w_A is read from already populated part of Z_A
                for m in range(i):
                    d = i - m
                    Z1 = Z1 + w_B * dist_cost[d] * Z_A[m]
            
            Z[j] = Z1

    return Z.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def xcorr_Z(FLOAT32_t [:,:] Z_A, FLOAT32_t [:,:] Z_B, UINT64_t k1, UINT64_t k2):
    cdef UINT64_t N = len(Z_A)
    assert N == len(Z_B)
    # assert k1 >= k2

    cdef UINT64_t L1 = len(Z_A[0])
    cdef UINT64_t L2 = len(Z_B[0])
    
    cdef UINT64_t L_max = max(L1, L2)

    # result will be stored here (Z = 'Zustandssumme' sum of states)
    cdef FLOAT32_t [:] Z_corr = np.zeros(2*L_max, dtype=np.float32)
    cdef FLOAT32_t [:] n_corr = np.zeros(2*L_max, dtype=np.float32)

    # helper variables to tell cython the types
    cdef FLOAT32_t a=0
    cdef int i=0, j=0, n=0, ofs = k1 - k2, d =0, shift = 0
    cdef UINT64_t L1_max, L2_max, L2_start

    # if k1 <= k2:
    #     ofs = k2 - k1
    #     L1_max = L1 - k2
    #     L2_start = k1 - ofs # first k2 mer that does not overlap first k1 mer
    #     L2_max = L2
    
    # else:
    #     ofs = k1 - k2
    #     L1_max = L2 + ofs - k1
    #     L2_start = 0
    #     L2_max = L2

    cdef UINT32_t index=0
    cdef FLOAT32_t w=0
    cdef FLOAT64_t Z1=0 # Single protein partition function

    # with nogil, parallel():
        # for j in prange(N, schedule='guided'):
    # with nogil:
    for n in range(N):
        # iterate over all k-mers
        for i in range(L1):
            for j in range(L2):
                d = j-i+ofs # separation between the two mers
                # print k1, k2, ofs, "i,j", i,j, "d",d
                Z_corr[d+L_max] += Z_A[n,i] * Z_B[n,j]
                n_corr[d+L_max] += 1
            

    # print L_max
    # return (Z_corr.base / n_corr.base)[L_max+1:]
    return Z_corr.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_counts_acc_weighted(UINT32_t [:,:] index_matrix, FLOAT32_t [:,:] acc_matrix, UINT64_t k, int n_max=0, int openen_ofs=0, int num_threads=8):
    assert k <= 16 # must fit into UINT32 kmer-indices!

    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]
    
    # result will be stored here
    cdef FLOAT32_t [:] weights = np.zeros(4**k, dtype=np.float32)

    # helper variables to tell cython the types
    cdef int thread_num
    cdef FLOAT32_t a=0
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index=0

    if n_max:
        N = min(N, n_max)

    with nogil, parallel():
        for j in prange(N, schedule='guided'):
            thread_num = openmp.omp_get_thread_num()
            # iterate over all k-mers
            for i in range(0, l):
                # assigned variables are thread-local
                index = index_matrix[j, i]
                a = acc_matrix[j, i + openen_ofs]
                weights[index] += a
            
    return weights.base

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def p_bound(FLOAT32_t [:] Z1, FLOAT32_t [:] rbp_conc_vector, int n_threads = 8):
    cdef UINT64_t N = Z1.base.shape[0]
    cdef UINT64_t n_conc = rbp_conc_vector.base.shape[0]

    # store weighted k-mer counts here (for each thread)
    cdef FLOAT32_t [:,:] p_bound = np.empty((n_conc, N), dtype = np.float32)

    # helper variables to tell cython the types
    cdef FLOAT32_t conc=0, Z=0
    cdef UINT64_t i=-1,j=-1

    with nogil, parallel(num_threads=8):
        for i in range(n_conc):
            conc = rbp_conc_vector[i]
            for j in prange(N, schedule='guided'):
                Z = Z1[j] * conc
                p_bound[i,j] = Z / (Z + 1.)
            
    return p_bound.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def index_matrix_kmer_counts(UINT32_t [:,:] index_matrix, UINT64_t k, int n_threads = 8):
    assert k <= 16 # must fit into UINT32 kmer-indices!
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]

    # store weighted k-mer counts here (for each thread)
    cdef UINT32_t [:,:] counts = np.zeros((n_threads, 4**k), dtype = np.uint32)

    # helper variables to tell cython the types
    cdef int thread_num
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index

    with nogil, parallel(num_threads=8):
        for j in prange(N, schedule='guided'):
            thread_num = openmp.omp_get_thread_num()

            for i in range(0, l):
                index = index_matrix[j, i]
                counts[thread_num, index] += 1
            
    return counts.base.sum(axis=0)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def weighted_kmer_counts(UINT32_t [:,:] index_matrix, FLOAT32_t [:] weights, UINT64_t k, int n_threads = 8):
    assert k <= 16 # must fit into UINT32 kmer-indices!
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    cdef UINT64_t N = index_matrix.base.shape[0]
    cdef UINT64_t l = index_matrix.base.shape[1]

    # store weighted k-mer counts here (for each thread)
    cdef FLOAT32_t [:,:] counts = np.zeros((n_threads, 4**k), dtype = np.float32)

    # helper variables to tell cython the types
    cdef int thread_num
    cdef UINT64_t i=0, j=0
    cdef UINT32_t index
    cdef FLOAT32_t w=0

    with nogil, parallel(num_threads=8):
        for j in prange(N, schedule='guided'):
            thread_num = openmp.omp_get_thread_num()

            w = weights[j]
            for i in range(0, l):
                index = index_matrix[j, i]
                counts[thread_num, index] += w
            
    return counts.base.sum(axis=0)





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_openen_counts(UINT8_t [:,:] seq_matrix, UINT8_t [:,:] openen_matrix, UINT64_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1
    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef UINT64_t l = L - k + 1

    # store joint frequencies here
    cdef UINT32_t [:,:] counts = np.zeros((4**k, 256), dtype = np.uint32)
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j
    
    with nogil:
        for j in range(N):
            
            # prepare index from first k-1 positions
            index = 0
            for i in range(k-1):
                index += seq_matrix[j, i] << 2 * (k - i - 2)

            # iterate over all k-mers, always adding next base to index
            for i in range(0, l):
                # get next "letter"
                s = seq_matrix[j, i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                counts[index, openen_matrix[j,i]] += 1
            
    return counts.base




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_mean_openen_profiles(UINT8_t [:,:] seq_matrix, UINT8_t [:,:] openen_matrix, FLOAT32_t [:] openen_lookup, int k_seq, int k_openen, int ofs):
    """
    Compute the mean open-energy levels relative to the 
    position of the kmer, for all kmers.
    
    returns a (4^k_seq, 2*(L-k_openen)+1) shaped array with mean 
    open-energies. If a kmer occurs multiple times in a read
    the open-energies will be counted multiple times 
    (into different position).
    """
    
    print k_seq, k_openen
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX_SEQ = 4**k_seq - 1
    cdef UINT64_t MAX_INDEX_OE = 4**k_openen - 1
    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef int l_seq = L - k_seq + 1
    cdef int l_openen = L - k_openen + 1

    # store observations here to compute means upon exit
    cdef UINT32_t [:,:,:] counts = np.zeros((4**k_seq, 2*l_openen+1, 256), dtype = np.uint32)
    #cdef FLOAT32_t [:,:] sums = np.zeros((4**k_seq, 2*l_openen+1), dtype = np.float32)
    #cdef FLOAT32_t [:] openens = np.zeros(l_openen, dtype=np.float32)
    cdef UINT8_t [:] openens = np.zeros(l_openen, dtype=np.uint8)
    cdef UINT64_t [:] indices = np.zeros(l_seq, dtype=np.uint64)
    
    # helper variables to tell cython the tqypes
    cdef UINT8_t s
    cdef int index_seq, index_oe, index, i, j, pos, x, m
    
    #with gil:
    for j in range(N):
        # iterate over all k-mers, always adding next base to index
        index = 0
        for i in range(L):
            s = seq_matrix[j, i]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s )
            
            if i >= k_seq-1:
                index_seq = index & MAX_INDEX_SEQ
                indices[i-k_seq+1] = index_seq
                
            if i >= k_openen-1:
                index_oe = index & MAX_INDEX_OE
                #openens[i-k_openen+1] = openen_lookup[openen_matrix[j,i-k_openen+1]]
                openens[i-k_openen+1] = openen_matrix[j,i-k_openen+1+ofs]

        for i in range(0,l_seq):
            index = indices[i]
            #print i, index, range(-i, l-i)
            for m in range(0, l_openen):
                x = m - i
                pos = l_openen + x
                #print x, pos
                #if index == 0b1001001110:
                    #print "i={0}, x={1}, pos={2}, openens[x+i] = {3}, sums[index, pos] = {4}, counts[index,pos]={5}".format(i, x, pos, openens[x+i], sums[index, pos], counts[index, pos])

                #sums[index, pos] += openens[m]
                counts[index, pos, openens[m]] += 1
            
    return counts.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_openen_profile(UINT32_t [:,:] index_matrix, UINT8_t [:,:] openen_matrix, int k_seq, UINT32_t kmer_index, int k_openen, int ofs):
    """
    Compute the mean open-energy levels relative to the 
    position of the kmer, for all kmers.
    
    returns a (4^k_seq, 2*(L-k_openen)+1) shaped array with mean 
    open-energies. If a kmer occurs multiple times in a read
    the open-energies will be counted multiple times 
    (into different position).
    """
    
    print k_seq, k_openen
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX_SEQ = 4**k_seq - 1
    cdef UINT64_t MAX_INDEX_OE = 4**k_openen - 1
    cdef UINT64_t N = len(index_matrix)
    cdef UINT64_t l_seq = len(index_matrix[0])
    cdef int l_openen = l_seq + k_seq - k_openen

    # store observations here to compute means upon exit
    cdef UINT32_t [:,:] counts = np.zeros((2*l_openen+1, 256), dtype = np.uint32)
    #cdef FLOAT32_t [:,:] sums = np.zeros((4**k_seq, 2*l_openen+1), dtype = np.float32)
    #cdef FLOAT32_t [:] openens = np.zeros(l_openen, dtype=np.float32)
    cdef UINT8_t [:] openens = np.zeros(l_openen, dtype=np.uint8)
    cdef UINT64_t [:] indices = np.zeros(l_seq, dtype=np.uint64)
    
    # helper variables to tell cython the tqypes
    cdef UINT8_t s
    cdef int index_seq, index_oe, index, i, j, pos, x, m
    
    #with gil:
    for j in range(N):
        # iterate over all k-mers, always adding next base to index
        #index = 0
        for i in range(l_seq):
            index = index_matrix[j, i]
            if index != kmer_index:
                continue
            
            for m in range(l_openen):
                pos = l_openen + m - i
                counts[pos, openen_matrix[j, m + ofs]] += 1

        #for i in range(0,l_seq):
            #index = indices[i]
            ##print i, index, range(-i, l-i)
            #for m in range(0, l_openen):
                #x = m - i
                #pos = l_openen + x
                ##print x, pos
                ##if index == 0b1001001110:
                    ##print "i={0}, x={1}, pos={2}, openens[x+i] = {3}, sums[index, pos] = {4}, counts[index,pos]={5}".format(i, x, pos, openens[x+i], sums[index, pos], counts[index, pos])

                ##sums[index, pos] += openens[m]
                #counts[index, pos, openens[m]] += 1
            
    return counts.base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def joint_kmer_profiles(UINT8_t [:,:] seq_matrix, int k_core, int k_flank, int pseudo=1):
    """
    Compute the co-occurrence frequency of k_flank mers relative to the 
    position of k_core mers, for all k_core mers.
    
    returns a (4^k_core, 2*(L-k_flank)+1) shaped array with frequencies.
    If a kmer occurs multiple times in a read it will be counted multiple 
    times (into different position).
    """
    
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX_CORE = 4**k_core - 1
    cdef UINT64_t MAX_INDEX_FLANK = 4**k_flank - 1
    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef int l_core = L - k_core + 1
    cdef int l_flank = L - k_flank + 1
    cdef int l_min = min(l_core, l_flank)
    cdef int l_max = max(l_core, l_flank)

    # store observations here to compute means upon exit
    cdef FLOAT32_t [:,:,::1] freqs = np.ones((l_core + l_flank, 4**k_core, 4**k_flank), dtype = np.float32) * pseudo
    cdef UINT64_t [::1] indices_flank = np.zeros(l_flank, dtype=np.uint64)
    cdef UINT64_t [::1] indices_core = np.zeros(l_core, dtype=np.uint64)
    
    # helper variables to tell cython the tqypes
    cdef UINT8_t s
    cdef int index_core, index_flank, index, i, j, pos, x, m
    
    #with gil:
    for j in range(N):
        # iterate over all k-mers, always adding next base to index
        index = 0
        for i in range(L):
            s = seq_matrix[j, i]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s )
            
            if i >= k_core-1:
                index_core = index & MAX_INDEX_CORE
                indices_core[i-k_core+1] = index_core
                
            if i >= k_flank-1:
                index_flank = index & MAX_INDEX_FLANK
                indices_flank[i-k_flank+1] = index_flank

        for i in range(0,l_core):
            index_core = indices_core[i]
            #print i, index, range(-i, l-i)
            for m in range(0, l_flank):
                x = m - i
                pos = l_core + x
                #print x, pos
                #if index == 0b1001001110:
                    #print "i={0}, x={1}, pos={2}, openens[x+i] = {3}, sums[index, pos] = {4}, counts[index,pos]={5}".format(i, x, pos, openens[x+i], sums[index, pos], counts[index, pos])

                index_flank = indices_flank[m]
                #print i, m, x, index_core, pos, index_flank
                freqs[pos, index_core, index_flank] += 1
                #counts[index_core, pos] += 1
            
    return freqs.base #/ counts.base[:,:,np.newaxis]


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def seq_set_SKA(np.ndarray[UINT8_t, ndim=2] seq_matrix, np.ndarray[FLOAT32_t] _weights, np.ndarray[FLOAT32_t] _background, UINT32_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1

    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])

    # store k-mer indices here
    _mer_indices = np.zeros(L-k+1, dtype=np.uint32)
    
    # store current k-mer weights here
    _mer_weights = np.zeros(L-k+1, dtype=np.float32)
    
    
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT32_t [::1] mer_indices = _mer_indices
    cdef FLOAT32_t [::1] mer_weights = _mer_weights
    cdef FLOAT32_t [::1] weights = _weights
    cdef FLOAT32_t [::1] background = _background

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef int index, i, j, ofs
    cdef FLOAT32_t w=0, total_w=0, current_weights_sum = 0, weights_sum = MAX_INDEX+1, Z=0
    
    current_weights_sum = _weights.sum()
    Z = weights_sum / current_weights_sum
    
    #with nogil, parallel(num_threads=8):
        #for j in prange(N):
    with nogil:
        for j in range(N):
            ofs = j*L

            # compute index of first k-1 mer by bit-shifts
            index = 0
            for i in range(k-1):
                index += _seq_matrix[ofs+i] << 2 * (k - i - 2)
            
            total_w = 0
            
            # iterate over k-mers
            for i in range(0, L-k+1):
                # get next "letter"
                s = _seq_matrix[ofs+i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                mer_indices[i] = index
                w = weights[index] / background[index] * Z
                mer_weights[i] = w
                total_w += w

            # update weights
            for i in range(0, L-k+1):
                weights[mer_indices[i]] += mer_weights[i]/total_w

            current_weights_sum += 1
            Z = weights_sum / current_weights_sum

    # normalize such that all weights sum up to 4**k
    _weights *= Z
    return _weights


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def count_best_ranked_hits(np.ndarray[UINT8_t, ndim=2] seq_matrix, np.ndarray[UINT32_t, ndim=1] _order):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t k = np.log2(len(_order))/2
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])
    cdef UINT32_t l = L-k+1

    # count reads covered by kmer with best rank
    cdef np.ndarray[UINT32_t, ndim=1] _hit_counts = np.zeros(len(_order) ,dtype=np.uint32)
    
    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT32_t [::1] order = _order
    cdef UINT32_t [::1] hit_counts = _hit_counts
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t best_order = MAX_INDEX
    cdef UINT64_t ofs, index, i, j, n_hits=0, best_index=0
    
    with nogil:
        for j in range(N):
            best_order=MAX_INDEX
            ofs = j*L

            # compute index of first k-1 mer by bit-shifts
            index = 0
            for i in range(k-1):
                index += _seq_matrix[ofs+i] << 2 * (k - i - 2)

            # iterate over remaining k-mers
            for i in range(0, l):
                # get next "letter"
                s = _seq_matrix[ofs+i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                
                # assign hit to kmer with best rank
                if order[index] < best_order:
                    best_index = index
                    best_order = order[index]

            hit_counts[best_index] += 1

    return _hit_counts



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def count_pure_hits(np.ndarray[UINT8_t, ndim=2] seq_matrix, np.ndarray[UINT32_t, ndim=1] _candidates, out_file=None, UINT32_t n_sample=100000):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t k = np.log2(len(_candidates))/2
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])
    cdef UINT32_t l = L-k+1

    # count reads with only one and no other kmer out of the candidates
    cdef np.ndarray[UINT32_t, ndim=1] _hit_counts = np.zeros(len(_candidates) ,dtype=np.uint32)
    cdef np.ndarray[UINT32_t] _n = np.zeros(4**k, dtype=np.uint32)
    cdef np.ndarray[UINT8_t] _seq = np.zeros(L, dtype=np.uint8)
    cdef np.ndarray[UINT8_t] _kmer = np.zeros(k, dtype=np.uint8)
    
        
    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT32_t [::1] candidates = _candidates
    cdef UINT32_t [::1] hit_counts = _hit_counts
    
    cdef UINT8_t [::1] seq = _seq
    cdef UINT8_t [::1] kmer = _kmer
    cdef UINT32_t [::1] n = _n
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t hit_id = 0
    cdef UINT64_t ofs, index, i, j, n_hits=0, best_index=0, best_i = 0, ov=0, best=MAX_INDEX, do_write=0
    
    if out_file:
        # avoid GIL issues if we want to write to this file
        do_write = 1
    
    with nogil:
        for j in range(N):
            hit_id = 0
            n_hits = 0
            ofs = j*L

            # compute index of first k-1 mer by bit-shifts
            index = 0
            for i in range(k-1):
                index += _seq_matrix[ofs+i] << 2 * (k - i - 2)

            # iterate over remaining k-mers
            for i in range(0, l):
                # get next "letter"
                s = _seq_matrix[ofs+i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                
                # assign hit to kmer with best rank
                if candidates[index] > 0:
                    if ov == 0:
                        # non-overlapping hit. Always counts!
                        n_hits += 1

                    if candidates[index] < best:
                        # attribute read to lowest-ranked kmer
                        best = candidates[index]
                        best_index = index
                        best_i = i
                    
                    ov = 7 # re-start overlap count-down

                if ov > 0:
                    ov -= 1
            
            if n_hits != 1:
                # do not count ambiguous or no hit
                continue

            hit_counts[best_index] += 1
            if do_write and n[best_index] < n_sample:
                with gil:
                    n[best_index] += 1

                    # convert seq entries back to string
                    for i in range(L):
                        seq[i] = bits_to_letters[ _seq_matrix[ofs+i] ]

                    # convert hit index back to kmer
                    for i in range(k):
                        s = best_index >> ((k - i-1) * 2)
                        kmer[i] = bits_to_letters[s & 3]
                    
                    out_file.write(">{0} | p={1} | r={2} | n={3}\n{4}\n".format(_kmer.tobytes(), best_i, best, n[best_index], _seq.tobytes()) )
                        

    return _hit_counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def count_reads_with_hits(np.ndarray[UINT8_t, ndim=2] seq_matrix, np.ndarray[UINT32_t, ndim=1] _candidates, out_file=None, UINT32_t n_sample=100000, str adap5='', str adap3=''):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t k = np.log2(len(_candidates))/2
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])
    cdef UINT32_t l = L-k+1
    cdef UINT64_t l_adap5 = len(adap5)

    # count reads with only one and no other kmer out of the candidates
    cdef np.ndarray[UINT32_t, ndim=1] _kmer_counts = np.zeros(len(_candidates) ,dtype=np.uint32)
    cdef np.ndarray[UINT32_t] _n = np.zeros(4**k, dtype=np.uint32)
    cdef np.ndarray[UINT8_t] _seq = np.zeros(L, dtype=np.uint8)
    cdef np.ndarray[UINT8_t] _kmer = np.zeros(l*(k+2), dtype=np.uint8) + ord(',')
    cdef np.ndarray[UINT64_t] _hit_indices = np.zeros(l, dtype=np.uint64)
    cdef np.ndarray[UINT64_t] _nonhit_indices = np.zeros(l, dtype=np.uint64)
    cdef np.ndarray[UINT64_t] _hit_pos = np.zeros(l, dtype=np.uint64)
    cdef np.ndarray[UINT64_t] _nonhit_pos = np.zeros(l, dtype=np.uint64)
        
    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT32_t [::1] candidates = _candidates
    cdef UINT32_t [::1] kmer_counts = _kmer_counts
    
    cdef UINT8_t [::1] seq = _seq
    cdef UINT8_t [::1] kmer = _kmer
    cdef UINT32_t [::1] n = _n
    cdef UINT64_t [::1] hit_indices = _hit_indices
    cdef UINT64_t [::1] nonhit_indices = _nonhit_indices
    cdef UINT64_t [::1] hit_pos = _hit_pos
    cdef UINT64_t [::1] nonhit_pos = _nonhit_pos
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t hit_id = 0
    cdef UINT64_t ofs, index, i, j, n_hits=0, n_nonhits=0, o=0, dont_need=0, do_write=0
    
    if out_file:
        # avoid GIL issues if we want to write to this file
        do_write = 1
    
    with nogil:
        for j in range(N):
            hit_id = 0
            n_hits = 0
            n_nonhits = 0
            dont_need = 0
            ofs = j*L

            # compute index of first k-1 mer by bit-shifts
            index = 0
            for i in range(k-1):
                index += _seq_matrix[ofs+i] << 2 * (k - i - 2)

            # iterate over remaining k-mers
            for i in range(0, l):
                # get next "letter"
                s = _seq_matrix[ofs+i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                
                # track hits
                if candidates[index] > 0:
                    hit_indices[n_hits] = index
                    hit_pos[n_hits] = i
                    n_hits += 1
                    if n[index] >= n_sample:
                        dont_need += 1
                
                # track non-hits
                else:
                    nonhit_indices[n_nonhits] = index
                    nonhit_pos[n_nonhits] = i
                    n_nonhits += 1
                    
            if not n_hits:
                # no candidate hits! count the non-hits 
                for i in range(n_nonhits):
                    index = nonhit_indices[i]
                    kmer_counts[index] += 1
            else:
                # one or more candidate hits. count only those
                for i in range(n_hits):
                    index = hit_indices[i]
                    kmer_counts[index] += 1
                    n[index] += 1
                    
            if do_write and dont_need < n_hits:

                # convert seq entries back to string
                for i in range(L):
                    seq[i] = bits_to_letters[ _seq_matrix[ofs+i] ]
                
                for o in range(n_hits):
                    index = hit_indices[o]

                    # convert hit index back to kmer
                    for i in range(k):
                        s = index >> ((k - i-1) * 2)
                        kmer[i+(k+1)*o] = bits_to_letters[s & 3]
                        
                with gil:
                    pos_str = ",".join( [str(p + l_adap5) for p in hit_pos[:n_hits]] )
                    out_file.write(">{0} | p={1}\n{2}{3}{4}\n".format(_kmer[:n_hits*(k+1)-1].tobytes(), pos_str, adap5, _seq.tobytes(), adap3))
                    

    return _kmer_counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def store_pure_reads(
        out_file,
        UINT64_t k,
        np.ndarray[UINT8_t, ndim=2] seq_matrix, 
        np.ndarray[INT32_t, ndim=1] _flags, # one entry for each read. 
        np.ndarray[UINT32_t, ndim=1] _indices, # one entry for each read.
        UINT32_t n_sample,
    ):
    
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])
    cdef UINT32_t l = L-k+1

    cdef np.ndarray[UINT32_t] _n = np.zeros(4**k, dtype=np.uint32)
    cdef np.ndarray[UINT8_t] _seq = np.zeros(L, dtype=np.uint8)
    cdef np.ndarray[UINT8_t] _kmer = np.zeros(k, dtype=np.uint8)
    
    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq = _seq
    cdef UINT8_t [::1] kmer = _kmer
    cdef INT32_t [::1] flags = _flags
    cdef UINT32_t [::1] indices = _indices
    cdef UINT32_t [::1] n = _n
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t hit_id = 0
    cdef INT32_t flag = 0
    cdef UINT64_t ofs, index, i, j, n_hits=0, hit_index=0
    
    for j in range(N):
        
        flag = flags[j]
        if flag < 1:
            # not pure
            continue

        hit_index = indices[j]
        if n[hit_index] >= n_sample:
            # already enough of these
            continue

        n[hit_index] += 1
        ofs = j*L

        # convert seq entries back to string
        for i in range(L):
            seq[i] = bits_to_letters[ _seq_matrix[ofs+i] ]

        # convert hit index back to kmer
        for i in range(k):
            s = hit_index >> ((k - i-1) * 2)
            kmer[i] = bits_to_letters[s & 3]
        
        out_file.write(">{0} | r={1} | n={2}\n{3}\n".format(_kmer.tobytes(), flag, n[hit_index], _seq.tobytes()) )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def count_reads_with_kmers(UINT32_t [:,:] index_matrix, UINT64_t k):
    cdef UINT32_t N = len(index_matrix)
    cdef UINT32_t L = len(index_matrix[0])

    # count reads containing a given kmer, for all kmers
    cdef UINT32_t [:] hit_counts = np.zeros(4**k ,dtype=np.uint32)
    # keep distinct kmer indices from each read here
    cdef UINT32_t [:] dindices = np.zeros(L ,dtype=np.uint32)
    
    # helper variables to tell cython the types
    cdef UINT64_t index, i, j, m, n_distinct=0, append=1
    
    with nogil:
        for j in range(N):
            n_distinct = 0
            # iterate over all k-mers in the read
            for i in range(L):
                index = index_matrix[j,i]

                # make sure we do not have this index already
                append = 1
                for m in range(n_distinct):
                    if index == dindices[m]:
                        append = 0
                
                # it's a new index
                if append:
                    dindices[n_distinct] = index
                    n_distinct += 1
                    # record the read for each of the contained kmers *once*
                    hit_counts[index] += 1
            

    return hit_counts.base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def count_reads_with_kmap_hit(UINT32_t [:,:] index_matrix, UINT8_t [:] kmap):
    cdef UINT32_t N = len(index_matrix)
    cdef UINT32_t L = len(index_matrix[0])

    # helper variables to tell cython the types
    cdef UINT64_t index, i, j, hit=0, n_reads=0
    
    with nogil:
        for j in range(N):
            hit = 0
            # iterate over all k-mers in the read
            for i in range(L):
                index = index_matrix[j,i]
                hit += kmap[index]

            n_reads += (hit > 0)

    return n_reads



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def joint_freq_at_distance(UINT32_t [:,:] index_matrix, UINT64_t k):
    cdef UINT32_t N = len(index_matrix)
    cdef UINT32_t L = len(index_matrix[0])
    cdef UINT32_t Nk = 4**k

    cdef UINT32_t [:,:,:] joint = np.zeros((Nk, Nk, L-k) ,dtype=np.uint32)

    # helper variables to tell cython the types
    cdef UINT64_t index_A, index_B, i, j, d, hit=0, n_reads=0
    
    
    with nogil:
        for j in range(N):
            hit = 0
            # iterate over all k-mers in the read
            for i in range(L-k):
                index_A = index_matrix[j,i]
                for d in range(k, L-i):
                    index_B = index_matrix[j,i+d]
                    joint[index_A, index_B, d-k] += 1

    return joint.base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_count_pos_per_read(UINT8_t [:,:] seq_matrix, UINT64_t kmer_index, UINT8_t k):
    """
    Examine each read for occurrences of the indicated kmer. Record the number of
    occurrences, as well as the the index of the last occurrence, for each read.
    """
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(seq_matrix.base)
    cdef UINT32_t L = len(seq_matrix.base[0])
    cdef UINT32_t l = L-k+1

    # count occurrences of the kmer, for each read
    cdef UINT8_t [:] counts = np.zeros(N, dtype=np.uint8)
    # keep position of first hit
    cdef UINT8_t [:] last_pos = np.zeros(N, dtype=np.uint8)
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, i, j, m
    
    with nogil:
        for j in range(N):
            # compute index of first k-1 mer by bit-shiftkmer_profile(self.seqm, k)s
            index = 0
            for i in range(k-1):
                index += seq_matrix[j,i] << 2 * (k - i - 2)

            # iterate over remaining k-mers
            for i in range(0, l):
                # get next "letter"
                s = seq_matrix[j,i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                
                if index == kmer_index:
                    counts[j] += 1
                    last_pos[j] = i

    return counts.base, last_pos.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def digitize_32fp_8bit(np.ndarray[FLOAT32_t, ndim=2] data, FLOAT32_t [:] bins):
    cdef UINT64_t n = len(bins.base)
    assert n <= 257
    cdef UINT64_t N = len(data)
    cdef UINT64_t L = len(data[0])
    cdef FLOAT32_t [:] flat = data.flatten()
    cdef UINT8_t [:] res = np.zeros(N*L, dtype=np.uint8)

    cdef UINT64_t i,j,m, pivot
    cdef FLOAT32_t x
    #res = np.zeros(data.size, dtype=int)
    with nogil, parallel(num_threads=8):
        for m in prange(N*L):
            x = flat[m]
            i = 0
            j = n-1
            #print "value", x
            #c = 0
            while j - i > 1:
                pivot = max(1, (j - i)/2 ) + i
                #print pivot, bins[pivot]
                #print "indices",i,j, pivot
                #print "values",'?', bins[i], bins[j], bins[pivot]
                
                if x >= bins[pivot]:
                    i = pivot
                else:
                    j = pivot
                #c += 1
                
            if x >= bins[j]:
                res[m] = j
            else:
                res[m] = i

            #print x,"->", res[n], "in {0} steps".format(c)
            #steps.append(c)
            
    #print np.array(steps).mean(), "average steps"
    return np.reshape(res.base, (N,L)) + 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def aggregate_binned_profiles(UINT8_t [:,:] bin_matrix, UINT8_t [:] pos, UINT8_t upstream, UINT8_t downstream):
    """
    Examine each read for occurrences of the indicated kmer. Record the number of
    occurrences, as well as the the index of the last occurrence, for each read.
    """
    
    cdef UINT64_t N = len(bin_matrix.base)
    cdef UINT64_t L = len(bin_matrix.base[0])

    cdef UINT64_t l = downstream + upstream + 1
    cdef UINT64_t rightmost = L - downstream - 1
    cdef UINT64_t leftmost = upstream

    ## count bin occupancies for downstream and upstream positions relative to the hit pos
    cdef UINT32_t [:,:] profile = np.zeros((l,2**8), dtype=np.uint32)
    
    ## keep position of first hit
    #cdef UINT8_t [:] last_pos = np.zeros(N, dtype=np.uint8)
    
    ## helper variables to tell cython the types
    cdef UINT8_t o
    cdef UINT64_t i, j, m
    
    with nogil:
        for j in range(N):
            i = pos[j]
            if i < leftmost or i > rightmost:
                # does not fit
                continue

            for m in range(l):
                # fetch the bin-value of the position rel to hit
                o = bin_matrix[j, i - leftmost + m] 
                # and record
                profile[m,o] += 1
                
    return profile.base



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_profiles(np.ndarray[UINT8_t, ndim=2] _seq_matrix, UINT64_t k):
    """
    count the occurrences of each kmer at each position from 0-L-k+1 
    across all reads.
    input: 
      _seq_matrix NxL UINT8_t array of all reads
      k : kmer size
    
    returns:
      4^k x (L-k+1) array of kmer counts along read positions (profiles)
    
    """
    #print "kmer_profiles"
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1

    cdef UINT32_t N = len(_seq_matrix)
    cdef UINT32_t L = len(_seq_matrix[0])
    cdef UINT32_t l = L-k+1 # maximum spacing

    # count reads containing a given kmer, for all kmers
    cdef np.ndarray[UINT32_t, ndim=2] _profiles = np.zeros( (4**k, l) ,dtype=np.uint32)

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    
    cdef UINT8_t [::1] seqm = _seq_matrix.flatten()
    cdef UINT32_t [:,:] profiles = _profiles
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef int ofs, index, i, j
    
    #with gil:
    for j in range(N):
        ofs = j*L
        
        # compute index of first k-1 mer by bit-shifts
        index = 0
        for i in range(k-1):
            index += seqm[ofs+i] << 2 * (k - i - 2)

        # iterate over k-mers
        for i in range(0, l):
            # get next "letter"
            s = seqm[ofs+i+k-1]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            # count occurrence
            profiles[index][i] += 1
            #print ">", index, index*l + 1, profiles[index*l + i]

    #print "outof",_profiles.sum()
    return _profiles

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_cooccurrence_distance_tensor(np.ndarray[UINT8_t, ndim=2] _seq_matrix, np.ndarray[UINT64_t, ndim=1] _kmer_lookup, UINT64_t k, UINT64_t n_kmers):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    
    cdef UINT32_t N = len(_seq_matrix)
    cdef UINT32_t L = len(_seq_matrix[0])
    cdef UINT32_t l = L-k+1 # maximum spacing

    # count reads containing a given kmer, for all kmers
    cdef np.ndarray[UINT32_t, ndim=3] _tensor = np.zeros( (n_kmers, n_kmers, l) ,dtype=np.uint32)

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    
    cdef UINT8_t [::1] seq_matrix = _seq_matrix.flatten()
    cdef UINT32_t [:,:,:] tensor = _tensor
    cdef UINT64_t [::1] kmer_lookup = _kmer_lookup.flatten()
    
    # running variables
    cdef UINT64_t [::1] encountered_vec = np.zeros(l, dtype=np.uint64)
    cdef UINT64_t [::1] spacing_vec = np.zeros(l, dtype=np.uint64)
    cdef UINT64_t n_tracing = 0
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef int ofs, index, i, j, m, kmer_i=0, kmer_j=0, spacing=0, kmer_hit=0, n2=n_kmers*n_kmers
    
    #with gil:
    for j in range(N):
        ofs = j*L
        
        # compute index of first k-1 mer by bit-shifts
        index = 0
        for i in range(k-1):
            index += seq_matrix[ofs+i] << 2 * (k - i - 2)

        kmer_i = 0
        kmer_j = 0
        
        n_tracing = 0
        # iterate over remaining k-mers
        for i in range(0, l):
            # get next "letter"
            s = seq_matrix[ofs+i+k-1]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            
            kmer_hit = kmer_lookup[index]
            
            if kmer_hit:
                for m in range(n_tracing):
                    # record occurrence relative to previously encountered kmers
                    kmer_i = encountered_vec[m]
                    kmer_j = kmer_hit
                    spacing = spacing_vec[m]
                    tensor[kmer_i-1, kmer_j-1, spacing] += 1
                    
                # record as new hit
                encountered_vec[n_tracing] = kmer_hit
                spacing_vec[n_tracing] = 0
                n_tracing += 1
            
            # and update all spacings
            for m in range(n_tracing):
                spacing_vec[m] += 1
                
                           
    return _tensor
      
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_flank_profiles(np.ndarray[UINT8_t, ndim=2] seq_matrix, str kmer, int k_flank=3):
    cdef UINT64_t k = len(kmer)
    
    # number of bits to shift right to convert k-mer index to k_flank-mer index
    cdef UINT64_t k_diff_bits = (k-k_flank)*2 

    # largest index in array of DNA/RNA k-mer counts
    cdef UINT64_t MAX_INDEX = 4**k - 1
    # the index we are looking for
    cdef UINT64_t k_index = seq_to_index(kmer)
    
    cdef UINT64_t N = len(seq_matrix)
    cdef UINT64_t L = len(seq_matrix[0])
    cdef UINT64_t l = L-k+1
    
    # aggregate flanking kmer counts - at each relative position - here
    _profile = np.zeros( ((4**k_flank) * 2*l), dtype=np.uint32)

    # store k-mer hits here
    _mask = np.zeros(N * l, dtype = np.uint8)

    # remember the hit positions
    _hit_pos = np.zeros(l, dtype = np.uint64)
    
    # buffer
    _indices = np.zeros(l, dtype = np.uint64)
    
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT8_t [::1] mask = _mask
    cdef UINT32_t [::1] profile = _profile
    cdef UINT64_t [::1] indices = _indices
    cdef UINT64_t [::1] hit_pos = _hit_pos

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT64_t index, findex, i, j, r, o, q
    cdef UINT64_t hits = 0

    with nogil:#, parallel(num_threads=8):
        for j in range(N):
            seq_bits = _seq_matrix[j*L:(j+1)*L]
            # compute index of first k-1-mer by bit-shifts
            index = kbits_to_index(seq_bits, k-1) 
            # iterate over remaining k-mers
            hits = 0
            #print list(_seq_matrix[j*L:(j+1)*L])
            for i in range(0, l):
                # get next "letter"
                s = seq_bits[i+k-1]
                # compute next index from previous by shift + next letter
                index = ((index << 2) | s ) & MAX_INDEX
                indices[i] = index
                #print i, index, index_to_seq(index, k), _indices[i]
                if index == k_index:
                    mask[j*l+i] = 1
                    hit_pos[hits] = i
                    hits += 1

            #print "pos 0 index", _indices[0]
            for q in range(0, hits):
                o = hit_pos[q]
                #print "hit at",o
                for i in range(0,l):
                    # compute shorter kmer index from longer ones
                    index = indices[i]
                    findex = index >> k_diff_bits
                    #print index, index_to_seq(index, k), index_to_seq(findex, k_flank), k_flank
                    r = findex*2*l + i-o+l-1
                    profile[r] += 1

    return _profile.reshape( (4**k_flank, 2*l) ), _mask.reshape( (N, l) )


