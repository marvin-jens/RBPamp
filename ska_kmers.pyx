__license__ = "MIT"
__version__ = "0.9"
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

from libc.math cimport exp, log


ctypedef np.uint8_t UINT8_t
ctypedef np.uint32_t UINT32_t
ctypedef np.uint64_t UINT64_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

# maps ASCII values of A,C,G,T to correct bits
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

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def read_raw_seqs(src, str pre="", str post="", UINT32_t n_max=0, UINT32_t n_skip=0):
    cdef char* l
    cdef UINT32_t i, N=0, L
    cdef list seqs = list()
    cdef np.ndarray[UINT8_t] _buf
    cdef UINT8_t [::1] buf
    
    for line in src:
        N += 1
        if n_skip and N <= n_skip:
            continue

        line = line.rstrip() # remove trailing new-line characters
        if pre or post:
            line = pre + line + post

        L = len(line)
        _buf = np.empty(L, dtype=np.uint8)
        buf = _buf # initialize the view
        
        l = line # extract raw string content
        for i in range(0,L):
            buf[i] = letter_to_bits[l[i]]
        
        seqs.append(_buf)
        if N >= n_max + n_skip and n_max:
            break

    return np.array(seqs)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def read_raw_seqs_chunked(src, str pre="", str post="", UINT32_t n_max=0, UINT32_t n_skip=0, chunklines=1000000):
    cdef char* l
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

    #for c in chunks:
        #print c.shape
        
    #print "concatenation", cat.shape
    #print "want", N,L, N*L
    return cat.reshape((N,L))

        
@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def read_fastq(src, str pre="", str post="", UINT32_t n_max=0, UINT32_t n_skip=0):
    cdef char* l
    cdef UINT32_t i, N, L, line_num = -1
    cdef list seqs = list()
    cdef np.ndarray[UINT8_t] _buf
    cdef UINT8_t [::1] buf
    
    N = 0
    for line in src:
        line_num += 1
        if line_num % 4 != 1:
            continue
        
        N += 1
        if n_skip and N <= n_skip:
            continue
        
        line = pre + line.rstrip() + post
        L = len(line)
        _buf = np.empty(L, dtype=np.uint8)
        buf = _buf # initialize the view
        
        l = line # extract raw string content
        for i in range(0,L):
            buf[i] = letter_to_bits[l[i]]
        
        seqs.append(_buf)
        if N >= n_max + n_skip and n_max:
            break
        
    return np.array(seqs)
            

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
cdef inline UINT32_t kbits_to_index(UINT8_t [:] kbits, UINT32_t k):
    cdef UINT32_t i, index = 0
    
    for i in range(k):
        index += kbits[i] << 2 * (k - i - 1)
    
    return index


def seq_to_index(seq):
    k = len(seq)
    bits = seq_to_bits(seq)
    return kbits_to_index(bits, k)

def index_to_seq(index, k):
    nucs = ['a','c','g','t']
    seq = []
    for i in range(k):
        j = index >> ((k-i-1) * 2)
        seq.append(nucs[j & 3])

    return "".join(seq)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def seq_set_kmer_count(np.ndarray[UINT8_t, ndim=2] seq_matrix, UINT32_t k):
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1

    # store k-mer counts here
    _counts = np.zeros(4**k, dtype = np.uint32)
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT32_t [::1] counts = _counts

    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t index, i, j
    
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


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_filter(np.ndarray[UINT8_t, ndim=2] seq_matrix, str kmer):
    cdef UINT32_t k = len(kmer)
    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    # the index we are looking for
    cdef UINT32_t k_index = seq_to_index(kmer)
    
    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])
    cdef UINT32_t l = L-k+1

    # store k-mer hits here
    _mask = np.zeros(N * l, dtype = np.uint8)
    
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT8_t [::1] mask = _mask


    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t index, i, j
    
    for j in range(N):
        seq_bits = _seq_matrix[j*L:(j+1)*L]
        # compute index of first k-1-mer by bit-shifts
        index = kbits_to_index(seq_bits, k-1) 
        # iterate over remaining k-mers
        for i in range(0, l):
            # get next "letter"
            s = seq_bits[i+k-1]
            # compute next index from previous by shift + next letter
            index = ((index << 2) | s ) & MAX_INDEX
            
            if index == k_index:
                _mask[j*l+i] = 1

    _mask = _mask.reshape( (N,l))
    rows = _mask.any(axis=1)
    return seq_matrix[rows], _mask[rows]


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def kmer_flank_profiles(np.ndarray[UINT8_t, ndim=2] seq_matrix, str kmer, int k_flank=3):
    
    cdef UINT32_t k = len(kmer)
    print ">>>",kmer, k, k_flank

    # largest index in array of DNA/RNA k-mer counts
    cdef UINT32_t MAX_INDEX = 4**k - 1
    # the index we are looking for
    cdef UINT32_t k_index = seq_to_index(kmer)
    
    cdef UINT32_t N = len(seq_matrix)
    cdef UINT32_t L = len(seq_matrix[0])
    cdef UINT32_t l = L-k+1
    
    # aggregate flanking kmer counts - at each relative position - here
    profile = np.zeros( (4**k_flank, 2*l), dtype=np.uint32)

    # store k-mer hits here
    _mask = np.zeros(N * l, dtype = np.uint8)

    # remember the hit positions
    _hit_pos = np.zeros(l, dtype = np.uint64)
    
    # buffer
    _indices = np.zeros(l, dtype = np.uint64)
    
    # make a cython MemoryView with fixed stride=1 for 
    # fastest possible indexing
    cdef UINT8_t [::1] mask = _mask
    cdef UINT64_t [::1] hit_pos = _hit_pos

    # a MemoryView into each sequence (already converted 
    # from letters to bits)
    cdef UINT8_t [::1] _seq_matrix = seq_matrix.flatten()
    cdef UINT8_t [::1] seq_bits
    
    # helper variables to tell cython the types
    cdef UINT8_t s
    cdef UINT32_t index, findex, i, j, r
    cdef UINT32_t hits = 0

    
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
            _indices[i] = index
            #print i, index, index_to_seq(index, k), _indices[i]
            if index == k_index:
                _mask[j*l+i] = 1
                hit_pos[hits] = i
                hits += 1

        #print "pos 0 index", _indices[0]
        for o in hit_pos[:hits]:
            #print "hit at",o
            for i in range(0,l):
                # compute shorter kmer index from longer ones
                #print _indices[i]
                index = _indices[i]
                
                findex = index >> ((k - k_flank)*2)
                #print index, index_to_seq(index, k), index_to_seq(findex, k_flank), k_flank
                profile[findex][i-o+l-1] += 1

    return profile




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


