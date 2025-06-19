import numpy as np
from RBPamp.params import ModelSetParams, ModelParametrization


# an empty PSAM, representing that information is not available
NA_model = ModelSetParams([ModelParametrization(11, 1, A0=np.nan)])


def load_model(fname, sort=True, n_samples=1):
    from RBPamp.params import ModelSetParams
    params = ModelSetParams.load(fname, n_samples, sort=sort)
    return params



def eval_model(params, seq, m=None):
    if len(seq) < params.k:
        return np.zeros(len(seq), dtype=np.float32)
    from RBPamp.cyska import seq_to_bits, PSAM_partition_function
    seqm = seq_to_bits(seq)
    seqm = seqm.reshape((1, len(seq)) )
    # print seqm, seqm.dtype
    # print params.psam_matrix
    accm = np.ones(seqm.shape, dtype=np.float32)
    # print accm, accm.dtype
    # print "seqm", seqm.shape, seqm.min(), seqm.max()
    if m is None:
        Z = np.array([PSAM_partition_function(seqm, accm, par.psam_matrix,
                                              single_thread=True)
                      * par.A0/params.A0 for par in params])
        # print Z.shape, "lseq", len(seq)
        return Z.sum(axis=0)[0, :]  # sum over all sub-motifs. we have only one sequence->index 0
    else:
        par = params.param_set[m]
        Z = PSAM_partition_function(seqm, accm, par.psam_matrix, single_thread=True) * par.A0/params.A0
        return Z[0, :]


def motif_peaks(Z, Amin=1e-3, pad=50, k=8):
    i = Z.argmax()
    # print Z[i], threshold
    # n_hits = 0
    while Z[i] >= Amin:
        start = max(0, i - pad)
        end = min(len(Z), i + k + pad)
        yield start, end
        # drop all scores in the padded region around the hit to zero and look at next-highest peak
        Z[start:end] = 0
        i = Z.argmax()


def ensure_path(full):
    import os
    path = os.path.dirname(full)
    try:
        os.makedirs(path)
    except OSError:
        pass

    return full


COMPLEMENT = {
    "a": "u",
    "t": "a",
    "u": "a",
    "c": "g",
    "g": "c",
    "k": "m",
    "m": "k",
    "r": "y",
    "y": "r",
    "s": "s",
    "w": "w",
    "b": "v",
    "v": "b",
    "h": "d",
    "d": "h",
    "n": "n",
    "A": "U",
    "T": "A",
    "U": "A",
    "C": "G",
    "G": "C",
    "K": "M",
    "M": "K",
    "R": "Y",
    "Y": "R",
    "S": "S",
    "W": "W",
    "B": "V",
    "V": "B",
    "H": "D",
    "D": "H",
    "N": "N",
    "-": "-",
    "=": "=",
    "+": "+",
}


def complement(s):
    return "".join([COMPLEMENT[x] for x in s])


def rev_comp(seq):
    return complement(seq)[::-1]


def yield_kmers(k, bases = 'ACGU'):
    import itertools
    """
    An iterater to all kmers of length k in alphabetical order
    """
    for kmer in itertools.product(bases, repeat=k):
        yield ''.join(kmer)


def load_kmers_from_file(fname):
    if not fname:
        return []
    else:
        return [line.rstrip() for line in open(fname)]


def kmers_from_seq(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def get_kmers_containing_cores(cores, k):
    match = set(cores)
    k_small = len(cores[0])
    assert k_small < k

    kmers = []
    for kmer in yield_kmers(k):
        if set(kmers_from_seq(kmer, k_small)) & match:
            kmers.append(kmer)
    
    return kmers

