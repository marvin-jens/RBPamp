import numpy as np

def load_model(fname):
    from cska.params import ModelSetParams
    params = ModelSetParams.load(fname, 1)
    return params

def eval_model(params, seq, m=None):
    if len(seq) < params.k:
        return np.zeros(len(seq), dtype=np.float32)
    from cska.cyska import seq_to_bits, PSAM_partition_function
    seqm = seq_to_bits(seq)
    seqm = seqm.reshape((1, len(seq)) )
    # print seqm, seqm.dtype
    # print params.psam_matrix
    accm = np.ones(seqm.shape, dtype=np.float32)
    # print accm, accm.dtype
    # print "seqm", seqm.shape, seqm.min(), seqm.max()
    if m is None:
        Z = np.array([PSAM_partition_function(seqm, accm, par.psam_matrix, single_thread=True) * par.A0/params.A0 for par in params])
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
