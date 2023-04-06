from RBPamp.reads import RBNSReads
from RBPamp.params import ModelParametrization, ModelSetParams
from RBPamp.partfunc import PartFuncModel
import numpy as np
import RBPamp.cyska
import logging
import os
logging.basicConfig(level=logging.DEBUG)
from argparse import ArgumentParser
import RBPamp.util as util


def sample_reads(reads, state, out_files=[]):
    print(f"psi: {state.psi.shape} min/max={state.psi.min():.3f} {state.psi.max():.3f}")
    print(f"betas: {state.params.betas}")

    def read_src(fname):
        for line in open(fname, 'rt'):
            if 'N' in line:
                continue
            else:
                # seq = (
                #     reads.adap5 + 
                #     line.lower().replace('t', 'u').rstrip() + 
                #     reads.adap3
                # )
                # print(seq)
                # i = seq.find('gcaug')
                # if i >= 0:
                #     print(" "*i + "*"*5)

                yield line

    rnd = RBPamp.cyska.fast_rand(state.psi.shape[1])
    for psi_row, seq, p in zip(state.psi.T, read_src(reads.fname), rnd):
        p_pd = 1 - (1-psi_row) * (1-state.params.betas)
        accept = p <= p_pd
        # print(f"p_bound={psi_row[0]:.2f} p_pulldown={p_pd[0]:.2f} rnd={p:.3f} accept={accept}")
        # if (psi_row > 0.5).any():
        #     break
    
        # print(accept.nonzero())
        for i in accept.nonzero()[0]:
            out_files[i].write(seq)

def main():
    parser = ArgumentParser()
    parser.add_argument("input", nargs='+', help="one or more input.reads files to sample from")
    parser.add_argument("--name", help="name of simulated RBP (default=synth", default='synth')
    parser.add_argument("--model", required=True, help="path to tsv file with PSAM data")
    parser.add_argument("--rbp-conc", help="comma-separated list of simulated RBP concentrations (default=20)")
    parser.add_argument("--betas", help="comma-separated list of background admixture values (default=0)")
    parser.add_argument("--out-dir", help="output-directory (default=./synth_reads)", default='./synth_reads')
    parser.add_argument("--out-mode", help="file open mode for outfiles (defalt='w') set to 'a' for append", default='w')
    parser.add_argument("--rnd-seed", help="pseudo-random generator seed value (default=313370815)", default=313370815, type=int)
    args = parser.parse_args()

    RBPamp.cyska.rand_seed(args.rnd_seed)

    #"/home/mjens/RBNS/RBFOX3/RBPamp/py3_seed1/opt_nostruct/optimized.tsv"

    rbp_conc = np.array([float(c) for c in args.rbp_conc.split(',')])
    if not args.betas:
        betas = np.zeros(len(rbp_conc), dtype=np.float32)
    else:
        betas = np.array([float(b) for b in args.betas.split(',')])

    assert (betas < 1).all()
    assert len(betas) == len(rbp_conc)
    
    params = util.load_model(args.model, n_samples=len(betas))
    print(params.betas)
    params.betas = betas

    util.ensure_path(args.out_dir + '/')
    out_fnames = [f"{args.name}_{int(conc)}.reads" for conc in rbp_conc]
    out_files = [open(os.path.join(args.out_dir, fname), args.out_mode) for fname in out_fnames]
    
    for freads in args.input:
        # "/home/mjens/RBNS/RBFOX3/RBFOX3_input.reads"
        reads = RBNSReads(freads)

        # class SamplingModel(PartFuncModel):

        model = PartFuncModel(
            reads,
            params,
            [],
            rbp_conc = rbp_conc,
        )
        model.n_samples = len(rbp_conc)

        # print(params.betas)
        state = model.predict(params, tune=False, beta_fixed=True, compute_R=False)
        # print(state)
        sample_reads(reads, state, out_files)
    
    for o in out_files:
        o.close()


if __name__ == "__main__":
    main()
