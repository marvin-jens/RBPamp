import os
import sys
import pysam
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("bamfile", help="path to bamfile")
parser.add_argument("--pos_min", default=87, type=int)
parser.add_argument("--pos_max", default=90, type=int)
args = parser.parse_args()

pmin = args.pos_min
pmax = args.pos_max

counts = defaultdict(int)
n_discard = 0
samfile = pysam.AlignmentFile(args.bamfile, "rb")
for read in samfile.fetch():
    if pmin <= read.pos <= pmax:
        counts[read.tid] += 1
    else:
        n_discard += 1

n_kept = 0
for tid in sorted(counts.keys()):
    print "{ref}\t{count}".format(ref=samfile.get_reference_name(tid), count=counts[tid])
    n_kept += counts[tid]

n_total = float(n_kept + n_discard)
kept_perc = 100. * n_kept / n_total
disc_perc = 100. * n_discard / n_total
bam_name = os.path.basename(args.bamfile)
sys.stderr.write("{bam_name}: kept {n_kept} ({kept_perc:.2f}%) well-aligned reads. Discarded {n_discard} ({disc_perc:.2f}%) mis-aligned reads\n".format(**locals()))


