from weblogolib import *
import numpy as np
fin = open('cap.fa')
seqs = read_seq_data(fin) 
#data = LogoData.from_seqs(seqs)
counts = np.array([
    [1,0,0,0],
    [0,1.,0,0],
    [0,0,1.,0],
    [0,0,0,1.],
    [.5,.5,0,0],
])
from corebio.seq import unambiguous_rna_alphabet
#data = LogoData(alphabet=unambiguous_rna_alphabet, length=5, counts=counts, entropy=np.ones(5), weight=np.ones(5))
data = LogoData.from_counts(unambiguous_rna_alphabet, counts)
import sys
sys.stderr.write(str( data))
options = LogoOptions(color_scheme=classic, fineprint="")
options.title = "A Logo Title"
format = LogoFormat(data, options)
eps = eps_formatter( data, format)
print eps
