from Bio import SeqIO

from spherical._protein import Protein

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

RVFV_MP12 = Protein(str(SeqIO.read("assets/RVFV.fasta", "fasta").seq))

ESM650M = "facebook/esm2_t33_650M_UR50D"
ESMFOLD = "facebook/esmfold_v1"

FAIR_ESM2_650M = "/raid/public/checkpoints/esm/esm2_t33_650M_UR50D.pt"
