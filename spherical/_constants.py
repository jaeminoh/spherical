from Bio import SeqIO

from spherical._protein_sequence import ReferenceProteinSequence

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

RVFV_MP12 = ReferenceProteinSequence(str(SeqIO.read("assets/RVFV.fasta", "fasta").seq))

ESM650M = "facebook/esm2_t33_650M_UR50D"
ESMFOLD = "facebook/esmfold_v1"
