import time
from Bio import SeqIO
from transformers import EsmForProteinFolding, AutoTokenizer

from spherical.esm2._fold import Fold
from spherical._protein_sequence import ReferenceProteinSequence


wt = SeqIO.read("assets/RVFV.fasta", "fasta").seq
rvfv_ref = ReferenceProteinSequence(str(wt))
all_mutations, positions, mutated_aas = rvfv_ref.get_all_single_mutations()

esm_name = "facebook/esmfold_v1"
fold = Fold(
    EsmForProteinFolding.from_pretrained(esm_name),
    AutoTokenizer.from_pretrained(esm_name),
)
fold.model.to("cuda")


def predict_and_save_pdb(sequence, filename):
    out = fold.model.infer(sequence)
    pdb = fold.model.output_to_pdb(out)
    with open(filename, "w") as f:
        f.write("".join(pdb))


tic = time.time()
filename = "results/rvfv.pdb"
predict_and_save_pdb(rvfv_ref.sequence, filename)
for seq, pos, aa in zip(all_mutations, positions, mutated_aas):
    filename = f"results/rvfv_{pos + 1}{aa}.pdb"  # 1-based indexing
    predict_and_save_pdb(seq, filename)
    toc = time.time()
    print(f"Saved: {pos + 1}{aa}, {toc - tic:.2f}s")
