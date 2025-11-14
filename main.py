import time

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from Bio import SeqIO

wt = SeqIO.read("assets/RVFV.fasta", "fasta").seq
wt = str(wt)

print("Loading model...")
tic = time.time()
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
toc = time.time()
print(f"Done! Took {toc - tic:.2f} seconds.")

input_ids = tokenizer.encode(
    wt, return_tensors="pt", add_special_tokens=False
)  # (1, 1197)
# input_ids = tokenizer.encode("MLKNVQVQLV", return_tensors="pt", add_special_tokens=False)

print("Moving tensors to gpus...")
tic = time.time()
model.to("cuda")
input_ids = input_ids.to("cuda")
toc = time.time()
print(f"Done! Took {toc - tic:.2f} seconds.")


print("Folding...")
tic = time.time()
with torch.no_grad():
    outputs = model(input_ids)
toc = time.time()
print(f"Done! Took {toc - tic:.2f} seconds.")
