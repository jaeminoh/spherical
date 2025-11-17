import torch
from transformers import EsmTokenizer, EsmForProteinFolding


class Fold:
    def __init__(self, model: EsmForProteinFolding, tokenizer: EsmTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, sequence: str):
        input_ids = self.tokenizer.encode(
            sequence, return_tensors="pt", add_special_tokens=False
        )
        return input_ids

    def __call__(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs  # Returns the folded structure or relevant outputs


if __name__ == "__main__":
    # example script, must be run at the repository root
    import time

    from transformers import AutoTokenizer
    from Bio import SeqIO

    wt = SeqIO.read("assets/RVFV.fasta", "fasta").seq
    wt = str(wt)

    print("Loading model...")
    tic = time.time()
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    toc = time.time()
    print(f"Done! Took {toc - tic:.2f} seconds.")

    folding = Fold(model, tokenizer)

    print("Moving tensors to gpus...")
    tic = time.time()
    input_ids = folding.tokenize(wt).to("cuda")
    folding.model.to("cuda")
    toc = time.time()
    print(f"Done! Took {toc - tic:.2f} seconds.")

    print("Folding...")
    tic = time.time()
    with torch.no_grad():
        outputs = folding(input_ids)
    toc = time.time()
    print(f"Done! Took {toc - tic:.2f} seconds.")
