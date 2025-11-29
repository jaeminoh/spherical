import torch
from transformers import EsmTokenizer, EsmForProteinFolding


class Fold:
    def __init__(
        self, model: EsmForProteinFolding, tokenizer: EsmTokenizer, device: str = "cuda"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer

    def tokenize(self, sequence: str):
        input_ids = self.tokenizer(
            sequence, return_tensors="pt", add_special_tokens=False
        )
        return input_ids

    def __call__(self, input_ids):
        with torch.no_grad():
            outputs = self.model(**input_ids)
        return outputs  # Returns the folded structure or relevant outputs

    def predict_and_save_pdb(self, sequence: str, filename: str):
        out = self.model.infer(sequence)
        pdb = self.model.output_to_pdb(out)
        with open(filename, "w") as f:
            f.write("".join(pdb))


def fold_and_save(
    model: EsmForProteinFolding, tokenizer: EsmTokenizer, sequence: str, filename: str
):
    sequence_tokens = tokenizer(
        sequence, return_tensors="pt", add_special_tokens=False
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(**sequence_tokens, num_recycles=4)
    pdb = model.output_to_pdb(out)
    with open(filename, "w") as f:
        f.write("".join(pdb))
    print(f"Saved folded structure to {filename}")
    print(f"Predicted TM-score: {out['ptm'].item():.2f}")
