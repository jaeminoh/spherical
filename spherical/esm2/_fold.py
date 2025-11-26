import torch
from transformers import EsmTokenizer, EsmForProteinFolding

import spherical


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


if __name__ == "__main__":
    # example usage: python spherical/esm2/_fold.py --position=335 --new_aa="S"
    import fire
    from tqdm import tqdm
    import pandas as pd

    import spherical

    model, tokenizer = spherical.esm2.load_model(
        EsmForProteinFolding, spherical.ESMFOLD
    )
    model.to("cuda")

    def main(csv_name: str, virus: str = "rvfv", method: str = "recycle4"):
        dir = f"results/{virus}"
        df = pd.read_csv(f"{dir}/{csv_name}.csv")
        positions = df["pos"].tolist()
        new_aas = df["mt"].tolist()
        protein = spherical.RVFV_MP12
        spherical.esm2.fold_and_save(
            model, tokenizer, protein.sequence, f"results/{virus}/{method}/wt.pdb"
        )

        for position, new_aa in tqdm(zip(positions, new_aas)):
            mutant = protein._mutate(position, new_aa)
            spherical.esm2.fold_and_save(
                model,
                tokenizer,
                mutant,
                f"results/{virus}/{method}/{position}{new_aa}.pdb",
            )

    fire.Fire(main)
