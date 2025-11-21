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


def fold_and_save(model: EsmForProteinFolding, sequence: str, filename: str):
    out = model.infer(sequence)
    pdb = model.output_to_pdb(out)
    with open(filename, "w") as f:
        f.write("".join(pdb))


if __name__ == "__main__":
    # example usage: python spherical/esm2/_fold.py --position=335 --new_aa="S"
    import fire
    from tqdm import tqdm
    import spherical

    model, _ = spherical.esm2.load_model(EsmForProteinFolding, spherical.ESMFOLD)
    model.to("cuda")

    def main(positions: list[int], new_aas: list[str]):
        protein = spherical.RVFV_MP12
        for position, new_aa in tqdm(zip(positions, new_aas)):
            mutant = protein._mutate(position, new_aa)
            spherical.esm2.fold_and_save(
                model, mutant, f"results/rvfv/{position}{new_aa}.pdb"
            )

    fire.Fire(main)

    # import time
    # import spherical

    # protein = spherical.RVFV_MP12
    # model, tokenizer = spherical.esm2.load_model(EsmForProteinFolding, spherical.ESMFOLD)
    # model.to("cuda")

    # all_mutations, positions, mutated_aas = protein.get_all_single_mutations(spherical.AMINO_ACIDS)

    # tic = time.time()
    # filename = "results//rvfv/wt.pdb"
    # spherical.esm2.fold_and_save(model, protein.sequence, filename)
    # for seq, pos, aa in zip(all_mutations, positions, mutated_aas):
    #     filename = f"results/rvfv/{pos + 1}{aa}.pdb"  # 1-based indexing
    #     spherical.esm2.fold_and_save(model, seq, filename)
    #     toc = time.time()
    #     print(f"Saved: {pos + 1}{aa}, {toc - tic:.2f}s")
