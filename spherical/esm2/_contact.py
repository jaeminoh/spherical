import torch
from esm import ESM2, BatchConverter
from tqdm import tqdm


class ContactPredictor:
    def __init__(self, model: ESM2, batch_converter: BatchConverter):
        self.model = model
        self.batch_converter = batch_converter

    def get_contact_matrix(self, name: str, sequence: str):
        *_, batch_tokens = self.batch_converter([(name, sequence)])
        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
        with torch.no_grad():
            predictions = self.model.predict_contacts(batch_tokens)[0]
        return predictions

    def get_contact_differences(
        self,
        wt_sequence: str,
        sequences: list[str],
        positions: list[int],
        aas: list[str],
    ):
        wt_contact = self.get_contact_matrix("wt", wt_sequence)  # wt

        vector_norms = {"l1": [], "l2": []}
        matrix_norms = {
            "operator_l1": [],
            "operator_l2": [],
            "frobenius_1": [],
            "frobenius_2": [],
        }

        for sequence, pos, aa in tqdm(zip(sequences, positions, aas)):
            name = f"{pos}{aa}"
            diff = wt_contact - self.get_contact_matrix(name, sequence)
            vector_norms["l1"].append(
                torch.linalg.norm(diff[pos], ord=1).to("cpu").item()
            )
            vector_norms["l2"].append(
                torch.linalg.norm(diff[pos], ord=2).to("cpu").item()
            )
            matrix_norms["operator_l1"].append(
                torch.linalg.norm(diff, ord=1).to("cpu").item()
            )
            matrix_norms["operator_l2"].append(
                torch.linalg.norm(diff, ord=2).to("cpu").item()
            )
            matrix_norms["frobenius_1"].append(abs(diff).sum().to("cpu").item())
            matrix_norms["frobenius_2"].append(
                torch.linalg.norm(diff, ord="fro").to("cpu").item()
            )
        return vector_norms, matrix_norms


if __name__ == "__main__":
    # Example usage: python spherical/esm2/_contact.py
    import pandas as pd
    import fire

    import spherical

    model, alphabet = spherical.esm2.load_model_and_alphabet(spherical.FAIR_ESM2_650M)
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    contact = ContactPredictor(model, batch_converter)

    def main(virus: str):
        if virus == "rvfv":
            wt = spherical.RVFV_MP12
        elif virus == "cov2":
            wt = spherical.SARS_COV2_SPIKE
        else:
            raise ValueError(f"Unknown virus: {virus}")

        sequences, positions, aas = wt.get_all_single_mutations(spherical.AMINO_ACIDS)

        vector_norms, matrix_norms = contact.get_contact_differences(
            wt.sequence, sequences, positions, aas
        )
        pd.DataFrame(
            {"pos": positions, "mt": aas, **vector_norms, **matrix_norms}
        ).to_csv(f"results/{virus}/all_single.csv", index=False)

    fire.Fire(main)
