from esm import ESM2, BatchConverter
from tqdm import tqdm
import numpy as np


class ContactPredictor:
    def __init__(self, model: ESM2, batch_converter: BatchConverter):
        self.model = model
        self.batch_converter = batch_converter

    def get_contact_matrix(self, name: str, sequence: str):
        *_, batch_tokens = self.batch_converter([(name, sequence)])
        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
        predictions = self.model.predict_contacts(batch_tokens)[0].cpu()
        return predictions.detach().numpy()

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
            vector_norms["l1"].append(np.linalg.norm(diff[pos], ord=1))
            vector_norms["l2"].append(np.linalg.norm(diff[pos], ord=2))
            matrix_norms["operator_l1"].append(np.linalg.norm(diff, ord=1))
            matrix_norms["operator_l2"].append(np.linalg.norm(diff, ord=2))
            matrix_norms["frobenius_1"].append(np.abs(diff).sum())
            matrix_norms["frobenius_2"].append(np.linalg.norm(diff, ord="fro"))

        return vector_norms, matrix_norms


if __name__ == "__main__":
    # Example usage: python spherical/esm2/_contact.py
    import pandas as pd

    import spherical

    model, alphabet = spherical.esm2.load_model_and_alphabet(spherical.FAIR_ESM2_650M)
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    contact = ContactPredictor(model, batch_converter)

    wt = spherical.RVFV_MP12
    sequences, positions, aas = wt.get_all_single_mutations(spherical.AMINO_ACIDS)

    vector_norms, matrix_norms = contact.get_contact_differences(
        wt.sequence, sequences, positions, aas
    )
    pd.DataFrame({"pos": positions, "mt": aas, **vector_norms, **matrix_norms}).to_csv(
        "results/rvfv/contact_matrix_diffs.csv", index=False
    )
