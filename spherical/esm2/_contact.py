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

    def get_contact_differences_vector(
        self,
        wt_sequence: str,
        sequences: list[str],
        positions: list[int],
        aas: list[str],
    ):
        wt_contact = self.get_contact_matrix("wt", wt_sequence)  # wt
        mse = []
        mae = []
        for sequence, pos, aa in tqdm(zip(sequences, positions, aas)):
            name = f"{pos}{aa}"
            contact = self.get_contact_matrix(name, sequence)
            diff = contact[pos] - wt_contact[pos]
            mse.append((diff**2).mean())
            mae.append(np.abs(diff).mean())
        return mse, mae

    def get_contact_differences_matrix(
        self,
        wt_sequence: str,
        sequences: list[str],
        positions: list[int],
        aas: list[str],
    ):
        wt_contact = self.get_contact_matrix("wt", wt_sequence)  # wt
        diffs = []

        for sequence, pos, aa in tqdm(zip(sequences, positions, aas)):
            name = f"{pos}{aa}"
            contact = self.get_contact_matrix(name, sequence)
            diff = np.linalg.norm(contact - wt_contact)  # Frobenius norm
            diffs.append(diff)


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

    diffs = contact.get_contact_differences_matrix(
        wt.sequence, sequences, positions, aas
    )
    pd.DataFrame({"pos": positions, "aa": aas, "diffs": diffs}).to_csv(
        "results/rvfv/contact_matrix_diffs.csv", index=False
    )

    # np.savez(
    #     "results/rvfv/esm2_contact_diffs.npz",
    #     diffs=np.stack(diffs),
    #     positions=positions,
    #     mt=aas,
    # )
