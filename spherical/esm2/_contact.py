from esm import ESM2, BatchConverter
from tqdm import tqdm


def _get_contacts(
    model: ESM2, batch_converter: BatchConverter, name: str, sequence: str
):
    *_, batch_tokens = batch_converter([(name, sequence)])
    batch_tokens = batch_tokens.to(next(model.parameters()).device)
    predictions = model.predict_contacts(batch_tokens)[0].cpu()
    return predictions.detach().numpy()


def get_contact_differences(
    model: ESM2,
    batch_converter: BatchConverter,
    wt_sequence: str,
    sequences: list[str],
    positions: list[int],
    aas: list[str],
):
    wt_contact = _get_contacts(model, batch_converter, "wt", wt_sequence)  # wt
    diffs = []

    for sequence, pos, aa in tqdm(zip(sequences, positions, aas)):
        name = f"{pos}{aa}"
        contact = _get_contacts(model, batch_converter, name, sequence)
        diff = contact[pos] - wt_contact[pos]
        diffs.append(diff)

    return diffs


if __name__ == "__main__":
    # Example usage: python spherical/esm2/_contact.py
    import numpy as np

    import spherical

    model, alphabet = spherical.esm2.load_model_and_alphabet(spherical.FAIR_ESM2_650M)
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    wt = spherical.RVFV_MP12
    sequences, positions, aas = wt.get_all_single_mutations(spherical.AMINO_ACIDS)

    diffs = spherical.esm2.get_contact_differences(
        model, batch_converter, wt.sequence, sequences, positions, aas
    )
    np.savez(
        "results/rvfv/esm2_contact_diffs.npz",
        diffs=np.stack(diffs),
        positions=positions,
        mt=aas,
    )
