import esm
import spherical
import numpy as np
from tqdm import tqdm

esm2, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm2 = esm2.eval().cuda()
esm2_batch_converter = esm2_alphabet.get_batch_converter()

wt = spherical.RVFV_MP12
sequences, positions, aas = wt.get_all_single_mutations(spherical.AMINO_ACIDS)


# inputs = [
#     ("wt", wt.sequence),
#     ("730Y", wt._mutate(730, "Y")),
#     ("116H", wt._mutate(116, "H")),
#     ("335S", wt._mutate(335, "S"))
# ]


def get_contacts(name, sequence):
    _input = (name, sequence)
    *_, esm2_batch_tokens = esm2_batch_converter([_input])
    esm2_batch_tokens = esm2_batch_tokens.to(next(esm2.parameters()).device)
    esm2_predictions = esm2.predict_contacts(esm2_batch_tokens)[0].cpu()
    return esm2_predictions


wt_contacts = get_contacts("wt", wt.sequence)
names = []
diffs = []

for seq, pos, aa in tqdm(zip(sequences, positions, aas)):
    name = f"{pos}{aa}"
    contacts = get_contacts(name, seq)
    diff = contacts[pos] - wt_contacts[pos]
    names.append(name)
    diffs.append(diff.detach().numpy())

np.savez("results/rvfv/esm2_contact_diffs.npz", names=names, diffs=np.stack(diffs))
