from esm.models.esmc import ESMC

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
)
import numpy as np
from data_summary import load_SAbDab, extract_sequences
import fire


def main(HIDDEN: int = 35):
    model = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"
    EMBEDDING_CONFIG = LogitsConfig(
        sequence=True,
        return_hidden_states=True,
        ith_hidden_layer=HIDDEN,  # 35 is the maximum (total 36, and -1 indicates all), for 600m model
    )

    def embed_sequence(model: ESM3InferenceClient, sequence: str) -> LogitsOutput:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = model.encode(protein)
        output = model.logits(protein_tensor, EMBEDDING_CONFIG)
        return output

    (ag_list, h_list, l_list), dg_list = extract_sequences(load_SAbDab())
    ag_features = []
    h_features = []
    l_features = []
    for i in range(len(dg_list)):
        ag_features.append(
            embed_sequence(model, str(ag_list[i]))
            .hidden_states.mean(-2)
            .squeeze()
            .float()
            .cpu()
        )
        h_features.append(
            embed_sequence(model, str(h_list[i]))
            .hidden_states.mean(-2)
            .squeeze()
            .float()
            .cpu()
        )
        l_features.append(
            embed_sequence(model, str(l_list[i]))
            .hidden_states.mean(-2)
            .squeeze()
            .float()
            .cpu()
        )

    ag_features = np.stack(ag_features)
    h_features = np.stack(h_features)
    l_features = np.stack(l_features)
    dG = np.stack(dg_list)
    np.savez("data.npz", ag=ag_features, hchain=h_features, lchain=l_features, dG=dG)


if __name__ == "__main__":
    fire.Fire(main)
