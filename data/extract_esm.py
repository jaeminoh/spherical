from esm.models.esmc import ESMC

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
)
import matplotlib.pyplot as plt

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

    ag_list, h_list, l_list = extract_sequences(load_SAbDab())

    ag1 = embed_sequence(model, str(ag_list[0]))
    h1 = embed_sequence(model, str(h_list[0]))
    l1 = embed_sequence(model, str(l_list[0]))

    # output.hidden_states.shape == [# hidden states, # sequences, len(sequence), # hidden features]
    print(
        ag1.hidden_states.mean(-2).squeeze(),
        h1.hidden_states.mean(-2).squeeze(),
        l1.hidden_states.mean(-2).squeeze(),
    )

    plt.plot(ag1.hidden_states.mean(-2).squeeze().float().cpu())
    plt.title(f"{HIDDEN}")
    plt.savefig(f"{HIDDEN}.pdf")


if __name__ == "__main__":
    fire.Fire(main)
