from esm.models.esmc import ESMC

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
)

from data_summary import load_SAbDab, extract_sequences

model = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True,
    return_hidden_states=True,  # 37 is the maximum, for 600m model
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

# [# hidden states = 36, # sequences = 1, len(sequence) = 12271, # hidden features = 1152]
print(
    ag1.hidden_states[-1].squeeze().mean(0),
    h1.hidden_states[-1].squeeze().mean(0).shape,
    l1.hidden_states[-1].squeeze().mean(0).shape,
)
