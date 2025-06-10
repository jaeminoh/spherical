from esm.models.esmc import ESMC
from concurrent.futures import ThreadPoolExecutor  # noqa
from typing import Sequence  # noqa

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,  # noqa
    LogitsConfig,
    LogitsOutput,
    ProteinType,  # noqa
)

sequence = "IRDFNNLTKGLCTINSWHIYGKDNAVRIGEDSDVLVTREPYVSCDPDECRFYALSQGTTIRGKHSNGTIHDRSQYRALISWPLSSPPTVYNSRVECIGWSSTSCHDGKTRMSICISGPNNNASAVIWYNRRPVTEINTWARNILRTQESECVCHNGVCPVVFTDGSATGPAETRIYYFKEGKILKWEPLAGTAKHIEECSCYGERAEITCTCRDNWQGSNRPVIRIDPVAMTHTSQYICSPVLTDNPRPNDPTVGKCNDPYPGNNNNGVKGFSYLDGVNTWLGRTISIASRSGYEMLKVPNALTDDKSKPTQGQTIVLNTDWSGYSGSFMDYWAEGECYRACFYVELIRGRPKEDKVWWTSNSIVSMCSSTEFLGQWDWPDGAKIEYFL"

model = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True,
    return_hidden_states=True,  # 37 is the maximum, for 600m model
)


def embed_sequence(
    model: ESM3InferenceClient, sequence: str = sequence
) -> LogitsOutput:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    return output


output = embed_sequence(model, sequence=sequence)
print(
    output.hidden_states.shape
)  # [# hidden states = 36, # sequences = 1, len(sequence) = 391, # hidden features = 1152]
