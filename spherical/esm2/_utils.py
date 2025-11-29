import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from ._types import EsmModels

import spherical


def load_model(esm: EsmModels, model_name: str):
    model = esm.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def infer(
    virus: str,
    in_csv_name: str,
    out_csv_name: str,
    callback_fn: callable,
    out_dicts: dict,
):
    if virus == "rvfv":
        protein = spherical.RVFV_MP12
    elif virus == "cov2":
        protein = spherical.SARS_COV2_SPIKE
    else:
        raise ValueError(f"Unknown virus: {virus}")

    csv_dir = f"results/{virus}"
    df = pd.read_csv(f"{csv_dir}/{in_csv_name}.csv")
    names = df["name"]

    for name in tqdm(names):
        p, a = spherical.split_name(name)
        mutant = protein.mutate(p, a)
        callback_fn(mutant, out_dicts)

    pd.DataFrame({"name": names, **out_dicts}).to_csv(
        f"{csv_dir}/{out_csv_name}.csv", index=False
    )
