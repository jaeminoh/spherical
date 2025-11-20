from transformers import AutoTokenizer
from ._types import EsmModels


def load_model(esm: EsmModels, model_name: str):
    model = esm.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
