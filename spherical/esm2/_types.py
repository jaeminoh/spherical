from transformers import EsmModel, EsmForMaskedLM, EsmForProteinFolding

type EsmModels = EsmModel | EsmForMaskedLM | EsmForProteinFolding
