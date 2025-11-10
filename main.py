from transformers import EsmTokenizer, EsmForMaskedLM

import spherical.esm2 as esm2


model_name = "facebook/esm2_t6_8M_UR50D"
model = EsmForMaskedLM.from_pretrained(model_name)
tokenizer = EsmTokenizer.from_pretrained(model_name)
wt = "MKTAYIAKQRQISFVKSHFSRQDILDLI"
mt = "MKTAYIAKQRQISFVKSHFSRQDILDLL" # last I -> L

print(esm2.MaskedMarginal(model, tokenizer)(wt, mt))
print(esm2.MutantMarginal(model, tokenizer)(wt, mt))
print(esm2.WildtypeMarginal(model, tokenizer)(wt, mt))
print(esm2.Entropy(model, tokenizer)(wt))
