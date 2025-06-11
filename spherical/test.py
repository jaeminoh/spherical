import time
import esm2quinox as esmqx
from utils import load_model

model = load_model(model_params=650)

protein = "SPIDERMAN"

tic = time.time()
_ = model(protein)
toc = time.time()
print(f"String: {toc - tic:.4f} seconds")

_ = model._call(esmqx.tokenise([protein])[0])
tic = time.time()
_ = model._call(esmqx.tokenise([protein])[0]).logits.block_until_ready()
toc = time.time()
print(f"Tokenised: {toc - tic:.4f} seconds")
