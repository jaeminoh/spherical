import jax
import jax.numpy as jnp
import esm2quinox as esmqx
import equinox as eqx
from utils import compute_esm_score_str


# TODO fit DMS dataset with this feature
# Example DMS dataset: https://github.com/debbiemarkslab/DeepSequence/blob/master/examples/mutations/PABP_YEAST_Fields2013-singles.csv
def score(model: esmqx.ESM2, wt: str, mt: str) -> float:
    grammaticality = compute_esm_score_str(model, wt, mt)
    semantic_wt, semantic_mt = jax.vmap(model)(esmqx.tokenise([wt, mt])).hidden
    semantic_change = jnp.abs(semantic_wt - semantic_mt).mean()
    print(f"Semantic change: {semantic_change:.4f}")
    print(f"Grammaticality score: {grammaticality:.4f}")
    return jnp.stack([grammaticality, semantic_change])


regressor = eqx.nn.MLP(2, "scalar", 2, 2, jnp.tanh)


if __name__ == "__main__":
    from utils import load_model
    from constants import SEQUENCE_WT, SEQUENCE_MT

    model = load_model(model_params=650)
    score(model, SEQUENCE_WT, SEQUENCE_MT)
