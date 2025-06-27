import jax
import jax.numpy as jnp
import jax.random as jr
import esm2quinox as esmqx
import equinox as eqx
from utils import compute_esm_score_str


# TODO fit DMS dataset with this feature
# Example DMS dataset: https://github.com/debbiemarkslab/DeepSequence/blob/master/examples/mutations/PABP_YEAST_Fields2013-singles.csv
def score(model: esmqx.ESM2, wt: str, mt: str) -> float:
    grammaticality = compute_esm_score_str(model, wt, mt)  # 1 forward pass
    semantic_change = SemanticChange(esmqx.ESM2).vanilla(
        wt, mt, ord=1
    )  # 2 forward passes
    return semantic_change, grammaticality


class SemanticChange:
    """
    Class to compute the semantic change between two sequences.
    """

    def __init__(self, model: esmqx.ESM2):
        self.model = model

    def vanilla(self, wt: str, mt: str, ord: int | str = 1) -> float:
        """
        Hie et al., 2021 Science, learning the language of viral evolution.
        semantic:= mean of hidden representation across all positions, (L+2, D) -> (D,)
        distance of semantic space: L^1 norm.
        """
        semantic_wt, semantic_mt = jax.vmap(model)(
            esmqx.tokenise([wt, mt])
        ).hidden  # 2 forward passes
        semantic_change = jnp.linalg.norm(
            semantic_wt.mean(axis=0) - semantic_mt.mean(axis=0), ord=ord
        )
        return semantic_change


regressor = eqx.nn.MLP(2, "scalar", 2, 2, jnp.tanh, key=jr.key(1234))


if __name__ == "__main__":
    from utils import load_model
    from constants import SEQUENCE_WT, SEQUENCE_MT

    model = load_model(model_params=650)
    print(score(model, SEQUENCE_WT, SEQUENCE_MT))
