import esm  # pip install fair-esm==2.0.0
import esm2quinox as esmqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Int, jaxtyped, Bool
from beartype import beartype
import random
import pandas as pd
from tqdm import trange
from constants import AA


def load_model(
    model_params: int = 8,
) -> esmqx.ESM2:
    """
    Load an ESM2 model with specified device and dtype.

    Args:
        model_name (str): Name of the ESM2 model to load.
        device (str): Device to load the model on (e.g., "cpu", "cuda").
        dtype (str): Data type for the model parameters (e.g., "float32", "bfloat16").

    Returns:
        esmqx.ESM2: Loaded ESM2 model.
    """
    if model_params == 8:
        torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    elif model_params == 650:
        torch_model, _ = esm.pretrained.esm2_t33_650M_UR50D()

    model = esmqx.from_torch(torch_model)
    return model


def find_mutation_information(
    wt_ids: Int[ArrayLike, " L"],
    mt_ids: Int[ArrayLike, " L"],
) -> Bool[ArrayLike, " L"]:
    positions = wt_ids != mt_ids
    return positions


def find_mutation_information_str(
    wt: str,
    mt: str,
) -> tuple[list[int], list[str], list[str]]:
    positions = []
    wt_aa = []
    mt_aa = []
    for i, (_wt, _mt) in enumerate(zip(wt, mt)):
        if _wt != _mt:
            positions.append(i)
            wt_aa.append(_wt)
            mt_aa.append(_mt)
    return positions, wt_aa, mt_aa


def mutate_id(wt_id, positions, mt_id):
    return (1 - positions) * wt_id + positions * mt_id


def mask_sequence(
    sequence_id: Int[ArrayLike, " L"],
    positions: Bool[ArrayLike, " L"],
) -> Int[ArrayLike, " L"]:
    mask_token: int = esmqx.tokenise(["#"])[0][1:-1]
    return mutate_id(sequence_id, positions, mask_token)


def mutate_str(wt: str, positions: list[int], mt: list[str] | str) -> str:
    sequence_masked = list(wt)
    if isinstance(mt, list):
        for i, m in zip(positions, mt):
            sequence_masked[i] = m
    elif isinstance(mt, str):
        for i in positions:
            sequence_masked[i] = mt
    return "".join(sequence_masked)


def mask_sequence_str(sequence: str, positions: list[int]) -> str:
    return mutate_str(sequence, positions, mt="#")


@jaxtyped(typechecker=beartype)
def compute_esm_score(
    model: esmqx.ESM2, wt_ids: Int[ArrayLike, " L"], mt_ids: Int[ArrayLike, " L"]
):
    mutated_positions = find_mutation_information(wt_ids, mt_ids)
    assert (wt_ids[0] == mt_ids[0]) and (wt_ids[-1] == mt_ids[-1]), (
        "First and last tokens must be the same for both sequences"
    )
    sequence_masked_id = mask_sequence(wt_ids, mutated_positions)
    out = model(sequence_masked_id)
    logits = out.logits
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    ids = jnp.nonzero(mutated_positions)
    # TODO faster way?
    wt_score = log_prob[ids, wt_ids[ids]].sum()
    mt_score = log_prob[ids, mt_ids[ids]].sum()
    return mt_score - wt_score


def compute_esm_score_str(model: esmqx.ESM2, wt: str, mt: str):
    positions, wt_aa, mt_aa = find_mutation_information_str(wt, mt)
    sequence_masked = mask_sequence_str(wt, positions)
    out = model(sequence_masked)
    logits = out.logits[1:-1]
    assert len(wt) == logits.shape[0], "different shape!"
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    ids = jnp.array(positions)
    wt_ids, mt_ids = esmqx.tokenise(["".join(wt_aa), "".join(mt_aa)])[:, 1:-1]
    wt_score = log_prob[ids, wt_ids].sum()
    mt_score = log_prob[ids, mt_ids].sum()
    return mt_score - wt_score


def mutate_randomly(
    wt: str,
    num_mutations: int = 1,
    seed: int = None,
) -> tuple[str, list]:
    assert num_mutations > 0, "Number of mutations must be greater than 0."
    """
    Generates a randomly mutated protein sequence.

    Args:
        original_sequence (str): The wild-type protein sequence.
        num_mutations (int): The exact number of mutations to introduce.
                             Ignored if mutation_rate is provided.
                               If provided, num_mutations is calculated from this.
        seed (int): Seed for random number generation for reproducibility.

    Returns:
        tuple: A tuple containing:
            - str: The generated mutant sequence.
            - list: A list of dictionaries, where each dictionary describes a mutation:
                    {"position": int, "original_aa": str, "mutated_aa": str}
    """
    if seed is not None:
        random.seed(seed)

    sequence_length = len(wt)

    if num_mutations > sequence_length:
        raise ValueError(
            f"Cannot introduce {num_mutations} mutations in a sequence of length {sequence_length}."
        )

    mutation_positions = random.sample(range(sequence_length), num_mutations)
    mutation_positions.sort()
    new_amino_acids = random.choices(AA, k=num_mutations)  # with replacement
    # Ensure new amino acids are different from original at those positions
    for pos in mutation_positions:
        original_aa = wt[pos]
        while True:
            new_aa = random.choice(AA)
            if new_aa != original_aa:
                break
        new_amino_acids[mutation_positions.index(pos)] = new_aa

    mt = mutate_str(
        wt,
        mutation_positions,
        new_amino_acids,
    )
    positions, wt_aa, mt_aa = find_mutation_information_str(wt, mt)

    return mt, {"positions": positions, "wt_aa": wt_aa, "mt_aa": mt_aa}


def format_mutations_for_multidms(mutation_info: dict) -> str:
    """Converts a list of detailed mutation dicts to a comma-separated string for multidms."""
    formatted_muts = []

    for _p, _wt, _mt in zip(
        mutation_info["positions"], mutation_info["wt_aa"], mutation_info["mt_aa"]
    ):
        # 1-indexed position
        formatted_muts.append(f"{_wt}{_p + 1}{_mt}")
    return " ".join(sorted(formatted_muts))


def run_monte_carlo_sampling_and_scoring(
    model: esmqx.ESM2,
    wt: str,
    num_samples: int,
    num_mutations: int,
    simulation_seed: int,
) -> pd.DataFrame:
    print(f"\n--- Starting Monte Carlo Simulation ({num_samples} samples) ---")
    all_sampled_data = []

    for it in trange(num_samples):
        mt, mutations_list_detailed = mutate_randomly(
            wt, num_mutations=num_mutations, seed=simulation_seed + it
        )
        esm_score = compute_esm_score_str(model, wt, mt)

        all_sampled_data.append(
            {
                "condition": "rMP12",  # Example condition for mutants
                "aa_substitutions": format_mutations_for_multidms(
                    mutations_list_detailed
                ),
                "func_score": esm_score,
            }
        )

    print(
        f"\nFinished processing {len(all_sampled_data) - 1} mutant samples (plus WT)."
    )
    return pd.DataFrame(all_sampled_data)


if __name__ == "__main__":
    from constants import SEQUENCE_WT, SEQUENCE_MT
    from utils import load_model

    # common routine
    torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    model = load_model(model_params=650)

    # comparing the mutant and wild-type sequences
    def comparing_wt_and_mt(model, wt: str, mt: str):
        wt_ids, mt_ids = esmqx.tokenise([wt, mt])
        esm_score = compute_esm_score(model, wt_ids, mt_ids)
        esm_score_str = compute_esm_score_str(model, wt, mt)
        assert jnp.allclose(esm_score, esm_score_str), (
            "Two methods should give the same result!"
        )

    comparing_wt_and_mt(model, SEQUENCE_WT, SEQUENCE_MT)

    # reconstruction capability
    def reconstruction(model: esmqx.ESM2, sequence: str):
        protein_ids = esmqx.tokenise([sequence])
        logits = jax.vmap(model)(protein_ids).logits.squeeze(0)[1:-1]
        output_ids = jnp.argmax(logits, axis=-1)
        input_ids = protein_ids[0][1:-1]
        print("Accuracy:", (output_ids == input_ids).mean())

    reconstruction(model, SEQUENCE_WT)
    reconstruction(model, SEQUENCE_MT)

    # random mutation
    _, info = mutate_randomly(SEQUENCE_WT, num_mutations=5)
    assert len(info["positions"]) == 5, "Number of mutations should be 5"
    print("mt_aa:", info["mt_aa"])

    sampled_df = run_monte_carlo_sampling_and_scoring(
        model,
        SEQUENCE_WT,
        num_samples=1500000,
        num_mutations=5,
        simulation_seed=42,
    )
    sampled_df.to_csv("data/sampled_mutant_data.csv", index=False)
