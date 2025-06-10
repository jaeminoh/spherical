# introduce single point mutations sequentially to a protein sequence
import numpy as np
import esm2quinox as esmqx
from tqdm import trange

from constants import AA, AA_to_ID, ID_to_AA
from utils import mutate_str, compute_esm_score_str


def scan_single_mutations(model: esmqx.ESM2, sequence: str):
    result = np.full((len(sequence), 33), -np.inf)  # total 33 tokens

    for it in trange(len(sequence)):
        # TODO vmap this
        for aa in AA:
            if sequence[it] != aa:
                new_sequence = mutate_str(sequence, [it], [aa])
                # TODO jit this
                score = compute_esm_score_str(model, sequence, new_sequence)
                result[it, AA_to_ID[aa]] = score

    return result


def recombination(
    model: esmqx.ESM2, sequence: str, max_mutations=5, save_path: str = None
) -> tuple[list[int], list[str]]:
    scan_result = scan_single_mutations(model, sequence)
    if save_path is not None:
        np.save(f"{save_path}.npy", scan_result)

    best_id_each_position = np.argmax(scan_result, axis=1)
    best_score_each_position = np.max(scan_result, axis=1)
    assert np.allclose(
        best_score_each_position,
        scan_result[np.arange(len(sequence)), best_id_each_position],
    )

    top_k_position = np.argsort(best_score_each_position)[-max_mutations:]
    top_k_id = best_id_each_position[top_k_position]

    mt_aa = [ID_to_AA[i] for i in top_k_id]
    return top_k_position, mt_aa


def forward_stepwise(model: esmqx.ESM2, sequence: str, max_mutations=5):
    """
    Perform forward stepwise mutation on a protein sequence.

    Args:
        sequence (str): The original protein sequence.
        model: The ESM model for scoring mutations.
        max_mutations (int): Maximum number of mutations to introduce.

    Returns:
        list: A list of mutated sequences.
    """
    for i in range(max_mutations):  # noqa
        print(f"Directed evolution, step {i + 1} of {max_mutations}")
        position, mt_aa = recombination(model, sequence, 1)
        mt = mutate_str(sequence, position, mt_aa)
        score = compute_esm_score_str(model, sequence, mt)
        print(f"position: {position}, substitution: {mt_aa}, score: {score}")
        sequence = mt
    return sequence


if __name__ == "__main__":
    from utils import find_mutation_information_str, load_model
    from constants import SEQUENCE_WT

    model = load_model(model_params=650)

    positions, mt_aa = recombination(
        model, SEQUENCE_WT, max_mutations=5, save_path="data/recombination_result"
    )
    print(f"Best mutations at positions {positions} with amino acids {mt_aa}")
    mt_recombination = mutate_str(SEQUENCE_WT, positions, mt_aa)
    score_recombination = compute_esm_score_str(model, SEQUENCE_WT, mt_recombination)
    print(f"score of recombination: {score_recombination}")

    final_sequence = forward_stepwise(model, SEQUENCE_WT)
    final_score = compute_esm_score_str(model, SEQUENCE_WT, final_sequence)
    print(f"score of directed evolution: {final_score}")
    print(
        f"Best sequence after forward stepwise mutation: {find_mutation_information_str(SEQUENCE_WT, final_sequence)}"
    )
