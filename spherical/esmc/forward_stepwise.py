# introduce single point mutations sequentially to a protein sequence
from tqdm import trange
from mutate_score import mutate_sequence, calculate_esm_llr_for_variant, AMINO_ACIDS


def forward_stepwise(sequence, model, max_mutations=5):
    """
    Perform forward stepwise mutation on a protein sequence.

    Args:
        sequence (str): The original protein sequence.
        model: The ESM model for scoring mutations.
        max_mutations (int): Maximum number of mutations to introduce.

    Returns:
        list: A list of mutated sequences.
    """
    for _ in (pbar := trange(max_mutations)):  # noqa
        best_score = float("-inf")

        for i in range(len(sequence)):
            for aa in AMINO_ACIDS:
                if sequence[i] != aa:
                    new_sequence, mutation_list_detailed = mutate_sequence(
                        sequence, [i], [aa]
                    )
                    score = calculate_esm_llr_for_variant(
                        model, new_sequence, mutation_list_detailed
                    )

                    if score > best_score:
                        best_score = score
                        best_sequence = new_sequence
                        print(
                            f"New best! Score: {best_score}, Mutation: {aa} at position {i}"
                        )

        if best_sequence is None or best_score <= 0:
            break
        sequence = best_sequence

    return best_sequence, best_score


if __name__ == "__main__":
    from mutate_score import load_esm_model, WT_SEQUENCE, find_mutation_information

    model = load_esm_model()
    final_sequence, final_score = forward_stepwise(WT_SEQUENCE, model)
    print(
        f"Best sequence after forward stepwise mutation: {find_mutation_information(WT_SEQUENCE, final_sequence)}"
    )
    print(f"Final score: {final_score}")
