# load fitted multidms model and collect model parameters
import pickle
import multidms

from mutate_score import (
    load_esm_model,
    calculate_esm_llr_for_variant,
    mutate_sequence,
    WT_SEQUENCE,
)

model_path = "data/full_models.pkl"

model: multidms.Model = pickle.load(open(model_path, "rb"))["model"][0]


def find_optimal_mutant(model: multidms.Model, num_mutations: int = 5) -> str:
    """
    Find the optimal mutant based on the model's beta parameters.
    The global epistasis model fits a beta parameter for each single point mutation,
    and models the effect of each mutation on the fitness score,
    by summing the beta parameters of the mutations in the mutant sequence followed by a monotonic increasing function.
    Therefore, the optimal mutant is the one with the highest beta parameters.

    Args:
        model (multidms.Model): The fitted multidms model.
        num_mutations (int): The number of mutations to consider in the optimal mutant.
    Returns:
        str: The optimal mutant sequence.
    """
    mutations_df = model.get_mutations_df()
    mutations_df = mutations_df.sort_values(by="beta", ascending=False)
    print("Top mutations by beta parameter:")
    print(mutations_df.head(num_mutations))

    positions = []
    new_amino_acids = []
    row_index = 0
    # mutation sites should be unique.
    while len(positions) < num_mutations:
        row = mutations_df.iloc[row_index]
        if row["sites"] not in positions:
            positions.append(row["sites"])
            new_amino_acids.append(row["muts"])
        row_index += 1

    # mutate sequence
    optimal_sequence, mutations_list_detailed = mutate_sequence(
        WT_SEQUENCE, positions, new_amino_acids
    )
    model = load_esm_model()
    # calculate the log likelihood ratio for the optimal mutant
    final_score = calculate_esm_llr_for_variant(
        model, optimal_sequence, mutations_list_detailed
    )
    print(f"Final score for the optimal mutant: {final_score}")


if __name__ == "__main__":
    find_optimal_mutant(model, num_mutations=5)
