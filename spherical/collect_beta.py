# load fitted multidms model and collect model parameters
import pickle
import multidms
from utils import mutate_str, compute_esm_score_str, load_model
from constants import SEQUENCE_WT

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
            positions.append(row["sites"] - 1)  # 1-indexed to 0-indexed
            new_amino_acids.append(row["muts"])
        row_index += 1

    # mutate sequence

    optimal_sequence = mutate_str(SEQUENCE_WT, positions, new_amino_acids)
    model = load_model(model_params=650)

    # calculate the log likelihood ratio for the optimal mutant
    final_score = compute_esm_score_str(model, SEQUENCE_WT, optimal_sequence)
    print(f"Final score for the optimal mutant: {final_score}")


if __name__ == "__main__":
    find_optimal_mutant(model, num_mutations=5)
