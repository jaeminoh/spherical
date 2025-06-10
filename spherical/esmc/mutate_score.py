import random
import torch
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import fire  # For command-line interface

# --- Global Constants (can be configured in main or passed as args) ---
WT_SEQUENCE = "MYVLLTILISVLVCEAVIRVSLSSTREETCFGDSTNPEMIEGAWDSLREEEMPEELSCSISGIREVKTSSQELYRALKAIIAADGLNNITCHGKDPEDKISLIKGPPHKKRVGIVRCERRRDAKQIGRETMAGIAMTVLPALAVFALAPVVFAEDPHLRNRPGKGHNYIDGMTQEDATCKPVTYAGACSSFDVLLEKGKFPLFQSYAHHRTLLEAVHDTIIAKADPPSCDLQSAHGNPCMKEKLVMKTHCPNDYQSAHYLNNDGKMASVKCPPKYELTEDCNFCRQMTGASLKKGSYPLQDLFCQSSEDDGSKLKTKMKGVCEVGVQALKKCDGQLSTAHEVVPFAVFKNSKKVYLDKLDLKTEENLLPDSFVCFEHKGQYKGTMDSGQTKRELKSFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHANGSGIVQIQVSGVWKKPLCVGYERVVVKRELSAKPIQRVEPCTTCITKCEPHGLVVRSTGFKISSAVACASGVCVTGSQSPSTEITLKYPGISQSSGGDIGVHMAHDDQSVSSKIVAHCPPQDPCLVHGCIVCAHGLINYQCHTALSAFVVVFVFSSIAIICLAVLYRVLKCLKIAPRKVLNPLMWITAFIRWIYKKMVARVADNINQVNREIGWMEGGQLVLGNPAPIPRHAPIPRYSTYLMLLLIVSYASACSELIQASSRITTCSTEGVNTKCRLSGTALIRAGSVGAEACLMLKGVKEDQTKFLKIKTVSSELSCREGQSYWTGSFSPKCLSSRRCHLVGECHVNRCLSWRDNETSAEFSFVGESTTMRENKCFEQCGGWGCGCFNVNPSCLFVHTYLQSVRKEALRVFNCIDWVHKLTLEITDFDGSVSTIDLGASSSRFTNWGSVSLSLDAEGISGSNSFSFIESPGKGYAIVDEPFSEIPRQGFLGEIRCNSESSVLSAHESCLRAPNLISYKPMIDQLECTTNLIDPFVVFERGSLPQTRNDKTFAASKGNRGVQAFSKGSVQADLTLMFDNFEVDFVGAAVSCDAAFLNLTGCYSCNAGARVCLSITSTGTGSLSAHNKDGSLHIVLPSENGTKDQCQILHFTVPEVEEEFMYSCDGDERPLLVKGTLIAIDPFDDRREAGGESTVVNPKSGSWNFFDWFSGLMSWFGGPLKTILLICLYVALSIGLFFLLIYLGRTGLSKMWLAATKKAS"
SEQUENCE_LENGTH = len(WT_SEQUENCE)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
MASK_TOKEN_CHAR = "[MASK]"

# CRITICAL: HARDCODED AMINO ACID TO ID MAPPING for ESM
# This is a common mapping for ESM models.
# If ESMC's internal vocabulary order differs, results will be inaccurate.
AA_TO_ID = {
    MASK_TOKEN_CHAR: 4,
    "A": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "K": 13,
    "L": 14,
    "M": 15,
    "N": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "V": 22,
    "W": 23,
    "Y": 24,
}


def find_mutation_information(
    original_sequence: str,
    mutant_sequence: str,
) -> list:
    """
    Finds the positions and amino acids of mutations between the original and mutant sequences.

    Args:
        original_sequence (str): The wild-type protein sequence.
        mutant_sequence (str): The mutated protein sequence.

    Returns:
        list: A list of dictionaries, where each dictionary describes a mutation:
              {"position": int, "original_aa": str, "mutated_aa": str}
    """
    mutations_introduced = []
    for i, (wt_aa, mt_aa) in enumerate(zip(original_sequence, mutant_sequence)):
        if wt_aa != mt_aa:
            mutations_introduced.append(
                {"position": i, "original_aa": wt_aa, "mutated_aa": mt_aa}
            )
    return mutations_introduced


def mutate_sequence(
    original_sequence: str,
    positions: list[int],
    new_amino_acids: list[str],
    return_detailed: bool = True,
) -> tuple[str, list] | str:
    """
    Mutates specific positions in a protein sequence with new amino acids.

    Args:
        original_sequence (str): The wild-type protein sequence.
        positions (list[int]): List of 0-indexed positions to mutate.
        new_amino_acids (list[str]): List of new amino acids to insert at the specified positions.

    Returns:
        str: The mutated protein sequence.
    """
    assert len(positions) == len(new_amino_acids), (
        "positions and new_amino_acids must have the same length."
    )
    mutant_list = list(original_sequence)
    mutations_introduced_detailed = []

    for pos, new_aa in zip(positions, new_amino_acids):
        if pos < 0 or pos >= len(original_sequence):
            raise ValueError(
                f"Position {pos} is out of bounds for sequence length {len(original_sequence)}."
            )
        mutant_list[pos] = new_aa
        mutations_introduced_detailed.append(
            {
                "position": pos,
                "original_aa": original_sequence[pos],
                "mutated_aa": new_aa,
            }
        )
    mutant_sequence = "".join(mutant_list)
    if return_detailed:
        return mutant_sequence, mutations_introduced_detailed
    else:
        return mutant_sequence


def create_random_mutant(
    original_sequence: str,
    num_mutations: int = 1,
    mutation_rate: float = None,
    seed: int = None,
) -> tuple[str, list]:
    """
    Generates a randomly mutated protein sequence.

    Args:
        original_sequence (str): The wild-type protein sequence.
        num_mutations (int): The exact number of mutations to introduce.
                             Ignored if mutation_rate is provided.
        mutation_rate (float): The percentage of positions to mutate (e.g., 0.01 for 1%).
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

    sequence_length = len(original_sequence)

    if mutation_rate is not None:
        num_mutations = max(1, int(sequence_length * mutation_rate))

    if num_mutations > sequence_length:
        raise ValueError(
            f"Cannot introduce {num_mutations} mutations in a sequence of length {sequence_length}."
        )
    if num_mutations == 0:
        return original_sequence, []

    mutation_positions = random.sample(range(sequence_length), num_mutations)
    mutation_positions.sort()
    new_amino_acids = random.choices(AMINO_ACIDS, k=num_mutations)  # with replacement
    # Ensure new amino acids are different from original at those positions
    for pos in mutation_positions:
        original_aa = original_sequence[pos]
        while True:
            new_aa = random.choice(AMINO_ACIDS)
            if new_aa != original_aa:
                break
        new_amino_acids[mutation_positions.index(pos)] = new_aa

    mutant_sequence, mutations_introduced_detailed = mutate_sequence(
        original_sequence,
        mutation_positions,
        new_amino_acids,
        return_detailed=True,
    )
    return mutant_sequence, mutations_introduced_detailed


# Helper to format mutations for multidms (e.g., "A10G,D20F")
def format_mutations_for_multidms(mutations_list_detailed: list) -> str:
    """Converts a list of detailed mutation dicts to a comma-separated string for multidms."""
    if not mutations_list_detailed:
        return ""  # multidms expects an empty string for WT, not "WT"
    formatted_muts = []
    for m in mutations_list_detailed:
        formatted_muts.append(
            f"{m['original_aa']}{m['position'] + 1}{m['mutated_aa']}"
        )  # 1-indexed position
    return " ".join(
        sorted(formatted_muts)
    )  # Use space as separator, not comma for 'M3A K5G' format


def load_esm_model(model_name: str = "esmc_600m") -> ESMC:
    """Loads the specified ESMC model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ESMC model '{model_name}' on {device}...")
    model = ESMC.from_pretrained(model_name).to(device)
    model.eval()  # Set the model to evaluation mode
    alphabet = model.tokenizer.get_vocab()
    print("ESMC model loaded.")
    return model, alphabet


def calculate_esm_llr_for_variant(
    client: ESMC, wt_seq: str, mutations_list_detailed: list
) -> float:
    """
    Calculates the total ESM Log-Likelihood Ratio (LLR) for a given set of mutations
    using simultaneous masking.

    Args:
        client (ESMC): The loaded ESMC model client.
        wt_seq (str): The wild-type protein sequence.
        mutations_list_detailed (list): List of mutation dictionaries.

    Returns:
        float: The total ESM LLR score for the variant.
    """
    if not mutations_list_detailed:
        return 0.0  # WT sequence has 0 LLR change

    # Create ONE masked version of the wild-type sequence with ALL mutations masked
    all_masked_wt_list = list(wt_seq)
    for mut_detail in mutations_list_detailed:
        all_masked_wt_list[mut_detail["position"]] = MASK_TOKEN_CHAR
    all_masked_wt_sequence_str = "".join(all_masked_wt_list)

    # Prepare the protein object for encoding
    all_masked_protein = ESMProtein(sequence=all_masked_wt_sequence_str)

    # Encode and get logits in ONE pass
    try:
        masked_protein_tensor = client.encode(all_masked_protein)
        masked_logits_output = client.logits(
            masked_protein_tensor, LogitsConfig(sequence=True, return_embeddings=False)
        )
        all_logits_tensor = masked_logits_output.logits.sequence.squeeze(0)
    except Exception as e:
        print(f"Error during ESM inference for variant: {e}. Returning NaN.")
        return float("nan")  # Return NaN if inference fails

    current_variant_total_llr = 0.0
    for mut_detail in mutations_list_detailed:
        pos_0_indexed = mut_detail["position"]
        wt_aa = mut_detail["original_aa"]
        mt_aa = mut_detail["mutated_aa"]

        logits_at_current_masked_pos = all_logits_tensor[
            pos_0_indexed + 1
        ]  # +1 for CLS token
        log_probs_at_pos = torch.nn.functional.log_softmax(
            logits_at_current_masked_pos, dim=-1
        )

        try:
            wt_aa_id = AA_TO_ID[wt_aa]
            mt_aa_id = AA_TO_ID[mt_aa]
        except KeyError as e:
            print(
                f"Error: Amino acid '{e}' not found in AA_TO_ID for mutation {wt_aa}{pos_0_indexed + 1}{mt_aa}. Skipping LLR contribution."
            )
            continue

        log_prob_wt_pred = log_probs_at_pos[wt_aa_id].item()
        log_prob_mt_pred = log_probs_at_pos[mt_aa_id].item()

        llr_for_mutation = log_prob_mt_pred - log_prob_wt_pred
        current_variant_total_llr += llr_for_mutation

    return current_variant_total_llr


def run_monte_carlo_sampling_and_scoring(
    num_samples: int,
    min_mutations_per_sample: int,
    max_mutations_per_sample: int,
    simulation_seed: int,
    esm_client: ESMC,
    wt_seq: str,
) -> pd.DataFrame:
    """
    Generates random mutants, calculates their ESM scores, and compiles results.

    Args:
        num_samples (int): Number of random mutants to generate.
        min_mutations_per_sample (int): Minimum number of mutations in a sampled variant.
        max_mutations_per_sample (int): Maximum number of mutations in a sampled variant.
        simulation_seed (int): Seed for random number generation.
        esm_client (ESMC): The loaded ESMC model client.
        wt_seq (str): The wild-type protein sequence.

    Returns:
        pd.DataFrame: DataFrame containing mutant sequences, ESM scores, and mutation strings,
                      formatted for multidms.
    """
    print(f"\n--- Starting Monte Carlo Simulation ({num_samples} samples) ---")
    print(
        f"Mutations per sample: {min_mutations_per_sample}-{max_mutations_per_sample}"
    )

    all_sampled_data = []

    # Add the wild-type sequence data point for multidms baseline
    # Renamed columns to match multidms example: 'aa_substitutions' and 'func_score'
    # 'condition' column added
    for i in range(num_samples):
        num_muts_for_this_sample = random.randint(
            min_mutations_per_sample, max_mutations_per_sample
        )

        mutant_sequence_str, mutations_list_detailed = create_random_mutant(
            wt_seq, num_mutations=num_muts_for_this_sample, seed=simulation_seed + i
        )

        if not mutations_list_detailed:
            continue

        total_llr = calculate_esm_llr_for_variant(
            esm_client, wt_seq, mutations_list_detailed
        )

        all_sampled_data.append(
            {
                "condition": "rMP12",  # Example condition for mutants
                "aa_substitutions": format_mutations_for_multidms(
                    mutations_list_detailed
                ),
                "func_score": total_llr,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{num_samples} samples.")

    print(
        f"\nFinished processing {len(all_sampled_data) - 1} mutant samples (plus WT)."
    )
    return pd.DataFrame(all_sampled_data)


def main(
    num_samples: int = 500,
    min_mutations: int = 1,
    max_mutations: int = 3,
    simulation_seed: int = 42,
    output_csv_path: str = "data/sampled_mutant_data.csv",  # Configurable output path
):
    """
    Main function to run the protein mutation simulation and MultiDMS analysis.

    Args:
        reuse (bool): If True, loads data from 'output_csv_path' instead of regenerating.
        num_samples (int): Number of random mutants to generate.
        min_mutations (int): Minimum number of mutations per sampled variant.
        max_mutations (int): Maximum number of mutations per sampled variant.
        simulation_seed (int): Seed for random number generation.
        multidms_model_type (str): Type of MultiDMS model to fit ('additive' or 'epistatic').
                                   (Note: its effect might be limited by multidms API)
        multidms_model_order (int): Order of epistasis for 'epistatic' model (e.g., 2 for pairwise).
                                     (Note: its effect might be limited by multidms API)
        fit_maxiter (int): Maximum iterations for the MultiDMS model fitting process.
        output_csv_path (str): Path to save/load the sampled data as a CSV file.
        output_parquet_path (str): Path to save the sampled data as a Parquet file (requires pyarrow).
    """
    print("\n--- Generating new data and running ESM scoring ---")
    # 1. Load ESM Model
    esm_client = load_esm_model()

    # 2. Run Monte Carlo Sampling and ESM Scoring
    sampled_df = run_monte_carlo_sampling_and_scoring(
        num_samples,
        min_mutations,
        max_mutations,
        simulation_seed,
        esm_client,
        WT_SEQUENCE,
    )

    # Saving sampled_df
    print(f"\n--- Saving sampled data to {output_csv_path} ---")
    sampled_df.to_csv(output_csv_path, index=False)
    print("Sampled data saved successfully as CSV.")


if __name__ == "__main__":
    fire.Fire(main)
