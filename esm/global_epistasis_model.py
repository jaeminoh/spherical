# TODO it does not work!!!!!

import pickle
import multidms  # Specific import for Data class
import pandas as pd
import fire  # For command-line interface

# --- Global Constants (can be configured in main or passed as args) ---
WT_SEQUENCE = "MYVLLTILISVLVCEAVIRVSLSSTREETCFGDSTNPEMIEGAWDSLREEEMPEELSCSISGIREVKTSSQELYRALKAIIAADGLNNITCHGKDPEDKISLIKGPPHKKRVGIVRCERRRDAKQIGRETMAGIAMTVLPALAVFALAPVVFAEDPHLRNRPGKGHNYIDGMTQEDATCKPVTYAGACSSFDVLLEKGKFPLFQSYAHHRTLLEAVHDTIIAKADPPSCDLQSAHGNPCMKEKLVMKTHCPNDYQSAHYLNNDGKMASVKCPPKYELTEDCNFCRQMTGASLKKGSYPLQDLFCQSSEDDGSKLKTKMKGVCEVGVQALKKCDGQLSTAHEVVPFAVFKNSKKVYLDKLDLKTEENLLPDSFVCFEHKGQYKGTMDSGQTKRELKSFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHANGSGIVQIQVSGVWKKPLCVGYERVVVKRELSAKPIQRVEPCTTCITKCEPHGLVVRSTGFKISSAVACASGVCVTGSQSPSTEITLKYPGISQSSGGDIGVHMAHDDQSVSSKIVAHCPPQDPCLVHGCIVCAHGLINYQCHTALSAFVVVFVFSSIAIICLAVLYRVLKCLKIAPRKVLNPLMWITAFIRWIYKKMVARVADNINQVNREIGWMEGGQLVLGNPAPIPRHAPIPRYSTYLMLLLIVSYASACSELIQASSRITTCSTEGVNTKCRLSGTALIRAGSVGAEACLMLKGVKEDQTKFLWIKTVSSELSCREGQSYWTGSFSPKCLSSRRCHLVGECHVNRCLSWRDNETSAEFSFVGESTTMRENKCFEQCGGWGCGCFNVNPSCLFVHTYLQSVRKEALRVFNCIDWVHKLTLEITDFDGSVSTIDLGASSSRFTNWGSVSLSLDAEGISGSNSFSFIESPGKGYAIVDEPFSEIPRQGFLGEIRCNSESSVLSAHESCLRAPNLISYKPMIDQLECTTNLIDPFVVFERGSLPQTRNDKTFAASKGNRGVQAFSKGSVQADLTLMFDNFEVDFVGAAVSCDAAFLNLTGCYSCNAGARVCLSITSTGTGSLSAHNKDGSLHIVLPSENGTKDQCQILHFTVPEVEEEFMYSCDGDERPLLVKGTLIAIDPFDDRREAGGESTVVNPKSGSWNFFDWFSGLMSWFGGPLKTILLICLYVALSIGLFFLLIYLGRTGLSKMWLAATKKAS"
SEQUENCE_LENGTH = len(WT_SEQUENCE)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids


# --- 4. MultiDMS Model Fitting Function ---


def fit_multidms_model(
    df: pd.DataFrame,
    wt_seq: str,  # This wt_seq is still passed, but used for internal verification below.
    model_type: str = "epistatic",
    model_order: int = 2,
    fit_maxiter: int = 5000,
) -> multidms.model.Model:
    """
    Prepares data for MultiDMS and fits a specified model.

    Args:
        df (pd.DataFrame): DataFrame containing sampled data, formatted with
                           'condition', 'aa_substitutions', 'func_score'.
        wt_seq (str): The wild-type protein sequence (used for verification purposes).
        model_type (str): Type of model to fit ('additive' or 'epistatic').
        model_order (int): The order of epistasis for the 'epistatic' model.
        fit_maxiter (int): Maximum iterations for the fitting process.

    Returns:
        multidms.model.Model: The fitted MultiDMS model object.
    """
    print("\n--- Preparing data for MultiDMS ---")

    # Ensure correct column types as multidms expects
    df["condition"] = df["condition"].astype(str)
    df["aa_substitutions"] = df["aa_substitutions"].astype(str)
    df["func_score"] = df["func_score"].astype(float)

    wt_row = df[df["condition"] == "WT_condition"]
    if not wt_row.empty:
        print(
            f"DEBUG: 'WT_condition' row found. 'aa_substitutions': '{wt_row['aa_substitutions'].iloc[0]}'"
        )
        if wt_row["aa_substitutions"].iloc[0] != "":
            print(
                "WARNING: 'WT_condition' row's 'aa_substitutions' is NOT an empty string. This is usually required!"
            )
    else:
        print(
            "WARNING: No 'WT_condition' row found in the DataFrame. MultiDMS requires a reference."
        )
    # --- END DIAGNOSTICS ---

    # The library is expected to infer it from the 'reference' condition and mutations.
    dms_dataset = multidms.Data(
        df,  # Pass the DataFrame directly
        reference="Simulated_Mutant",  # Set the wild-type condition as per the DataFrame
        alphabet=multidms.AAS,  # Specify amino acid alphabet
        verbose=True,
        nb_workers=None,  # Set number of workers for data processing
    )
    print(f"MultiDMS dataset created with {len(dms_dataset.variants_df)} variants.")

    print(
        f"\n--- Fitting MultiDMS {model_type} model (order={model_order}, maxiter={fit_maxiter}) ---"
    )

    # define the fitting parameters
    scale_coeff_lasso_shift = [
        0.0,
        5.00e-6,
        1.00e-05,
        2.00e-05,
        4.00e-05,
        8.00e-05,
        1.60e-04,
        3.20e-04,
        6.40e-04,
    ]  # the sweep of lasso coefficient params
    alpha_d = True
    scale_coeff_ridge_alpha_d = 1e-3
    num_training_steps = 30
    iterations_per_step = 1000  # default 20000
    # init_beta_naught = 5.0 # We've found that we need to start with a higher beta_naught to get the model to converge correctly,
    scale_coeff_ridge_beta = 0.0  # the sweep of ridge coefficient params
    # train_frac = 0.8  # fraction of data to use for cross validation training.
    # lasso_choice = 4.00e-05  # the lasso coefficient to use for the final model

    fit_params = {
        "scale_coeff_lasso_shift": scale_coeff_lasso_shift,
        "alpha_d": [alpha_d],
        "scale_coeff_ridge_alpha_d": [scale_coeff_ridge_alpha_d],
        "scale_coeff_ridge_beta": [scale_coeff_ridge_beta],
        "num_training_steps": [num_training_steps],
        "iterations_per_step": [iterations_per_step],
    }

    fit_params["dataset"] = [dms_dataset]
    _, _, models = multidms.fit_models(fit_params, n_threads=-1)
    pickle.dump(models, open("full_models.pkl", "wb"))
    return models


def main(
    multidms_model_type: str = "epistatic",
    multidms_model_order: int = 2,
    fit_maxiter: int = 5000,
    csv_path: str = "sampled_mutant_data.csv",
    parquet_path: str = "sampled_mutant_data.parquet",  # Not used in this specific script but kept for consistency
) -> None:
    """
    Main function for the MultiDMS fitting and analysis part of the workflow.

    Args:
        multidms_model_type (str): Type of MultiDMS model to fit ('additive' or 'epistatic').
                                   Passed to the Model constructor.
        multidms_model_order (int): Order of epistasis for 'epistatic' model (e.g., 2 for pairwise).
                                     Passed to the Model constructor.
        fit_maxiter (int): Maximum iterations for the MultiDMS model fitting process.
        csv_path (str): Path to the CSV file containing the sampled data.
        parquet_path (str): Path to the Parquet file (not used here, but kept for consistency).
    """
    print(f"\n--- Load data from {csv_path} ---")

    # Load data directly. If file not found or other issues, it will raise an error.
    sampled_df = pd.read_csv(csv_path)

    # --- CRITICAL FIX: Handle NaN in 'aa_substitutions' column ---
    # Fill any NaN values with an empty string before type conversion
    sampled_df["aa_substitutions"] = sampled_df["aa_substitutions"].fillna("")
    # --- End CRITICAL FIX ---

    # Ensure loaded DataFrame has correct types for multidms
    sampled_df["condition"] = sampled_df["condition"].astype(str)
    sampled_df["aa_substitutions"] = sampled_df["aa_substitutions"].astype(
        str
    )  # Re-convert after fillna
    sampled_df["func_score"] = sampled_df["func_score"].astype(float)
    print("Data loaded successfully from CSV.")

    # 3. Fit MultiDMS Model
    fitted_model = fit_multidms_model(
        sampled_df,
        WT_SEQUENCE,  # WT_SEQUENCE is still needed here for context and passed to fit_multidms_model
        model_type=multidms_model_type,
        model_order=multidms_model_order,
        fit_maxiter=fit_maxiter,
    )
    return fitted_model


if __name__ == "__main__":
    fire.Fire(main)
