import pickle
import multidms  # Specific import for Data class
import pandas as pd
import fire
from scipy.stats import pearsonr

# --- Global Constants (can be configured in main or passed as args) ---
WT_SEQUENCE = "MYVLLTILISVLVCEAVIRVSLSSTREETCFGDSTNPEMIEGAWDSLREEEMPEELSCSISGIREVKTSSQELYRALKAIIAADGLNNITCHGKDPEDKISLIKGPPHKKRVGIVRCERRRDAKQIGRETMAGIAMTVLPALAVFALAPVVFAEDPHLRNRPGKGHNYIDGMTQEDATCKPVTYAGACSSFDVLLEKGKFPLFQSYAHHRTLLEAVHDTIIAKADPPSCDLQSAHGNPCMKEKLVMKTHCPNDYQSAHYLNNDGKMASVKCPPKYELTEDCNFCRQMTGASLKKGSYPLQDLFCQSSEDDGSKLKTKMKGVCEVGVQALKKCDGQLSTAHEVVPFAVFKNSKKVYLDKLDLKTEENLLPDSFVCFEHKGQYKGTMDSGQTKRELKSFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHANGSGIVQIQVSGVWKKPLCVGYERVVVKRELSAKPIQRVEPCTTCITKCEPHGLVVRSTGFKISSAVACASGVCVTGSQSPSTEITLKYPGISQSSGGDIGVHMAHDDQSVSSKIVAHCPPQDPCLVHGCIVCAHGLINYQCHTALSAFVVVFVFSSIAIICLAVLYRVLKCLKIAPRKVLNPLMWITAFIRWIYKKMVARVADNINQVNREIGWMEGGQLVLGNPAPIPRHAPIPRYSTYLMLLLIVSYASACSELIQASSRITTCSTEGVNTKCRLSGTALIRAGSVGAEACLMLKGVKEDQTKFLWIKTVSSELSCREGQSYWTGSFSPKCLSSRRCHLVGECHVNRCLSWRDNETSAEFSFVGESTTMRENKCFEQCGGWGCGCFNVNPSCLFVHTYLQSVRKEALRVFNCIDWVHKLTLEITDFDGSVSTIDLGASSSRFTNWGSVSLSLDAEGISGSNSFSFIESPGKGYAIVDEPFSEIPRQGFLGEIRCNSESSVLSAHESCLRAPNLISYKPMIDQLECTTNLIDPFVVFERGSLPQTRNDKTFAASKGNRGVQAFSKGSVQADLTLMFDNFEVDFVGAAVSCDAAFLNLTGCYSCNAGARVCLSITSTGTGSLSAHNKDGSLHIVLPSENGTKDQCQILHFTVPEVEEEFMYSCDGDERPLLVKGTLIAIDPFDDRREAGGESTVVNPKSGSWNFFDWFSGLMSWFGGPLKTILLICLYVALSIGLFFLLIYLGRTGLSKMWLAATKKAS"
SEQUENCE_LENGTH = len(WT_SEQUENCE)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids


# --- 4. MultiDMS Model Fitting Function ---


def fit_multidms_model(
    df: pd.DataFrame,
    fit_maxiter: int = 5000,
) -> multidms.model.Model:
    """
    Prepares data for MultiDMS and fits a specified model.

    Args:
        df (pd.DataFrame): DataFrame containing sampled data, formatted with
                           'condition', 'aa_substitutions', 'func_score'.
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

    # The library is expected to infer it from the 'reference' condition and mutations.
    dms_dataset = multidms.Data(
        df,  # Pass the DataFrame directly
        reference="rMP12",  # Set the wild-type condition as per the DataFrame
        alphabet=multidms.AAS,  # Specify amino acid alphabet
        verbose=True,
        nb_workers=None,  # Set number of workers for data processing
    )
    print(f"MultiDMS dataset created with {len(dms_dataset.variants_df)} variants.")

    print(f"\n--- Fitting MultiDMS. maxiter={fit_maxiter}) ---")

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
    pickle.dump(models, open("data/full_models.pkl", "wb"))
    print("MultiDMS model fitting completed and saved to 'data/full_models.pkl'.")


def main(
    fit_maxiter: int = 5000,
    csv_path: str = "data/sampled_mutant_data.csv",
    model_path: str = "data/full_models.pkl",
    mode: str = "test",
    max_mutations: int = 15,
) -> None:
    """
    Main function for the MultiDMS fitting and analysis part of the workflow.

    Args:
        fit_maxiter (int): Maximum iterations for the MultiDMS model fitting process.
        csv_path (str): Path to the CSV file containing the sampled data.
    """
    sampled_df = pd.read_csv(csv_path)
    if mode == "train":
        print(f"\n--- Load data from {csv_path} ---")
        sampled_df["condition"] = sampled_df["condition"].astype(str)
        sampled_df["aa_substitutions"] = sampled_df["aa_substitutions"].astype(str)
        sampled_df["func_score"] = sampled_df["func_score"].astype(float)
        print("Data loaded successfully from CSV.")

        fit_multidms_model(sampled_df, fit_maxiter=fit_maxiter)

    print("\n--- Testing the fitted model ---")
    test_df = pd.read_csv(f"data/test{max_mutations}.csv")
    print(test_df.head())
    print("predictions:")
    models: list[multidms.Model] = pickle.load(open(model_path, "rb"))

    for model in models["model"]:
        df = model.add_phenotypes_to_df(test_df)
        print(df)

    y_pred = df["predicted_func_score"]
    y_true = test_df["func_score"]
    print(f"Pearson correlation: {pearsonr(y_pred, y_true)}")
    df.iloc[:, :-1].head().to_csv(
        f"data/test{max_mutations}_predictions.csv", index=False
    )


if __name__ == "__main__":
    fire.Fire(main)
