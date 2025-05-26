import pickle
import multidms  # Specific import for Data class
import pandas as pd

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
        reference="Simulated_Mutant",  # Set the wild-type condition as per the DataFrame
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
    return models
