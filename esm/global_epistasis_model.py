# TODO it does not work!!!!!


import multidms.data  # Specific import for Data class
import multidms.model  # Specific import for Model class
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

    # --- DIAGNOSTICS FOR WT_SEQUENCE AND REFERENCE ROW ---
    print(f"DEBUG: WT_SEQUENCE length: {len(wt_seq)}")
    print(f"DEBUG: First 10 chars of WT_SEQUENCE: {wt_seq[:10]}")
    print(f"DEBUG: Last 10 chars of WT_SEQUENCE: {wt_seq[-10:]}")

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
    dms_dataset = multidms.data.Data(
        df,  # Pass the DataFrame directly
        reference="WT_condition",  # Set the wild-type condition as per the DataFrame
        alphabet=multidms.data.AAS,  # Specify amino acid alphabet
        verbose=True,
        nb_workers=None,  # Set number of workers for data processing
    )
    print(f"MultiDMS dataset created with {len(dms_dataset.variants_df)} variants.")

    print(
        f"\n--- Fitting MultiDMS {model_type} model (order={model_order}, maxiter={fit_maxiter}) ---"
    )

    # Initializing multidms.model.Model directly with model_type and order.
    model = multidms.model.Model(
        dms_dataset,
        conditional_shifts=False,
    )

    # Fit the model directly.
    model.fit(maxiter=fit_maxiter)
    print("MultiDMS model fitted successfully.")
    return model


# --- 5. Coefficient Analysis Function ---


def analyze_coefficients(model: multidms.model.Model) -> None:
    """
    Extracts, filters, and displays coefficients from the fitted MultiDMS model.

    Args:
        model: The fitted MultiDMS model object.
    """
    print("\n--- Collecting and displaying model coefficients ---")
    # MultiDMS stores parameters in model.param_df
    coefficients = model.param_df

    if "pvalue" in coefficients.columns:
        significant_coefficients = coefficients[
            coefficients["pvalue"].notna() & (coefficients["pvalue"] < 0.05)
        ].copy()
        print(
            f"Found {len(significant_coefficients)} significant coefficients (p < 0.05)."
        )
    else:
        significant_coefficients = coefficients.copy()
        print(
            "No 'pvalue' column found in coefficients. Displaying all non-zero coefficients."
        )

    # MultiDMS coefficients are often already named like 'beta', 'alpha', etc.
    # The example does not explicitly show 'coefficient' column.
    # Assuming 'coefficient' or similar is the main value.
    if "coefficient" in significant_coefficients.columns:
        significant_coefficients["abs_coefficient"] = significant_coefficients[
            "coefficient"
        ].abs()
        significant_coefficients = significant_coefficients.sort_values(
            by="abs_coefficient", ascending=False
        )
    elif (
        "beta" in significant_coefficients.columns
    ):  # MultiDMS might use 'beta' for coefficients
        significant_coefficients["abs_coefficient"] = significant_coefficients[
            "beta"
        ].abs()
        significant_coefficients = significant_coefficients.sort_values(
            by="abs_coefficient", ascending=False
        )
    else:
        print(
            "Warning: Neither 'coefficient' nor 'beta' column found for sorting. Displaying as-is."
        )
        # Attempt to sort by the first numeric column if no standard coefficient column
        numeric_cols = significant_coefficients.select_dtypes(
            include=["number"]
        ).columns
        if len(numeric_cols) > 0:
            significant_coefficients = significant_coefficients.sort_values(
                by=numeric_cols[0].iloc[0], ascending=False
            )

    print("\nTop 20 Coefficients (most impactful):")
    # MultiDMS's 'param_df' columns vary based on model.
    # Typical columns might be 'beta', 'mutation', 'site', 'type', 'pvalue'
    preferred_display_cols = ["beta", "mutation", "site", "type", "pvalue"]

    # Filter to only include columns that actually exist in the DataFrame
    final_display_cols = [
        col for col in preferred_display_cols if col in significant_coefficients.columns
    ]

    print(significant_coefficients.head(20)[final_display_cols])

    # Check if 'type' column exists before filtering
    if "type" in coefficients.columns:
        additive_coefficients = coefficients[coefficients["type"] == "additive"]
        print("\nTop 10 Additive Coefficients (Single Mutations):")
        if not additive_coefficients.empty:
            print(
                additive_coefficients.sort_values(by="beta", ascending=False).head(10)[
                    final_display_cols
                ]
            )
        else:
            print("No additive coefficients found.")

        pairwise_coefficients = coefficients[coefficients["type"] == "epistatic"]
        print("\nTop 10 Pairwise Coefficients (Epistatic Interactions):")
        if not pairwise_coefficients.empty:
            # MultiDMS might have 'epistatic' type but need 'mut1'/'mut2' columns
            # if 'beta' exists, sort by beta
            print(
                pairwise_coefficients.sort_values(by="beta", ascending=False).head(10)[
                    final_display_cols
                ]
            )
        else:
            print("No epistatic coefficients found.")
    else:
        print(
            "\n'type' column not found in coefficients; cannot separate additive/pairwise effects."
        )


# --- Main Execution Block ---


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

    # 4. Analyze Coefficients
    analyze_coefficients(fitted_model)


if __name__ == "__main__":
    fire.Fire(main)
