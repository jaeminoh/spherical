from spherical.global_epistasis_model import fit_multidms_model
import fire
import pandas as pd


def main(
    fit_maxiter: int = 5000,
    csv_path: str = "data/sampled_mutant_data.csv",
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
        fit_maxiter=fit_maxiter,
    )
    return fitted_model


if __name__ == "__main__":
    fire.Fire(main)
