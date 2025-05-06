import numpy as np
import pandas as pd
import sys
import os


def read_rosetta_scorefile(scorefile_path):
    """
    Reads a Rosetta .sc score file into a pandas DataFrame.

    Args:
        scorefile_path (str): The path to the .sc file.

    Returns:
        pandas.DataFrame: A DataFrame containing the score data, or None if an error occurs.
    """
    if not os.path.exists(scorefile_path):
        print(f"Error: Score file not found at '{scorefile_path}'", file=sys.stderr)
        return None

    try:
        header_line_index = None
        with open(scorefile_path, "r") as f:
            for i, line in enumerate(f):
                # Strip leading/trailing whitespace and check if it starts with "SCORE:"
                if line.strip().startswith("SCORE:"):
                    header_line_index = i
                    break

        if header_line_index is None:
            print(
                f"Error: Could not find 'SCORE:' header line in '{scorefile_path}'",
                file=sys.stderr,
            )
            return None

        df = pd.read_csv(
            scorefile_path,
            sep=r"\s+",
            skiprows=header_line_index,
            comment="#",
            header=0,
            engine="python",
        )
        if not df.empty and df.columns[0] == "SCORE:":
            df = df.drop(columns=[df.columns[0]])

        return df

    except Exception as e:
        print(
            f"An error occurred while reading '{scorefile_path}': {e}", file=sys.stderr
        )
        return None


def compute_correlation(dG, dR):
    """
    Computes the correlation between two lists of values.

    Args:
        dG (list): List of delta G values.
        dR (list): List of Rosetta scores.

    Returns:
        float: The correlation coefficient.
    """
    if len(dG) != len(dR):
        print(
            "Error: Lists must be of the same length for correlation calculation.",
            file=sys.stderr,
        )
        return None

    dG_series = pd.Series(dG)
    dR_series = pd.Series(dR)

    return dG_series.corr(dR_series)


def compute_R2(dG, dR):
    """
    Computes the R^2 value between two lists of values.

    Args:
        dG (list): List of delta G values.
        dR (list): List of Rosetta scores.

    Returns:
        float: The R^2 value.
    """
    if len(dG) != len(dR):
        print(
            "Error: Lists must be of the same length for R^2 calculation.",
            file=sys.stderr,
        )
        return None

    from sklearn.linear_model import LinearRegression

    # Reshape for sklearn
    dG = np.array(dG).reshape(-1, 1)
    dR = np.array(dR).reshape(-1, 1)

    model = LinearRegression()
    model.fit(dG, dR)
    r_squared = model.score(dG, dR)
    return r_squared


if __name__ == "__main__":
    from data_summary import load_SAbDab

    d = load_SAbDab()
    # 2uzi contains unknown amino acids GTP
    d = d[d.pdb != "2uzi"]

    dG = []
    dR = []
    pdb_id = []

    for _, _d in d.iterrows():
        dG.append(_d.delta_g)
        dR.append(
            read_rosetta_scorefile(f"SAbDab/{_d.pdb}.sc")["dG_separated"]
        )  # /dSASAx100"])
        # dR.append(read_rosetta_scorefile(f"SAbDab/{_d.pdb}_total.sc").total_score[0])

    print(len(dG), len(dR))
    # print("correlation coefficient: ", compute_correlation(dG, dR))
    print("R^2: ", compute_R2(dG, dR))
