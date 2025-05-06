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
        # Rosetta score files typically have a header line starting with "SCORE:"
        # followed by column names. The data lines also start with "SCORE:".
        # We can use pandas.read_csv, specifying the separator and skipping comments.

        # Read the file, using space as a separator (might need adjustment for tabs)
        # Skip lines starting with '#' which might be comments
        # The 'comment' parameter handles lines starting with #
        # The 'sep' parameter handles the delimiter. Rosetta uses spaces, sometimes multiple.
        # 'skipinitialspace=True' helps with multiple spaces.
        # 'header=0' means the first line after comments is the header.

        # Find the header line (the first line starting with "SCORE:")
        header_line_index = None
        with open(scorefile_path, 'r') as f:
            for i, line in enumerate(f):
                # Strip leading/trailing whitespace and check if it starts with "SCORE:"
                if line.strip().startswith("SCORE:"):
                    header_line_index = i
                    break

        if header_line_index is None:
            print(f"Error: Could not find 'SCORE:' header line in '{scorefile_path}'", file=sys.stderr)
            return None

        # Use pandas to read the file starting from the header line
        # Use a regex separator to handle multiple spaces/tabs
        df = pd.read_csv(
            scorefile_path,
            sep=r'\s+',          # Use one or more whitespace characters as separator
            skiprows=header_line_index, # Start reading from the header line
            comment='#',          # Skip lines starting with '#'
            header=0,             # The first line after skipping is the header
            engine='python'       # 'python' engine is needed for regex separator
        )

        # The first column is usually "SCORE:", which is not needed in the data.
        # The last column is usually "description" or the input file name.
        # We can drop the first column.
        if not df.empty and df.columns[0] == 'SCORE:':
             # If 'description' column exists, it's the last one and contains filenames
             # If not, the last column might be the filename itself depending on Rosetta version/flags
             # Let's assume the last column is the identifier/description and keep it.
             # Drop the first column which is just "SCORE:"
             df = df.drop(columns=[df.columns[0]])


        return df

    except Exception as e:
        print(f"An error occurred while reading '{scorefile_path}': {e}", file=sys.stderr)
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
        print("Error: Lists must be of the same length for correlation calculation.", file=sys.stderr)
        return None

    # Convert to pandas Series for correlation calculation
    dG_series = pd.Series(dG)
    dR_series = pd.Series(dR)

    # Calculate and return the correlation coefficient
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
        print("Error: Lists must be of the same length for R^2 calculation.", file=sys.stderr)
        return None
    
    from sklearn.linear_model import LinearRegression
    import numpy as np

    dG = np.array(dG).reshape(-1, 1)  # Reshape for sklearn
    dR = np.array(dR).reshape(-1, 1)  # Reshape for sklearn
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

    for (_, _d) in d.iterrows():
        dG.append(_d.delta_g)
        #dR.append(read_rosetta_scorefile(f"SAbDab/{_d.pdb}.sc")["dG_separated"])#/dSASAx100"])
        dR.append(read_rosetta_scorefile(f"SAbDab/{_d.pdb}_total.sc").total_score[0])

    print(len(dG), len(dR))
    #print("correlation coefficient: ", compute_correlation(dG, dR))
    print("R^2: ", compute_R2(dG, dR))

