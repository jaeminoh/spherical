import os
# Assuming data_summary module and load_SAbDab function exist and work
from data_summary import load_SAbDab
import concurrent.futures
import subprocess
import time # Optional: to time the execution

# --- Configuration ---
SCORING_SCRIPT = "./scoring.sh" # Make sure this is the correct path to your script
MAX_WORKERS = 200 # Set the number of concurrent processes (adjust based on your CPU cores and system load)
# Ensure ROSETTA_BIN and ROSETTA_TOOLS are set as environment variables
# in the shell where you run this Python script, or are hardcoded/set
# within the scoring.sh script itself.

# --- Function to run the scoring script for a single item ---
def run_scoring_script(pdb_id, antigen_chain, h_chain, l_chain):
    """
    Constructs the command and runs the scoring.sh script for a single PDB entry.
    Uses subprocess.run for better control over output and return codes.
    Returns a tuple: (pdb_id, return_code, stdout_str, stderr_str)
    """
    # Construct the command list
    # Note: chain IDs should be passed as separate arguments
    command = [
        SCORING_SCRIPT,
        pdb_id,
        antigen_chain,
        h_chain,
        l_chain
    ]

    # Convert arguments to strings just in case they are not (e.g. pandas might return specific types)
    command = [str(arg) for arg in command]

    print(f"Submitting job for {pdb_id}...") # Indicate job submission

    try:
        # Execute the command
        # capture_output=True captures stdout and stderr
        # text=True decodes stdout/stderr as text
        # check=False prevents raising an exception for non-zero exit codes (we check result.returncode instead)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ # Pass the current environment variables (including ROSETTA paths)
        )

        # Print completion status
        if result.returncode == 0:
            print(f"Job for {pdb_id} completed successfully.")
        else:
            print(f"Job for {pdb_id} finished with error code {result.returncode}.")
            print(f"--- Stderr for {pdb_id} ---")
            print(result.stderr)
            print(f"--- Stdout for {pdb_id} ---")
            print(result.stdout)

        return (pdb_id, result.returncode, result.stdout, result.stderr)

    except FileNotFoundError:
        print(f"Error: Scoring script '{SCORING_SCRIPT}' not found.")
        return (pdb_id, -2, "", f"Scoring script '{SCORING_SCRIPT}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred while running job for {pdb_id}: {e}")
        return (pdb_id, -1, "", str(e))


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time() # Start timing

    d = load_SAbDab()

    # Check if DataFrame is empty
    if d.empty:
        print("No data loaded from load_SAbDab(). Exiting.")
        exit(0)

    print(f"Loaded {len(d)} entries. Starting parallel scoring with {MAX_WORKERS} workers.")

    # Use ProcessPoolExecutor to manage a pool of worker processes
    # Each worker process can run one instance of the scoring script
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        # Submit the scoring tasks to the executor
        # We store a mapping from the Future object back to the PDB ID
        # so we know which job finished when we get the result
        future_to_pdb = {
            executor.submit(
                run_scoring_script,
                row["pdb"],
                row["antigen_chain"],
                row["Hchain"],
                row["Lchain"]
            ): row["pdb"]
            for index, row in d.iterrows()
        }

        # Process the results as they complete (they might finish in any order)
        results = []
        for future in concurrent.futures.as_completed(future_to_pdb):
            pdb_id = future_to_pdb[future]
            try:
                # Get the result from the completed future
                # If an exception occurred within run_scoring_script, .result() will re-raise it
                result = future.result()
                results.append(result)
                # You can process the result here if needed (e.g., log status, check return code)
                # run_scoring_script already prints basic status

            except Exception as exc:
                # This catches exceptions raised by the run_scoring_script function itself
                print(f'Job for {pdb_id} generated an exception during execution: {exc}')
                results.append((pdb_id, -3, "", f"Python exception: {exc}")) # Append an error result

    end_time = time.time() # End timing
    duration = end_time - start_time

    print("\n--- All Jobs Finished ---")
    print(f"Total time elapsed: {duration:.2f} seconds")

    # Optional: You can iterate through 'results' list here to summarize outcomes
    successful_count = sum(1 for res in results if res[1] == 0)
    error_count = len(results) - successful_count
    print(f"Successful jobs: {successful_count}")
    print(f"Jobs with errors: {error_count}")
    # Note: A non-zero return code from scoring.sh doesn't necessarily mean
    # Rosetta failed completely, check the log files produced by scoring.sh.
    # The stderr/stdout captured here can also help diagnose issues.