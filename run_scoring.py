import os

import concurrent.futures
import subprocess
import time

from data_summary import load_SAbDab

SCORING_SCRIPT = "./scoring.sh"
MAX_WORKERS = 200


def run_scoring_script(pdb_id, antigen_chain, h_chain, l_chain):
    """
    Constructs the command and runs the scoring.sh script for a single PDB entry.
    Uses subprocess.run for better control over output and return codes.
    Returns a tuple: (pdb_id, return_code, stdout_str, stderr_str)
    """
    command = [SCORING_SCRIPT, pdb_id, antigen_chain, h_chain, l_chain]

    command = [str(arg) for arg in command]

    print(f"Submitting job for {pdb_id}...")

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, env=os.environ
        )

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


if __name__ == "__main__":
    start_time = time.time()

    d = load_SAbDab()

    if d.empty:
        print("No data loaded from load_SAbDab(). Exiting.")
        exit(0)

    print(
        f"Loaded {len(d)} entries. Starting parallel scoring with {MAX_WORKERS} workers."
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdb = {
            executor.submit(
                run_scoring_script,
                row["pdb"],
                row["antigen_chain"],
                row["Hchain"],
                row["Lchain"],
            ): row["pdb"]
            for index, row in d.iterrows()
        }

        results = []
        for future in concurrent.futures.as_completed(future_to_pdb):
            pdb_id = future_to_pdb[future]
            try:
                result = future.result()
                results.append(result)

            except Exception as exc:
                print(
                    f"Job for {pdb_id} generated an exception during execution: {exc}"
                )
                results.append((pdb_id, -3, "", f"Python exception: {exc}"))

    end_time = time.time()
    duration = end_time - start_time

    print("\n--- All Jobs Finished ---")
    print(f"Total time elapsed: {duration:.2f} seconds")

    successful_count = sum(1 for res in results if res[1] == 0)
    error_count = len(results) - successful_count
    print(f"Successful jobs: {successful_count}")
    print(f"Jobs with errors: {error_count}")
