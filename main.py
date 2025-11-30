import os
import fire
import pandas as pd
from tqdm import tqdm
from transformers import EsmForProteinFolding
import torch
from PDAnalysis import Deformation, Protein

import spherical


def _get_protein(virus: str):
    """Get protein based on virus name."""
    if virus == "rvfv":
        return spherical.RVFV_MP12
    elif virus == "cov2":
        return spherical.SARS_COV2_SPIKE
    else:
        raise ValueError(f"Unknown virus: {virus}")


def _load_csv(csv_name: str, virus: str):
    """Load CSV file and return dataframe with csv_dir."""
    csv_dir = f"results/{virus}"
    df = pd.read_csv(f"{csv_dir}/{csv_name}.csv")
    return df, csv_dir


def compute_scores(csv_name: str, method: str, virus: str):
    """Compute scores for mutations listed in CSV using specified method."""
    wt = _get_protein(virus)
    df, csv_dir = _load_csv(csv_name, virus)

    model, alphabet = spherical.esm2.load_model_and_alphabet(spherical.FAIR_ESM2_650M)
    model = model.eval().cuda()

    if method == "masked_marginal":
        score_fn = spherical.esm2.MaskedMarginal(model, alphabet)
    elif method == "wildtype_marginal":
        score_fn = spherical.esm2.WildtypeMarginal(model, alphabet)
    elif method == "mutant_marginal":
        score_fn = spherical.esm2.MutantMarginal(model, alphabet)
    else:
        raise ValueError(f"Unknown method: {method}")

    scores = []
    for name in tqdm(df["name"]):
        positions, aas = spherical.split_name(name)
        mt = wt.mutate(positions, aas)
        scores.append(score_fn(wt.sequence, mt.sequence).item())

    df[method] = scores
    df.to_csv(f"{csv_dir}/{csv_name}_{method}.csv", index=False)


def compute_contacts(csv_name: str, virus: str):
    """Compute contact differences for mutations listed in CSV."""
    wt = _get_protein(virus)
    df, csv_dir = _load_csv(csv_name, virus)

    model, alphabet = spherical.esm2.load_model_and_alphabet(spherical.FAIR_ESM2_650M)
    model = model.eval().cuda()
    contact = spherical.esm2.ContactPredictor(model, alphabet)

    vector_norms = {"l1": [], "l2": []}
    matrix_norms = {
        "operator_l1": [],
        "operator_l2": [],
        "frobenius_1": [],
        "frobenius_2": [],
    }

    wt_contact = contact.get_contact_matrix("wt", wt.sequence)

    for name in tqdm(df["name"]):
        positions, aas = spherical.split_name(name)
        mt = wt.mutate(positions, aas)

        diff = wt_contact - contact.get_contact_matrix(name, mt.sequence)

        # Use corresponding positions
        pos = torch.tensor(positions) if isinstance(positions, list) else positions
        vector_norms["l1"].append(abs(diff[pos]).sum().item())
        vector_norms["l2"].append((diff[pos] ** 2).sum().item() ** 0.5)
        matrix_norms["operator_l1"].append(torch.linalg.norm(diff, ord=1).item())
        matrix_norms["operator_l2"].append(torch.linalg.norm(diff, ord=2).item())
        matrix_norms["frobenius_1"].append(abs(diff).sum().item())
        matrix_norms["frobenius_2"].append((diff**2).sum().item() ** 0.5)

    for key, values in {**vector_norms, **matrix_norms}.items():
        df[key] = values

    df.to_csv(f"{csv_dir}/{csv_name}_contact.csv", index=False)


def merge_csv(csv_name: str, virus: str = "rvfv"):
    """Merge CSV files for different scoring methods into a single CSV."""
    csv_dir = f"results/{virus}"
    df = pd.read_csv(f"{csv_dir}/{csv_name}.csv")
    length = len(df)

    methods = [
        "masked_marginal",
        "mutant_marginal",
        "wildtype_marginal",
        "contact",
        "rmsd",
    ]

    for method in methods:
        csv_path = f"{csv_dir}/{csv_name}_{method}.csv"
        if os.path.exists(csv_path):
            method_df = pd.read_csv(csv_path)
            df = df.merge(method_df, on="name")

    assert len(df) == length, (
        "Merged dataframe length does not match original dataframe length."
    )
    df.to_csv(f"{csv_dir}/{csv_name}_merged_{len(methods)}.csv", index=False)


def fold_structures(csv_name: str, virus: str = "rvfv"):
    """Fold protein structures for mutations listed in CSV."""
    protein = _get_protein(virus)
    df, csv_dir = _load_csv(csv_name, virus)

    model, tokenizer = spherical.esm2.load_model(
        EsmForProteinFolding, spherical.ESMFOLD
    )
    model.to("cuda")

    csv_dir = f"results/{virus}"
    df = pd.read_csv(f"{csv_dir}/{csv_name}.csv")

    out_dir = f"{csv_dir}/esmfold"
    os.makedirs(out_dir, exist_ok=True)

    # wild type
    print("Folding wild type...")
    spherical.esm2.fold_and_save(
        model, tokenizer, protein.sequence, f"{out_dir}/wt.pdb"
    )

    # mutants
    for name in tqdm(df["name"]):
        positions, aas = spherical.split_name(name)
        mutant = protein.mutate(positions, aas)
        spherical.esm2.fold_and_save(
            model,
            tokenizer,
            mutant.sequence,
            f"{out_dir}/{name}.pdb",
        )


def compute_rmsd(
    csv_name: str,
    virus: str,
    folding_method: str = "esmfold",
    deformation_method: str = "rmsd",
):
    df, csv_dir = _load_csv(csv_name, virus)
    wt = Protein(f"{csv_dir}/{folding_method}/wt.pdb")

    # protein_kwargs = load_protein_kwargs(args)
    # deform_kwargs = load_deformation_kwargs(args)

    rmsd_list = []
    # wt = load_protein_object([wt], **protein_kwargs)
    for name in tqdm(df["name"]):
        mt = Protein(f"{csv_dir}/{folding_method}/{name}.pdb")
        rmsd = Deformation(wt, mt, method=deformation_method)
        rmsd.run()
        rmsd_list.append(rmsd.rmsd)
    df["rmsd"] = rmsd_list
    df.to_csv(f"{csv_dir}/{csv_name}_rmsd.csv", index=False)


if __name__ == "__main__":
    fire.Fire(
        {
            "score": compute_scores,
            "contact": compute_contacts,
            "fold": fold_structures,
            "rmsd": compute_rmsd,
            "merge": merge_csv,
        }
    )
