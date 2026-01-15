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
    elif virus == "gfp":
        return spherical.GFP
    elif virus == "zh501":
        return spherical.RVFV_ZH501
    else:
        raise ValueError(f"Unknown virus: {virus}")


def _load_csv(csv_name: str, virus: str):
    """Load CSV file and return dataframe with csv_dir."""
    csv_dir = f"results/{virus}"
    df = pd.read_csv(f"{csv_dir}/{csv_name}.csv")
    return df, csv_dir


def _drop(df: pd.DataFrame, endswith: str = "_y"):
    columns = df.columns
    to_be_dropped = []
    for col in columns:
        if col.endswith(endswith):
            to_be_dropped.append(col)
    return df.drop(columns=to_be_dropped)


def get_mutations(
    protein: str,
    csv_name: str | None = None,
    all_single: bool = True,
    n: int | None = None,
    num_mutations: int | None = None,
):
    wt = _get_protein(protein)
    names = []
    if all_single:
        _, pos, aas = wt.get_all_single_mutations(spherical.AMINO_ACIDS)
        for p, a in zip(pos, aas):
            names.append(f"{p}{a}")
        pd.DataFrame({"name": names}).to_csv(f"results/{protein}/all_single.csv")
    else:
        _, pos, aas = wt.get_some_multiple_mutations(
            n, num_mutations, spherical.AMINO_ACIDS
        )
        for p, a in zip(pos, aas):
            names.append(spherical.make_name(p, a))
        pd.DataFrame({"name": names}).to_csv(
            f"results/{protein}/{csv_name}.csv", index=False
        )


def sort_and_crop(
    protein: str,
    csv_name: str,
    method: str = "embedding_distance",
    n: int = 100,
):
    methods_list = [
        "masked_marginal",
        "wildtype_marginal",
        "mutant_marginal",
        "embedding_distance",
        "l1",
        "l2",
        "operator_l1",
        "operator_l2",
        "frobenius_1",
        "frobenius_2",
    ]
    assert method in methods_list, (
        f"Method {method} is not supported. Choose from {methods_list}."
    )
    df, csv_dir = _load_csv(f"{csv_name}_{method}", protein)
    df = df.sort_values(by=method)
    df.head(n).to_csv(f"{csv_dir}/{csv_name}_{method}_bottom{n}.csv", index=False)
    df.tail(n).to_csv(f"{csv_dir}/{csv_name}_{method}_top{n}.csv", index=False)


def drop_columns(protein: str, csv_name: str, endswith: str = "_y"):
    df, csv_dir = _load_csv(csv_name, protein)
    df = _drop(df, endswith=endswith)
    df.to_csv(f"{csv_dir}/{csv_name}_dropped.csv", index=False)


def compute_scores(protein: str, csv_name: str, method: str):
    """Compute scores for mutations listed in CSV using specified method."""
    wt = _get_protein(protein)
    df, csv_dir = _load_csv(csv_name, protein)

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


def compute_embedding_distances(protein: str, csv_name: str):
    wt = _get_protein(protein)
    df, csv_dir = _load_csv(csv_name, protein)

    model, alphabet = spherical.esm2.load_model_and_alphabet(spherical.FAIR_ESM2_650M)
    model = model.eval().cuda()

    embedding_distance = spherical.esm2.EmbeddingDistance(model, alphabet)
    wt_embedding = embedding_distance.get_embedding(wt.sequence)

    l1 = []
    cosine = []

    for name in tqdm(df["name"]):
        positions, aas = spherical.split_name(name)
        mt = wt.mutate(positions, aas)
        mt_embedding = embedding_distance.get_embedding(mt.sequence)
        l1.append(torch.abs(wt_embedding - mt_embedding).sum().item())
        cosine.append(
            torch.nn.functional.cosine_similarity(
                wt_embedding.ravel(), mt_embedding.ravel(), dim=0
            ).item()
        )

    df["embedding_distance"] = l1
    df["cosine_similarity"] = cosine
    df.to_csv(f"{csv_dir}/{csv_name}_embedding_distance.csv", index=False)


def compute_contacts(protein: str, csv_name: str):
    """Compute contact differences for mutations listed in CSV."""
    wt = _get_protein(protein)
    df, csv_dir = _load_csv(csv_name, protein)

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


def merge_csv(protein: str, csv_name: str):
    """Merge CSV files for different scoring methods into a single CSV."""
    csv_dir = f"results/{protein}"
    df = pd.read_csv(f"{csv_dir}/{csv_name}.csv")
    length = len(df)

    df = _drop(df)

    methods = [
        "embedding_distance",
        "masked_marginal",
        "mutant_marginal",
        "wildtype_marginal",
        "contact",
        "rmsd",
        "strain",
    ]

    for method in methods:
        csv_path = f"{csv_dir}/{csv_name}_{method}.csv"
        if os.path.exists(csv_path):
            method_df = pd.read_csv(csv_path)
            df = method_df.merge(df, on="name")

    assert len(df) == length, (
        "Merged dataframe length does not match original dataframe length."
    )
    df.to_csv(f"{csv_dir}/{csv_name}_merged.csv", index=False)


def fold_structures(protein: str, csv_name: str):
    """Fold protein structures for mutations listed in CSV."""
    wt = _get_protein(protein)
    df, csv_dir = _load_csv(csv_name, protein)

    model, tokenizer = spherical.esm2.load_model(
        EsmForProteinFolding, spherical.ESMFOLD
    )
    model.to("cuda")

    csv_dir = f"results/{protein}"
    df = pd.read_csv(f"{csv_dir}/{csv_name}.csv")

    out_dir = f"{csv_dir}/esmfold"
    os.makedirs(out_dir, exist_ok=True)

    # wild type
    print("Folding wild type...")
    spherical.esm2.fold_and_save(model, tokenizer, wt.sequence, f"{out_dir}/wt.pdb")

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
    protein: str,
    csv_name: str,
    override: bool = False,
    folding_method: str = "esmfold",
    deformation_method: str = "rmsd",
):
    df, csv_dir = _load_csv(csv_name, protein)
    wt = Protein(f"{csv_dir}/{folding_method}/wt.pdb")

    list = []
    for name in tqdm(df["name"]):
        mt = Protein(f"{csv_dir}/{folding_method}/{name}.pdb")
        deformation = Deformation(wt, mt, method=deformation_method)
        deformation.run()
        list.append(deformation)

    if deformation_method == "rmsd":
        list = [d.rmsd for d in list]
    elif deformation_method == "strain":
        list = [d.strain.mean() for d in list]

    df[deformation_method] = list
    if not override:
        csv_name = f"{csv_name}_{deformation_method}"
    df.to_csv(f"{csv_dir}/{csv_name}.csv", index=False)


def compare_pdb_sequences(
    protein: str, name1: str, name2: str, method: str = "alphafold"
):
    dir_path = f"results/{protein}/{method}"
    path_pdb_1 = f"{dir_path}/{name1}.pdb"
    path_pdb_2 = f"{dir_path}/{name2}.pdb"

    positions, originals, mutated = spherical.compare_two_pdbs(path_pdb_1, path_pdb_2)
    print("Positions:", positions)
    print("Original residues:", originals)
    print("Mutated residues:", mutated)


def prepare_af2_dummy_csv(
    protein: str,
    csv_name: str,
):
    wt = _get_protein(protein)
    df, csv_dir = _load_csv(csv_name, protein)
    sequences = []
    for name in df["name"]:
        positions, aas = spherical.split_name(name)
        mt = wt.mutate(positions, aas)
        sequences.append(mt.sequence)

    pd.DataFrame({"sample_name": df["name"], "sequence": sequences}).to_csv(
        f"{csv_dir}/{csv_name}_dummy.csv", index=False
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "score": compute_scores,
            "embedding_distance": compute_embedding_distances,
            "contact": compute_contacts,
            "fold": fold_structures,
            "rmsd": compute_rmsd,
            "merge": merge_csv,
            "drop": drop_columns,
            "mutate": get_mutations,
            "rank": sort_and_crop,
            "compare_pdb_sequences": compare_pdb_sequences,
            "prepare_af2_csv": prepare_af2_dummy_csv,
        }
    )
