import re
import csv

from Bio import SeqIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import Residue
import numpy as np


def read_fasta(fname: str):
    seqs = SeqIO.parse(fname, "fasta")
    return seqs


def read_gbk(fname: str):
    seqs = SeqIO.parse(fname, "genbank")
    return seqs


def write_fasta_gbk_protein(fname: str, replace_with: str = "X"):
    """
    Extract translated sequence information (.fasta) from genbank file (.gb, .gbk).
    """
    assert ".gb" in fname, "The file must be in (.gb, .gbk) format!"
    seqs = read_gbk(fname)
    id_list = []
    protein_list = []
    for seq in seqs:
        for f in seq.features:
            if f.type == "CDS":
                (id,) = f.qualifiers["protein_id"]
                id_list.append(id)
                (protein,) = f.qualifiers["translation"]
                protein_list.append(re.sub("X", replace_with, protein))
    print(f"There are {len(protein_list)} proteins!")

    with open("data/rvfv.fasta", "w") as f:
        for id, protein in zip(id_list, protein_list):
            f.write(">" + id + "\n" + protein + "\n")


def remove_lowercase(text):
    """Removes lowercase letters from a string.

    Args:
    text: The input string.

    Returns:
    A new string with lowercase letters removed.
    """
    return re.sub(r"[a-z]", "", text)


def where_uppercase(seq: str):
    """WARNING: ChimeraX supports 1-based indexing."""
    positions = []
    for i, letter in enumerate(seq):
        if letter.isupper():
            positions.append(i)
    return positions


def where_lowercase(seq: str):
    """WARNING: ChimeraX supports 1-based indexing."""
    positions = []
    for i, letter in enumerate(seq):
        if letter.islower():
            positions.append(i)
    return positions


def save_csv(result: list, path: str):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(result)


def read_cif(path: str = "data/alphafold/fold_2025_02_10_08_03_model_0.cif"):
    # FastMMCIFParser does not parses residues.
    st = MMCIFParser().get_structure("tmp", path)
    return st


def get_first_atom(res: Residue):
    atom = next(res.get_atoms())  # `get_atoms` method returns a generator
    return atom


def rr_distance(r1: Residue, r2: Residue):
    """Compute residue-residue distance. \n

    Reference: https://rbvi.github.io/chimerax-recipes/rrdist/rrdist.html
    """
    a1 = get_first_atom(r1).coord
    a2 = get_first_atom(r2).coord
    return np.linalg.norm(a1 - a2)  # Euclidean distance (3D)


def rs_distance(r: Residue, rr, pp):
    """Compute minimum `rr_distance` for a residue and a structure."""
    distance = np.inf
    for p in pp:
        if (d := rr_distance(r, rr[p])) < distance:
            distance = d
    return distance


def ss_distance(rr: list, pp1: list[int], pp2: list[int]) -> float:
    """Compute minimum `rr_distance` between two structures."""
    distance = np.inf
    for p1 in pp1:
        for p2 in pp2:
            if (d := rr_distance(rr[p1], rr[p2])) < distance:
                distance = d
    return distance


# TODO performance optimization
def _merge_if_close(rr, epitopes, thr):
    merged_epitopes = []
    while epitopes:
        pp1 = epitopes.pop(0)
        i = 0
        while i < len(epitopes):
            pp2 = epitopes[i]
            if ss_distance(rr, pp1, pp2) < thr:
                pp1 = pp1 + pp2
                epitopes.pop(i)
            else:
                i += 1
        merged_epitopes.append(pp1)
    return merged_epitopes


def group_residues_by_distance(epi_seq: str, path_cif: str, thr: float = 4.0):
    positions = where_uppercase(epi_seq)
    st = read_cif(path_cif)
    rr = list(st.get_residues())
    epitopes = [[p] for p in positions]
    return _merge_if_close(rr, epitopes, thr)


def add_nearby_ones_to_groups(
    path_cif: str, epi_seq: str, gg: list[list[int]], thr: float
):
    st = read_cif(path_cif)
    rr = list(st.get_residues())
    nonepi_positions = where_lowercase(epi_seq)
    gg_len = len(gg)
    gg_new = []
    while gg:
        g = gg.pop(0)
        for i in nonepi_positions:
            if rs_distance(rr[i], rr, g) < thr:
                g.append(i)
        g.sort()
        gg_new.append(g)
    assert gg_len == len(gg_new)  # unit test!
    return gg_new
