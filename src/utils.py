from Bio import SeqIO
import re


def read_fasta(fname: str):
    seqs = SeqIO.parse(fname, "fasta")
    return seqs


def read_gbk(fname: str):
    seqs = SeqIO.parse(fname, "genbank")
    return seqs


def write_fasta_gbk_protein(fname: str):
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
                protein_list.append(re.sub("X", "A", protein))
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


def file_reader(_input):
    content_arry = []
    f = open(_input, "r")
    if f.mode == "r":
        content_arry = f.read().splitlines()
        f.close()
    return content_arry
