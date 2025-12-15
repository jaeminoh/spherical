from Bio import SeqIO


def split_name(name: str) -> tuple[list[int], list[str]]:
    """Parse mutation name string into positions and amino acids.

    Example: "730A;650V" -> positions=[730, 650], aas=["A", "V"]
    """
    mutations = name.split(";")
    positions = []
    aas = []

    for mut in mutations:
        pos = int("".join(c for c in mut if c.isdigit()))
        aa = "".join(c for c in mut if c.isalpha())
        positions.append(pos)
        aas.append(aa)

    return positions, aas


def make_name(positions: list[int], aas: list[str]) -> str:
    """Construct mutation name string from positions and amino acids.

    Example: positions=[730, 650], aas=["A", "V"] -> "730A;650V"
    """
    return ";".join(f"{pos}{aa}" for pos, aa in zip(positions, aas))


def compare_two_pdbs(
    path_pdb_1: str, path_pdb_2: str
) -> tuple[list[int], list[str], list[str]]:
    seq_1 = next(SeqIO.parse(path_pdb_1, "pdb-atom")).seq
    seq_2 = next(SeqIO.parse(path_pdb_2, "pdb-atom")).seq
    assert len(seq_1) == len(seq_2), "Sequences have different lengths"

    positions = []
    origianls = []
    mutated = []

    for i, (res_1, res_2) in enumerate(zip(seq_1, seq_2)):
        if res_1 != res_2:
            positions.append(i)
            origianls.append(res_1)
            mutated.append(res_2)

    return positions, origianls, mutated
