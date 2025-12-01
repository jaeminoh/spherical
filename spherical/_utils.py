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
