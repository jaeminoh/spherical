class BaseMutation:
    def __init__(self):
        pass

    def mutate_str(wt: str, positions: list[int], mt_aa: list[str] | str) -> str:
        sequence_masked = list(wt)
        if isinstance(mt_aa, list):
            for i, m in zip(positions, mt_aa):
                sequence_masked[i] = m
        elif isinstance(mt_aa, str):
            for i in positions:
                sequence_masked[i] = mt_aa
        return "".join(sequence_masked)


class Recombination:
    def __init__(self):
        pass
