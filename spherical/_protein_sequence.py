from spherical._constants import AMINO_ACIDS


class ReferenceProteinSequence:
    def __init__(self, sequence: str):
        self.sequence = sequence

    def _mutate(self, position: int, new_aa: str) -> str:
        seq_list = list(self.sequence)
        seq_list[position] = new_aa
        return "".join(seq_list)

    def mutate(self, poisitions: list[int], new_aas: list[str]) -> str:
        """Mutate multiple positions in the sequence.

        Args:
            poisitions (list[int]): List of positions to mutate.
            new_aas (list[str]): List of new amino acids corresponding to the positions.
        """
        seq_list = list(self.sequence)
        for pos, aa in zip(poisitions, new_aas):
            seq_list[pos] = aa
        return "".join(seq_list)

    def get_all_single_mutations(self) -> list[str]:
        """Generate all single amino acid mutations for the sequence."""
        single_mutations = []
        positions = []
        mutated_aas = []
        seq = list(self.sequence)
        for i, original_aa in enumerate(seq):
            for aa in AMINO_ACIDS:
                if aa != original_aa:
                    mutated_seq = self._mutate(i, aa)
                    single_mutations.append(mutated_seq)
                    positions.append(i)
                    mutated_aas.append(aa)
        return single_mutations, positions, mutated_aas


if __name__ == "__main__":
    from Bio import SeqIO

    wt = SeqIO.read("assets/RVFV.fasta", "fasta").seq
    rvfv_ref = ReferenceProteinSequence(str(wt))
    all_mutations, *_ = rvfv_ref.get_all_single_mutations()
    print(f"should be 1197 * 19 = 22743, {len(all_mutations)}")
