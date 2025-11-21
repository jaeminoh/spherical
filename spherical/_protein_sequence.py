from dataclasses import dataclass


@dataclass
class ReferenceProteinSequence:
    sequence: str

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

    def get_all_single_mutations(
        self, all_amino_acids: str
    ) -> tuple[list[str], list[int], list[str]]:
        """Generate all single amino acid mutations for the sequence."""
        single_mutations = []
        positions = []
        mutated_aas = []
        seq = list(self.sequence)
        for i, original_aa in enumerate(seq):
            for aa in all_amino_acids:
                if aa != original_aa:
                    mutated_seq = self._mutate(i, aa)
                    single_mutations.append(mutated_seq)
                    positions.append(i)
                    mutated_aas.append(aa)
        return single_mutations, positions, mutated_aas
