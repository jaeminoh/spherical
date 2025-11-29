from dataclasses import dataclass
import numpy as np


@dataclass
class Protein:
    sequence: str

    def __post_init__(self):
        assert isinstance(self.sequence, str), "Sequence must be a string"

    def _mutate(self, position: int, new_aa: str) -> str:
        seq_list = list(self.sequence)
        seq_list[position] = new_aa
        return "".join(seq_list)

    def mutate(self, positions: list[int], new_aas: list[str]) -> str:
        """Mutate multiple positions in the sequence.

        Args:
            positions (list[int]): List of positions to mutate.
            new_aas (list[str]): List of new amino acids corresponding to the positions.
        """
        if isinstance(positions, int) and isinstance(new_aas, str):
            mutant_sequence = self._mutate(positions, new_aas)
        else:
            seq_list = list(self.sequence)
            for pos, aa in zip(positions, new_aas):
                seq_list[pos] = aa
            mutant_sequence = "".join(seq_list)
        return Protein(mutant_sequence)

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

    def get_some_multiple_mutations(
        self,
        n: int,
        num_substitutions: int,
        all_amino_acids: str,
        seed: int = 42,
    ) -> tuple[list[str], list[list[int]], list[list[str]]]:
        """Generate n random multiple mutations with specified number of substitutions.

        Args:
            n: Number of mutant sequences to generate
            num_substitutions: Number of positions to mutate in each sequence
            all_amino_acids: String of amino acids to choose from
            seed: Random seed for reproducibility

        Returns:
            Tuple of (mutated_sequences, positions, mutated_amino_acids)
        """
        np.random.seed(seed)
        pos = np.sort(
            np.random.choice(len(self.sequence), size=(n, num_substitutions)), axis=1
        )
        mt_aas_indices = np.random.choice(
            len(all_amino_acids), size=(n, num_substitutions)
        )

        sequences = []
        positions = []
        mutated_aas = []

        for p, a in zip(pos, mt_aas_indices):
            pos_list = p.tolist()
            aa_list = [all_amino_acids[idx] for idx in a]
            mutated_seq = self.mutate(pos_list, aa_list).sequence

            sequences.append(mutated_seq)
            positions.append(pos_list)
            mutated_aas.append(aa_list)

        return sequences, positions, mutated_aas


if __name__ == "__main__":
    import fire
    import pandas as pd

    import spherical

    def main(virus: str, n: int = 200, num_substitutions: int = 5):
        if virus == "rvfv":
            protein = spherical.RVFV_MP12
        elif virus == "cov2":
            protein = spherical.SARS_COV2_SPIKE
        _, positions, mutated_aas = protein.get_some_multiple_mutations(
            n, num_substitutions, spherical.AMINO_ACIDS
        )
        names = []
        for position, mt_aa in zip(positions, mutated_aas):
            names.append(spherical.make_name(position, mt_aa))

        pd.DataFrame({"name": names}).to_csv(
            f"results/{virus}/some_multiple.csv", index=False
        )

    fire.Fire(main)
