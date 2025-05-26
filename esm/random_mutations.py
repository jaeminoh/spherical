import random


def create_random_mutant(
    original_sequence: str,
    num_mutations: int = 1,
    mutation_rate: float = None,  # If rate is provided, num_mutations is ignored
    seed: int = None,
) -> tuple[str, list]:
    """
    Generates a randomly mutated protein sequence.

    Args:
        original_sequence (str): The wild-type protein sequence.
        num_mutations (int): The exact number of mutations to introduce.
                             Ignored if mutation_rate is provided.
        mutation_rate (float): The percentage of positions to mutate (e.g., 0.01 for 1%).
                               If provided, num_mutations is calculated from this.
        seed (int): Seed for random number generation for reproducibility.

    Returns:
        tuple: A tuple containing:
            - str: The generated mutant sequence.
            - list: A list of dictionaries, where each dictionary describes a mutation:
                    {"position": int, "original_aa": str, "mutated_aa": str}
    """
    # 20 standard amino acids
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

    if seed is not None:
        random.seed(seed)

    mutant_list = list(original_sequence)
    sequence_length = len(original_sequence)
    mutations_introduced = []

    if mutation_rate is not None:
        num_mutations = max(
            1, int(sequence_length * mutation_rate)
        )  # Ensure at least 1 mutation

    if num_mutations > sequence_length:
        raise ValueError(
            f"Cannot introduce {num_mutations} mutations in a sequence of length {sequence_length}."
        )

    mutation_positions = random.sample(range(sequence_length), num_mutations)
    mutation_positions.sort()

    for pos in mutation_positions:
        original_aa = mutant_list[pos]

        # Select a new amino acid different from the original one
        possible_mutations = [aa for aa in AMINO_ACIDS if aa != original_aa]
        if not possible_mutations:  # Should not happen if AMINO_ACIDS has >1 element
            continue  # Skip if no other AA is available (e.g., if AMINO_ACIDS was just one AA)

        mutated_aa = random.choice(possible_mutations)

        mutant_list[pos] = mutated_aa
        mutations_introduced.append(original_aa + str(pos) + mutated_aa)

    mutant_sequence = "".join(mutant_list)
    return mutant_sequence, mutations_introduced


if __name__ == "__main__":
    # Rift Valley fever virus
    wt_sequence = "MYVLLTILISVLVCEAVIRVSLSSTREETCFGDSTNPEMIEGAWDSLREEEMPEELSCSISGIREVKTSSQELYRALKAIIAADGLNNITCHGKDPEDKISLIKGPPHKKRVGIVRCERRRDAKQIGRETMAGIAMTVLPALAVFALAPVVFAEDPHLRNRPGKGHNYIDGMTQEDATCKPVTYAGACSSFDVLLEKGKFPLFQSYAHHRTLLEAVHDTIIAKADPPSCDLQSAHGNPCMKEKLVMKTHCPNDYQSAHYLNNDGKMASVKCPPKYELTEDCNFCRQMTGASLKKGSYPLQDLFCQSSEDDGSKLKTKMKGVCEVGVQALKKCDGQLSTAHEVVPFAVFKNSKKVYLDKLDLKTEENLLPDSFVCFEHKGQYKGTMDSGQTKRELKSFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHANGSGIVQIQVSGVWKKPLCVGYERVVVKRELSAKPIQRVEPCTTCITKCEPHGLVVRSTGFKISSAVACASGVCVTGSQSPSTEITLKYPGISQSSGGDIGVHMAHDDQSVSSKIVAHCPPQDPCLVHGCIVCAHGLINYQCHTALSAFVVVFVFSSIAIICLAVLYRVLKCLKIAPRKVLNPLMWITAFIRWIYKKMVARVADNINQVNREIGWMEGGQLVLGNPAPIPRHAPIPRYSTYLMLLLIVSYASACSELIQASSRITTCSTEGVNTKCRLSGTALIRAGSVGAEACLMLKGVKEDQTKFLKIKTVSSELSCREGQSYWTGSFSPKCLSSRRCHLVGECHVNRCLSWRDNETSAEFSFVGESTTMRENKCFEQCGGWGCGCFNVNPSCLFVHTYLQSVRKEALRVFNCIDWVHKLTLEITDFDGSVSTIDLGASSSRFTNWGSVSLSLDAEGISGSNSFSFIESPGKGYAIVDEPFSEIPRQGFLGEIRCNSESSVLSAHESCLRAPNLISYKPMIDQLECTTNLIDPFVVFERGSLPQTRNDKTFAASKGNRGVQAFSKGSVQADLTLMFDNFEVDFVGAAVSCDAAFLNLTGCYSCNAGARVCLSITSTGTGSLSAHNKDGSLHIVLPSENGTKDQCQILHFTVPEVEEEFMYSCDGDERPLLVKGTLIAIDPFDDRREAGGESTVVNPKSGSWNFFDWFSGLMSWFGGPLKTILLICLYVALSIGLFFLLIYLGRTGLSKMWLAATKKAS"

    # Mutate exactly 5 random positions
    print("--- Generating a mutant with 5 random mutations ---")
    random_mutant_sequence_5, mutations_5 = create_random_mutant(
        wt_sequence, num_mutations=5, seed=42
    )
    print(f"Original Sequence Length: {len(wt_sequence)}")
    print(f"Random Mutant (5 mutations): {random_mutant_sequence_5}")
    print("Mutations (5 mutations):")
    for mut in mutations_5:
        print(mut)
