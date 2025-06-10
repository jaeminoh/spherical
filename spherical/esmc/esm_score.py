import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

wt_protein_obj = ESMProtein(
    sequence="MYVLLTILISVLVCEAVIRVSLSSTREETCFGDSTNPEMIEGAWDSLREEEMPEELSCSISGIREVKTSSQELYRALKAIIAADGLNNITCHGKDPEDKISLIKGPPHKKRVGIVRCERRRDAKQIGRETMAGIAMTVLPALAVFALAPVVFAEDPHLRNRPGKGHNYIDGMTQEDATCKPVTYAGACSSFDVLLEKGKFPLFQSYAHHRTLLEAVHDTIIAKADPPSCDLQSAHGNPCMKEKLVMKTHCPNDYQSAHYLNNDGKMASVKCPPKYELTEDCNFCRQMTGASLKKGSYPLQDLFCQSSEDDGSKLKTKMKGVCEVGVQALKKCDGQLSTAHEVVPFAVFKNSKKVYLDKLDLKTEENLLPDSFVCFEHKGQYKGTMDSGQTKRELKSFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHANGSGIVQIQVSGVWKKPLCVGYERVVVKRELSAKPIQRVEPCTTCITKCEPHGLVVRSTGFKISSAVACASGVCVTGSQSPSTEITLKYPGISQSSGGDIGVHMAHDDQSVSSKIVAHCPPQDPCLVHGCIVCAHGLINYQCHTALSAFVVVFVFSSIAIICLAVLYRVLKCLKIAPRKVLNPLMWITAFIRWIYKKMVARVADNINQVNREIGWMEGGQLVLGNPAPIPRHAPIPRYSTYLMLLLIVSYASACSELIQASSRITTCSTEGVNTKCRLSGTALIRAGSVGAEACLMLKGVKEDQTKFLKIKTVSSELSCREGQSYWTGSFSPKCLSSRRCHLVGECHVNRCLSWRDNETSAEFSFVGESTTMRENKCFEQCGGWGCGCFNVNPSCLFVHTYLQSVRKEALRVFNCIDWVHKLTLEITDFDGSVSTIDLGASSSRFTNWGSVSLSLDAEGISGSNSFSFIESPGKGYAIVDEPFSEIPRQGFLGEIRCNSESSVLSAHESCLRAPNLISYKPMIDQLECTTNLIDPFVVFERGSLPQTRNDKTFAASKGNRGVQAFSKGSVQADLTLMFDNFEVDFVGAAVSCDAAFLNLTGCYSCNAGARVCLSITSTGTGSLSAHNKDGSLHIVLPSENGTKDQCQILHFTVPEVEEEFMYSCDGDERPLLVKGTLIAIDPFDDRREAGGESTVVNPKSGSWNFFDWFSGLMSWFGGPLKTILLICLYVALSIGLFFLLIYLGRTGLSKMWLAATKKAS"
)
mt_protein_obj = ESMProtein(
    sequence="MYVLLTILTSVLVCEAIIRVSLSSTREETCFGDSTNPEMIEGAWDSLREEEMPEELSCSISGIREVKTSSQELYRALKAIIAADGLNNITCHGKDPEDKISLIKGPPHKKRVGIVRCERRRDAKQIGRETMAGIAMTVLPALAVFALAPVVFAEDPHLRNRPGKGHNYIDGMTQEDATCKPVTYAGACSSFDVLLEKGKFPLFQSYAHHRTLLEAVHDTIIAKADPPSCDLLSAHGNPCMKEKLVMKTHCPNDYQSAHHLNNDGKMASVKCPPKYELTEDCNFCRQMTGASLKKGSYPLQDLFCQSSEDDGSKLKTKMKGVCEVGVQALKKCDGQLSTAHEVVPFAVFKNSKKVYLDKLDLKTEENLLPDSFVCFEHKGQYKGTMDSGQTKRELKSFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHANGSGIVQIQVSGVWKKPLCVGYERVVVKRELSAKPIQRVEPCTTCITKCEPHGLVVRSTGFKISSAVACASGVCVTGSQSPSTEITLKYPGISQSSGGDIGVHMAHDDQSVSSKIVAHCPPQDPCLVHDCIVCAHGLINYQCHTALSAFVVVFVFSSIAIICLAILYRVLKCLKIAPRKVLNPLMWITAFIRWIYKKMVARVADNINQVNREIGWMEGGQLVLGNPAPIPRHAPIPRYSTYLMLLLIVSYASACSELIQASSRITTCSTEGVNTKCRLSGTALIRAGSVGAEACLMLKGVKEDQTKFLKLKTVSSELSCREGQSYWTGSFSPKCLSSRRCHLVGECHVNRCLSWRDNETSAEFSFVGESTTMRENKCFEQCGGWGCGCFNVNPSCLFVHTYLQSVRKEALRVFNCIDWVHKLTLEITDFDGSVSTIDLGASSSRFTNWGSVSLSLDAEGISGSNSFSFIESPGKGYAIVDEPFSEIPRQGFLGEIRCNSESSVLSAHESCLRAPNLISYKPMIDQLECTTNLIDPFVVFERGSLPQTRNDKTFAASKGNRGVQAFSKGSVQADLTLMFDNFEVDFVGAAVSCDAAFLNLTGCYSCNAGARVCLSITSTGTGSLSAHNKDGSLHIVLPSENGTKDQCQILHFTVPEVEEEFMYSCDGDERPLLVKGTLIAIDPFDDRREAGGESTVVNPKSGSWNFFDWFSGLMSWFGGPLKTILLICLYVALSIGLFFLLIYLGGTGLSKMWLAATKKAS"
)

# Extract the string sequences from the ESMProtein objects
wt_sequence = wt_protein_obj.sequence
mt_sequence = mt_protein_obj.sequence

# Ensure sequences are of the same length
if len(wt_sequence) != len(mt_sequence):
    raise ValueError("Wild-type and mutant sequences must have the same length.")

# Load the ESMC model
device = "cuda" if torch.cuda.is_available() else "cpu"
client = ESMC.from_pretrained("esmc_600m").to(device)
client.eval()  # Set the model to evaluation mode

# --- CRITICAL: HARDCODED AMINO ACID TO ID MAPPING ---
# This is a common mapping for ESM models (and many protein MLMs).
# If ESMC uses a different internal ordering, these IDs will be incorrect.
# The `[MASK]` token ID is assumed to be 4.
AA_TO_ID = {
    "[MASK]": 4,  # Common ID for mask token
    "A": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "K": 13,
    "L": 14,
    "M": 15,
    "N": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "V": 22,
    "W": 23,
    "Y": 24,
    # Add other possible tokens if your sequences might contain them (e.g., 'X', 'B', 'Z')
    # If the logits dimension is larger, you might need to extend this.
}
# Reverse mapping for sanity checks (optional)
ID_TO_AA = {
    v: k for k, v in AA_TO_ID.items()
}  # Not strictly needed for calculation but useful for debugging

# Get the mask token character for string manipulation
mask_token_char = "[MASK]"

total_llr_score = 0.0
mutated_positions_details = []

print("---")
print(
    "Identifying Mutations and Calculating LLRs using Masking (ESMC, hardcoded alphabet):"
)
print("WARNING: This code relies on a hardcoded amino acid to ID mapping.")
print(
    "If the ESMC model's internal vocabulary order differs, results will be inaccurate."
)
print("---")

# Identify all mutation sites first
mutations = []
for i in range(len(wt_sequence)):
    if wt_sequence[i] != mt_sequence[i]:
        mutations.append((i, wt_sequence[i], mt_sequence[i]))

if not mutations:
    print("No mutations found between the wild-type and mutant sequences.")
else:
    # --- SIMULTANEOUS MASKING START ---
    # Create ONE masked version of the wild-type sequence with ALL mutations masked
    all_masked_wt_list = list(wt_sequence)
    masked_indices = []

    for i, wt_aa, mt_aa in mutations:
        all_masked_wt_list[i] = mask_token_char  # Mask all mutation sites
        masked_indices.append(i)  # Keep track of masked indices

    all_masked_wt_sequence_str = "".join(all_masked_wt_list)

    # Prepare the protein object for encoding
    all_masked_protein = ESMProtein(sequence=all_masked_wt_sequence_str)

    # Encode and get logits in ONE pass
    all_masked_protein_tensor = client.encode(all_masked_protein)
    all_masked_logits_output = client.logits(
        all_masked_protein_tensor, LogitsConfig(sequence=True, return_embeddings=False)
    )
    all_logits_tensor = all_masked_logits_output.logits.sequence.squeeze(
        0
    )  # Get the full logits tensor

    total_llr_score = 0.0
    mutated_positions_details = []

    # Iterate through the mutations to calculate LLRs from the single logits tensor
    for i, wt_aa, mt_aa in mutations:
        # Get logits for this specific masked position from the full tensor
        logits_at_current_masked_pos = all_logits_tensor[
            i + 1
        ]  # i+1 because of CLS token

        log_probs_at_pos = torch.nn.functional.log_softmax(
            logits_at_current_masked_pos, dim=-1
        )

        try:
            wt_aa_id = AA_TO_ID[wt_aa]
            mt_aa_id = AA_TO_ID[mt_aa]
        except KeyError as e:
            print(
                f"Error: Amino acid '{e}' not found in the hardcoded AA_TO_ID mapping. Skipping this mutation."
            )
            continue

        log_prob_wt_pred = log_probs_at_pos[wt_aa_id].item()
        log_prob_mt_pred = log_probs_at_pos[mt_aa_id].item()

        llr_for_mutation = log_prob_mt_pred - log_prob_wt_pred
        total_llr_score += llr_for_mutation

        mutated_positions_details.append(
            {
                "index": i,
                "position_1_indexed": i + 1,
                "wild_type_aa": wt_aa,
                "mutant_aa": mt_aa,
                "log_prob_wt_pred": log_prob_wt_pred,
                "log_prob_mt_pred": log_prob_mt_pred,
                "llr": llr_for_mutation,
            }
        )

        print(f"Position {i + 1} (0-indexed: {i}): {wt_aa} -> {mt_aa}")
        print(f"  Log-Prob({wt_aa} | all masked context): {log_prob_wt_pred:.4f}")
        print(f"  Log-Prob({mt_aa} | all masked context): {log_prob_mt_pred:.4f}")
        print(f"  LLR: {llr_for_mutation:.4f}")
    # --- SIMULTANEOUS MASKING END ---

print("---")
if mutated_positions_details:
    print(
        f"Total ESMC Log-Likelihood Ratio (LLR) Score for all mutations (Simultaneous Masking): {total_llr_score:.4f}"
    )
else:
    print("No mutations found between the wild-type and mutant sequences.")
print("---")
