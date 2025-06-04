import esm  # pip install fair-esm==2.0.0
import esm2quinox as esmqx
import jax
import jax.numpy as jnp


def find_mutation_information(
    sequence_wt: str,
    sequence_mt: str,
) -> dict:
    """
    Finds the positions and amino acids of mutations between the original and mutant sequences.

    Args:
        sequence_wt (str): The wild-type protein sequence.
        sequence_mt (str): The mutated protein sequence.

    Returns:
        list: A list of dictionaries, where each dictionary describes a mutation:
              {"position": int, "original_aa": str, "mutated_aa": str}
    """
    positions = []
    wt_aa = []
    mt_aa = []
    for i, (_wt, _mt) in enumerate(zip(sequence_wt, sequence_mt)):
        if _wt != _mt:
            positions.append(i)
            wt_aa.append(_wt)
            mt_aa.append(_mt)
    return {"positions": positions, "wt": wt_aa, "mt": mt_aa}


def mask_sequence(sequence: str, positions: list[int]) -> str:
    sequence = list(sequence)
    for pos in positions:
        sequence[pos] = "m"
    return "".join(sequence)


def compute_esm_score(model: esmqx.ESM2, wt: str, mt: str):
    information = find_mutation_information(wt, mt)

    positions = jnp.array(information["positions"])
    wt_ids = jnp.array(list(model.alphabet[aa] for aa in information["wt"]))
    mt_ids = jnp.array(list(model.alphabet[aa] for aa in information["mt"]))

    masked_wt = mask_sequence(wt, information["positions"])
    proteins = esmqx.tokenise([masked_wt])
    logits = jax.vmap(model)(proteins).logits.squeeze(0)[1:-1][
        positions
    ]  # remove batch dimension and special tokens

    log_prob = jax.nn.log_softmax(logits, axis=-1)
    wt_score = log_prob[wt_ids].sum()
    mt_score = log_prob[mt_ids].sum()

    return mt_score - wt_score


def comparing_wt_and_mt(model, wt, mt):
    esm_score = compute_esm_score(model, wt, mt)
    print(f"ESM Score: {esm_score:.4f}")


def reconstruction(model: esmqx.ESM2, sequence: str):
    proteins = esmqx.tokenise([sequence])
    logits = jax.vmap(model)(proteins).logits.squeeze(0)[
        1:-1
    ]  # remove batch dimension and special tokens
    output_ids = jnp.argmax(logits, axis=-1)
    input_ids = proteins[0][1:-1]  # remove special tokens
    print("Accuracy:", (output_ids == input_ids).mean())


if __name__ == "__main__":
    # common routine
    torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    model = esmqx.from_torch(torch_model)

    # Example sequences
    sequence_wt = "MYVLLTILISVLVCEAVIRVSLSSTREETCFGDSTNPEMIEGAWDSLREEEMPEELSCSISGIREVKTSSQELYRALKAIIAADGLNNITCHGKDPEDKISLIKGPPHKKRVGIVRCERRRDAKQIGRETMAGIAMTVLPALAVFALAPVVFAEDPHLRNRPGKGHNYIDGMTQEDATCKPVTYAGACSSFDVLLEKGKFPLFQSYAHHRTLLEAVHDTIIAKADPPSCDLQSAHGNPCMKEKLVMKTHCPNDYQSAHYLNNDGKMASVKCPPKYELTEDCNFCRQMTGASLKKGSYPLQDLFCQSSEDDGSKLKTKMKGVCEVGVQALKKCDGQLSTAHEVVPFAVFKNSKKVYLDKLDLKTEENLLPDSFVCFEHKGQYKGTMDSGQTKRELKSFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHANGSGIVQIQVSGVWKKPLCVGYERVVVKRELSAKPIQRVEPCTTCITKCEPHGLVVRSTGFKISSAVACASGVCVTGSQSPSTEITLKYPGISQSSGGDIGVHMAHDDQSVSSKIVAHCPPQDPCLVHGCIVCAHGLINYQCHTALSAFVVVFVFSSIAIICLAVLYRVLKCLKIAPRKVLNPLMWITAFIRWIYKKMVARVADNINQVNREIGWMEGGQLVLGNPAPIPRHAPIPRYSTYLMLLLIVSYASACSELIQASSRITTCSTEGVNTKCRLSGTALIRAGSVGAEACLMLKGVKEDQTKFLKIKTVSSELSCREGQSYWTGSFSPKCLSSRRCHLVGECHVNRCLSWRDNETSAEFSFVGESTTMRENKCFEQCGGWGCGCFNVNPSCLFVHTYLQSVRKEALRVFNCIDWVHKLTLEITDFDGSVSTIDLGASSSRFTNWGSVSLSLDAEGISGSNSFSFIESPGKGYAIVDEPFSEIPRQGFLGEIRCNSESSVLSAHESCLRAPNLISYKPMIDQLECTTNLIDPFVVFERGSLPQTRNDKTFAASKGNRGVQAFSKGSVQADLTLMFDNFEVDFVGAAVSCDAAFLNLTGCYSCNAGARVCLSITSTGTGSLSAHNKDGSLHIVLPSENGTKDQCQILHFTVPEVEEEFMYSCDGDERPLLVKGTLIAIDPFDDRREAGGESTVVNPKSGSWNFFDWFSGLMSWFGGPLKTILLICLYVALSIGLFFLLIYLGRTGLSKMWLAATKKAS"
    sequence_mt = "MYVLLTILTSVLVCEAIIRVSLSSTREETCFGDSTNPEMIEGAWDSLREEEMPEELSCSISGIREVKTSSQELYRALKAIIAADGLNNITCHGKDPEDKISLIKGPPHKKRVGIVRCERRRDAKQIGRETMAGIAMTVLPALAVFALAPVVFAEDPHLRNRPGKGHNYIDGMTQEDATCKPVTYAGACSSFDVLLEKGKFPLFQSYAHHRTLLEAVHDTIIAKADPPSCDLLSAHGNPCMKEKLVMKTHCPNDYQSAHHLNNDGKMASVKCPPKYELTEDCNFCRQMTGASLKKGSYPLQDLFCQSSEDDGSKLKTKMKGVCEVGVQALKKCDGQLSTAHEVVPFAVFKNSKKVYLDKLDLKTEENLLPDSFVCFEHKGQYKGTMDSGQTKRELKSFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHANGSGIVQIQVSGVWKKPLCVGYERVVVKRELSAKPIQRVEPCTTCITKCEPHGLVVRSTGFKISSAVACASGVCVTGSQSPSTEITLKYPGISQSSGGDIGVHMAHDDQSVSSKIVAHCPPQDPCLVHDCIVCAHGLINYQCHTALSAFVVVFVFSSIAIICLAILYRVLKCLKIAPRKVLNPLMWITAFIRWIYKKMVARVADNINQVNREIGWMEGGQLVLGNPAPIPRHAPIPRYSTYLMLLLIVSYASACSELIQASSRITTCSTEGVNTKCRLSGTALIRAGSVGAEACLMLKGVKEDQTKFLKLKTVSSELSCREGQSYWTGSFSPKCLSSRRCHLVGECHVNRCLSWRDNETSAEFSFVGESTTMRENKCFEQCGGWGCGCFNVNPSCLFVHTYLQSVRKEALRVFNCIDWVHKLTLEITDFDGSVSTIDLGASSSRFTNWGSVSLSLDAEGISGSNSFSFIESPGKGYAIVDEPFSEIPRQGFLGEIRCNSESSVLSAHESCLRAPNLISYKPMIDQLECTTNLIDPFVVFERGSLPQTRNDKTFAASKGNRGVQAFSKGSVQADLTLMFDNFEVDFVGAAVSCDAAFLNLTGCYSCNAGARVCLSITSTGTGSLSAHNKDGSLHIVLPSENGTKDQCQILHFTVPEVEEEFMYSCDGDERPLLVKGTLIAIDPFDDRREAGGESTVVNPKSGSWNFFDWFSGLMSWFGGPLKTILLICLYVALSIGLFFLLIYLGGTGLSKMWLAATKKAS"

    # comparing the mutant and wild-type sequences
    comparing_wt_and_mt(model, sequence_wt, sequence_mt)

    # reconstruction capability
    reconstruction(model, sequence_wt)
    reconstruction(model, sequence_mt)
