import esm  # pip install fair-esm==2.0.0
import esm2quinox as esmqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Bool, Int, jaxtyped
from beartype import beartype


def find_mutation_information(
    wt_ids: Int[ArrayLike, " L"],
    mt_ids: Int[ArrayLike, " L"],
) -> Bool[ArrayLike, " L"]:
    positions = wt_ids != mt_ids
    return positions


def find_mutation_information_str(
    wt: str,
    mt: str,
) -> tuple[list[int], list[str], list[str]]:
    positions = []
    wt_aa = []
    mt_aa = []
    for i, (_wt, _mt) in enumerate(zip(wt, mt)):
        if _wt != _mt:
            positions.append(i)
            wt_aa.append(_wt)
            mt_aa.append(_mt)
    return positions, wt_aa, mt_aa


def mask_sequence(
    sequence_ids: Int[ArrayLike, " L"],
    positions: Bool[ArrayLike, " L"],
    mask_token: int,
) -> Int[ArrayLike, " L"]:
    return (1 - positions) * sequence_ids + positions * mask_token


def mask_sequence_str(sequence: str, positions: list[int]) -> str:
    sequence_masked = list(sequence)
    for pos in positions:
        sequence_masked[pos] = "m"  # Masking with 'm' for the mutated positions
    return "".join(sequence_masked)


@jaxtyped(typechecker=beartype)
def compute_esm_score(
    model: esmqx.ESM2, wt_ids: Int[ArrayLike, " L"], mt_ids: Int[ArrayLike, " L"]
):
    mutated_positions = find_mutation_information(wt_ids, mt_ids)
    assert (wt_ids[0] == mt_ids[0]) and (wt_ids[-1] == mt_ids[-1]), (
        "First and last tokens must be the same for both sequences"
    )
    sequence_masked_id = mask_sequence(
        wt_ids, mutated_positions, mask_token=model.alphabet["m"]
    )
    out = model(sequence_masked_id)  # model input
    logits = out.logits
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    ids = jnp.nonzero(mutated_positions)
    wt_score = log_prob[ids, wt_ids[ids]].sum()
    mt_score = log_prob[ids, mt_ids[ids]].sum()
    return mt_score - wt_score


def compute_esm_score_str(model: esmqx.ESM2, wt: str, mt: str):
    positions, wt_aa, mt_aa = find_mutation_information_str(wt, mt)
    sequence_masked = mask_sequence_str(wt, positions)
    # sequence_masked_id = esmqx.tokenise([sequence_masked]).squeeze(0)  # Tokenize the masked sequence
    out = model(sequence_masked)  # model input
    "NOTE: tokenize does not properly. It converts masking token to M residue."
    logits = out.logits[1:-1]  # remove batch dimension and special tokens
    assert len(wt) == logits.shape[0], "different shape!"
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    ids = jnp.array(positions)  # Convert positions to jnp array
    wt_ids = jnp.array(list(model.alphabet[_aa] for _aa in wt_aa))
    mt_ids = jnp.array(list(model.alphabet[_aa] for _aa in mt_aa))
    wt_score = log_prob[ids, wt_ids].sum()
    mt_score = log_prob[ids, mt_ids].sum()
    return mt_score - wt_score


def comparing_wt_and_mt(model, wt: str, mt: str):
    wt_ids, mt_ids = esmqx.tokenise([wt, mt])
    esm_score = compute_esm_score(model, wt_ids, mt_ids)
    print(f"ESM Score: {esm_score:.4f}")
    esm_score_str = compute_esm_score_str(model, wt, mt)
    print(f"ESM Score (string): {esm_score_str:.4f}")


def reconstruction(model: esmqx.ESM2, sequence: str):
    protein_ids = esmqx.tokenise([sequence])
    logits = jax.vmap(model)(protein_ids).logits.squeeze(0)[
        1:-1
    ]  # remove batch dimension and special tokens
    output_ids = jnp.argmax(logits, axis=-1)
    input_ids = protein_ids[0][1:-1]  # remove special tokens
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
