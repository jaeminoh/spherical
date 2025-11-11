import torch
from transformers import EsmTokenizer, EsmModel
from Bio.Seq import MutableSeq
from Bio import SeqIO

import spherical.esm2 as esm2


class Main:
    """
    if esm2_size == "small": 8M
    if esm2_size == "medium": 650M
    """

    def __init__(self, esm2_size: str = "small", wt_path: str = "assets/RVFV.fasta"):
        wt = SeqIO.read(wt_path, "fasta").seq
        mt = MutableSeq(wt)
        mt[12] = "M"
        self.wt = str(wt)
        self.mt = str(mt)
        self.esm2_size = esm2_size
        if esm2_size == "small":
            model_name = "facebook/esm2_t6_8M_UR50D"
        elif esm2_size == "medium":
            model_name = "facebook/esm2_t33_650M_UR50D"
        self.model = EsmModel.from_pretrained(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)

    def test(self):
        wt_ids, mt_ids, _ = esm2.MaskedMarginal(self.model, self.tokenizer).compute_ids(
            self.wt, self.mt
        )
        with torch.no_grad():
            inputs_id = torch.stack([wt_ids, mt_ids])
            wt_embedding, mt_embedding = self.model(inputs_id).last_hidden_state.numpy()

        # embedding size
        print(f"embedding size: {wt_embedding.shape}")

        # bos
        wt_bos = wt_embedding[0]
        mt_bos = mt_embedding[0]
        print(f"bos difference: {max(abs(wt_bos - mt_bos))}")

        # eos
        wt_eos = wt_embedding[-1]
        mt_eos = mt_embedding[-1]
        print(f"eos difference: {max(abs(wt_eos - mt_eos))}")

        # mean
        wt_mean = wt_embedding.mean(axis=0)
        mt_mean = mt_embedding.mean(axis=0)
        print(f"mean difference: {max(abs(wt_mean - mt_mean))}")


if __name__ == "__main__":
    import fire

    fire.Fire(Main)
