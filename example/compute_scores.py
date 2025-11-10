import os

from transformers import EsmTokenizer, EsmForMaskedLM
from Bio.Seq import MutableSeq
from Bio import SeqIO
import pandas as pd
import numpy as np

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
        self.model = EsmForMaskedLM.from_pretrained(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)

    def test(self):
        # usage: python compute_scores.py --esm2_size="small" - test
        print(esm2.MaskedMarginal(self.model, self.tokenizer)(self.wt, self.mt))
        print(esm2.MutantMarginal(self.model, self.tokenizer)(self.wt, self.mt))
        print(esm2.WildtypeMarginal(self.model, self.tokenizer)(self.wt, self.mt))

    def compute_entropy(self):
        # usage: python compute_scores.py --esm2_size="small" - compute_entropy
        entropy = esm2.Entropy(self.model, self.tokenizer)(self.wt).numpy()
        os.makedirs("results", exist_ok=True)
        data = pd.DataFrame(
            {"index": np.arange(1, len(entropy) + 1), "entropy": entropy}
        )
        data.to_csv(f"results/rmp12_entropy_{self.esm2_size}.csv", index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(Main)
