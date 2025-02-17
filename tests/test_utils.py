import spherical.utils as spu
import pandas as pd


class TestSequences:
    seq = "abcdEFGH"
    pp = spu.where_uppercase(seq)

    def test_sequence(self):
        assert self.pp == [5, 6, 7, 8]

    def test_save_routine(self, tmp_path):
        path = f"{tmp_path}/tmp.csv"
        spu.save_csv(self.pp, path)  # save result
        ppp = pd.read_csv(path, header=None)
        assert self.pp == ppp.values.squeeze().tolist()
