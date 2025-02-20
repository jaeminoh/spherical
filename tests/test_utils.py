import spherical.utils as spu
import pandas as pd


class TestSequences:
    seq = "abcdEFGH"
    pp0 = spu.where_uppercase(seq)
    pp1 = [5, 6, 7, 8]

    def test_upper_and_lower(self):
        assert self.pp0 == [4, 5, 6, 7]
        assert spu.where_lowercase(self.seq) == [0, 1, 2, 3]

    def test_indexing_conversion(self):
        assert spu.convert_indexing(self.pp0, 0, 1) == self.pp1
        assert spu.convert_indexing(self.pp1, 1, 0) == self.pp0

    def test_remove_lowercases(self):
        assert spu.remove_lowercase(self.seq) == "EFGH"

    def test_save_routine(self, tmp_path):
        path = f"{tmp_path}/tmp.csv"
        spu.save_csv(self.pp0, path)  # save result
        ppp = pd.read_csv(path, header=None)
        assert self.pp0 == ppp.values.squeeze().tolist()
