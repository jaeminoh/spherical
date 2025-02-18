import fire

import spherical.utils as spu


def main():
    st = spu.read_cif("../data/alphafold/fold_2025_02_10_08_03_model_0.cif")
    rr = st.get_residues()
    r1 = next(rr)
    r2 = next(rr)
    d = spu.rr_distance(r1, r2)
    print(d)


if __name__ == "__main__":
    fire.Fire(main)
