import spherical.utils as spu


def main(thr: float = 4.0):
    fasta = spu.read_fasta("../data/result/Bcell_epitope_preds.fasta")
    epi_seq = next(fasta).seq
    epi_groups = spu.group_residues_by_distance(
        epi_seq, path_cif="../data/alphafold/fold_2025_02_10_08_03_model_0.cif", thr=thr
    )
    print(epi_groups)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
