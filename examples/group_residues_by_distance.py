import spherical.utils as spu


def main(thr: float = 8.0):
    fasta = spu.read_fasta("../data/result/Bcell_epitope_preds.fasta")
    epi_seq = next(fasta).seq
    path_cif = "../data/alphafold/fold_2025_02_10_08_03_model_0.cif"
    epi_groups = spu.group_residues_by_distance(epi_seq, path_cif=path_cif, thr=thr)
    print("epitope groups")
    print(epi_groups)

    # TODO what would be the best cutoff for nearby residue?
    groups = spu.add_nearby_ones_to_groups(path_cif, epi_seq, epi_groups, thr=thr / 2.2)
    print("\n epitope regions")
    print(groups)

    groups_flat = [index for group in groups for index in group]
    print(f"How many residues: {len(set(groups_flat))}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
