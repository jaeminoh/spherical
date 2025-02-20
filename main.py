import spherical.utils as spu


def main(thr: float = 8.0):
    epi_fasta = spu.read_fasta("data/result/Bcell_epitope_preds.fasta")
    epi_seq = next(epi_fasta).seq
    path_cif = "data/alphafold/fold_2025_02_10_08_03_model_0.cif"
    st = spu.read_cif(path_cif)
    rr = list(st.get_residues())
    epi_groups = spu.group_residues_by_distance(rr, epi_seq, thr=thr)
    print("epitope groups")
    print(epi_groups)

    # TODO what would be the best cutoff for nearby residue?
    groups = spu.add_nearby_ones_to_groups(rr, epi_seq, epi_groups, thr=thr / 2.2)

    groups = [spu.convert_indexing(group, 0, 1) for group in groups]
    print("\n epitope regions")
    print(groups)

    groups_flat = [index for group in groups for index in group]
    print(f"How many involved residues are: {len(set(groups_flat))}")

    for i, group in enumerate(groups):
        spu.write_selected_residues_to_pdb(st, group, out_path=f"data/pdb/{i}.pdb")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
