export CUDA_VISIBLE_DEVICES=0

model=esm2_t33_650M_UR50D.pt
family=Phenuiviridae


# for genus in Bandavirus Coguvirus Phasivirus Phlebovirus Rubodvirus Tenuivirus Uukuvirus Unclassified
# do
#     python scripts/extract.py checkpoints/$model assets/$family/$genus.fasta embeddings/$family/$genus --include mean
# done

python scripts/umap_clustering.py --family=$family --repr_layers=33