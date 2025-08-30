export CUDA_VISIBLE_DEVICES=3

model=esm2_t6_8M_UR50D.pt
family=Phenuiviridae


# for genus in Bandavirus Beidvirus Phlebovirus Rubodvirus
# do
#     python scripts/extract.py checkpoints/$model assets/$family/$genus.fasta embeddings/$family/$genus --include mean
# done

python scripts/umap_clustering.py --family=$family