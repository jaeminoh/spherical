export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

python spherical/esm2/_fold.py --filepath="results/rvfv/bottom100_contact_mse.csv"
python spherical/esm2/_fold.py --filepath="results/rvfv/top100_contact_mse.csv"