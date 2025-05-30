#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python mutate_score.py --num_samples=100000 --min_mutations=3 --max_mutations=7 # training dataset
#python mutate_score.py --num_samples=100 --min_mutations=10 --max_mutations=15 --output_csv_path="data/test.csv"
python global_epistasis_model.py --mode="train" 
#python mutate_score.py --num_samples=100 --min_mutations=1 --max_mutations=3 --output_csv_path="data/test3.csv"
#python global_epistasis_model.py --mode="test" --max_mutations=3
python collect_beta.py