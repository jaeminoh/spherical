#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python mutate_score.py --num_samples=10000 --min_mutations=10 --max_mutations=30 # training dataset
python mutate_score.py --num_samples=100 --min_mutations=10 --max_mutations=15 --output_csv_path="data/test.csv"
python global_epistasis_model.py --mode="train" 
python global_epistasis_model.py --mode="test"