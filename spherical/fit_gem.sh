#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python mutate_score.py --num_samples=10000 --min_mutations=5 --max_mutations=5 # training dataset
python global_epistasis_model.py --mode="train" 
python collect_beta.py