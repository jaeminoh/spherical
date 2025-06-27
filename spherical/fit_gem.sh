#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

#python directed_evolution.py
#python utils.py
python global_epistasis_model.py --num_samples=100
python collect_beta.py

python global_epistasis_model.py --num_samples=1500
python collect_beta.py