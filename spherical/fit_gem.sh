#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python directed_evolution.py
python utils.py
python global_epistasis_model.py
python collect_beta.py