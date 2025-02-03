#!/bin/bash
# NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=jax_flow
#SBATCH --time=05:30:00
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --output=slurm.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1

cd $SCRATCH/SPHERICAL
ml purge
source .venv/bin/activate
python discrete_flow_matching.py