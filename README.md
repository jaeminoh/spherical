# SPHERICAL

## Requirements

1. To perform AutoDock, [Boost](https://www.boost.org/users/download/) is required.

## Installation

1. clone this repository.
2. `uv sync`, see [uv docs](https://docs.astral.sh/uv/).

## Some commands for ChimeraX

- `select :12, 13-15` command selects (12, 13, 14, 15)th residues.
- `save xxx.pdb selectedOnly true` saves only selected ones to `.pdb` format.

## Some notes for RFDiffusion
[colab](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/rf/examples/diffusion.ipynb)

- contigs `5-10/A15-30/25-30` fixes (15, ..., 30)th residues of the A chain, samples 5~10 residues, and samples 25~30 residues randomly.
- sampled protein structures are validated with protein MPNN (3D structure to sequence) + AF prediction (sequence to 3D structure).

## ZDock

- from ZDock pdb to ChimeraX compatible PDB: `awk '{printf("%54.54s\n", $0)}' < complex.1.pdb > c.1.pdb`