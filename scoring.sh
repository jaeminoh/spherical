#!/bin/bash
# $1 : pdb
# $2 : Ag chain
# $3 : Ab heavy chain
# $4 : Ab light chain

export RP="rosetta/source/bin"
# relaxation
#$RP/relax.default.linuxgccrelease -s "SAbDab/$1.pdb" -nstruct 1 -relax:default_repeats 5

# dR
$RP/InterfaceAnalyzer.linuxgccrelease -s "$1_0001.pdb" -interface "$2_$3$4" -out:file:score_only "SAbDab/$1.sc" -ignore_unrecognized_res

# total energy
#$RP/score_jd2.linuxgccrelease -ignore_unrecognized_res -s "$1_0001.pdb" -out:file:score_only "SAbDab/$1_total.sc" -ignore_unrecognized_res