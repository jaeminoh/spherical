#!/bin/bash

export RP="rosetta/source"
export F="data/antigen"

# relaxation
#$RP/bin/score_jd2.linuxgccrelease -ignore_unrecognized_res -s $F".pdb" -out:suffix _crystal
#$RP/bin/relax.default.linuxgccrelease -s $F".pdb" -out:suffix _relaxed -nstruct 1 -relax:default_repeats 5 -out:path:pdb data

# score
$RP/bin/score_jd2.linuxgccrelease -ignore_unrecognized_res -in:file:s $F"_relaxed_0001.pdb" -out:prefix $F
#$RP/bin/score_jd2.linuxgccrelease -ignore_unrecognized_res -in:file:s $F"_relaxed_0002.pdb" -out:suffix _0002
