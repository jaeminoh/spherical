pos_list=(1006 1022 367 1006 894 29 29 29 29 27)
aa_list=("I" "M" "V" "L" "Q" "S" "E" "N" "P" "C")

for i in "${!pos_list[@]}"
do
    x=${pos_list[i]}
    y=${aa_list[i]}
    postfix="${x}${y}"
    python spherical/_deformation.py --protA results/rvfv/wt.pdb --protB results/rvfv/${postfix}.pdb --output results/rvfv/${postfix}.csv
done



