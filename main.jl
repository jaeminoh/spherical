using AntigenDesign

p = read_rvfv_data()
r = compute_bcell_epitope_score_iedb(p[1])
print(r)