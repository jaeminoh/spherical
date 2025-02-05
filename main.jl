using AntigenDesign
using PyCall

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")

# load protein sequence
pp = read_rvfv_data() # 375

# predict epitopes
#rr = compute_bcell_epitope_score_iedb.(pp) # too many requests


function max_score(result)
    i = argmax(result.Score)
    return result[i,:] # row of maximum score
end

ee = max_score.(rr[1:5])