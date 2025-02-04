module AntigenDesign

include("genbank.jl")
export read_rvfv_data
include("iedb_api.jl")
export compute_bcell_epitope_score_iedb

end