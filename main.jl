using DelimitedFiles, DataFrames

"""
IEDB Antibody Epitope Prediction (http://tools.iedb.org/main/tools-api/)
"""
function compute_bcell_epitope_score_iedb(
    sequence::String="VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTE",
    method::String = "Emini";
    window_size::Int=6
    )
    result = read(`curl --data "method=$(method)&sequence_text=$(sequence)&window_size=$(window_size)" http://tools-cluster-interface.iedb.org/tools_api/bcell/`, String)
    open("temp.txt", "w") do io
        write(io, result)
    end
    df = readdlm("temp.txt", '\t')
    rm("temp.txt")
    result = DataFrame(df[2:end,:], df[1,:])
    return result
end
