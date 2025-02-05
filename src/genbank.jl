using GenomicAnnotations
using FASTX: FASTA
"""
Read GenBank data (.gb file) located in `data/`, and return translated protein.
"""
function read_rvfv_data(file_name::String="sequence")
    chr = readgbk("data/$(file_name).gb")
    protein = @genes(chr, CDS).translation
    return protein
end

function convert_gb_to_fasta(file_name::String="sequence")
    chr = readgbk("data/$(file_name).gb")
    open(FASTA.Writer, "data/$(file_name).fasta") do w
        for gene in @genes(chr, CDS)
            seq = replace(gene.translation, "X"=>"")
            write(w, FASTA.Record(gene.protein_id, seq))
        end
    end
end