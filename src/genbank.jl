using GenomicAnnotations

function read_rvfv_data()
    chr = readgbk("data/sequence.gb")
    protein = @genes(chr, CDS).translation
    return protein
end

