from esm import Alphabet
from transformers import AutoTokenizer
from Bio import SeqIO


# token list
alphabet = Alphabet.from_architecture("ESM-1b")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/final_20250911_124633")
assert alphabet.all_toks == tokenizer.all_tokens, "Token lists do not match"

# tokenization
test_file = "assets/test.fasta"
sequences = []
for record in SeqIO.parse(test_file, "fasta"):
    sequences.append(str(record.seq))

batch_converter = alphabet.get_batch_converter()
labels, strs, toks = batch_converter([("protein1", sequences[0])])

assert (toks == tokenizer(sequences, return_tensors="pt")["input_ids"]).sum() == len(sequences[0]) + 2, "Tokenization does not match"