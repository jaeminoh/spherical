import torch
from transformers import EsmModel, AutoTokenizer
from esm import FastaBatchedDataset, Alphabet



# load model from Hugging Face
model_name = "checkpoints/final_20250911_124633"
model = EsmModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
alphabet = Alphabet.from_architecture("ESM-1b")

fasta_file = "assets/test.fasta"
dataset = FastaBatchedDataset.from_file(fasta_file)
batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
)
print(f"Read {fasta_file} with {len(dataset)} sequences")


model.num_layers = model.config.num_hidden_layers

assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [-1])
repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [-1]]

for batch_idx, (labels, strs, toks) in enumerate(data_loader):
    print(
        f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
    )

    out = model(toks)
    print(out["last_hidden_state"].shape)
    if batch_idx == 0:
        break