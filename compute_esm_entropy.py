from transformers import EsmForMaskedLM, EsmForProteinFolding
import pandas as pd
from tqdm import tqdm
import spherical

protein = spherical.RVFV_MP12
model, tokenizer = spherical.esm2.load_model(EsmForMaskedLM, spherical.ESM650M)
entropy = spherical.esm2.Entropy(model, tokenizer)
e = entropy(protein.sequence)

df = pd.DataFrame({"pos": [i + 1 for i in range(len(e))], "entropy": e.tolist()})
df.to_csv("results/rvfv/entropy.csv", index=False)

df = df.sort_values(by="entropy")

# lowest entropy position
aa_list = list(spherical.AMINO_ACIDS)
pos = df.iloc[0]["pos"].astype(int) - 1
positions = [pos for _ in aa_list]

model, _ = spherical.esm2.load_model(EsmForProteinFolding, spherical.ESMFOLD)
model.to("cuda")

for position, aa in tqdm(zip(positions, aa_list)):
    mutant = protein._mutate(position, aa)
    file_name = f"results/rvfv/{position}{aa}.pdb"
    spherical.esm2.fold_and_save(model, mutant, file_name)


# highest entropy position
pos = df.iloc[-1]["pos"].astype(int) - 1
positions = [pos for _ in aa_list]
for position, aa in tqdm(zip(positions, aa_list)):
    mutant = protein._mutate(position, aa)
    file_name = f"results/rvfv/{position}{aa}.pdb"
    spherical.esm2.fold_and_save(model, mutant, file_name)
