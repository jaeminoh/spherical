import os
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt

def load_genus(
        family: str,
        genus: str,):
    dir_path = f"embeddings/{family}/{genus}"
    embeddings_path = os.listdir(dir_path)
    embeddings = []
    for f in embeddings_path:
        file  = torch.load(f"{dir_path}/{f}")
        embeddings.append(file["mean_representations"][6])
    return np.stack(embeddings)

def main(family: str, seed: int = 2025):
    dir_path = f"embeddings/{family}"
    genera_name = [f.name for f in os.scandir(dir_path) if f.is_dir()]
    embeddings = []
    for genus in genera_name:
        embeddings.append(load_genus(family, genus))

    reducer = umap.UMAP(random_state=seed)
    reducer.fit(np.concatenate(embeddings))
    for i, genus in enumerate(genera_name):
        reduced = reducer.transform(embeddings[i])
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, marker="x", label=f"{genus}")
    plt.title(f"UMAP projection of {family}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"umap_clustering_{family}.pdf", format="pdf")

if __name__ == "__main__":
    import fire
    fire.Fire(main)