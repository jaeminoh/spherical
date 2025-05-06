from Bio import SeqIO
import pandas as pd


def load_SAbDab():
    df = pd.read_csv("SAbDab/SAbDab.tsv", sep="\t")
    df = df[
        ~pd.isna(df.affinity)
        & ~pd.isna(df.delta_g)
        & ~pd.isna(df.temperature)
        & ~pd.isna(df.Hchain)
        & ~pd.isna(df.Lchain)
        & ~pd.isna(df.antigen_chain)
    ]
    return df


def return_record_id(pdb_id: str, record_id: str):
    record_id = record_id.lower()
    if pdb_id not in record_id:
        record_id = f"{pdb_id}:{record_id}"
    return record_id


def extract_sequences(df):
    ag_list = []
    h_list = []
    l_list = []
    dg_list = []

    for _, row in df.iterrows():
        pdb_id = f"{row.pdb}"
        ag_chain = row.antigen_chain.lower()
        if "|" not in ag_chain:  # some antigen has multiple chains
            h_chain = row.Hchain.lower()
            l_chain = row.Lchain.lower()
            chain = {
                return_record_id(pdb_id, record.id): record.seq
                for record in SeqIO.parse(f"SAbDab/{pdb_id}.pdb", "pdb-seqres")
            }
            ag_list.append(chain[pdb_id + ":" + ag_chain])
            h_list.append(chain[pdb_id + ":" + h_chain])
            l_list.append(chain[pdb_id + ":" + l_chain])
            dg_list.append(row.delta_g)

    return (ag_list, h_list, l_list), dg_list


if __name__ == "__main__":
    import numpy as np

    df = load_SAbDab()
    ag_list, h_list, l_list = extract_sequences(df)

    ag_len = np.array([len(_ag) for _ag in ag_list])
    h_len = np.array([len(_h) for _h in h_list])
    l_len = np.array([len(_l) for _l in l_list])

    print(f"""
        There are total {len(ag_len)} AgAb complexes.
        """)
    print("stats (mean, min, max)")
    print(f"antigen sequence: ({ag_len.mean()}, {ag_len.min()}, {ag_len.max()})")
    print(f"antibody heavy-chain: ({h_len.mean()}, {h_len.min()}, {h_len.max()})")
    print(f"antibody light-chain: ({l_len.mean()}, {l_len.min()}, {l_len.max()})")
