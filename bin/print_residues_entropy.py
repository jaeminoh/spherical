import pandas as pd


def main(top_k: int = 50, file_name: str = "rmp12_entropy_medium.csv"):
    entropy = pd.read_csv(f"results/{file_name}")
    entropy = entropy.sort_values(by="entropy")

    low = entropy[:top_k]
    high = entropy[-top_k:]

    print(f"{top_k} Low entropy indices:")
    print(low["index"].to_list())
    print(f"{top_k} High entropy indices:")
    print(high["index"].to_list())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
