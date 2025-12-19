import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("labeled_dataset.parquet")

splits = []

for tissue, g in df.groupby("tissue"):
    if len(g) < 200:
        continue

    train, test = train_test_split(
        g, test_size=0.2, stratify=g["synergy_label"], random_state=42
    )

    train["split"] = "train"
    test["split"] = "test"

    splits.append(train)
    splits.append(test)

final_df = pd.concat(splits)
final_df.to_parquet("final_split_dataset.parquet")
