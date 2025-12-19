import pandas as pd

df = pd.read_parquet("clean_dataset.parquet")

df["synergy_label"] = (df["synergy_score"] > 5).astype(int)

df.to_parquet("labeled_dataset.parquet")
