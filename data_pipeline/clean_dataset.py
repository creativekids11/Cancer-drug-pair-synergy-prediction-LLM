import pandas as pd
import numpy as np

df = pd.read_parquet("raw_integration.parquet")

# Drop invalid rows
df = df.dropna(subset=["drugA", "drugB", "cell_line", "synergy_score"])

# Canonical ordering of drug pairs
df[["drugA", "drugB"]] = np.sort(df[["drugA", "drugB"]], axis=1)

# Remove duplicates
df = df.drop_duplicates(
    subset=["drugA", "drugB", "cell_line"]
)

# Ensure numeric
df["synergy_score"] = pd.to_numeric(df["synergy_score"], errors="coerce")
df = df.dropna(subset=["synergy_score"])

df.to_parquet("clean_dataset.parquet")
print("Clean dataset:", df.shape)
