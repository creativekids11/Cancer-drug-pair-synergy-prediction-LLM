import pandas as pd

df = pd.read_parquet("final_split_dataset.parquet")

def build_prompt(r):
    return (
        f"Drug 1: {r.drugA}\n"
        f"Drug 2: {r.drugB}\n"
        f"Cell line: {r.cell_line}\n"
        f"Tissue: {r.tissue}\n\n"
        "Is this drug combination synergistic?"
    )

df["prompt"] = df.apply(build_prompt, axis=1)
df["answer"] = df["synergy_label"].map({1: "Yes", 0: "No"})

# JSONL for LLMs
df[["prompt", "answer", "split", "tissue"]].to_json(
    "cancergpt_dataset.jsonl",
    orient="records",
    lines=True
)

# CSV for ML baselines
df.to_csv("cancergpt_dataset.csv", index=False)
