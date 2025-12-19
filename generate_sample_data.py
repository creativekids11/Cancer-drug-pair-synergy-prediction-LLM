#!/usr/bin/env python3
"""
generate_sample_data.py

Generate synthetic DrugCombDB-like sample data for testing the CancerGPT pipeline
This creates realistic-looking data without needing actual DrugCombDB access

Usage:
    python generate_sample_data.py --output sample_data.csv --samples 2000
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("sample_data")

# Drug names (real examples)
DRUGS = [
    "BRAF inhibitor", "MEK inhibitor", "EGFR inhibitor", "HER2 inhibitor",
    "KRAS inhibitor", "ALK inhibitor", "PDGFR inhibitor", "FLT3 inhibitor",
    "MET inhibitor", "PI3K inhibitor", "mTOR inhibitor", "CDK4/6 inhibitor",
    "PD-L1 inhibitor", "CTLA-4 inhibitor", "IL-2 inhibitor", "TNF-alpha inhibitor",
    "BCL-2 inhibitor", "MCL-1 inhibitor", "PROTAC degrader", "DNA-PKcs inhibitor",
    "ATM inhibitor", "PARP inhibitor", "Topoisomerase inhibitor", "Microtubule stabilizer",
    "Histone deacetylase inhibitor", "Proteasome inhibitor", "Heat shock protein inhibitor",
    "NF-kappa-B inhibitor", "JAK inhibitor", "STAT3 inhibitor"
]

# Cell line names (real examples)
CELL_LINES = [
    "A375", "A2780", "A431", "A549", "Caco-2", "HeLa", "HepG2", "HL60",
    "Huh-7", "K562", "MCF-7", "MDA-MB-231", "MDA-MB-453", "MEWO", "NALM-6",
    "OVCAR-3", "PC-3", "SK-MEL-2", "SW480", "U251", "U87", "COLO-829",
    "JIMT-1", "SUM-159", "ZR-75-1", "MV4-11", "KG-1", "MOLT-4", "CCRF-CEM", "RS4-11"
]

# Cancer tissues/types (rare tissues as per paper)
TISSUES = [
    "pancreas", "endometrium", "liver", "soft tissue", "stomach",
    "urinary tract", "bone",
    # Add some common tissues for potential pretraining
    "breast", "melanoma", "lung", "colorectal", "ovarian", "prostate",
    "head and neck", "gastric", "renal", "esophageal"
]

def generate_sample_data(
    num_samples: int = 2000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic DrugCombDB-like data
    
    Args:
        num_samples: Number of samples to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with drug pair data
    """
    np.random.seed(random_state)
    
    log.info(f"Generating {num_samples} synthetic samples...")
    
    # Generate basic data
    data = {
        "drugA": np.random.choice(DRUGS, num_samples),
        "drugB": np.random.choice(DRUGS, num_samples),
        "cell_line": np.random.choice(CELL_LINES, num_samples),
        "tissue": np.random.choice(TISSUES, num_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure drugA != drugB (can't pair a drug with itself)
    same_drug_mask = df["drugA"] == df["drugB"]
    while same_drug_mask.any():
        df.loc[same_drug_mask, "drugB"] = np.random.choice(DRUGS, same_drug_mask.sum())
        same_drug_mask = df["drugA"] == df["drugB"]
    
    # Generate sensitivities (IC50 values, typically 0-10 in log scale)
    df["sensitivity_A"] = np.random.exponential(2, num_samples)
    df["sensitivity_B"] = np.random.exponential(2, num_samples)
    
    # Generate Loewe synergy scores (typically -10 to 10, >5 is synergistic)
    df["loewe_score"] = np.random.normal(0, 4, num_samples)
    
    # Create binary synergy label (synergistic if Loewe > 5)
    df["synergy_label"] = (df["loewe_score"] > 5).astype(int)
    
    # Adjust distribution: make synergy less common (more realistic)
    # Rare tissues have even lower synergy rate
    common_tissues = ["breast", "melanoma", "lung", "colorectal"]
    rare_tissues = ["pancreas", "endometrium", "liver", "soft tissue", "stomach",
                    "urinary tract", "bone"]
    
    # For common tissues: 20% synergy
    common_mask = df["tissue"].isin(common_tissues)
    df.loc[common_mask, "synergy_label"] = np.random.binomial(1, 0.2, common_mask.sum())
    
    # For rare tissues: 10% synergy (harder to find synergies)
    rare_mask = df["tissue"].isin(rare_tissues)
    df.loc[rare_mask, "synergy_label"] = np.random.binomial(1, 0.1, rare_mask.sum())
    
    # Recalculate Loewe scores to match labels
    df["loewe_score"] = np.where(
        df["synergy_label"] == 1,
        np.random.uniform(5, 15, num_samples),  # Synergistic
        np.random.uniform(-10, 5, num_samples)   # Not synergistic
    )
    
    # Reorder columns
    df = df[["drugA", "drugB", "cell_line", "tissue", "sensitivity_A", 
             "sensitivity_B", "loewe_score", "synergy_label"]]
    
    log.info(f"Generated data shape: {df.shape}")
    log.info(f"Synergy distribution: {df['synergy_label'].value_counts().to_dict()}")
    log.info(f"Tissues: {df['tissue'].nunique()} unique")
    log.info(f"  Common tissues: {df[df['tissue'].isin(common_tissues)].shape[0]} samples")
    log.info(f"  Rare tissues: {df[df['tissue'].isin(rare_tissues)].shape[0]} samples")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic DrugCombDB sample data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_data.csv",
        help="Output file path"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    log.info("\n" + "="*70)
    log.info("SAMPLE DATA GENERATION")
    log.info("="*70)
    
    # Generate data
    df = generate_sample_data(
        num_samples=args.samples,
        random_state=args.seed
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    log.info(f"\nSaved to: {output_path}")
    log.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    log.info("\n" + "="*70)
    log.info("NEXT STEPS")
    log.info("="*70)
    log.info(f"1. Prepare data:")
    log.info(f"   python prepare_data.py --input {args.output} --output data_prepared")
    log.info(f"\n2. Run experiments:")
    log.info(f"   python run_experiments.py --data-path data_prepared/full.csv")
    log.info(f"\n3. Evaluate:")
    log.info(f"   python evaluate_cancergpt.py --data-path data_prepared/full.csv --plot")


if __name__ == "__main__":
    main()
