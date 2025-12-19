#!/usr/bin/env python3
"""
prepare_data.py

Data preparation utilities for CancerGPT
- Load DrugCombDB data
- Validate and clean data
- Create train/test/val splits
- Balance classes
- Feature engineering

Based on: Li et al. "CancerGPT for few shot drug pair synergy prediction"
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split, StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("data_prep")

# -----------------------
# DATA VALIDATION
# -----------------------

def validate_drugcombdb_data(df: pd.DataFrame) -> bool:
    """
    Validate DrugCombDB format
    
    Expected columns:
    - drugA: str - First drug name
    - drugB: str - Second drug name
    - cell_line: str - Cell line name
    - tissue: str - Tissue/cancer type
    - synergy_label: binary - Synergistic (1) or not (0)
    
    Optional columns:
    - sensitivity_A: float - Drug A sensitivity
    - sensitivity_B: float - Drug B sensitivity
    - loewe_score: float - Loewe synergy score
    - ic50_A: float - IC50 for drug A
    - ic50_B: float - IC50 for drug B
    """
    required = {"drugA", "drugB", "tissue", "synergy_label"}
    available = set(df.columns)
    
    if not required.issubset(available):
        missing = required - available
        log.error(f"Missing required columns: {missing}")
        return False
    
    # Validate synergy_label is binary
    unique_labels = df["synergy_label"].unique()
    if not set(unique_labels).issubset({0, 1}):
        log.error(f"synergy_label must be binary (0/1), got: {unique_labels}")
        return False
    
    log.info("Data validation passed")
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data"""
    df = df.copy()
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    log.info(f"Removed {initial_len - len(df)} duplicates")
    
    # Handle missing values
    # Drop rows with missing critical columns
    critical_cols = ["drugA", "drugB", "tissue", "synergy_label"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    log.info(f"Removed {before - len(df)} rows with missing critical values")
    
    # Fill optional numeric columns with mean
    numeric_cols = ["sensitivity_A", "sensitivity_B", "ic50_A", "ic50_B"]
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Convert string columns to lowercase
    string_cols = ["drugA", "drugB", "cell_line", "tissue"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
    
    log.info(f"Data cleaning complete: {len(df)} rows remaining")
    return df


def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """Calculate dataset statistics"""
    stats = {
        "total_samples": len(df),
        "unique_tissues": df["tissue"].nunique(),
        "unique_drugs_A": df["drugA"].nunique(),
        "unique_drugs_B": df["drugB"].nunique(),
        "class_distribution": df["synergy_label"].value_counts().to_dict(),
        "samples_by_tissue": df["tissue"].value_counts().to_dict()
    }
    
    log.info(f"\nDataset Statistics:")
    log.info(f"  Total samples: {stats['total_samples']}")
    log.info(f"  Tissues: {stats['unique_tissues']}")
    log.info(f"  Class distribution: {stats['class_distribution']}")
    
    return stats


def split_by_tissue(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by tissue (tissue-aware split)
    
    Returns:
        train_df, val_df, test_df
    """
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for tissue in df["tissue"].unique():
        tissue_data = df[df["tissue"] == tissue]
        
        if len(tissue_data) < 2:
            log.warning(f"Tissue {tissue} has only {len(tissue_data)} samples, skipping split")
            continue
        
        # Split with stratification
        temp, test = train_test_split(
            tissue_data,
            test_size=test_size,
            stratify=tissue_data["synergy_label"],
            random_state=random_state
        )
        
        train_size_adjusted = val_size / (1 - test_size)
        
        train, val = train_test_split(
            temp,
            test_size=train_size_adjusted,
            stratify=temp["synergy_label"],
            random_state=random_state
        )
        
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    log.info(f"\nData split (tissue-aware):")
    log.info(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    log.info(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    log.info(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def balance_classes(
    df: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance classes by undersampling majority class
    
    Args:
        df: DataFrame with synergy_label column
        random_state: For reproducibility
    
    Returns:
        Balanced DataFrame
    """
    counts = df["synergy_label"].value_counts()
    
    if len(counts) == 1:
        log.warning("Only one class present, skipping balancing")
        return df
    
    min_class = counts.idxmin()
    max_class = counts.idxmax()
    min_count = counts.min()
    
    # Get all minority class samples
    minority_df = df[df["synergy_label"] == min_class]
    
    # Subsample majority class
    majority_df = df[df["synergy_label"] == max_class].sample(
        n=min_count,
        random_state=random_state
    )
    
    balanced_df = pd.concat([minority_df, majority_df], ignore_index=True)
    
    log.info(f"\nClass balancing:")
    log.info(f"  Before: {counts[0]} vs {counts[1]}")
    log.info(f"  After: {len(minority_df)} vs {len(majority_df)}")
    
    return balanced_df


# -----------------------
# FEATURE ENGINEERING
# -----------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to dataset"""
    df = df.copy()
    
    # Drug pair similarity (placeholder - can use actual drug similarity)
    # This would typically use chemical similarity or bioactivity data
    if "drugA" in df.columns and "drugB" in df.columns:
        df["drug_pair_hash"] = (
            df["drugA"] + "_" + df["drugB"]
        ).apply(hash)
    
    # Tissue-specific features
    if "tissue" in df.columns:
        tissue_synergy_rates = df.groupby("tissue")["synergy_label"].mean()
        df["tissue_synergy_rate"] = df["tissue"].map(tissue_synergy_rates)
    
    # Sensitivity difference (if available)
    if "sensitivity_A" in df.columns and "sensitivity_B" in df.columns:
        df["sensitivity_diff"] = abs(
            df["sensitivity_A"] - df["sensitivity_B"]
        )
        df["sensitivity_product"] = (
            df["sensitivity_A"] * df["sensitivity_B"]
        )
    
    log.info(f"Added engineered features: {df.shape[1]} total columns")
    
    return df


# -----------------------
# FILE I/O
# -----------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from various formats"""
    path = Path(path)
    
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".xlsx":
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def save_dataset(
    df: pd.DataFrame,
    path: str,
    format: str = "csv"
) -> Path:
    """Save dataset to various formats"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    log.info(f"Saved dataset to {path}")
    return path


# -----------------------
# MAIN PIPELINE
# -----------------------

def prepare_dataset(
    input_path: str,
    output_dir: str = "data_prepared",
    test_size: float = 0.2,
    val_size: float = 0.1,
    balance: bool = True,
    add_features: bool = True,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Complete data preparation pipeline
    
    Args:
        input_path: Path to input dataset
        output_dir: Output directory
        test_size: Test set fraction
        val_size: Validation set fraction
        balance: Whether to balance classes
        add_features: Whether to add engineered features
        random_state: Random seed
    
    Returns:
        Dict with paths to prepared datasets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"\n{'='*70}")
    log.info("DATA PREPARATION PIPELINE")
    log.info(f"{'='*70}")
    
    # Load
    log.info(f"\nLoading data from {input_path}")
    df = load_dataset(input_path)
    log.info(f"Loaded {len(df)} samples")
    
    # Validate
    if not validate_drugcombdb_data(df):
        raise ValueError("Data validation failed")
    
    # Clean
    df = clean_data(df)
    get_dataset_statistics(df)
    
    # Add features
    if add_features:
        df = add_engineered_features(df)
    
    # Split by tissue
    train_df, val_df, test_df = split_by_tissue(
        df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    # Balance training set
    if balance:
        train_df = balance_classes(train_df, random_state=random_state)
    
    # Save splits
    paths = {}
    
    paths["train"] = str(save_dataset(train_df, output_dir / "train.csv"))
    paths["val"] = str(save_dataset(val_df, output_dir / "val.csv"))
    paths["test"] = str(save_dataset(test_df, output_dir / "test.csv"))
    
    # Save full dataset for k-shot experiments
    paths["full"] = str(save_dataset(df, output_dir / "full.csv"))
    
    # Save statistics
    stats = get_dataset_statistics(df)
    stats_file = output_dir / "statistics.json"
    
    import json
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    log.info(f"\n{'='*70}")
    log.info("DATA PREPARATION COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Output directory: {output_dir}")
    
    return paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare data for CancerGPT"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_prepared",
        help="Output directory"
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing"
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Disable feature engineering"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_path=args.input,
        output_dir=args.output,
        balance=not args.no_balance,
        add_features=not args.no_features
    )
