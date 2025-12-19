#!/usr/bin/env python3
"""
cancergpt_kshot_finetuning.py

K-shot fine-tuning for rare tissue drug synergy prediction
Based on CancerGPT paper methodology

Implements:
- k-shot data sampling with class balance
- Common tissue pre-training
- Rare tissue fine-tuning
- Comparison of full vs last-layer training
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from cancergpt_model import (
    CancerGPTModel, DrugSynergyDataset,
    CancerGPTTrainer, CancerGPTPredictor
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("kshot_finetuning")

# -----------------------
# CONFIGURATION
# -----------------------

class KShotConfig:
    """K-shot fine-tuning configuration"""
    
    # Model settings
    MODEL_NAME = "gpt2"
    HIDDEN_SIZE = 768
    DROPOUT_RATE = 0.1
    
    # Training settings
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 4
    BATCH_SIZE = 8
    
    # K-shot settings
    K_SHOTS = [0, 2, 4, 8, 16, 32, 64, 128]
    MIN_POSITIVE_SAMPLES = 1  # Ensure at least one positive
    MIN_NEGATIVE_SAMPLES = 1  # Ensure at least one negative
    
    # Data settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    SYNERGY_THRESHOLD = 5.0  # Loewe score > 5 = synergistic
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output
    OUTPUT_DIR = Path("cancergpt_outputs")


# -----------------------
# K-SHOT SAMPLING
# -----------------------

class KShotSampler:
    """Sample k-shot data with class balance"""
    
    @staticmethod
    def sample_balanced(
        df: pd.DataFrame,
        k: int,
        label_col: str = "synergy_label",
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Sample k examples with balanced classes
        
        Args:
            df: DataFrame with labels
            k: Number of samples to select
            label_col: Column name for labels
            random_state: Random seed
        
        Returns:
            DataFrame with k balanced samples
        """
        if k == 0:
            return pd.DataFrame(columns=df.columns)
        
        np.random.seed(random_state)
        
        # Separate by class
        positives = df[df[label_col] == 1]
        negatives = df[df[label_col] == 0]
        
        # Calculate samples per class
        samples_per_class = k // 2
        
        # Sample with balance
        pos_sample = positives.sample(
            n=min(samples_per_class, len(positives)),
            random_state=random_state
        )
        neg_sample = negatives.sample(
            n=min(samples_per_class, len(negatives)),
            random_state=random_state
        )
        
        samples = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)
        
        log.info(
            f"Sampled k={k}: {len(pos_sample)} positive, "
            f"{len(neg_sample)} negative"
        )
        
        return samples
    
    @staticmethod
    def incremental_sampling(
        df: pd.DataFrame,
        max_k: int,
        k_values: List[int],
        random_state: int = 42
    ) -> Dict[int, pd.DataFrame]:
        """
        Sample multiple k values while maintaining consistency
        
        For k=2 → k=4, we keep the original 2 samples and add 2 new ones
        
        Args:
            df: Full training data
            max_k: Maximum k to sample
            k_values: List of k values to sample
            random_state: Random seed
        
        Returns:
            Dictionary mapping k -> sampled DataFrame
        """
        np.random.seed(random_state)
        
        samples_dict = {}
        prev_indices = []
        
        for k in k_values:
            if k == 0:
                samples_dict[0] = pd.DataFrame(columns=df.columns)
                continue
            
            # Separate by class
            positives = df[df["synergy_label"] == 1]
            negatives = df[df["synergy_label"] == 0]
            
            samples_per_class = k // 2
            
            # Sample with balance, ensuring we don't repeat previous samples
            available_pos = positives.drop(prev_indices, errors='ignore')
            available_neg = negatives.drop(prev_indices, errors='ignore')
            
            pos_sample = available_pos.sample(
                n=min(samples_per_class, len(available_pos)),
                random_state=random_state
            )
            neg_sample = available_neg.sample(
                n=min(samples_per_class, len(available_neg)),
                random_state=random_state
            )
            
            current_samples = pd.concat([pos_sample, neg_sample])
            samples_dict[k] = current_samples.reset_index(drop=True)
            
            # Track indices for next iteration
            prev_indices = list(current_samples.index)
            
            log.info(
                f"k={k}: {len(pos_sample)} positive, "
                f"{len(neg_sample)} negative"
            )
        
        return samples_dict


# -----------------------
# RARE TISSUE EVALUATION
# -----------------------

class RareTissueEvaluator:
    """Evaluate model on rare tissues"""
    
    def __init__(
        self,
        config: KShotConfig = KShotConfig()
    ):
        self.config = config
        self.results = {}
    
    def evaluate_tissue(
        self,
        tissue_name: str,
        tissue_data: pd.DataFrame,
        model_checkpoint: Optional[str] = None,
        fine_tuning_strategy: str = "full"
    ) -> Dict:
        """
        Evaluate on a rare tissue
        
        Args:
            tissue_name: Name of tissue
            tissue_data: DataFrame with tissue data
            model_checkpoint: Path to pretrained model
            fine_tuning_strategy: "full" or "last_layer"
        
        Returns:
            Results dictionary
        """
        log.info(f"\n{'='*70}")
        log.info(f"Evaluating tissue: {tissue_name}")
        log.info(f"{'='*70}")
        log.info(f"Total samples: {len(tissue_data)}")
        
        # Split into train and test
        train_data, test_data = train_test_split(
            tissue_data,
            test_size=self.config.TEST_SIZE,
            stratify=tissue_data["synergy_label"],
            random_state=self.config.RANDOM_STATE
        )
        
        log.info(
            f"Train: {len(train_data)} | Test: {len(test_data)} | "
            f"Synergy ratio: {train_data['synergy_label'].mean():.2%}"
        )
        
        tissue_results = {
            "tissue": tissue_name,
            "total_samples": len(tissue_data),
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "synergy_ratio": train_data["synergy_label"].mean(),
            "k_shot_results": {}
        }
        
        # K-shot experiments
        for k in self.config.K_SHOTS:
            if k == 0:
                # Zero-shot: test on pretrained model without fine-tuning
                log.info(f"\nZero-shot evaluation (k={k})")
                results = self._zero_shot_eval(test_data, model_checkpoint)
            else:
                # K-shot: fine-tune then evaluate
                if k > len(train_data):
                    log.warning(f"k={k} exceeds available training samples ({len(train_data)}), skipping")
                    continue
                
                log.info(f"\nK-shot evaluation (k={k})")
                results = self._kshot_eval(
                    train_data, test_data, k,
                    model_checkpoint, fine_tuning_strategy
                )
            
            tissue_results["k_shot_results"][k] = results
        
        self.results[tissue_name] = tissue_results
        return tissue_results
    
    def _zero_shot_eval(
        self,
        test_data: pd.DataFrame,
        model_checkpoint: Optional[str]
    ) -> Dict:
        """Zero-shot evaluation without fine-tuning"""
        
        # Load model
        if model_checkpoint:
            model = CancerGPTModel(self.config.MODEL_NAME)
            checkpoint = torch.load(model_checkpoint, map_location=self.config.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = CancerGPTModel(self.config.MODEL_NAME)
        
        # Create dataset and evaluate
        test_dataset = DrugSynergyDataset(test_data, model.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE)
        
        trainer = CancerGPTTrainer(
            model, device=self.config.DEVICE,
            num_epochs=1  # Not used in evaluation
        )
        
        metrics = trainer.evaluate(test_loader)
        
        log.info(
            f"Zero-shot - Accuracy: {metrics['accuracy']:.4f}, "
            f"AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}"
        )
        
        return metrics
    
    def _kshot_eval(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        k: int,
        model_checkpoint: Optional[str],
        fine_tuning_strategy: str
    ) -> Dict:
        """K-shot fine-tuning and evaluation"""
        
        # Sample k examples
        sampler = KShotSampler()
        kshot_train = sampler.sample_balanced(
            train_data, k,
            random_state=self.config.RANDOM_STATE
        )
        
        # Load model
        if model_checkpoint:
            model = CancerGPTModel(
                self.config.MODEL_NAME,
                freeze_backbone=(fine_tuning_strategy == "last_layer")
            )
            checkpoint = torch.load(model_checkpoint, map_location=self.config.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = CancerGPTModel(
                self.config.MODEL_NAME,
                freeze_backbone=(fine_tuning_strategy == "last_layer")
            )
        
        # Create datasets
        train_dataset = DrugSynergyDataset(kshot_train, model.tokenizer)
        test_dataset = DrugSynergyDataset(test_data, model.tokenizer)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE)
        
        # Fine-tune
        trainer = CancerGPTTrainer(
            model,
            device=self.config.DEVICE,
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            num_epochs=self.config.NUM_EPOCHS
        )
        
        history = trainer.fit(train_loader, test_loader)
        
        # Evaluate
        metrics = trainer.evaluate(test_loader)
        
        log.info(
            f"K-shot={k} ({fine_tuning_strategy}) - "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}"
        )
        
        return metrics
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        """Save results to JSON"""
        if output_path is None:
            output_path = self.config.OUTPUT_DIR / "kshot_results.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        results_json = {}
        for tissue, tissue_results in self.results.items():
            results_json[tissue] = {
                "tissue": tissue_results["tissue"],
                "total_samples": tissue_results["total_samples"],
                "train_samples": tissue_results["train_samples"],
                "test_samples": tissue_results["test_samples"],
                "synergy_ratio": float(tissue_results["synergy_ratio"]),
                "k_shot_results": {
                    str(k): {
                        "accuracy": float(v.get("accuracy", 0)),
                        "auroc": float(v.get("auroc", 0)),
                        "auprc": float(v.get("auprc", 0))
                    }
                    for k, v in tissue_results["k_shot_results"].items()
                }
            }
        
        with open(output_path, "w") as f:
            json.dump(results_json, f, indent=2)
        
        log.info(f"Results saved to {output_path}")


# -----------------------
# COMMON TISSUE PRETRAINING
# -----------------------

def pretrain_on_common_tissues(
    common_tissue_data: pd.DataFrame,
    config: KShotConfig = KShotConfig()
) -> str:
    """
    Pretrain CancerGPT on common tissues
    
    This creates a "warm" initialization for fine-tuning on rare tissues
    
    Args:
        common_tissue_data: DataFrame with common tissue data
        config: Configuration
    
    Returns:
        Path to saved model checkpoint
    """
    log.info(f"\n{'='*70}")
    log.info("Pretraining on common tissues")
    log.info(f"{'='*70}")
    log.info(f"Total samples: {len(common_tissue_data)}")
    
    # Split data
    train_data, val_data = train_test_split(
        common_tissue_data,
        test_size=0.2,
        stratify=common_tissue_data["synergy_label"],
        random_state=config.RANDOM_STATE
    )
    
    log.info(f"Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Create model
    model = CancerGPTModel(
        model_name=config.MODEL_NAME,
        freeze_backbone=False
    )
    
    # Create datasets
    train_dataset = DrugSynergyDataset(train_data, model.tokenizer)
    val_dataset = DrugSynergyDataset(val_data, model.tokenizer)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    # Train
    trainer = CancerGPTTrainer(
        model,
        device=config.DEVICE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        num_epochs=config.NUM_EPOCHS
    )
    
    history = trainer.fit(train_loader, val_loader)
    
    # Save model
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.OUTPUT_DIR / "cancergpt_pretrained.pt"
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": config.MODEL_NAME,
            "hidden_size": config.HIDDEN_SIZE
        }
    }, checkpoint_path)
    
    log.info(f"Pretrained model saved to {checkpoint_path}")
    
    return str(checkpoint_path)


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="K-shot fine-tuning for drug synergy prediction"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to dataset (CSV or parquet)"
    )
    parser.add_argument(
        "--tissue",
        type=str,
        help="Specific tissue to evaluate (if not provided, evaluate all)"
    )
    parser.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8, 16],
        help="K values to evaluate"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["full", "last_layer"],
        default="full",
        help="Fine-tuning strategy"
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Pretrain on common tissues first"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cancergpt_outputs",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load data
    log.info(f"Loading data from {args.data_path}")
    if args.data_path.endswith(".csv"):
        df = pd.read_csv(args.data_path)
    else:
        df = pd.read_parquet(args.data_path)
    
    # Configure
    config = KShotConfig()
    config.K_SHOTS = args.k_shots
    config.OUTPUT_DIR = Path(args.output_dir)
    
    # Pretrain if requested
    pretrained_model = None
    if args.pretrain:
        common_tissues = df[~df["tissue"].isin(["pancreas", "endometrium", "liver", 
                                                   "soft tissue", "stomach", "urinary tract", "bone"])]
        if len(common_tissues) > 0:
            pretrained_model = pretrain_on_common_tissues(common_tissues, config)
    
    # Evaluate
    evaluator = RareTissueEvaluator(config)
    
    if args.tissue:
        # Evaluate specific tissue
        tissue_data = df[df["tissue"] == args.tissue]
        evaluator.evaluate_tissue(
            args.tissue,
            tissue_data,
            pretrained_model,
            args.strategy
        )
    else:
        # Evaluate all tissues
        rare_tissues = ["pancreas", "endometrium", "liver", "soft tissue", 
                        "stomach", "urinary tract", "bone"]
        for tissue in rare_tissues:
            tissue_data = df[df["tissue"] == tissue]
            if len(tissue_data) > 0:
                evaluator.evaluate_tissue(
                    tissue,
                    tissue_data,
                    pretrained_model,
                    args.strategy
                )
    
    # Save results
    evaluator.save_results()
    log.info("Evaluation complete!")


if __name__ == "__main__":
    main()
