#!/usr/bin/env python3
"""
evaluate_cancergpt.py

Comprehensive evaluation script for CancerGPT
- Compare against baselines (XGBoost, TabTransformer, Collaborative Filtering)
- Evaluate on rare tissues
- Generate results and visualizations
- Fact-check LLM reasoning

Based on: Li et al. "CancerGPT for few shot drug pair synergy prediction"
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from cancergpt_model import CancerGPTModel, DrugSynergyDataset, CancerGPTTrainer
from baseline_models import compare_baselines
from cancergpt_kshot_finetuning import RareTissueEvaluator, KShotConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("evaluation")

# -----------------------
# EVALUATION PIPELINE
# -----------------------

class CancerGPTEvaluator:
    """Comprehensive evaluation of CancerGPT"""
    
    def __init__(
        self,
        output_dir: str = "cancergpt_evaluation",
        config: Optional[KShotConfig] = None
    ):
        """Initialize evaluator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or KShotConfig()
        self.results = {}
    
    def evaluate_tissue_kshot(
        self,
        tissue_name: str,
        tissue_data: pd.DataFrame,
        k_values: Optional[List[int]] = None
    ) -> Dict:
        """
        Evaluate CancerGPT on tissue with k-shot learning
        
        Args:
            tissue_name: Name of tissue
            tissue_data: Data for tissue
            k_values: k values to test
        
        Returns:
            Results dictionary
        """
        from sklearn.model_selection import train_test_split
        
        log.info(f"\nEvaluating {tissue_name}")
        log.info(f"Total samples: {len(tissue_data)}")
        
        if k_values is None:
            k_values = [0, 2, 4, 8, 16]
        
        # Split data
        train_data, test_data = train_test_split(
            tissue_data,
            test_size=0.2,
            stratify=tissue_data["synergy_label"],
            random_state=42
        )
        
        tissue_results = {
            "tissue": tissue_name,
            "total_samples": len(tissue_data),
            "k_shot_results": {},
            "baseline_comparison": {}
        }
        
        # Test on baseline models with full training data
        log.info(f"Evaluating baselines on {tissue_name}...")
        baseline_results = compare_baselines(train_data, test_data)
        
        for model_name, metrics in baseline_results.items():
            tissue_results["baseline_comparison"][model_name] = {
                "accuracy": float(metrics.get("accuracy", 0)),
                "auroc": float(metrics.get("auroc", 0)),
                "auprc": float(metrics.get("auprc", 0))
            }
            log.info(
                f"{model_name}: AUROC={metrics['auroc']:.4f}, "
                f"AUPRC={metrics['auprc']:.4f}"
            )
        
        # K-shot evaluation
        evaluator = RareTissueEvaluator(self.config)
        k_results = evaluator.evaluate_tissue(
            tissue_name,
            tissue_data,
            model_checkpoint=None,
            fine_tuning_strategy="full"
        )
        
        tissue_results["k_shot_results"] = {
            str(k): {
                "accuracy": float(v.get("accuracy", 0)),
                "auroc": float(v.get("auroc", 0)),
                "auprc": float(v.get("auprc", 0))
            }
            for k, v in k_results["k_shot_results"].items()
        }
        
        self.results[tissue_name] = tissue_results
        return tissue_results
    
    def compare_finetuning_strategies(
        self,
        tissue_name: str,
        tissue_data: pd.DataFrame,
        k: int = 16
    ) -> Dict:
        """
        Compare full fine-tuning vs last-layer training
        
        Args:
            tissue_name: Name of tissue
            tissue_data: Data for tissue
            k: k-shot value
        
        Returns:
            Comparison results
        """
        from sklearn.model_selection import train_test_split
        
        log.info(f"\nComparing fine-tuning strategies on {tissue_name} (k={k})")
        
        train_data, test_data = train_test_split(
            tissue_data,
            test_size=0.2,
            stratify=tissue_data["synergy_label"],
            random_state=42
        )
        
        evaluator = RareTissueEvaluator(self.config)
        
        # Full fine-tuning
        full_results = evaluator.evaluate_tissue(
            f"{tissue_name}_full",
            tissue_data,
            fine_tuning_strategy="full"
        )
        
        # Last-layer training
        last_layer_results = evaluator.evaluate_tissue(
            f"{tissue_name}_last_layer",
            tissue_data,
            fine_tuning_strategy="last_layer"
        )
        
        comparison = {
            "tissue": tissue_name,
            "k": k,
            "full_finetuning": full_results["k_shot_results"].get(k, {}),
            "last_layer_training": last_layer_results["k_shot_results"].get(k, {})
        }
        
        log.info(
            f"Full: AUROC={comparison['full_finetuning'].get('auroc', 0):.4f}, "
            f"Last-layer: AUROC={comparison['last_layer_training'].get('auroc', 0):.4f}"
        )
        
        return comparison
    
    def evaluate_zero_shot_performance(
        self,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Evaluate zero-shot performance on diverse tissues
        
        Args:
            test_df: Test data
        
        Returns:
            Zero-shot results by tissue
        """
        log.info("\nEvaluating zero-shot performance...")
        
        model = CancerGPTModel(model_name="gpt2")
        trainer = CancerGPTTrainer(model, device=self.config.DEVICE)
        
        results = {}
        
        for tissue in test_df["tissue"].unique():
            tissue_data = test_df[test_df["tissue"] == tissue]
            if len(tissue_data) < 10:
                continue
            
            dataset = DrugSynergyDataset(tissue_data, model.tokenizer)
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=32)
            
            metrics = trainer.evaluate(loader)
            results[tissue] = {
                "accuracy": float(metrics.get("accuracy", 0)),
                "auroc": float(metrics.get("auroc", 0)),
                "auprc": float(metrics.get("auprc", 0)),
                "num_samples": len(tissue_data)
            }
            
            log.info(
                f"{tissue}: AUROC={metrics['auroc']:.4f}, "
                f"AUPRC={metrics['auprc']:.4f} (n={len(tissue_data)})"
            )
        
        return results
    
    def generate_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            output_path: Path to save report
        
        Returns:
            Report path
        """
        if output_path is None:
            output_path = self.output_dir / "evaluation_report.json"
        
        report = {
            "evaluation_date": pd.Timestamp.now().isoformat(),
            "config": {
                "model": self.config.MODEL_NAME,
                "learning_rate": self.config.LEARNING_RATE,
                "num_epochs": self.config.NUM_EPOCHS,
                "batch_size": self.config.BATCH_SIZE,
                "k_shots": self.config.K_SHOTS
            },
            "results": self.results
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        log.info(f"Report saved to {output_path}")
        return str(output_path)
    
    def plot_results(self, output_path: Optional[str] = None):
        """
        Plot evaluation results
        
        Args:
            output_path: Path to save plots
        """
        if not HAS_PLOTTING:
            log.warning("Plotting requires matplotlib/seaborn. Skipping...")
            return
        
        if not self.results:
            log.warning("No results to plot")
            return
        
        if output_path is None:
            output_path = self.output_dir / "results.png"
        
        # Prepare data
        tissues = []
        k_values = []
        aurocs = []
        auprc = []
        
        for tissue, tissue_results in self.results.items():
            for k, metrics in tissue_results.get("k_shot_results", {}).items():
                tissues.append(tissue)
                k_values.append(k)
                aurocs.append(metrics.get("auroc", 0))
                auprc.append(metrics.get("auprc", 0))
        
        if not tissues:
            log.warning("No data to plot")
            return
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # AUROC
        pivot_auroc = pd.DataFrame({
            "tissue": tissues,
            "k": k_values,
            "auroc": aurocs
        }).pivot(index="tissue", columns="k", values="auroc")
        
        sns.heatmap(pivot_auroc, annot=True, fmt=".3f", cmap="RdYlGn",
                    ax=axes[0], cbar_kws={"label": "AUROC"})
        axes[0].set_title("AUROC by Tissue and K-shot")
        axes[0].set_ylabel("Tissue")
        axes[0].set_xlabel("Number of Shots (K)")
        
        # AUPRC
        pivot_auprc = pd.DataFrame({
            "tissue": tissues,
            "k": k_values,
            "auprc": auprc
        }).pivot(index="tissue", columns="k", values="auprc")
        
        sns.heatmap(pivot_auprc, annot=True, fmt=".3f", cmap="RdYlGn",
                    ax=axes[1], cbar_kws={"label": "AUPRC"})
        axes[1].set_title("AUPRC by Tissue and K-shot")
        axes[1].set_ylabel("Tissue")
        axes[1].set_xlabel("Number of Shots (K)")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info(f"Plots saved to {output_path}")
        plt.close()


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of CancerGPT"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cancergpt_evaluation",
        help="Output directory"
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["kshot", "zero_shot", "baselines", "all"],
        default="all",
        help="Type of evaluation to run"
    )
    parser.add_argument(
        "--tissues",
        type=str,
        nargs="+",
        help="Specific tissues to evaluate"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots"
    )
    
    args = parser.parse_args()
    
    # Load data
    log.info(f"Loading data from {args.data_path}")
    if args.data_path.endswith(".csv"):
        df = pd.read_csv(args.data_path)
    else:
        df = pd.read_parquet(args.data_path)
    
    # Initialize evaluator
    evaluator = CancerGPTEvaluator(output_dir=args.output_dir)
    
    # Determine tissues to evaluate
    if args.tissues:
        tissues_to_eval = args.tissues
    else:
        tissues_to_eval = df["tissue"].unique()
    
    # Run evaluations
    if args.eval_type in ["kshot", "all"]:
        log.info("\n" + "="*70)
        log.info("K-SHOT EVALUATION")
        log.info("="*70)
        
        for tissue in tissues_to_eval:
            tissue_data = df[df["tissue"] == tissue]
            if len(tissue_data) > 0:
                evaluator.evaluate_tissue_kshot(tissue, tissue_data)
    
    if args.eval_type in ["zero_shot", "all"]:
        log.info("\n" + "="*70)
        log.info("ZERO-SHOT EVALUATION")
        log.info("="*70)
        
        zero_shot_results = evaluator.evaluate_zero_shot_performance(df)
    
    # Generate report and plots
    evaluator.generate_report()
    
    if args.plot:
        evaluator.plot_results()
    
    log.info("\n" + "="*70)
    log.info("EVALUATION COMPLETE")
    log.info("="*70)


if __name__ == "__main__":
    main()
