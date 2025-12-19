#!/usr/bin/env python3
"""
run_experiments.py

Master experiment runner for CancerGPT
- Load DrugCombDB data
- Run k-shot experiments on rare tissues
- Compare against baselines
- Generate comprehensive reports

Based on: Li et al. "CancerGPT for few shot drug pair synergy prediction"
          npj Digital Medicine (2024)
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from cancergpt_kshot_finetuning import (
    RareTissueEvaluator, KShotConfig, pretrain_on_common_tissues
)
from evaluate_cancergpt import CancerGPTEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("experiments")

# -----------------------
# CONFIGURATION
# -----------------------

DEFAULT_RARE_TISSUES = [
    "pancreas",
    "endometrium",
    "liver",
    "soft tissue",
    "stomach",
    "urinary tract",
    "bone"
]

class ExperimentConfig:
    """Configuration for experiment runs"""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str = "results",
        k_shots: Optional[list] = None,
        rare_tissues: Optional[list] = None,
        fine_tuning_strategies: Optional[list] = None,
        run_pretraining: bool = False,
        compare_baselines: bool = True,
        save_checkpoints: bool = True
    ):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.k_shots = k_shots or [0, 2, 4, 8, 16, 32, 64, 128]
        self.rare_tissues = rare_tissues or DEFAULT_RARE_TISSUES
        self.fine_tuning_strategies = fine_tuning_strategies or ["full", "last_layer"]
        self.run_pretraining = run_pretraining
        self.compare_baselines = compare_baselines
        self.save_checkpoints = save_checkpoints
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"experiment_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        return {
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "k_shots": self.k_shots,
            "rare_tissues": self.rare_tissues,
            "fine_tuning_strategies": self.fine_tuning_strategies,
            "run_pretraining": self.run_pretraining,
            "compare_baselines": self.compare_baselines,
            "save_checkpoints": self.save_checkpoints,
            "timestamp": self.timestamp
        }


# -----------------------
# EXPERIMENT RUNNER
# -----------------------

class CancerGPTExperimentRunner:
    """Master runner for CancerGPT experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {
            "config": config.to_dict(),
            "experiments": {},
            "summary": {}
        }
        
        # Load data
        log.info(f"Loading data from {config.data_path}")
        self.df = self._load_data(config.data_path)
        log.info(f"Loaded {len(self.df)} samples with columns: {self.df.columns.tolist()}")
        
        # Initialize evaluator
        self.evaluator = CancerGPTEvaluator(
            output_dir=str(config.experiment_dir / "evaluations")
        )
    
    @staticmethod
    def _load_data(path: str) -> pd.DataFrame:
        """Load data from various formats"""
        path = Path(path)
        
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".xlsx":
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def validate_data(self) -> bool:
        """Validate data has required columns"""
        required_cols = {"drugA", "drugB", "tissue", "synergy_label"}
        available_cols = set(self.df.columns)
        
        if not required_cols.issubset(available_cols):
            missing = required_cols - available_cols
            log.error(f"Missing required columns: {missing}")
            return False
        
        log.info(f"Data validation passed: {len(self.df)} samples")
        return True
    
    def run_kshot_experiments(self) -> Dict:
        """Run k-shot learning experiments on rare tissues"""
        log.info("\n" + "="*80)
        log.info("K-SHOT LEARNING EXPERIMENTS")
        log.info("="*80)
        
        kshot_results = {}
        
        for tissue in self.config.rare_tissues:
            tissue_data = self.df[self.df["tissue"] == tissue]
            
            if len(tissue_data) == 0:
                log.warning(f"No data for tissue: {tissue}")
                continue
            
            if len(tissue_data) < 10:
                log.warning(
                    f"Insufficient data for {tissue}: {len(tissue_data)} samples "
                    f"(minimum 10 required)"
                )
                continue
            
            log.info(f"\nEvaluating {tissue} ({len(tissue_data)} samples)")
            
            # Test all fine-tuning strategies
            for strategy in self.config.fine_tuning_strategies:
                strategy_key = f"{tissue}_{strategy}"
                
                evaluator = RareTissueEvaluator(KShotConfig())
                results = evaluator.evaluate_tissue(
                    tissue,
                    tissue_data,
                    fine_tuning_strategy=strategy
                )
                
                kshot_results[strategy_key] = results
                
                # Log results for each k
                log.info(f"  Strategy: {strategy}")
                for k, metrics in sorted(results.get("k_shot_results", {}).items()):
                    log.info(
                        f"    k={k}: AUROC={metrics.get('auroc', 0):.4f}, "
                        f"AUPRC={metrics.get('auprc', 0):.4f}"
                    )
        
        self.results["experiments"]["kshot"] = kshot_results
        return kshot_results
    
    def run_pretraining(self) -> Dict:
        """Pretraining on common tissues"""
        log.info("\n" + "="*80)
        log.info("PRETRAINING ON COMMON TISSUES")
        log.info("="*80)
        
        # Identify common tissues (tissues with >100 samples)
        tissue_counts = self.df["tissue"].value_counts()
        common_tissues = tissue_counts[tissue_counts > 100].index.tolist()
        
        if not common_tissues:
            log.warning("No common tissues found for pretraining (need >100 samples)")
            return {}
        
        common_data = self.df[self.df["tissue"].isin(common_tissues)]
        log.info(f"Found {len(common_tissues)} common tissues with {len(common_data)} samples")
        
        # Run pretraining
        pretrain_results = pretrain_on_common_tissues(
            common_data,
            num_epochs=self.config.k_shots  # Use k_shots as proxy for epochs
        )
        
        self.results["experiments"]["pretraining"] = pretrain_results
        return pretrain_results
    
    def run_baseline_comparison(self) -> Dict:
        """Compare against baselines on each rare tissue"""
        log.info("\n" + "="*80)
        log.info("BASELINE COMPARISON")
        log.info("="*80)
        
        baseline_results = {}
        
        for tissue in self.config.rare_tissues:
            tissue_data = self.df[self.df["tissue"] == tissue]
            
            if len(tissue_data) < 10:
                continue
            
            log.info(f"\nComparing baselines on {tissue}")
            
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                tissue_data,
                test_size=0.2,
                stratify=tissue_data["synergy_label"],
                random_state=42
            )
            
            from baseline_models import compare_baselines
            tissue_baselines = compare_baselines(train_data, test_data)
            
            baseline_results[tissue] = tissue_baselines
            
            for model_name, metrics in tissue_baselines.items():
                log.info(
                    f"  {model_name}: AUROC={metrics.get('auroc', 0):.4f}, "
                    f"AUPRC={metrics.get('auprc', 0):.4f}"
                )
        
        self.results["experiments"]["baselines"] = baseline_results
        return baseline_results
    
    def generate_summary_report(self) -> Dict:
        """Generate summary statistics and comparisons"""
        log.info("\n" + "="*80)
        log.info("GENERATING SUMMARY REPORT")
        log.info("="*80)
        
        summary = {
            "dataset_statistics": {
                "total_samples": len(self.df),
                "num_tissues": self.df["tissue"].nunique(),
                "tissues": self.df["tissue"].value_counts().to_dict(),
                "synergy_distribution": self.df["synergy_label"].value_counts().to_dict()
            },
            "performance_summary": {},
            "best_models": {}
        }
        
        # Aggregate k-shot results
        if "kshot" in self.results["experiments"]:
            best_auroc = 0
            best_config = None
            
            auroc_by_k = {}
            auprc_by_k = {}
            
            for strategy_key, results in self.results["experiments"]["kshot"].items():
                for k, metrics in results.get("k_shot_results", {}).items():
                    auroc = metrics.get("auroc", 0)
                    auprc = metrics.get("auprc", 0)
                    
                    if k not in auroc_by_k:
                        auroc_by_k[k] = []
                        auprc_by_k[k] = []
                    
                    auroc_by_k[k].append(auroc)
                    auprc_by_k[k].append(auprc)
                    
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_config = {
                            "strategy": strategy_key,
                            "k": k,
                            "auroc": auroc,
                            "auprc": auprc
                        }
            
            # Calculate averages
            summary["performance_summary"]["kshot"] = {
                "mean_auroc_by_k": {
                    str(k): float(np.mean(v)) for k, v in auroc_by_k.items()
                },
                "mean_auprc_by_k": {
                    str(k): float(np.mean(v)) for k, v in auprc_by_k.items()
                }
            }
            
            if best_config:
                summary["best_models"]["kshot"] = best_config
        
        self.results["summary"] = summary
        return summary
    
    def save_results(self) -> Path:
        """Save all results to JSON"""
        results_file = self.config.experiment_dir / "results.json"
        
        # Convert numpy types to native Python types
        results_serializable = json.loads(
            json.dumps(self.results, default=str)
        )
        
        with open(results_file, "w") as f:
            json.dump(results_serializable, f, indent=2)
        
        log.info(f"Results saved to {results_file}")
        return results_file
    
    def run_all(self):
        """Run complete experiment pipeline"""
        log.info("\n" + "="*80)
        log.info("CANCERGPT EXPERIMENT PIPELINE")
        log.info("="*80)
        log.info(f"Experiment directory: {self.config.experiment_dir}")
        
        # Validate data
        if not self.validate_data():
            log.error("Data validation failed")
            return
        
        # Run pretraining if requested
        if self.config.run_pretraining:
            self.run_pretraining()
        
        # Run k-shot experiments
        self.run_kshot_experiments()
        
        # Compare baselines if requested
        if self.config.compare_baselines:
            self.run_baseline_comparison()
        
        # Generate summary
        self.generate_summary_report()
        
        # Save results
        self.save_results()
        
        log.info("\n" + "="*80)
        log.info("EXPERIMENT COMPLETE")
        log.info("="*80)
        log.info(f"Results saved to: {self.config.experiment_dir}")


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Master experiment runner for CancerGPT"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to dataset (csv, parquet, or xlsx)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8, 16, 32, 64, 128],
        help="K values for k-shot learning"
    )
    parser.add_argument(
        "--rare-tissues",
        type=str,
        nargs="+",
        default=DEFAULT_RARE_TISSUES,
        help="Tissues to evaluate"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        choices=["full", "last_layer"],
        default=["full", "last_layer"],
        help="Fine-tuning strategies to compare"
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline comparison"
    )
    parser.add_argument(
        "--with-pretraining",
        action="store_true",
        help="Run pretraining on common tissues"
    )
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        k_shots=args.k_shots,
        rare_tissues=args.rare_tissues,
        fine_tuning_strategies=args.strategies,
        run_pretraining=args.with_pretraining,
        compare_baselines=not args.skip_baselines
    )
    
    runner = CancerGPTExperimentRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
