#!/usr/bin/env python3
"""
Complete CancerGPT Pipeline Orchestrator
========================================
Master script that handles the entire workflow:
1. Generate synthetic data (if needed)
2. Prepare and clean data
3. Train CancerGPT with k-shot learning
4. Evaluate on rare tissues
5. Generate comprehensive reports
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cancergpt_pipeline.log')
    ]
)
logger = logging.getLogger("pipeline_orchestrator")


class CancerGPTPipeline:
    """Master orchestrator for the entire CancerGPT pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.data_dir = Path(args.output_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Key file paths
        self.sample_data_path = self.data_dir / "sample_data.csv"
        self.prepared_data_dir = self.data_dir / "data_prepared"
        self.prepared_full_path = self.prepared_data_dir / "full.csv"
        self.results_dir = Path("results")
        
        logger.info(f"Pipeline initialized with output directory: {self.data_dir}")
    
    def generate_data(self):
        """Step 1: Generate synthetic sample data if it doesn't exist."""
        logger.info("=" * 80)
        logger.info("STEP 1: GENERATING SYNTHETIC DATA")
        logger.info("=" * 80)
        
        if self.sample_data_path.exists() and not self.args.regenerate_data:
            logger.info(f"Data already exists at {self.sample_data_path}, skipping generation.")
            return True
        
        try:
            from generate_sample_data import generate_sample_data
            
            logger.info(f"Generating {self.args.num_samples} synthetic samples...")
            df = generate_sample_data(
                num_samples=self.args.num_samples,
                random_seed=self.args.random_seed
            )
            
            df.to_csv(self.sample_data_path, index=False)
            logger.info(f"✓ Generated data saved to {self.sample_data_path}")
            logger.info(f"  - Total samples: {len(df)}")
            logger.info(f"  - Synergy distribution: {df['synergy_label'].value_counts().to_dict()}")
            logger.info(f"  - File size: {self.sample_data_path.stat().st_size / 1024:.1f} KB")
            
            return True
        except Exception as e:
            logger.error(f"✗ Data generation failed: {e}")
            return False
    
    def prepare_data(self):
        """Step 2: Prepare and preprocess data (clean, validate, split)."""
        logger.info("=" * 80)
        logger.info("STEP 2: PREPARING AND SPLITTING DATA")
        logger.info("=" * 80)
        
        if self.prepared_full_path.exists() and not self.args.regenerate_data:
            logger.info(f"Prepared data already exists, skipping preparation.")
            return True
        
        try:
            from prepare_data import (
                load_data, validate_data, clean_data, 
                split_by_tissue, balance_dataset
            )
            
            logger.info(f"Loading data from {self.sample_data_path}...")
            df = load_data(str(self.sample_data_path))
            logger.info(f"✓ Loaded {len(df)} samples")
            
            # Validate
            logger.info("Validating data format...")
            is_valid, errors = validate_data(df)
            if not is_valid:
                logger.error(f"Validation errors: {errors}")
                return False
            logger.info("✓ Data validation passed")
            
            # Clean
            logger.info("Cleaning data (removing duplicates, handling missing values)...")
            df_clean = clean_data(df)
            removed = len(df) - len(df_clean)
            logger.info(f"✓ Removed {removed} duplicate/invalid rows, {len(df_clean)} remain")
            
            # Split by tissue
            logger.info("Splitting by tissue (stratified train/val/test)...")
            train_df, val_df, test_df = split_by_tissue(
                df_clean,
                train_ratio=0.7,
                val_ratio=0.1,
                test_ratio=0.2
            )
            logger.info(f"✓ Split complete:")
            logger.info(f"  - Train: {len(train_df)} samples (70%)")
            logger.info(f"  - Val: {len(val_df)} samples (10%)")
            logger.info(f"  - Test: {len(test_df)} samples (20%)")
            
            # Balance classes in training data
            logger.info("Balancing classes in training data...")
            train_df_balanced = balance_dataset(train_df, method='undersample')
            logger.info(f"✓ Training data balanced:")
            logger.info(f"  - Before: {train_df['synergy_label'].value_counts().to_dict()}")
            logger.info(f"  - After: {train_df_balanced['synergy_label'].value_counts().to_dict()}")
            
            # Save all splits
            self.prepared_data_dir.mkdir(exist_ok=True, parents=True)
            train_df_balanced.to_csv(self.prepared_data_dir / "train.csv", index=False)
            val_df.to_csv(self.prepared_data_dir / "val.csv", index=False)
            test_df.to_csv(self.prepared_data_dir / "test.csv", index=False)
            df_clean.to_csv(self.prepared_full_path, index=False)
            
            logger.info(f"✓ Data saved to {self.prepared_data_dir}/")
            logger.info(f"  - train.csv, val.csv, test.csv, full.csv")
            
            return True
        except Exception as e:
            logger.error(f"✗ Data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_and_evaluate(self):
        """Step 3: Run k-shot learning experiments on rare tissues."""
        logger.info("=" * 80)
        logger.info("STEP 3: TRAINING AND EVALUATING (K-SHOT LEARNING)")
        logger.info("=" * 80)
        
        try:
            # Build command for run_experiments.py
            cmd = [
                "python", "run_experiments.py",
                "--data-path", str(self.prepared_full_path),
                "--k-shots"] + [str(k) for k in self.args.k_shots]
            
            if self.args.skip_baselines:
                cmd.append("--skip-baselines")
            
            if self.args.with_pretraining:
                cmd.append("--with-pretraining")
            
            if self.args.rare_tissues:
                cmd.extend(["--rare-tissues"] + self.args.rare_tissues)
            
            cmd.extend(["--strategies", "full", "last_layer"])
            
            logger.info(f"Running experiments with command:")
            logger.info(f"  {' '.join(cmd)}")
            logger.info("")
            
            result = subprocess.run(cmd, cwd=os.getcwd())
            
            if result.returncode == 0:
                logger.info("✓ Training and evaluation completed successfully")
                return True
            else:
                logger.error(f"✗ Training failed with return code {result.returncode}")
                return False
        except Exception as e:
            logger.error(f"✗ Training execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_report(self):
        """Step 4: Generate summary report."""
        logger.info("=" * 80)
        logger.info("STEP 4: GENERATING SUMMARY REPORT")
        logger.info("=" * 80)
        
        try:
            # Find the most recent results
            if not self.results_dir.exists():
                logger.warning("No results directory found")
                return False
            
            experiments = sorted(self.results_dir.glob("experiment_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not experiments:
                logger.warning("No experiment results found")
                return False
            
            latest_exp = experiments[0]
            results_json = latest_exp / "results.json"
            
            if not results_json.exists():
                logger.warning(f"Results file not found: {results_json}")
                return False
            
            logger.info(f"Found latest results: {latest_exp.name}")
            
            with open(results_json) as f:
                results = json.load(f)
            
            # Generate summary
            summary_path = latest_exp / "SUMMARY.txt"
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("CANCERGPT EXPERIMENT SUMMARY REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Experiment: {latest_exp.name}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                # Configuration
                if 'config' in results:
                    f.write("CONFIGURATION\n")
                    f.write("-" * 80 + "\n")
                    for key, val in results['config'].items():
                        f.write(f"  {key}: {val}\n")
                    f.write("\n")
                
                # Results by tissue
                if 'tissues' in results:
                    f.write("RESULTS BY TISSUE\n")
                    f.write("-" * 80 + "\n")
                    for tissue, tissue_results in results['tissues'].items():
                        f.write(f"\n{tissue.upper()}\n")
                        for strategy, k_results in tissue_results.items():
                            f.write(f"  Strategy: {strategy}\n")
                            for k, metrics in k_results.items():
                                f.write(f"    k={k}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            logger.info(f"✓ Summary report saved to {summary_path}")
            
            # Print key findings
            logger.info("\nKEY FINDINGS:")
            logger.info("-" * 80)
            if 'tissues' in results:
                for tissue, tissue_results in list(results['tissues'].items())[:2]:  # Show first 2 tissues
                    logger.info(f"\n{tissue}:")
                    for strategy, k_results in tissue_results.items():
                        best_k = max(k_results.items(), key=lambda x: float(x[1].get('auroc', 0)))
                        logger.info(f"  {strategy}: best AUROC={best_k[1]['auroc']:.4f} at k={best_k[0]}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Report generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Execute the complete pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("CANCERGPT COMPLETE PIPELINE")
        logger.info("=" * 80 + "\n")
        
        start_time = datetime.now()
        
        # Execute pipeline steps
        steps = [
            ("Data Generation", self.generate_data),
            ("Data Preparation", self.prepare_data),
            ("Training & Evaluation", self.train_and_evaluate),
            ("Report Generation", self.generate_report),
        ]
        
        results = {}
        for step_name, step_func in steps:
            logger.info("")
            success = step_func()
            results[step_name] = "✓ PASS" if success else "✗ FAIL"
            
            if not success and not self.args.continue_on_error:
                logger.error(f"\nPipeline stopped due to failure in: {step_name}")
                break
        
        # Print summary
        elapsed = datetime.now() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        for step, status in results.items():
            logger.info(f"{status} {step}")
        logger.info(f"\nTotal time: {elapsed}")
        logger.info("=" * 80 + "\n")
        
        all_passed = all("✓" in status for status in results.values())
        return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Complete CancerGPT Pipeline: Data Generation → Preparation → Training → Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  # Quick test with sample data generation (10 min)
  python run.py --num-samples 1000 --k-shots 0 2 4 --skip-baselines

  # Full pipeline with pretraining (2-5 hours)
  python run.py --num-samples 5000 --with-pretraining

  # Custom rare tissue evaluation
  python run.py --rare-tissues pancreas endometrium liver --k-shots 0 2 4 8

  # Continue on error instead of stopping
  python run.py --continue-on-error
        """
    )
    
    # Data generation args
    parser.add_argument('--num-samples', type=int, default=2000,
                        help='Number of synthetic samples to generate (default: 2000)')
    parser.add_argument('--regenerate-data', action='store_true',
                        help='Regenerate data even if it exists')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Training args
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for data and results (default: current directory)')
    parser.add_argument('--k-shots', type=int, nargs='+', default=[0, 2, 4, 8],
                        help='K-shot values to evaluate (default: [0, 2, 4, 8])')
    parser.add_argument('--rare-tissues', type=str, nargs='+', default=None,
                        help='Specific rare tissues to evaluate')
    parser.add_argument('--with-pretraining', action='store_true',
                        help='Pretrain on common tissues before evaluating rare tissues')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Skip baseline model comparisons (XGBoost, TabTransformer)')
    
    # Control args
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue pipeline even if a step fails')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = CancerGPTPipeline(args)
    exit_code = pipeline.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
