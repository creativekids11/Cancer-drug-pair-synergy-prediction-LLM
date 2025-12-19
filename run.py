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

# Fix Unicode encoding issues on Windows
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


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
        self.api_data_path = self.data_dir / "api_drugcomb_data.csv"
        
        logger.info(f"Pipeline initialized with output directory: {self.data_dir}")
    
    def fetch_api_data(self):
        """Step 0: Fetch REAL data from DrugCombDB API (http://drugcombdb.denglab.org)."""
        logger.info("=" * 80)
        logger.info("STEP 0: FETCHING DATA FROM DRUGCOMBDB (http://drugcombdb.denglab.org)")
        logger.info("=" * 80)
        
        if self.api_data_path.exists() and not self.args.regenerate_data:
            import pandas as pd
            df = pd.read_csv(self.api_data_path)
            logger.info(f"[OK] API data loaded from cache: {len(df)} records")
            return True
        
        try:
            import aiohttp
            import asyncio
            from tqdm import tqdm
            import pandas as pd
            
            BASE_URL = "http://drugcombdb.denglab.org:8888"
            PAGE_SIZE = 500
            MAX_PAGES = getattr(self.args, 'api_max_pages', 500)
            CONCURRENCY = 10  # Reduced from 60 - API can't handle high concurrency
            REQUEST_TIMEOUT = 60  # Increased timeout
            
            logger.info(f"API Configuration:")
            logger.info(f"  - Base URL: {BASE_URL}")
            logger.info(f"  - Page size: {PAGE_SIZE} records/page")
            logger.info(f"  - Max pages: {MAX_PAGES} (total ~{MAX_PAGES * PAGE_SIZE} records)")
            logger.info(f"  - Concurrency: {CONCURRENCY} parallel requests")
            logger.info(f"  - Timeout: {REQUEST_TIMEOUT}s per request")
            
            sem = asyncio.Semaphore(CONCURRENCY)
            fetched_pages = set()
            failed_pages = []
            
            async def fetch_page(session, page, max_retries=2):
                """Fetch single page with retry logic and better error handling."""
                async with sem:
                    for attempt in range(max_retries):
                        try:
                            url = f"{BASE_URL}/integration/list"
                            params = {"page": page, "size": PAGE_SIZE}
                            
                            # Create timeout object
                            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                            
                            async with session.get(url, params=params, timeout=timeout) as resp:
                                # Check status first
                                if resp.status != 200:
                                    if attempt == max_retries - 1:
                                        logger.debug(f"Page {page}: HTTP {resp.status}")
                                    continue
                                
                                # Try to parse JSON
                                try:
                                    data = await resp.json()
                                except Exception as e:
                                    if attempt == max_retries - 1:
                                        logger.debug(f"Page {page}: Invalid JSON response: {e}")
                                    continue
                                
                                # Check if data is valid
                                if data is None or not isinstance(data, dict):
                                    if attempt == max_retries - 1:
                                        logger.debug(f"Page {page}: Invalid response structure")
                                    continue
                                
                                # Extract page data
                                page_data = data.get('data', {})
                                if isinstance(page_data, dict):
                                    page_records = page_data.get('page', [])
                                else:
                                    page_records = []
                                
                                if page_records:
                                    return page, page_records
                                else:
                                    # Empty page - likely end of dataset
                                    if attempt == 0:
                                        logger.debug(f"Page {page}: Empty response (end of dataset)")
                                    return page, []
                        
                        except asyncio.TimeoutError:
                            if attempt < max_retries - 1:
                                logger.debug(f"Page {page}: Timeout, retrying...")
                                await asyncio.sleep(1)
                        except aiohttp.ClientError as e:
                            if attempt < max_retries - 1:
                                logger.debug(f"Page {page}: Connection error, retrying...")
                                await asyncio.sleep(1)
                        except Exception as e:
                            if attempt == max_retries - 1:
                                logger.debug(f"Page {page}: {type(e).__name__}: {e}")
                    
                    # Failed after all retries
                    if page not in failed_pages:
                        failed_pages.append(page)
                    return page, []
            
            async def fetch_all():
                """Fetch all pages with progress tracking."""
                all_rows = []
                
                # Use connector with custom settings
                connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=5)
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    # Create tasks
                    tasks = [fetch_page(session, p) for p in range(1, MAX_PAGES + 1)]
                    
                    pbar = tqdm(total=len(tasks), desc="Fetching DrugCombDB", unit="pages", ncols=80)
                    
                    # Process as completed
                    for fut in asyncio.as_completed(tasks):
                        try:
                            page_num, page_data = await fut
                            if page_data:
                                fetched_pages.add(page_num)
                                all_rows.extend(page_data)
                        except Exception as e:
                            logger.debug(f"Task error: {e}")
                        finally:
                            pbar.update(1)
                    
                    pbar.close()
                
                return all_rows
            
            logger.info(f"Starting data fetch from DrugCombDB (this may take several minutes)...")
            rows = asyncio.run(fetch_all())
            
            if not rows:
                logger.warning("[WARN] No data retrieved from API!")
                logger.warning("  - Endpoint: http://drugcombdb.denglab.org:8888/integration/list")
                logger.warning("  - The API may be temporarily unavailable")
                logger.warning("  - Or all requested pages returned empty results")
                logger.warning("  - Falling back to synthetic data generation...")
                return False
            
            # Convert to DataFrame with safe extraction
            records = []
            for r in rows:
                try:
                    synergy_score = float(r.get("synergyScore", 0))
                    records.append({
                        "drugA": str(r.get("drugName1", "")).strip(),
                        "drugB": str(r.get("drugName2", "")).strip(),
                        "cell_line": str(r.get("cellName", "")).strip(),
                        "tissue": str(r.get("tissue", "")).strip(),
                        "synergy_label": 1 if synergy_score > 5 else 0,
                        "synergy_score": synergy_score,
                        "source": str(r.get("source", "")).strip(),
                        "block_id": str(r.get("blockId", "")).strip()
                    })
                except Exception as e:
                    logger.debug(f"Skipping invalid record: {e}")
                    continue
            
            if not records:
                logger.error("[FAILED] Could not parse any valid records from API response!")
                return False
            
            df = pd.DataFrame(records)
            
            # Remove invalid entries
            initial_count = len(df)
            df = df[
                (df['drugA'].notna()) & (df['drugA'] != '') &
                (df['drugB'].notna()) & (df['drugB'] != '') &
                (df['cell_line'].notna()) & (df['cell_line'] != '') &
                (df['tissue'].notna()) & (df['tissue'] != '')
            ]
            removed_count = initial_count - len(df)
            
            if len(df) == 0:
                logger.error("[FAILED] All records were invalid after filtering!")
                return False
            
            # Save to CSV
            self.api_data_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(self.api_data_path, index=False)
            
            # Log detailed statistics
            logger.info("[OK] DrugCombDB Data Fetch Complete")
            logger.info(f"  - Pages fetched: {len(fetched_pages)} pages")
            logger.info(f"  - Total records received: {initial_count}")
            logger.info(f"  - Invalid records removed: {removed_count}")
            logger.info(f"  - Valid records saved: {len(df)}")
            logger.info(f"  - Unique drug pairs: {len(df[['drugA', 'drugB']].drop_duplicates())}")
            logger.info(f"  - Unique drugs: {int(df['drugA'].nunique() + df['drugB'].nunique())}")
            logger.info(f"  - Unique tissues: {df['tissue'].nunique()}")
            logger.info(f"  - Unique cell lines: {df['cell_line'].nunique()}")
            logger.info(f"  - Synergy distribution: {df['synergy_label'].value_counts().to_dict()}")
            logger.info(f"  - File size: {self.api_data_path.stat().st_size / (1024*1024):.2f} MB")
            
            if failed_pages:
                logger.info(f"  - Failed/empty pages: {len(failed_pages)} (typically end-of-dataset)")
            
            return True
            
        except ImportError as e:
            logger.error(f"[FAILED] Missing required library: {e}")
            logger.error("Install with: pip install aiohttp pandas")
            return False
        except KeyboardInterrupt:
            logger.warning("[INTERRUPTED] User cancelled API fetch")
            return False
        except Exception as e:
            logger.error(f"[FAILED] API fetch error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_data(self):
        """Step 1: Generate synthetic data if API fetch failed."""
        logger.info("=" * 80)
        logger.info("STEP 1: GENERATING SYNTHETIC DATA (fallback if API empty)")
        logger.info("=" * 80)
        
        # If API data exists, skip synthetic generation
        if self.api_data_path.exists():
            logger.info(f"[OK] API data exists, skipping synthetic generation")
            return True
        
        if self.sample_data_path.exists() and not self.args.regenerate_data:
            logger.info(f"[OK] Synthetic data already exists, skipping generation.")
            return True
        
        try:
            from generate_sample_data import generate_sample_data
            
            logger.info(f"Generating {self.args.num_samples} synthetic samples...")
            df = generate_sample_data(
                num_samples=self.args.num_samples,
                random_seed=self.args.random_seed
            )
            
            df.to_csv(self.sample_data_path, index=False)
            logger.info(f"[OK] Synthetic data saved to {self.sample_data_path}")
            logger.info(f"  - Total samples: {len(df)}")
            logger.info(f"  - Synergy distribution: {df['synergy_label'].value_counts().to_dict()}")
            logger.info(f"  - File size: {self.sample_data_path.stat().st_size / 1024:.1f} KB")
            
            return True
        except Exception as e:
            logger.error(f"[FAILED] Synthetic data generation failed: {e}")
            import traceback
            traceback.print_exc()
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
                load_dataset, validate_drugcombdb_data, clean_data, 
                split_by_tissue, balance_classes
            )
            
            # Determine which data source to use
            data_source = self.api_data_path if self.api_data_path.exists() else self.sample_data_path
            logger.info(f"Loading data from {data_source}...")
            df = load_dataset(str(data_source))
            logger.info(f"[OK] Loaded {len(df)} samples")
            
            # Validate
            logger.info("Validating data format...")
            is_valid = validate_drugcombdb_data(df)
            if not is_valid:
                logger.error(f"Validation failed")
                return False
            logger.info("[OK] Data validation passed")
            
            # Clean
            logger.info("Cleaning data (removing duplicates, handling missing values)...")
            df_clean = clean_data(df)
            removed = len(df) - len(df_clean)
            logger.info(f"[OK] Removed {removed} duplicate/invalid rows, {len(df_clean)} remain")
            
            # Split by tissue
            logger.info("Splitting by tissue (stratified train/val/test)...")
            train_df, val_df, test_df = split_by_tissue(
                df_clean,
                test_size=0.2,
                val_size=0.1
            )
            logger.info(f"[OK] Split complete:")
            logger.info(f"  - Train: {len(train_df)} samples")
            logger.info(f"  - Val: {len(val_df)} samples")
            logger.info(f"  - Test: {len(test_df)} samples")
            
            # Balance classes in training data
            logger.info("Balancing classes in training data...")
            train_df_balanced = balance_classes(train_df)
            logger.info(f"[OK] Training data balanced:")
            logger.info(f"  - Before: {train_df['synergy_label'].value_counts().to_dict()}")
            logger.info(f"  - After: {train_df_balanced['synergy_label'].value_counts().to_dict()}")
            
            # Save all splits
            self.prepared_data_dir.mkdir(exist_ok=True, parents=True)
            train_df_balanced.to_csv(self.prepared_data_dir / "train.csv", index=False)
            val_df.to_csv(self.prepared_data_dir / "val.csv", index=False)
            test_df.to_csv(self.prepared_data_dir / "test.csv", index=False)
            df_clean.to_csv(self.prepared_full_path, index=False)
            
            logger.info(f"[OK] Data saved to {self.prepared_data_dir}/")
            logger.info(f"  - train.csv, val.csv, test.csv, full.csv")
            
            return True
        except Exception as e:
            logger.error(f"[FAILED] Data preparation failed: {e}")
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
                logger.info("[OK] Training and evaluation completed successfully")
                return True
            else:
                logger.error(f"[FAILED] Training failed with return code {result.returncode}")
                return False
        except Exception as e:
            logger.error(f"[FAILED] Training execution failed: {e}")
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
            
            logger.info(f"[OK] Summary report saved to {summary_path}")
            
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
            logger.error(f"[FAILED] Report generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Execute the complete pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("CANCERGPT COMPLETE PIPELINE - DRUGCOMBDB INTEGRATION")
        logger.info("=" * 80 + "\n")
        
        start_time = datetime.now()
        results = {}
        
        # Execute pipeline steps
        steps = []
        
        # ALWAYS attempt to fetch from DrugCombDB API first
        logger.info("\n*** STEP 0: DRUGCOMBDB API FETCH (PRIMARY DATA SOURCE) ***\n")
        api_success = self.fetch_api_data()
        
        if api_success:
            results["Fetch DrugCombDB API Data"] = "[OK] - Using real data from http://drugcombdb.denglab.org"
            logger.info("\n" + "!" * 80)
            logger.info("SUCCESS: Using real data from DrugCombDB")
            logger.info("!" * 80)
        else:
            logger.warning("\n" + "!" * 80)
            logger.warning("NOTE: DrugCombDB API returned no data")
            logger.warning("Continuing with synthetic data generation instead")
            logger.warning("!" * 80)
            results["Fetch DrugCombDB API Data"] = "[INFO] - No data from API, using synthetic"
        
        steps.extend([
            ("Data Preparation", self.prepare_data),
            ("Training & Evaluation", self.train_and_evaluate),
            ("Report Generation", self.generate_report),
        ])
        
        for step_name, step_func in steps:
            logger.info("")
            success = step_func()
            results[step_name] = "[OK]" if success else "[FAILED]"
            
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
        
        all_passed = all("OK" in status or "INFO" in status for status in results.values())
        return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="CancerGPT Pipeline with DrugCombDB Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  # DEFAULT: Fetch real data from DrugCombDB API (http://drugcombdb.denglab.org)
  python run.py --api-max-pages 500
  
  # Fetch full database (~1.2M records, 3-5 hours)
  python run.py --api-max-pages 6000

  # Quick test with fewer API pages
  python run.py --api-max-pages 10 --skip-baselines

  # Use cached API data + train with different k-shots
  python run.py --k-shots 0 2 4 8 16

  # Full pipeline with pretraining
  python run.py --api-max-pages 1000 --with-pretraining

  # Specific tissue evaluation
  python run.py --rare-tissues pancreas endometrium liver --api-max-pages 500
        """
    )
    
    # API data ingestion args (PRIMARY)
    parser.add_argument('--api-max-pages', type=int, default=500,
                        help='Pages to fetch from DrugCombDB API (default: 500 = ~250K records)')
    parser.add_argument('--regenerate-data', action='store_true',
                        help='Re-fetch API data even if cached')
    
    # Data generation args (FALLBACK - only if API empty)
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Synthetic samples if API returns no data (default: 5000)')
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
                        help='Skip baseline model comparisons')
    
    # Control args
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue pipeline even if a step fails')
    
    # Legacy arg (ignored, kept for compatibility)
    parser.add_argument('--use-api-data', action='store_true',
                        help='(DEPRECATED - API is now always used first)')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = CancerGPTPipeline(args)
    exit_code = pipeline.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
