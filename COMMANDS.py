#!/usr/bin/env python3
"""
CANCERGPT COMPLETE PIPELINE - QUICK COMMAND REFERENCE
======================================================

Copy and paste any of these commands to run the pipeline:
"""

# ============================================================================
# QUICK START - CHOOSE ONE OF THESE
# ============================================================================

# 1. MINIMAL TEST (10 minutes)
# Good for: Verifying everything works
# Use: First time testing
"""
python run.py --num-samples 1000 --k-shots 0 2 4 --skip-baselines
"""

# 2. STANDARD TEST (30 minutes)
# Good for: Testing with reasonable data
# Use: Normal experiments
"""
python run.py --num-samples 2000 --k-shots 0 2 4 8 --skip-baselines
"""

# 3. FULL PIPELINE (2-5 hours)
# Good for: Complete evaluation with all features
# Use: Publication-quality results
"""
python run.py --num-samples 5000 --with-pretraining
"""

# 4. SPECIFIC TISSUES (1-2 hours)
# Good for: Focus on particular cancer types
# Use: Tissue-specific analysis
"""
python run.py --rare-tissues pancreas endometrium liver --k-shots 0 2 4 8 16
"""

# 5. MAXIMUM K-SHOT RANGE (2-3 hours)
# Good for: Complete k-shot learning curves
# Use: Detailed analysis
"""
python run.py --num-samples 3000 --k-shots 0 2 4 8 16 32 64 128
"""

# 6. WITH YOUR OWN DATA
# Good for: Use your dataset
# Use: Production deployment
"""
# Step 1: Prepare CSV with columns: drugA, drugB, tissue, synergy_label
# Step 2: Edit generate_sample_data.py to load your file
# Step 3: Run:
python run.py --num-samples 0 --regenerate-data
"""

# ============================================================================
# ADVANCED OPTIONS - COMBINE FOR CUSTOM WORKFLOWS
# ============================================================================

# Parameter efficient (faster, fewer parameters):
"""
python run.py --num-samples 2000 --skip-baselines
# Only trains last layer of classifier, freezes GPT-2 backbone
"""

# Include baseline comparisons:
"""
python run.py --num-samples 2000 --k-shots 0 2 4 8
# Compares: CancerGPT vs XGBoost vs TabTransformer vs Collaborative Filtering
"""

# Resume on error:
"""
python run.py --continue-on-error
# Continues pipeline even if one step fails
"""

# Custom random seed:
"""
python run.py --random-seed 12345
# For reproducibility with different seed
"""

# Save to specific directory:
"""
python run.py --output-dir /path/to/results
# Saves data and results to custom location
"""

# ============================================================================
# RESULT INSPECTION - AFTER RUNNING
# ============================================================================

# View execution log in real-time:
"""
# On Windows PowerShell:
Get-Content -Path cancergpt_pipeline.log -Wait

# On Linux/Mac:
tail -f cancergpt_pipeline.log
"""

# View summary report:
"""
# On Windows:
type results/experiment_*/SUMMARY.txt

# On Linux/Mac:
cat results/experiment_*/SUMMARY.txt
"""

# View full metrics (JSON):
"""
# On Windows:
python -m json.tool results/experiment_*/results.json

# On Linux/Mac:
cat results/experiment_*/results.json | python -m json.tool
"""

# Check data files:
"""
# View generated data
python -c "import pandas as pd; df = pd.read_csv('sample_data.csv'); print(df.head()); print(f'Shape: {df.shape}')"

# View prepared data
python -c "import pandas as pd; df = pd.read_csv('data_prepared/train.csv'); print(f'Training set: {len(df)} samples'); print(df['synergy_label'].value_counts())"
"""

# ============================================================================
# COMMON WORKFLOWS
# ============================================================================

"""
WORKFLOW 1: Quick Validation (10 minutes)
=========================================
# Run this first to make sure everything works
python run.py --num-samples 1000 --k-shots 0 2 4 --skip-baselines

# Expected output:
# ✓ sample_data.csv created (1000 samples)
# ✓ data_prepared/ directory created
# ✓ Training completed on 1-2 tissues
# ✓ results/ directory with metrics
# ✓ Summary report generated

# Check results:
type results/experiment_*/SUMMARY.txt
"""

"""
WORKFLOW 2: Standard Experiment (30 minutes)
=============================================
# Run with reasonable dataset and k-shot schedule
python run.py --num-samples 2000 --k-shots 0 2 4 8 --skip-baselines

# Expected output:
# ✓ 2000 synthetic samples generated
# ✓ Data prepared and split (train: 1388, val: 206, test: 406)
# ✓ Training on multiple tissues with k=[0,2,4,8]
# ✓ Comprehensive metrics saved

# View results:
python -m json.tool results/experiment_*/results.json | more
"""

"""
WORKFLOW 3: Full Analysis (2-5 hours)
======================================
# Run complete pipeline with pretraining and all features
python run.py --num-samples 5000 --with-pretraining

# What this does:
# 1. Generates 5000 synthetic samples
# 2. Pretrain on common tissues (50-100 tissues)
# 3. Fine-tune on rare tissues (pancreas, endometrium, etc.)
# 4. Test full k-shot range [0, 2, 4, 8, 16, 32, 64, 128]
# 5. Compare full vs. last-layer fine-tuning
# 6. Include baseline models (XGBoost, TabTransformer, CF)

# View comprehensive results:
type results/experiment_*/SUMMARY.txt
python -m json.tool results/experiment_*/results.json
"""

"""
WORKFLOW 4: Tissue-Specific Focus (1-2 hours)
==============================================
# Evaluate only specific tissues of interest
python run.py \\
    --num-samples 3000 \\
    --rare-tissues pancreas endometrium liver \\
    --k-shots 0 2 4 8 16 \\
    --skip-baselines

# This will:
# - Focus on 3 specific tissues
# - Test k values: 0, 2, 4, 8, 16 (skip larger values)
# - Skip baseline comparisons (faster)
# - Provide faster results while still thorough
"""

"""
WORKFLOW 5: Full K-Shot Learning Curves (2-3 hours)
===================================================
# Evaluate complete k-shot range for learning curves
python run.py \\
    --num-samples 3000 \\
    --k-shots 0 2 4 8 16 32 64 128

# This will:
# - Test all k values from 0 to 128
# - Generate complete learning curves
# - Show how performance improves with more examples
# - Identify saturation points
"""

"""
WORKFLOW 6: Production Deployment (Your Data)
==============================================
# Step 1: Prepare your CSV file
# Required columns: drugA, drugB, tissue, synergy_label
# Example:
#   drugA,drugB,tissue,synergy_label
#   Paclitaxel,Carboplatin,pancreas,1
#   Docetaxel,Cisplatin,liver,0

# Step 2: Create a Python script to load your data
# Edit generate_sample_data.py and change:
#   def generate_sample_data(num_samples=2000, random_seed=42):
#       df = pd.read_csv("your_data.csv")
#       return df

# Step 3: Run with your data
python run.py --num-samples 0 --regenerate-data --with-pretraining

# This will:
# - Load your data instead of generating synthetic data
# - Prepare and split your data
# - Train on your real samples
# - Generate evaluation metrics
"""

# ============================================================================
# PARAMETER COMBINATIONS
# ============================================================================

"""
FAST MODE (5 minutes)
====================
python run.py --num-samples 500 --k-shots 0 2 --skip-baselines
- Minimal samples and k values
- No baselines
- Fastest execution
- Use for: Quick testing, infrastructure validation
"""

"""
BALANCED MODE (30 minutes)
==========================
python run.py --num-samples 2000 --k-shots 0 2 4 8 --skip-baselines
- Reasonable data size
- Standard k-shot range
- No baselines (save time)
- Use for: Most experiments
"""

"""
THOROUGH MODE (2 hours)
=======================
python run.py --num-samples 3000 --k-shots 0 2 4 8 16 32
- Larger dataset
- Extended k-shot range
- Include baselines
- Use for: Publication-quality results
"""

"""
COMPREHENSIVE MODE (5 hours)
=============================
python run.py \\
    --num-samples 5000 \\
    --k-shots 0 2 4 8 16 32 64 128 \\
    --with-pretraining
- Large dataset
- All k values
- Pretraining enabled
- All baselines included
- Use for: Final evaluation, benchmarking
"""

# ============================================================================
# TROUBLESHOOTING COMMANDS
# ============================================================================

"""
IF: Data file not found
DO: python run.py --num-samples 2000 --regenerate-data
"""

"""
IF: Out of memory (CUDA)
DO: python run.py --num-samples 1000 --k-shots 0 2 4 --skip-baselines
"""

"""
IF: Missing dependencies
DO: pip install torch transformers scikit-learn pandas numpy xgboost
"""

"""
IF: Need to check what tissues are available
DO: python -c "import pandas as pd; df=pd.read_csv('sample_data.csv'); print(df['tissue'].unique())"
"""

"""
IF: Need to verify data preparation worked
DO: python -c "import pandas as pd; print('Train:', len(pd.read_csv('data_prepared/train.csv'))); print('Val:', len(pd.read_csv('data_prepared/val.csv'))); print('Test:', len(pd.read_csv('data_prepared/test.csv')))"
"""

"""
IF: Need to view last few lines of log
DO: tail -n 50 cancergpt_pipeline.log  (Linux/Mac)
    Get-Content cancergpt_pipeline.log -Tail 50  (Windows)
"""

# ============================================================================
# ONE-LINERS FOR COMMON TASKS
# ============================================================================

# Generate synthetic data only
"""
python -c "from generate_sample_data import generate_sample_data; df = generate_sample_data(2000); df.to_csv('sample_data.csv', index=False); print(f'Generated {len(df)} samples')"
"""

# Check data quality
"""
python -c "import pandas as pd; df = pd.read_csv('sample_data.csv'); print(f'Samples: {len(df)}'); print(f'Columns: {list(df.columns)}'); print(f'Tissues: {df[\"tissue\"].nunique()}'); print(f'Synergy ratio: {(df[\"synergy_label\"].sum() / len(df)):.1%}')"
"""

# View experiment results
"""
python -c "import json; r = json.load(open('results/experiment_20251219_113333/results.json')); [print(f'{tissue}: best AUROC = {max(v[\"full\"].values(), key=lambda x: x[\"auroc\"])[\"auroc\"]:.3f}') for tissue, v in r['tissues'].items()]"
"""

# List all experiments
"""
ls -la results/experiment_*/SUMMARY.txt  (Linux/Mac)
dir results\experiment_*\SUMMARY.txt     (Windows)
"""

# ============================================================================
# GETTING HELP
# ============================================================================

"""
# Show all available options
python run.py --help

# Run with verbose logging (Linux/Mac)
python run.py --num-samples 1000 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log

# Run in background (Linux/Mac)
nohup python run.py --num-samples 5000 &

# Run in background (Windows PowerShell)
Start-Process -NoNewWindow python -ArgumentList "run.py --num-samples 5000"
"""

# ============================================================================
# DOCUMENTATION REFERENCES
# ============================================================================

"""
Quick Reference:
- This file: Commands and workflows

Quick Setup:
- README_COMPLETE_SETUP.md: Setup instructions

Comprehensive Guide:
- README_COMPLETE.md: Everything about CancerGPT
  - Model architecture (detailed technical breakdown)
  - K-shot methodology (learning curves, rare tissues)
  - Usage guide (all commands and options)
  - Results interpretation (what metrics mean)
  - Troubleshooting (common issues and solutions)

Project Status:
- PROJECT_SUMMARY.md: Complete project overview
- 00_START_HERE_README.md: Master entry point
- READY_TO_USE.md: Quick reference
"""

# ============================================================================
# EXAMPLE OUTPUT
# ============================================================================

"""
After running: python run.py --num-samples 1000 --k-shots 0 2 4 --skip-baselines

You'll see:
================================================================================
CANCERGPT COMPLETE PIPELINE
================================================================================

================================================================================
STEP 1: GENERATING SYNTHETIC DATA
================================================================================
Generating 1000 synthetic samples...
✓ Generated data saved to sample_data.csv
  - Total samples: 1000
  - Synergy distribution: {0: 893, 1: 107}
  - File size: 107.5 KB

================================================================================
STEP 2: PREPARING AND SPLITTING DATA
================================================================================
Loading data from sample_data.csv...
✓ Loaded 1000 samples
Validating data format...
✓ Data validation passed
Cleaning data...
✓ Removed 0 duplicate/invalid rows, 1000 remain
Splitting by tissue...
✓ Split complete:
  - Train: 694 samples (70%)
  - Val: 103 samples (10%)
  - Test: 203 samples (20%)

================================================================================
STEP 3: TRAINING AND EVALUATING (K-SHOT LEARNING)
================================================================================
Running experiments with command:
  python run_experiments.py --data-path data_prepared\\full.csv --k-shots 0 2 4 ...

2025-12-19 11:33:33 | experiments | INFO | Loaded 1000 samples
Evaluating pancreas (54 samples)
  K-shot=0: AUROC=0.75, AUPRC=0.25
  K-shot=2: AUROC=0.92, AUPRC=0.50
  K-shot=4: AUROC=0.67, AUPRC=0.20

================================================================================
STEP 4: GENERATING SUMMARY REPORT
================================================================================
✓ Summary report saved to results/experiment_20251219_113333/SUMMARY.txt

================================================================================
PIPELINE SUMMARY
================================================================================
✓ PASS Data Generation
✓ PASS Data Preparation
✓ PASS Training & Evaluation
✓ PASS Report Generation

Total time: 0:10:45
================================================================================
"""

# ============================================================================
# CONTACT & SUPPORT
# ============================================================================

"""
For detailed information:
- See README_COMPLETE.md (1060+ lines of comprehensive documentation)
- Check logs in cancergpt_pipeline.log
- View results in results/experiment_*/

For troubleshooting:
- Section 10 of README_COMPLETE.md
- Troubleshooting quick reference in README_COMPLETE_SETUP.md
"""

print(__doc__)
