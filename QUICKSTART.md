# CancerGPT QUICKSTART

Get started with CancerGPT in 5 minutes!

## Prerequisites

```bash
pip install torch transformers scikit-learn pandas xgboost matplotlib seaborn
```

## Step 1: Prepare Your Data

Your data should have these columns:
- `drugA`: First drug name
- `drugB`: Second drug name  
- `tissue`: Cancer tissue type
- `synergy_label`: Binary label (0 = not synergistic, 1 = synergistic)

Example:
```csv
drugA,drugB,tissue,synergy_label
BRAF inhibitor,MEK inhibitor,melanoma,1
EGFR inhibitor,HER2 inhibitor,breast,0
```

## Step 2: Run Data Preparation

```bash
python prepare_data.py --input your_data.csv --output data_prepared
```

This will:
- Validate data format
- Remove duplicates and missing values
- Create train/val/test splits
- Balance classes
- Add engineered features

Output files:
- `data_prepared/train.csv`
- `data_prepared/val.csv`
- `data_prepared/test.csv`
- `data_prepared/full.csv`

## Step 3: Run Experiments

### Basic (Minimal)
```bash
python run_experiments.py --data-path data_prepared/full.csv
```

### Full Pipeline
```bash
python run_experiments.py \
    --data-path data_prepared/full.csv \
    --output-dir results \
    --strategies full last_layer \
    --with-pretraining
```

### Custom Tissues
```bash
python run_experiments.py \
    --data-path data_prepared/full.csv \
    --rare-tissues pancreas liver breast \
    --k-shots 4 8 16 32
```

Results will be saved to:
```
results/experiment_YYYYMMDD_HHMMSS/results.json
```

## Step 4: Evaluate Results

```bash
python evaluate_cancergpt.py \
    --data-path data_prepared/full.csv \
    --output-dir evaluation_results \
    --plot
```

This generates:
- `evaluation_report.json` - Metrics for each tissue
- `results.png` - Visualization heatmaps

## Step 5: View Results

```bash
# View JSON results
cat results/experiment_YYYYMMDD_HHMMSS/results.json

# Best performing configuration:
python -c "import json; r=json.load(open('results/experiment_YYYYMMDD_HHMMSS/results.json')); print(r['summary']['best_models'])"
```

## Example: Full End-to-End

```bash
# 1. Prepare data
python prepare_data.py --input drugcombdb.csv --output data_prepared

# 2. Run experiments
python run_experiments.py \
    --data-path data_prepared/full.csv \
    --output-dir results \
    --with-pretraining

# 3. Evaluate
python evaluate_cancergpt.py \
    --data-path data_prepared/full.csv \
    --output-dir evaluation_results \
    --plot

# 4. Check results
ls -la results/experiment_*/results.json
```

## Using Python API Directly

```python
import pandas as pd
from cancergpt_model import CancerGPTModel, DrugSynergyDataset
from cancergpt_kshot_finetuning import RareTissueEvaluator, KShotConfig
from torch.utils.data import DataLoader

# Load data
df = pd.read_csv("data_prepared/train.csv")

# Initialize model
model = CancerGPTModel(model_name="gpt2")

# Create dataset
dataset = DrugSynergyDataset(df, model.tokenizer)
loader = DataLoader(dataset, batch_size=8)

# Evaluate on rare tissue
tissue_data = pd.read_csv("data_pancreas.csv")
evaluator = RareTissueEvaluator(KShotConfig())
results = evaluator.evaluate_tissue("pancreas", tissue_data)

# Print results
for k, metrics in results["k_shot_results"].items():
    print(f"k={k}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")
```

## Expected Output

After `run_experiments.py`:

```
================================================================================
CANCERGPT EXPERIMENT PIPELINE
================================================================================
Experiment directory: results/experiment_20240101_120000

Loading data from data_prepared/full.csv
Loaded 5000 samples with columns: ['drugA', 'drugB', 'tissue', ...]

================================================================================
K-SHOT LEARNING EXPERIMENTS
================================================================================

Evaluating pancreas (425 samples)
  Strategy: full
    k=0: AUROC=0.7200, AUPRC=0.6800
    k=2: AUROC=0.7450, AUPRC=0.7050
    k=4: AUROC=0.7680, AUPRC=0.7280
    ...
    k=128: AUROC=0.8200, AUPRC=0.7800

[... more tissues ...]

================================================================================
BASELINE COMPARISON
================================================================================

Comparing baselines on pancreas
  XGBoost: AUROC=0.7100, AUPRC=0.6700
  TabTransformer: AUROC=0.7300, AUPRC=0.6900
  CollaborativeFiltering: AUROC=0.6800, AUPRC=0.6400

[... more tissues ...]

================================================================================
EXPERIMENT COMPLETE
================================================================================
Results saved to: results/experiment_20240101_120000
```

## Key Configuration Parameters

Edit these values in scripts if needed:

```python
# Learning rate (model fine-tuning)
LEARNING_RATE = 5e-5

# Training epochs
NUM_EPOCHS = 4

# Batch size (reduce if OOM)
BATCH_SIZE = 8

# K-shot values to test
K_SHOTS = [0, 2, 4, 8, 16, 32, 64, 128]

# Synergy threshold (Loewe score)
SYNERGY_THRESHOLD = 5.0
```

## Troubleshooting

**Issue: "CUDA out of memory"**
```bash
# Use smaller batch size
python run_experiments.py --batch-size 4

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
python run_experiments.py --data-path data_prepared/full.csv
```

**Issue: "Not enough data for tissue"**
- Minimum 10 samples required per tissue
- k cannot exceed available minority class samples
- Check `data_prepared/statistics.json` for sample counts

**Issue: "Missing required columns"**
- Ensure CSV has: `drugA`, `drugB`, `tissue`, `synergy_label`
- Use `prepare_data.py` to validate and clean data first

## Output Interpretation

**AUROC (Area Under ROC Curve)**
- Measure of classifier discriminative ability
- Range: 0-1 (0.5 = random)
- Higher is better

**AUPRC (Area Under Precision-Recall Curve)**
- Better for imbalanced datasets
- Range: 0-1
- Higher is better

**k-shot Learning**
- k=0: Zero-shot (no fine-tuning)
- k=N: Fine-tuned with N examples
- k increases = better performance (generally)

## Next Steps

1. **Customize tissues**: Modify `rare_tissues` parameter
2. **Adjust k-shots**: Change `k_shots` parameter  
3. **Compare strategies**: Try both `full` and `last_layer` fine-tuning
4. **Pretraining**: Enable `--with-pretraining` for common tissues
5. **Baselines**: Compare with XGBoost, TabTransformer, Collaborative Filtering

## Citation

Li et al. "CancerGPT: An Effective Multimodal Deep Learning Tool for Synergistic Drug Pair Prediction in Rare Cancers" npj Digital Medicine (2024)

## Support

See `README.md` for detailed documentation.
