# CancerGPT: Few-Shot Learning for Drug Synergy Prediction

Based on research paper: **Li et al. "CancerGPT: An Effective Multimodal Deep Learning Tool for Synergistic Drug Pair Prediction in Rare Cancers"** (npj Digital Medicine, 2024)

## Overview

CancerGPT leverages large language models (specifically GPT-2) to predict drug synergy in cancer cell lines using few-shot learning. The model is particularly effective for rare cancer tissues with limited training data.

### Key Features

- **LLM-based Architecture**: Uses pretrained GPT-2 (124M parameters) as feature extractor
- **Few-Shot Learning**: Achieves strong performance with k=[0,2,4,8,16,32,64,128] examples
- **Tabular-to-Text Conversion**: Converts structured drug pair data to natural language
- **Multiple Fine-Tuning Strategies**: Full parameter vs last-layer fine-tuning
- **Baseline Comparisons**: XGBoost, TabTransformer, Collaborative Filtering

## Project Structure

```
d:\cancer_drug_pair\
├── cancergpt_model.py              # Core CancerGPT architecture
├── cancergpt_kshot_finetuning.py   # K-shot fine-tuning pipeline
├── baseline_models.py              # Comparison baseline models
├── run_experiments.py              # Master experiment runner
├── evaluate_cancergpt.py            # Evaluation framework
├── prepare_data.py                 # Data preparation utilities
├── data_pipeline/                  # Data ingestion pipeline
│   ├── orchestrator.py
│   ├── orchestrator_tissue_restricted.py
│   ├── ollama_template_generator.py
│   ├── prompt_utils.py
│   └── ...
└── results/                        # Experiment outputs (auto-created)
    └── experiment_YYYYMMDD_HHMMSS/
        ├── results.json
        ├── evaluations/
        └── ...
```

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install torch torchvision torchaudio
pip install transformers scikit-learn pandas numpy
pip install xgboost matplotlib seaborn
```

### 2. Data Preparation

```bash
# Prepare your DrugCombDB data
python prepare_data.py \
    --input your_drugcombdb_data.csv \
    --output data_prepared

# Expected columns in input CSV:
# - drugA: str (drug A name)
# - drugB: str (drug B name)
# - tissue: str (cancer tissue type)
# - synergy_label: int (0 = not synergistic, 1 = synergistic)
# - cell_line: str (optional)
# - sensitivity_A, sensitivity_B: float (optional)
```

### 3. Run Experiments

```bash
# Run complete experiment pipeline
python run_experiments.py \
    --data-path data_prepared/full.csv \
    --output-dir results \
    --with-pretraining \
    --strategies full last_layer

# Or specify custom parameters:
python run_experiments.py \
    --data-path data_prepared/full.csv \
    --k-shots 0 2 4 8 16 32 64 128 \
    --rare-tissues pancreas endometrium liver "soft tissue" stomach \
    --strategies full last_layer \
    --with-pretraining
```

### 4. Evaluate Results

```bash
# Comprehensive evaluation with plots
python evaluate_cancergpt.py \
    --data-path data_prepared/full.csv \
    --output-dir results/evaluations \
    --eval-type all \
    --plot
```

## Detailed Usage

### CancerGPT Model Architecture

```python
from cancergpt_model import CancerGPTModel, DrugSynergyDataset, CancerGPTTrainer
import pandas as pd

# Load your data
df = pd.read_csv("data_prepared/train.csv")

# Initialize model
model = CancerGPTModel(
    model_name="gpt2",
    freeze_backbone=False,  # Full fine-tuning
    hidden_size=768,
    num_classes=2
)

# Create dataset
dataset = DrugSynergyDataset(df, model.tokenizer)

# Initialize trainer
trainer = CancerGPTTrainer(
    model,
    device="cuda",  # or "cpu"
    learning_rate=5e-5,
    weight_decay=0.01,
    max_grad_norm=1.0
)

# Train
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=8)
val_loader = DataLoader(dataset, batch_size=8)

trainer.fit(train_loader, val_loader, num_epochs=4)
```

### K-Shot Fine-Tuning

```python
from cancergpt_kshot_finetuning import RareTissueEvaluator, KShotConfig

# Configure k-shot learning
config = KShotConfig(
    K_SHOTS=[0, 2, 4, 8, 16, 32, 64, 128],
    LEARNING_RATE=5e-5,
    NUM_EPOCHS=4,
    BATCH_SIZE=8
)

# Initialize evaluator
evaluator = RareTissueEvaluator(config)

# Load data for rare tissue
tissue_data = pd.read_csv("data_pancreas.csv")

# Evaluate
results = evaluator.evaluate_tissue(
    tissue_name="pancreas",
    tissue_data=tissue_data,
    fine_tuning_strategy="full"  # or "last_layer"
)

# Results contain:
# - Zero-shot performance (k=0)
# - K-shot performance for each k
# - AUROC and AUPRC metrics
print(results["k_shot_results"])
```

### Baseline Comparison

```python
from baseline_models import compare_baselines
from sklearn.model_selection import train_test_split

# Split data
train_df, test_df = train_test_split(df, test_size=0.2)

# Compare all baselines
results = compare_baselines(train_df, test_df)

# Returns dict with:
# - XGBoost: accuracy, auroc, auprc
# - TabTransformer: accuracy, auroc, auprc
# - CollaborativeFiltering: accuracy, auroc, auprc

for model_name, metrics in results.items():
    print(f"{model_name}: AUROC={metrics['auroc']:.4f}")
```

## Expected Results

Based on the research paper, expected performance on rare tissues:

| Tissue | AUROC (k=128) | AUPRC (k=128) |
|--------|---------------|---------------|
| Pancreas | 0.82 | 0.78 |
| Endometrium | 0.85 | 0.81 |
| Liver | 0.79 | 0.75 |
| Soft Tissue | 0.81 | 0.77 |
| Stomach | 0.80 | 0.76 |
| Urinary Tract | 0.83 | 0.79 |
| Bone | 0.77 | 0.72 |

*Actual results depend on data quality and quantity*

## Model Architecture Details

### CancerGPT Components

1. **Tabular-to-Text Converter** (`TabularToText`)
   - Converts structured drug pairs to natural language
   - Template: "Drug A: {drugA}, Drug B: {drugB}, Cell Line: {cell_line}, Tissue: {tissue}"
   - Optional: Incorporates drug sensitivities

2. **GPT-2 Feature Extractor**
   - Pretrained 124M parameter model from HuggingFace
   - Takes text prompt as input
   - Extracts last token embedding (768-dim)

3. **Classification Head**
   - Linear layer on top of embeddings
   - Input: 768-dim embedding
   - Output: 2-class logits (synergistic/not-synergistic)

4. **Fine-Tuning Strategies**
   - **Full Fine-Tuning**: Update all parameters
   - **Last-Layer**: Freeze GPT-2 backbone, train only classifier

### K-Shot Learning

- **K-shot Sampler**: Ensures balanced class distribution
  - Minimum 1 positive, 1 negative example per k
  - Incremental sampling maintains consistency across k values
  
- **Evaluation**: Train on k examples, test on held-out set
  
- **K Values**: [0, 2, 4, 8, 16, 32, 64, 128]

## Configuration

Key hyperparameters in `KShotConfig`:

```python
MODEL_NAME = "gpt2"                    # Pretrained model
LEARNING_RATE = 5e-5                  # AdamW learning rate
WEIGHT_DECAY = 0.01                   # L2 regularization
NUM_EPOCHS = 4                         # Training epochs
BATCH_SIZE = 8                         # Batch size
K_SHOTS = [0, 2, 4, 8, 16, 32, 64, 128]
SYNERGY_THRESHOLD = 5.0                # Loewe score threshold
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Data Format

### Input CSV Requirements

**Minimal columns:**
```
drugA,drugB,tissue,synergy_label
BRAF inhibitor,MEK inhibitor,melanoma,1
```

**Recommended columns:**
```
drugA,drugB,cell_line,tissue,sensitivity_A,sensitivity_B,loewe_score,synergy_label
```

**Data Validation:**
- `synergy_label` must be binary (0 or 1)
- `tissue` and `drugA`/`drugB` should be non-empty strings
- Minimum 10 samples per tissue for meaningful k-shot evaluation

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python run_experiments.py --batch-size 4

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
python run_experiments.py
```

### CUDA Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
export CUDA_VISIBLE_DEVICES=""
```

### Insufficient Data

- Minimum 10 samples per tissue for k-shot evaluation
- k must be ≤ min(samples_positive, samples_negative)
- If k > available samples, evaluation is skipped

## Output Files

After running experiments, you'll find:

```
results/experiment_YYYYMMDD_HHMMSS/
├── results.json                    # Complete results
├── evaluations/
│   ├── evaluation_report.json     # Evaluation summary
│   └── results.png               # Visualization plots
├── checkpoints/                   # Saved model checkpoints
└── logs/                          # Training logs
```

## Citation

If you use CancerGPT in your research, please cite:

```bibtex
@article{li2024cancergpt,
  title={CancerGPT: An Effective Multimodal Deep Learning Tool for Synergistic Drug Pair Prediction in Rare Cancers},
  author={Li, et al.},
  journal={npj Digital Medicine},
  volume={7},
  pages={189},
  year={2024}
}
```

## References

- **GPT-2**: Radford et al., 2019
- **Transformers Library**: Wolf et al., 2019
- **DrugCombDB**: Zaharija et al., 2021

## Support

For issues or questions:
1. Check data format matches requirements
2. Verify all required packages are installed
3. Review logs in results directory
4. Check GPU memory availability

## License

This implementation is provided for research purposes.
