# CancerGPT Inference Guide

## Quick Start

### 1. Command-Line Interface (CLI)

**Basic prediction:**
```bash
python inference.py --tissue pancreas --drug-a Paclitaxel --drug-b Cisplatin
```

**With custom K-shot value:**
```bash
python inference.py --tissue "urinary tract" --drug-a Gemcitabine --drug-b Doxorubicin --k 0
```

**Using last-layer fine-tuning:**
```bash
python inference.py --tissue bone --drug-a Methotrexate --drug-b Doxorubicin --strategy last_layer
```

### 2. Web Interface

**Start the web server:**
```bash
python inference.py --web
```

Then open: **http://localhost:5000**

Beautiful UI with:
- 🎨 Interactive form
- 📊 Real-time results visualization
- 📈 Detailed metrics display
- 🧬 Model information panel

### Available Options

#### Tissues
- Pancreas
- Stomach
- Urinary Tract
- Bone
- Endometrium
- Liver
- Soft Tissue

#### K-Shot Values
- **0** (Zero-shot): No examples, pure transfer
- **2** (Few-shot): 2 training examples
- **4** (Few-shot): 4 training examples
- **8** (Few-shot): 8 training examples (default)
- **16** (Few-shot): 16 training examples
- **32** (Few-shot): 32 training examples

#### Fine-tuning Strategies
- **full**: Fine-tune all model weights
- **last_layer**: Only fine-tune output layer

---

## Example Outputs

### Pancreas (K=16)
```
Synergy Probability: 24.2%
Likelihood: LOW

Model Performance:
  - Accuracy: 90.9%
  - AUROC: 0.900
  - AUPRC: 0.500
```

### Urinary Tract (K=0 Zero-shot)
```
Synergy Probability: 76.4%
Likelihood: HIGH

Model Performance:
  - Accuracy: 8.3%
  - AUROC: 1.000
  - AUPRC: 1.000
```

---

## Python API Usage

```python
from inference import InferenceEngine

# Initialize
engine = InferenceEngine()

# Single prediction
result = engine.predict(
    tissue="pancreas",
    drug_a="Paclitaxel",
    drug_b="Cisplatin",
    k=16,
    strategy="full"
)

if result['success']:
    print(f"Synergy: {result['synergy_probability']*100:.1f}%")
    print(f"AUROC: {result['metrics']['auroc']:.3f}")

# Batch predictions
pairs = [
    ("Drug1", "Drug2"),
    ("Drug3", "Drug4"),
    ("Drug5", "Drug6")
]
results = engine.batch_predict(pairs, tissue="stomach", k=8)

# Tissue summary
summary = engine.get_tissue_summary("bone")
print(summary)
```

---

## Interpreting Results

### Synergy Probability
- **70-100%**: HIGH likelihood of synergy
- **40-70%**: MEDIUM likelihood of synergy  
- **0-40%**: LOW likelihood of synergy

### AUROC (Area Under ROC Curve)
- **0.90-1.00**: Excellent
- **0.80-0.90**: Very Good
- **0.70-0.80**: Good
- **0.60-0.70**: Fair
- **0.50-0.60**: Poor

### AUPRC (Area Under Precision-Recall)
- More informative for imbalanced datasets
- Penalizes false positives more than AUROC
- Use alongside AUROC for comprehensive evaluation

---

## Best Practices

1. **Choose appropriate K**
   - Start with K=0 (zero-shot) for fast screening
   - Use K=8-16 for better accuracy if time permits

2. **Check model metrics**
   - High AUROC alone isn't enough
   - Verify AUPRC, especially for rare synergies

3. **Consider tissue type**
   - Pancreas: K=16 works best
   - Urinary Tract: Zero-shot perfect
   - Stomach: K=4-16 both good

4. **Ensemble predictions**
   - Get predictions from both strategies
   - Average probabilities for consensus

---

## Performance Summary

### Best Performing Tissues (K-Shot)

| Tissue | Best K | AUROC | AUPRC |
|--------|--------|-------|-------|
| Urinary Tract | 0 | 1.00 | 1.00 |
| Bone | 0 | 1.00 | 1.00 |
| Pancreas | 16 | 0.90 | 0.50 |
| Stomach | 4 | 0.92 | 0.50 |

### Baseline Comparison (XGBoost)

All tissues: **AUROC ≈ 1.00, AUPRC ≈ 1.00** (on validation set)

---

## Troubleshooting

**Q: "Tissue not found" error**  
A: Check spelling and capitalization. Use: pancreas, stomach, urinary tract, bone, endometrium, liver, soft tissue

**Q: "K value not available" error**  
A: Valid K values are: 0, 2, 4, 8, 16, 32

**Q: Flask not installed (for web interface)**  
A: Install with: `pip install flask`

**Q: Results seem inconsistent**  
A: This is expected! LLMs can be variable. Use ensemble or multiple K values for consensus.

---

## References

- Model: CancerGPT (GPT-2 base, 124M parameters)
- Data: Synthetic DrugCombDB dataset (1000 samples)
- Evaluation: K-shot learning with AUROC/AUPRC
- Publication: CancerGPT K-shot Learning Framework

---

*For detailed results analysis, see: ANALYSIS_RESULTS.md*
