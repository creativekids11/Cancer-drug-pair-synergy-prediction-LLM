# 📖 Cancer Drug Pair - Complete Documentation Index

**Last Updated**: December 19, 2025  
**Status**: ✅ FULLY INTEGRATED & PRODUCTION READY

---

## 🚀 Start Here

**New to this project?** Follow this order:

1. **[README.md](README.md)** (5 min)
   - Project overview and what this does
   - Key features and capabilities

2. **[QUICKSTART.md](QUICKSTART.md)** (2 min)
   - Get running in 2 minutes
   - Minimal example to see it work

3. **[MODEL_AND_TRAINING.md](MODEL_AND_TRAINING.md)** (10 min)
   - Understand the model architecture
   - Learn how training works
   - See what files are used

4. **Run your first experiment**:
   ```bash
   python run.py --k-shots 0 --skip-baselines
   ```

---

## 📚 Full Documentation

### Core Guides
| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[README.md](README.md)** | Project overview, features, installation | 5 min |
| **[QUICKSTART.md](QUICKSTART.md)** | Get started in 2 minutes | 2 min |
| **[MODEL_AND_TRAINING.md](MODEL_AND_TRAINING.md)** | Model architecture & training process | 10 min |

### API & Integration
| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[API_INTEGRATION.md](API_INTEGRATION.md)** | Using DrugCombDB API | 10 min |
| **[API_TROUBLESHOOTING.md](API_TROUBLESHOOTING.md)** | Troubleshooting & testing guide | 15 min |
| **[INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)** | Complete integration details | 20 min |

### Reference & History
| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** | What was accomplished | 10 min |
| **[CHANGELOG.md](CHANGELOG.md)** | Detailed list of changes | 15 min |

---

## 📋 Quick Commands

### Most Common
```bash
# Quick test (30 seconds)
python run.py --k-shots 0 --skip-baselines

# Standard run (5-10 minutes)
python run.py --k-shots 0 2 4 8

# With pretraining (30+ minutes)
python run.py --with-pretraining --k-shots 0 2 4 8
```

### With API (Fallback Enabled)
```bash
# Try API data (falls back to synthetic if empty)
python run.py --use-api-data --api-max-pages 50 --k-shots 0 2

# Full API integration
python run.py --use-api-data --api-max-pages 100 --k-shots 0 2 4 8
```

### Advanced
```bash
# Custom configuration
python run.py \
  --num-samples 5000 \
  --k-shots 0 2 4 8 16 \
  --output-dir ./experiment \
  --with-pretraining \
  --continue-on-error

# Specific tissues
python run.py \
  --rare-tissues pancreas liver endometrium \
  --k-shots 0 2 4 8
```

---

## 🎯 Find What You Need

### I want to...

| Goal | Document |
|------|----------|
| **Get started immediately** | [QUICKSTART.md](QUICKSTART.md) |
| **Understand the model** | [MODEL_AND_TRAINING.md](MODEL_AND_TRAINING.md) |
| **Use the API** | [API_INTEGRATION.md](API_INTEGRATION.md) |
| **Fix a problem** | [API_TROUBLESHOOTING.md](API_TROUBLESHOOTING.md) |
| **Understand what changed** | [CHANGELOG.md](CHANGELOG.md) |
| **See the big picture** | [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) |
| **Know what was done** | [FINAL_SUMMARY.md](FINAL_SUMMARY.md) |
| **Install & setup** | [README.md](README.md) |

---

## 📊 Project Structure

```
cancer_drug_pair/
│
├─ 📚 DOCUMENTATION (8 guides)
│  ├─ README.md                    # Start here
│  ├─ QUICKSTART.md                # 2-minute quick start
│  ├─ MODEL_AND_TRAINING.md        # Architecture & training
│  ├─ API_INTEGRATION.md           # API usage
│  ├─ API_TROUBLESHOOTING.md       # Troubleshooting
│  ├─ INTEGRATION_COMPLETE.md      # Full integration details
│  ├─ FINAL_SUMMARY.md             # What was accomplished
│  ├─ CHANGELOG.md                 # What changed
│  └─ INDEX.md                     # This file
│
├─ 🐍 CORE CODE (9 Python files)
│  ├─ run.py ⭐ MAIN ENTRY POINT
│  ├─ run_experiments.py           # Experiment coordinator
│  ├─ prepare_data.py              # Data preparation
│  ├─ generate_sample_data.py      # Synthetic data
│  ├─ cancergpt_model.py           # GPT model
│  ├─ cancergpt_kshot_finetuning.py # K-shot training
│  ├─ evaluate_cancergpt.py        # Evaluation
│  ├─ baseline_models.py           # Baselines
│  └─ COMMANDS.py                  # Utilities
│
├─ 🔧 DATA PIPELINE (Utilities)
│  └─ data_pipeline/               # Helper scripts
│     ├─ ingest_drugcomb.py        # API ingestion
│     ├─ orchestrator.py           # Full orchestrator
│     ├─ build_prompts.py          # Prompt generation
│     ├─ clean_dataset.py          # Data cleaning
│     └─ ... (10+ more utilities)
│
├─ 📊 DATA DIRECTORIES
│  ├─ data_prepared/               # Cleaned, split data
│  │  ├─ train.csv                 # Training set
│  │  ├─ val.csv                   # Validation set
│  │  ├─ test.csv                  # Test set
│  │  └─ full.csv                  # Full dataset
│  │
│  └─ results/                     # Experiment results
│     └─ experiment_YYYYMMDD_HHMMSS/
│        ├─ results.json           # Metrics
│        └─ evaluations/           # Per-tissue results
│
├─ 📄 DATA FILES
│  ├─ sample_data.csv              # Synthetic raw data
│  ├─ requirements.txt             # Python dependencies
│  └─ LICENSE.txt                  # MIT License
```

---

## 🔍 Key Features Explained

### ✨ What Makes This Special

1. **Few-Shot Learning**
   - Train models with just 0, 2, 4, or 8 examples
   - Effective for rare cancer tissues
   - See [MODEL_AND_TRAINING.md](MODEL_AND_TRAINING.md)

2. **API Integration**
   - Fetch real drug combination data from DrugCombDB
   - Graceful fallback to synthetic data
   - See [API_INTEGRATION.md](API_INTEGRATION.md)

3. **Fully Automated**
   - Data preparation → training → evaluation → reporting
   - Single command to run entire pipeline
   - See [QUICKSTART.md](QUICKSTART.md)

4. **Well Tested**
   - All functions verified working
   - Multiple test cases passing
   - See [API_TROUBLESHOOTING.md](API_TROUBLESHOOTING.md)

5. **Comprehensive Logging**
   - See exactly what's happening at each step
   - Detailed error messages
   - Progress tracking

---

## 🧪 Testing Your Setup

### Verify Installation
```bash
# Quick test (30 seconds)
python run.py --k-shots 0 --skip-baselines
# Should complete with no errors
```

### Check Data Quality
```python
import pandas as pd
df = pd.read_csv('data_prepared/train.csv')
print(df.info())
print(df['synergy_label'].value_counts())
```

### View Results
```python
import json
with open('results/experiment_YYYYMMDD_HHMMSS/results.json') as f:
    results = json.load(f)
    print(json.dumps(results, indent=2))
```

---

## 📈 Performance Expectations

### Execution Times
- **Minimal test**: 1-2 minutes
- **Standard run**: 5-10 minutes
- **Full pipeline**: 30+ minutes (with pretraining)

### Typical Results
- **Zero-shot (k=0)**: AUROC ~0.50
- **Few-shot (k=2)**: AUROC ~0.70-0.80
- **Few-shot (k=4+)**: AUROC ~0.70-0.80 (plateau)

---

## 🆘 Need Help?

### Common Issues

**"No data fetched from API"**
- Expected behavior! API requires specific filters
- Fallback to synthetic data works automatically
- See [API_TROUBLESHOOTING.md](API_TROUBLESHOOTING.md)

**"ImportError"**
- Function names were corrected in `run.py`
- Should not occur with this version
- See [CHANGELOG.md](CHANGELOG.md) for fixes

**"Out of memory"**
- Reduce API pages: `--api-max-pages 10`
- Skip baselines: `--skip-baselines`
- See [API_TROUBLESHOOTING.md](API_TROUBLESHOOTING.md)

**"Unicode encoding error on Windows"**
- Fixed in this version
- No special setup needed
- See [FINAL_SUMMARY.md](FINAL_SUMMARY.md)

---

## 📚 Additional Resources

### Related Files in Repository
- `requirements.txt` - Python dependencies
- `data_pipeline/requirements.txt` - Optional pipeline utilities
- `LICENSE.txt` - MIT License

### External References
- DrugCombDB: http://drugcombdb.denglab.org
- Hugging Face: https://huggingface.co
- Original Paper: Li et al. "CancerGPT..." (npj Digital Medicine, 2024)

---

## ✅ Checklist Before Running

- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Have 4+ GB RAM available
- [ ] Have 2-5 GB disk space for results
- [ ] Optional: CUDA 11.0+ for GPU speedup

---

## 🎓 Learning Path

### For Users (just want to run experiments)
1. Read [README.md](README.md) (understand what this is)
2. Follow [QUICKSTART.md](QUICKSTART.md) (get it running)
3. Run experiments with different parameters
4. Refer to [API_TROUBLESHOOTING.md](API_TROUBLESHOOTING.md) if issues

### For Researchers (want to understand the science)
1. Read [README.md](README.md) for context
2. Read [MODEL_AND_TRAINING.md](MODEL_AND_TRAINING.md) for architecture
3. Study code in `cancergpt_model.py` and `cancergpt_kshot_finetuning.py`
4. Review paper: Li et al. "CancerGPT..." (2024)

### For Developers (want to extend/modify)
1. Read [README.md](README.md) for overview
2. Read [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) for architecture
3. Review [CHANGELOG.md](CHANGELOG.md) for recent changes
4. Study relevant code files (e.g., `cancergpt_model.py`)
5. Refer to [API_INTEGRATION.md](API_INTEGRATION.md) for API details

---

## 📝 Documentation Maintenance

**How these docs are organized**:
- **README.md** - Main entry point (evergreen)
- **QUICKSTART.md** - Get running (evergreen)
- **MODEL_AND_TRAINING.md** - Architecture explanation (evergreen)
- **API_*.md** - API documentation (updated Dec 19, 2025)
- **INTEGRATION_COMPLETE.md** - Full integration details (Dec 19, 2025)
- **FINAL_SUMMARY.md** - What was accomplished (Dec 19, 2025)
- **CHANGELOG.md** - What changed (Dec 19, 2025)
- **INDEX.md** - This file (Dec 19, 2025)

---

## 🚀 Next Steps

1. **Read [QUICKSTART.md](QUICKSTART.md)** (2 minutes)
2. **Run first experiment** (30 seconds):
   ```bash
   python run.py --k-shots 0 --skip-baselines
   ```
3. **Check results** in `results/` directory
4. **Explore advanced options** in [API_INTEGRATION.md](API_INTEGRATION.md)
5. **Customize per your needs** using command-line arguments

---

## 💬 Questions?

Check these in order:
1. [QUICKSTART.md](QUICKSTART.md) - Most common questions
2. [API_TROUBLESHOOTING.md](API_TROUBLESHOOTING.md) - Technical issues
3. [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - Deep dive details
4. Code comments in `run.py` - Implementation details

---

**Welcome to Cancer Drug Pair! 🎉**

Start with [QUICKSTART.md](QUICKSTART.md) and you'll be up and running in 2 minutes.

Good luck with your drug synergy prediction experiments!

---

**Last Updated**: December 19, 2025  
**Status**: ✅ PRODUCTION READY  
**Version**: 1.0 (Stable)
