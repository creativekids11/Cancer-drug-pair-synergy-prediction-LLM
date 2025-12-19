# Cancer Drug Pair Model and Training Process

## Overview
This project predicts the synergistic effect of drug pairs on cancer cell lines using machine learning and large language models (LLMs). The workflow includes data preparation, model architecture, training, evaluation, and inference.

---

## 1. Data Preparation
- **Source:** Drug combination datasets (e.g., DrugComb, custom CSVs)
- **Scripts:**
  - `prepare_data.py`: Cleans and splits raw data into train/val/test sets.
  - `data_pipeline/`: Contains utilities for prompt building, dataset cleaning, and tissue-specific splitting.
- **Outputs:**
  - `data_prepared/`: Contains `train.csv`, `val.csv`, `test.csv`, and statistics.

---

## 2. Model Architecture
- **Baseline Models:** Implemented in `baseline_models.py` (e.g., Random Forest, XGBoost, simple neural nets).
- **CancerGPT Model:**
  - Defined in `cancergpt_model.py`
  - Utilizes LLMs (e.g., GPT-based) for prompt-based prediction.
  - Supports k-shot finetuning (`cancergpt_kshot_finetuning.py`).
- **Prompt Engineering:**
  - `data_pipeline/build_prompts.py` and `prompt_utils.py` generate input prompts for LLMs.

---

## 3. Training Process
- **Main Training Script:** `run_experiments.py` or `run.py`
- **Steps:**
  1. Load and preprocess data.
  2. Build prompts for LLM input.
  3. Train baseline or LLM-based models.
  4. Optionally finetune LLMs with k-shot learning.
  5. Save results and evaluation metrics.
- **Configuration:**
  - Hyperparameters and settings are managed via config files (see `data_pipeline/config.example.json`).

---

## 4. Evaluation
- **Script:** `evaluate_cancergpt.py`
- **Metrics:**
  - Synergy prediction accuracy
  - ROC-AUC, F1-score, etc.
- **Results:**
  - Saved in `results/` with per-experiment folders and JSON summaries.

---

## 5. Inference
- **Script:** `run.py` or custom scripts
- **Process:**
  - Load trained model
  - Prepare new drug pair/cell line data
  - Generate predictions using the model

---

## 6. Key Files and Directories
- `README.md`: Project overview and quickstart
- `baseline_models.py`: Baseline ML models
- `cancergpt_model.py`: Main LLM model
- `cancergpt_kshot_finetuning.py`: Finetuning logic
- `data_pipeline/`: Data and prompt utilities
- `run_experiments.py`: Main experiment runner
- `evaluate_cancergpt.py`: Evaluation logic
- `requirements.txt`: Dependencies

---

## 7. How to Train
1. Prepare data: `python prepare_data.py`
2. Run experiments: `python run_experiments.py`
3. Evaluate: `python evaluate_cancergpt.py`

---

## 8. Customization
- Modify `data_pipeline/` scripts for new datasets or prompt styles.
- Adjust model hyperparameters in config files.
- Add new models in `baseline_models.py` or extend `cancergpt_model.py`.

---

## 9. References
- See `README.md` and `QUICKSTART.md` for setup and usage.
- Check `data_pipeline/README_IMPROVEMENTS.md` for ideas on extending the pipeline.

---

For further details, review the code and comments in each script. This document provides a high-level map for understanding and extending the project.