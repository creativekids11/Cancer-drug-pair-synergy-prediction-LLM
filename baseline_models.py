#!/usr/bin/env python3
"""
baseline_models.py

Baseline models for comparison with CancerGPT
Implements: XGBoost, TabTransformer, Collaborative Filtering

Based on paper: Li et al. "CancerGPT for few shot drug pair synergy prediction"
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup

log = logging.getLogger("baselines")

# -----------------------
# XGBOOST
# -----------------------

class XGBoostModel:
    """XGBoost baseline for tabular data"""
    
    def __init__(
        self,
        learning_rate: float = 0.3,
        max_depth: int = 20,
        n_estimators: int = 1000,
        random_state: int = 42
    ):
        """Initialize XGBoost model"""
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=random_state,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for XGBoost"""
        # Encode categorical features
        df_copy = df.copy()
        
        encoders = {}
        for col in ["drugA", "drugB", "cell_line", "tissue"]:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                encoders[col] = le
        
        # Extract features
        feature_cols = [col for col in df_copy.columns if col not in ["synergy_label"]]
        X = df_copy[feature_cols].values.astype(np.float32)
        y = df_copy["synergy_label"].values
        
        return X, y
    
    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train XGBoost"""
        X_train, y_train = self.prepare_features(train_df)
        
        eval_set = None
        if val_df is not None:
            X_val, y_val = self.prepare_features(val_df)
            eval_set = [(X_val, y_val)]
        
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    def predict(self, test_df: pd.DataFrame) -> Dict:
        """Predict"""
        X_test, y_test = self.prepare_features(test_df)
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        try:
            auroc = roc_auc_score(y_test, y_proba)
            auprc = average_precision_score(y_test, y_proba)
        except:
            auroc, auprc = 0.5, 0.5
        
        return {
            "predictions": y_pred,
            "probabilities": y_proba,
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc
        }


# -----------------------
# TABTRANSFORMER
# -----------------------

class TabularDataset(Dataset):
    """Dataset for tabular models"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize dataset"""
        self.df = df.copy()
        
        # Encode categorical features
        self.encoders = {}
        for col in ["drugA", "drugB", "cell_line", "tissue"]:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
        
        # Separate features and labels
        self.labels = self.df["synergy_label"].values
        self.features = self.df.drop("synergy_label", axis=1).values.astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class TabTransformerModel(nn.Module):
    """
    TabTransformer: Transformer for Tabular Data
    
    Architecture:
    - Embedding layers for categorical features
    - Multi-head attention layers
    - MLP head for classification
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        num_classes: int = 2
    ):
        """Initialize TabTransformer"""
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Embedding
        x = self.embedding(x)  # [batch_size, hidden_dim]
        x = x.unsqueeze(1)      # [batch_size, 1, hidden_dim]
        
        # Transformer
        x = self.transformer(x)  # [batch_size, 1, hidden_dim]
        x = x.mean(dim=1)        # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(x)  # [batch_size, num_classes]
        
        return logits


class TabTransformerTrainer:
    """Trainer for TabTransformer"""
    
    def __init__(
        self,
        model: TabTransformerModel,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize trainer"""
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation set"""
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                
                logits = self.model(features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_preds == all_labels)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
            auprc = average_precision_score(all_labels, all_probs)
        except:
            auroc, auprc = 0.5, 0.5
        
        return {
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc,
            "predictions": all_preds,
            "probabilities": all_probs
        }
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 50):
        """Full training"""
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            if (epoch + 1) % 10 == 0:
                log.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Loss: {train_loss:.4f}, AUROC: {val_metrics['auroc']:.4f}"
                )


# -----------------------
# COLLABORATIVE FILTERING
# -----------------------

class CollaborativeFilteringModel:
    """
    Collaborative Filtering for drug synergy prediction
    
    Intuition: If drug pair (A, B) is synergistic in cell line C1,
    it's likely to be synergistic in cell line C2 if A and B have
    similar effects in C2 as in C1.
    """
    
    def __init__(self):
        """Initialize model"""
        self.drug_similarities = {}
        self.cell_line_similarities = {}
        self.training_data = None
    
    def train(self, train_df: pd.DataFrame):
        """Train collaborative filtering"""
        self.training_data = train_df.copy()
        
        # Compute drug similarities based on co-occurrence in synergistic pairs
        drugs = set(train_df["drugA"].unique()) | set(train_df["drugB"].unique())
        
        for drug1 in drugs:
            for drug2 in drugs:
                if drug1 == drug2:
                    self.drug_similarities[(drug1, drug2)] = 1.0
                else:
                    # Count synergistic pairs
                    mask1 = ((train_df["drugA"] == drug1) | (train_df["drugB"] == drug1)) & \
                            (train_df["synergy_label"] == 1)
                    mask2 = ((train_df["drugA"] == drug2) | (train_df["drugB"] == drug2)) & \
                            (train_df["synergy_label"] == 1)
                    
                    common = len(train_df[mask1 & mask2])
                    total1 = len(train_df[mask1])
                    total2 = len(train_df[mask2])
                    
                    if total1 > 0 and total2 > 0:
                        similarity = common / max(total1, total2)
                    else:
                        similarity = 0.0
                    
                    self.drug_similarities[(drug1, drug2)] = similarity
    
    def predict(self, test_df: pd.DataFrame) -> Dict:
        """Predict using collaborative filtering"""
        predictions = []
        probabilities = []
        
        for _, row in test_df.iterrows():
            drug1 = row["drugA"]
            drug2 = row["drugB"]
            cell_line = row["cell_line"]
            
            # Find similar drug pairs in training data
            similar_pairs = []
            
            for _, train_row in self.training_data.iterrows():
                # Check if drug pair has similar drugs
                if ((train_row["drugA"] == drug1 or train_row["drugB"] == drug1) and
                    (train_row["drugA"] == drug2 or train_row["drugB"] == drug2)):
                    
                    # Weight by synergy label
                    weight = train_row["synergy_label"]
                    similar_pairs.append(weight)
            
            # Make prediction
            if similar_pairs:
                prob = np.mean(similar_pairs)
            else:
                prob = 0.5  # Default to 0.5 if no similar pairs found
            
            pred = 1 if prob > 0.5 else 0
            predictions.append(pred)
            probabilities.append(prob)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        labels = test_df["synergy_label"].values
        
        accuracy = np.mean(predictions == labels)
        try:
            auroc = roc_auc_score(labels, probabilities)
            auprc = average_precision_score(labels, probabilities)
        except:
            auroc, auprc = 0.5, 0.5
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc
        }


# -----------------------
# EVALUATION UTILITIES
# -----------------------

def compare_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: Optional[list] = None
) -> Dict:
    """
    Compare multiple baseline models
    
    Args:
        train_df: Training data
        test_df: Test data
        models: List of model names to evaluate
    
    Returns:
        Dictionary with results for each model
    """
    if models is None:
        models = ["xgboost", "tabtransformer", "collaborative_filtering"]
    
    results = {}
    
    if "xgboost" in models:
        log.info("Training XGBoost...")
        xgb = XGBoostModel()
        xgb.train(train_df)
        results["xgboost"] = xgb.predict(test_df)
    
    if "tabtransformer" in models:
        log.info("Training TabTransformer...")
        
        train_dataset = TabularDataset(train_df)
        test_dataset = TabularDataset(test_df)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        input_dim = train_dataset.features.shape[1]
        model = TabTransformerModel(input_dim=input_dim)
        trainer = TabTransformerTrainer(model)
        
        trainer.fit(train_loader, test_loader, num_epochs=50)
        results["tabtransformer"] = trainer.evaluate(test_loader)
    
    if "collaborative_filtering" in models:
        log.info("Training Collaborative Filtering...")
        cf = CollaborativeFilteringModel()
        cf.train(train_df)
        results["collaborative_filtering"] = cf.predict(test_df)
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    
    drugs = ["Drug_A", "Drug_B", "Drug_C", "Drug_D"]
    cell_lines = ["HeLa", "MCF7", "A549"]
    tissues = ["breast", "lung"]
    
    train_df = pd.DataFrame({
        "drugA": np.random.choice(drugs, n_samples),
        "drugB": np.random.choice(drugs, n_samples),
        "cell_line": np.random.choice(cell_lines, n_samples),
        "tissue": np.random.choice(tissues, n_samples),
        "sensitivity_A": np.random.rand(n_samples),
        "sensitivity_B": np.random.rand(n_samples),
        "synergy_label": np.random.randint(0, 2, n_samples)
    })
    
    test_df = train_df.sample(20)
    
    # Compare baselines
    results = compare_baselines(train_df, test_df)
    
    for model_name, metrics in results.items():
        print(f"{model_name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")
