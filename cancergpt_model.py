#!/usr/bin/env python3
"""
cancergpt_model.py

Implementation of CancerGPT - Few-shot drug pair synergy prediction using LLMs
Based on: Li et al. "CancerGPT for few shot drug pair synergy prediction using 
large pretrained language models" (npj Digital Medicine, 2024)

Key Components:
- Tabular to natural language conversion
- LLM-based feature extraction
- Classification head for binary synergy prediction
- k-shot fine-tuning strategy
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import (
    GPT2Tokenizer, GPT2Model,
    AutoTokenizer, AutoModel,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

log = logging.getLogger("cancergpt")

# -----------------------
# DATA CONVERSION
# -----------------------

class TabularToText:
    """Convert tabular data to natural language text for LLM input"""
    
    @staticmethod
    def text_template(row: Dict) -> str:
        """
        Convert row to natural language using text template strategy
        
        Format: "The first drug is {drug1}. The second drug is {drug2}. 
                 The cell line is {cell_line}. Tissue is {tissue}. 
                 The first drug's sensitivity using relative inhibition is {sensitivity1}. 
                 The second drug's sensitivity using relative inhibition is {sensitivity2}."
        """
        return (
            f"The first drug is {row['drugA']}. "
            f"The second drug is {row['drugB']}. "
            f"The cell line is {row['cell_line']}. "
            f"Tissue is {row['tissue']}. "
            f"The first drug's sensitivity using relative inhibition is {row.get('sensitivity_A', 0):.3f}. "
            f"The second drug's sensitivity using relative inhibition is {row.get('sensitivity_B', 0):.3f}."
        )
    
    @staticmethod
    def create_prompt(text_input: str, task: str = "synergy") -> str:
        """Create prompt for LLM"""
        if task == "synergy":
            return f"Decide in a single word if the synergy of the drug combination in the cell line is positive or not. {text_input} Synergy:"
        else:
            return text_input


# -----------------------
# DATASET
# -----------------------

class DrugSynergyDataset(Dataset):
    """Dataset for drug synergy prediction"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
        add_sensitivity: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            df: DataFrame with columns [drugA, drugB, cell_line, tissue, synergy_label]
            tokenizer: Tokenizer for LLM
            max_length: Maximum token length
            add_sensitivity: Add drug sensitivity scores if available
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_sensitivity = add_sensitivity
        
        log.info(f"Loaded dataset with {len(df)} samples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Convert to text
        text = TabularToText.text_template(row)
        prompt = TabularToText.create_prompt(text)
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(row["synergy_label"], dtype=torch.long),
            "drug_pair": f"{row['drugA']}_{row['drugB']}",
            "cell_line": row["cell_line"],
            "tissue": row["tissue"]
        }


# -----------------------
# MODEL ARCHITECTURE
# -----------------------

class CancerGPTModel(nn.Module):
    """
    CancerGPT: LLM-based drug synergy predictor
    
    Architecture:
    - GPT-2 encoder (frozen or trainable)
    - Classification head
    - Binary classification (synergistic / not synergistic)
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        hidden_size: int = 768,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Initialize model
        
        Args:
            model_name: Pretrained model name (gpt2, gpt2-medium, etc.)
            hidden_size: Hidden dimension of LLM
            num_classes: Number of classification classes (2 for binary)
            dropout_rate: Dropout rate
            freeze_backbone: Freeze LLM parameters (last-layer training)
        """
        super().__init__()
        
        # Load pretrained LLM
        self.lm = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.lm.parameters():
                param.requires_grad = False
            log.info("Backbone LLM frozen (last-layer training)")
        else:
            log.info("Full fine-tuning enabled")
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            return_embeddings: Return LLM embeddings instead of logits
        
        Returns:
            logits [batch_size, num_classes] or embeddings [batch_size, hidden_size]
        """
        # Get LLM output
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state of final token (as per paper)
        # Find last non-padded token for each sequence
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        
        # Get embeddings of last token
        batch_indices = torch.arange(last_hidden_state.size(0))
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 to get last index
        seq_lengths = torch.clamp(seq_lengths, min=0)
        
        embeddings = last_hidden_state[batch_indices, seq_lengths]  # [batch_size, hidden_size]
        
        if return_embeddings:
            return embeddings
        
        # Classification head
        pooled = self.dropout(embeddings)
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        
        return logits
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get LLM embeddings for downstream tasks"""
        return self.forward(input_ids, attention_mask, return_embeddings=True)


# -----------------------
# TRAINING
# -----------------------

class CancerGPTTrainer:
    """Trainer for k-shot fine-tuning"""
    
    def __init__(
        self,
        model: CancerGPTModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 4
    ):
        """
        Initialize trainer
        
        Args:
            model: CancerGPT model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        log.info(f"Trainer initialized on {device}")
    
    def setup_optimizer(self, train_loader: DataLoader) -> Tuple:
        """Setup optimizer and scheduler"""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        scheduler
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        log.info(f"Training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = np.mean(all_preds == all_labels)
        
        # Handle cases where there's only one class in the set
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.5
        
        try:
            auprc = average_precision_score(all_labels, all_probs)
        except:
            auprc = 0.5
        
        return {
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc,
            "preds": all_preds,
            "probs": all_probs,
            "labels": all_labels
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        
        Returns:
            Training history
        """
        optimizer, scheduler = self.setup_optimizer(train_loader)
        
        history = {
            "train_loss": [],
            "val_metrics": []
        }
        
        for epoch in range(self.num_epochs):
            log.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            history["train_loss"].append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_metrics"].append(val_metrics)
                log.info(
                    f"Val - Accuracy: {val_metrics['accuracy']:.4f}, "
                    f"AUROC: {val_metrics['auroc']:.4f}, "
                    f"AUPRC: {val_metrics['auprc']:.4f}"
                )
        
        return history


# -----------------------
# INFERENCE
# -----------------------

class CancerGPTPredictor:
    """Inference module for drug synergy prediction"""
    
    def __init__(
        self,
        model: CancerGPTModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize predictor"""
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(
        self,
        dataset: DrugSynergyDataset,
        batch_size: int = 32
    ) -> Dict:
        """
        Make predictions on dataset
        
        Returns:
            predictions, probabilities, embeddings
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_probs = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get predictions
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Get embeddings
                embeddings = self.model.get_embeddings(input_ids, attention_mask)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_embeddings.extend(embeddings.cpu().numpy())
        
        return {
            "predictions": np.array(all_preds),
            "probabilities": np.array(all_probs),
            "embeddings": np.array(all_embeddings)
        }
    
    def predict_single(
        self,
        drug_a: str,
        drug_b: str,
        cell_line: str,
        tissue: str,
        sensitivity_a: float = 0.0,
        sensitivity_b: float = 0.0
    ) -> Dict:
        """
        Predict synergy for single drug pair
        
        Returns:
            prediction (0/1), probability of synergy, embedding
        """
        row = {
            "drugA": drug_a,
            "drugB": drug_b,
            "cell_line": cell_line,
            "tissue": tissue,
            "sensitivity_A": sensitivity_a,
            "sensitivity_B": sensitivity_b,
            "synergy_label": 0  # Dummy label
        }
        
        # Create dummy dataframe
        df = pd.DataFrame([row])
        dataset = DrugSynergyDataset(df, self.model.tokenizer)
        
        result = self.predict(dataset, batch_size=1)
        
        return {
            "prediction": result["predictions"][0],
            "probability_synergistic": result["probabilities"][0, 1],
            "probability_antagonistic": result["probabilities"][0, 0],
            "embedding": result["embeddings"][0]
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = CancerGPTModel(model_name="gpt2", freeze_backbone=False)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy data
    df = pd.DataFrame({
        "drugA": ["Aspirin", "Ibuprofen"],
        "drugB": ["Acetaminophen", "Naproxen"],
        "cell_line": ["HeLa", "MCF7"],
        "tissue": ["breast", "lung"],
        "sensitivity_A": [0.5, 0.6],
        "sensitivity_B": [0.7, 0.8],
        "synergy_label": [1, 0]
    })
    
    # Create dataset
    dataset = DrugSynergyDataset(df, model.tokenizer)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test forward pass
    batch = dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        embeddings = model.get_embeddings(input_ids, attention_mask)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
