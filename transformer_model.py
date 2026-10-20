"""
===============================================================================
Week 3: Transformer Models (SafeAeroBERT & DistilBERT)
===============================================================================
Reference Papers:
  1. Andrade & Walsh (2023) -- "SafeAeroBERT" (Section IV)
  2. Liu, Li, Ng, Han & Feng (2025) -- "Accident investigation via LLMs reasoning:
     HFACS-guided Chain-of-Thoughts" (Appendix C/D for structural reference)

Implements:
  - A self-contained PyTorch training loop for sequence classification.
  - Binary relevance (multi-label) fine-tuning:
      one dense layer per HFACS class with sigmoid activation
      and binary cross-entropy loss.
  - Checkpoint switch: DistilBERT (base) vs. SafeAeroBERT (domain-pretrained).
  - Per-class and aggregate metrics (Precision, Recall, F1).

Usage:
    python transformer_model.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
)
import warnings

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    # Model selection: "distilbert-base-uncased" or a SafeAeroBERT checkpoint
    # Example SafeAeroBERT: "andradewalsh/SafeAeroBERT" (if public on HF Hub)
    "BASE_CHECKPOINT": "distilbert-base-uncased",
    # If using a local checkpoint, provide the path instead.
    # "BASE_CHECKPOINT": "./safeaerobert-checkpoint",

    # Training hyperparameters
    "MAX_LENGTH": 128,
    "BATCH_SIZE": 16,
    "EPOCHS": 3,
    "LEARNING_RATE": 2e-5,
    "WARMUP_RATIO": 0.1,  # Fraction of total steps for linear warmup
    "WEIGHT_DECAY": 0.01,

    # Data paths
    "TRAIN_PATH": "weak_labeling_results/dataset_majority_train.csv",
    "VAL_PATH": "weak_labeling_results/dataset_majority_validation.csv",
    "TEST_PATH": "weak_labeling_results/dataset_majority_test.csv",

    # Label set
    "HFACS_CLASSES": [
        "Skill-Based Error",
        "Decision Error",
        "Violation",
        "Perceptual Error"
    ],

    # Output directory
    "OUTPUT_DIR": "week3_results",

    # Device (auto-detect)
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class HFACSDataset(Dataset):
    """Dataset for HFACS multi-label classification."""

    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float),
        }


def load_csv_data(csv_path: str) -> Tuple[List[str], List[List[int]]]:
    """Load text and multi-label binarized HFACS labels from CSV."""
    df = pd.read_csv(csv_path)

    texts = df["text"].astype(str).tolist()

    # Convert string labels to binary vectors
    label_to_idx = {label: idx for idx, label in enumerate(CONFIG["HFACS_CLASSES"])}
    binary_labels = []
    for label_str in df["label"]:
        # Assume label_str is one of the HFACS classes (single-label for now)
        vec = [0] * len(CONFIG["HFACS_CLASSES"])
        if label_str in label_to_idx:
            vec[label_to_idx[label_str]] = 1
        binary_labels.append(vec)

    return texts, binary_labels


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class HFACSClassifier(nn.Module):
    """Transformer encoder + binary relevance classification head."""

    def __init__(self, checkpoint: str, n_classes: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch, n_classes)
        return logits


# -----------------------------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(data_loader)


def eval_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    loss_avg = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    hloss = hamming_loss(all_labels, all_preds)

    # Per-class metrics
    per_class = {}
    for idx, class_name in enumerate(CONFIG["HFACS_CLASSES"]):
        y_true = all_labels[:, idx]
        y_pred = all_preds[:, idx]
        per_class[class_name] = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    metrics = {
        "loss": loss_avg,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "hamming_loss": hloss,
        "per_class": per_class,
    }

    return loss_avg, metrics


def main():
    print("=" * 70)
    print("WEEK 3: TRANSFORMER MODEL TRAINING (SafeAeroBERT / DistilBERT)")
    print("=" * 70)
    print(f"Using device: {CONFIG['DEVICE']}")
    print(f"Base checkpoint: {CONFIG['BASE_CHECKPOINT']}")
    print()

    # Create output directory
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(exist_ok=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["BASE_CHECKPOINT"])

    # Load datasets
    print("[INFO] Loading datasets...")
    train_texts, train_labels = load_csv_data(CONFIG["TRAIN_PATH"])
    val_texts, val_labels = load_csv_data(CONFIG["VAL_PATH"])
    test_texts, test_labels = load_csv_data(CONFIG["TEST_PATH"])

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    print(f"  Test samples: {len(test_texts)}")

    # Create datasets and dataloaders
    train_dataset = HFACSDataset(train_texts, train_labels, tokenizer, CONFIG["MAX_LENGTH"])
    val_dataset = HFACSDataset(val_texts, val_labels, tokenizer, CONFIG["MAX_LENGTH"])
    test_dataset = HFACSDataset(test_texts, test_labels, tokenizer, CONFIG["MAX_LENGTH"])

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2
    )

    # Initialize model
    model = HFACSClassifier(
        checkpoint=CONFIG["BASE_CHECKPOINT"],
        n_classes=len(CONFIG["HFACS_CLASSES"])
    )
    model = model.to(CONFIG["DEVICE"])

    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
    )
    total_steps = len(train_loader) * CONFIG["EPOCHS"]
    warmup_steps = int(total_steps * CONFIG["WARMUP_RATIO"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = -1.0  # allow saving even when F1 is 0 (tiny validation sets)
    best_model_path = output_dir / f"best_model_{CONFIG['BASE_CHECKPOINT'].replace('/', '_')}.pt"
    for epoch in range(CONFIG["EPOCHS"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['EPOCHS']}")
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, CONFIG["DEVICE"]
        )
        val_loss, val_metrics = eval_model(
            model, val_loader, loss_fn, CONFIG["DEVICE"]
        )

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | F1 (macro): {val_metrics['f1_macro']:.4f}")

        # Save best model (use > so F1=0 still saves on first epoch)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  ** New best model saved! (F1 macro: {best_val_f1:.4f})")

    # If no model was saved (e.g. empty validation set), save the current model
    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)
        print("[INFO] Saved current model (no better model found during training)")

    # Load best model and evaluate on test set
    print("\n[INFO] Loading best model for final evaluation...")
    model.load_state_dict(
        torch.load(best_model_path, map_location=CONFIG["DEVICE"])
    )
    test_loss, test_metrics = eval_model(
        model, test_loader, loss_fn, CONFIG["DEVICE"]
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"  Loss:         {test_metrics['loss']:.4f}")
    print(f"  Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Precision:    {test_metrics['precision']:.4f}")
    print(f"  Recall:       {test_metrics['recall']:.4f}")
    print(f"  F1 (macro):   {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (micro):   {test_metrics['f1_micro']:.4f}")
    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    print("\nPer-Class F1 Scores:")
    for class_name, scores in test_metrics["per_class"].items():
        print(f"  {class_name:<20}: {scores['f1']:.4f}")

    # Save results to JSON
    results = {
        "config": CONFIG,
        "test_metrics": test_metrics,
    }
    with open(output_dir / "transformer_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[INFO] Results saved to {output_dir / 'transformer_results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()