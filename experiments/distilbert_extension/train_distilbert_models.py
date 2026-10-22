"""
=================================================================================
Phase 3B: Train DistilBERT Models (Crisp vs Fuzzy)
=================================================================================
Trains two DistilBERT classifiers:
- Model A: Crisp weak labels
- Model B: Fuzzy weak labels
=================================================================================
"""

import sys
import os
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.distilbert_extension.prepare_fuzzy_splits import HFACS_CLASSES

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "distilbert_extension"
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Training config
CONFIG = {
    "BASE_CHECKPOINT": "distilbert-base-uncased",
    "MAX_LENGTH": 128,
    "BATCH_SIZE": 16,
    "EPOCHS": 3,
    "LEARNING_RATE": 2e-5,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.1,
    "RANDOM_SEED": 42,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "HFACS_CLASSES": HFACS_CLASSES,
}

def load_data(csv_path, label_col='label'):
    """Load text and labels from CSV."""
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()

    label_to_idx = {c: i for i, c in enumerate(CONFIG["HFACS_CLASSES"])}
    labels = df[label_col].tolist()
    y_indices = np.array([label_to_idx[lbl] for lbl in labels if lbl in label_to_idx])

    # Filter texts to match labels
    valid_texts = [texts[i] for i, lbl in enumerate(labels) if lbl in label_to_idx]

    return valid_texts, y_indices

def get_embeddings(text_list, tokenizer, base_model, device, batch_size=64):
    """Extract DistilBERT CLS embeddings."""
    all_emb = []
    base_model.eval()
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            b_texts = text_list[i:i+batch_size]
            inputs = tokenizer(
                b_texts, padding=True, truncation=True,
                max_length=CONFIG["MAX_LENGTH"], return_tensors="pt"
            ).to(device)
            outputs = base_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            cls_rep = outputs.last_hidden_state[:, 0, :]
            all_emb.append(cls_rep.cpu())
    return torch.cat(all_emb, dim=0)

def train_model(model_name, train_csv, val_csv, test_csv, label_col, epochs=None):
    """Train a DistilBERT classification head on frozen embeddings."""
    epochs = epochs or CONFIG["EPOCHS"]
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*80}")

    device = CONFIG["DEVICE"]
    print(f"[INFO] Device: {device}")

    # Load data
    print(f"[INFO] Loading data from {train_csv}...")
    train_texts, y_train = load_data(train_csv, label_col)
    val_texts, y_val = load_data(val_csv, label_col)
    test_texts, y_test = load_data(test_csv, label_col)

    print(f"[INFO] Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # Load tokenizer and base model
    print("[INFO] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["BASE_CHECKPOINT"])
    base_model = AutoModel.from_pretrained(CONFIG["BASE_CHECKPOINT"])
    base_model.to(device)
    base_model.eval()

    # Extract embeddings
    print("[INFO] Extracting embeddings...")
    t0 = time.time()
    X_train_emb = get_embeddings(train_texts, tokenizer, base_model, device)
    X_val_emb = get_embeddings(val_texts, tokenizer, base_model, device)
    X_test_emb = get_embeddings(test_texts, tokenizer, base_model, device)
    print(f"[INFO] Embeddings extracted in {time.time() - t0:.1f}s")
    print(f"  Train: {X_train_emb.shape} | Val: {X_val_emb.shape} | Test: {X_test_emb.shape}")

    # Free base model
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create data loaders
    train_dataset = TensorDataset(X_train_emb, torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

    val_dataset = TensorDataset(X_val_emb, torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    test_dataset = TensorDataset(X_test_emb, torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    # Classification head
    head = nn.Linear(768, len(CONFIG["HFACS_CLASSES"])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_f1 = 0.0
    best_weights = None
    best_bias = None
    history = {"train_loss": [], "val_f1": [], "val_acc": []}

    print(f"[INFO] Training classification head for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # Train
        head.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validate
        head.eval()
        with torch.no_grad():
            val_logits = head(X_val_emb.to(device))
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val, val_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_f1"].append(val_macro_f1)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_weights = head.weight.data.clone()
            best_bias = head.bias.data.clone()

    # Restore best weights
    head.weight.data = best_weights
    head.bias.data = best_bias

    # Evaluate on test set
    head.eval()
    with torch.no_grad():
        test_logits = head(X_test_emb.to(device))
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
    test_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)

    print(f"\n[RESULTS] {model_name} Test Performance:")
    print(f"  Accuracy:      {test_acc:.4f}")
    print(f"  Macro-F1:      {test_macro_f1:.4f}")
    print(f"  Weighted-F1:   {test_weighted_f1:.4f}")
    print(f"  Precision:     {test_prec:.4f}")
    print(f"  Recall:        {test_rec:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=CONFIG["HFACS_CLASSES"], digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Save model
    model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_head.pt"
    torch.save({
        'head_weight': best_weights,
        'head_bias': best_bias,
        'config': CONFIG,
        'history': history,
        'test_metrics': {
            'accuracy': test_acc,
            'macro_f1': test_macro_f1,
            'weighted_f1': test_weighted_f1,
            'precision': test_prec,
            'recall': test_rec
        }
    }, model_path)
    print(f"[INFO] Model saved to {model_path}")

    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': [CONFIG["HFACS_CLASSES"][i] for i in y_test],
        'pred_label': [CONFIG["HFACS_CLASSES"][i] for i in test_preds],
        'correct': y_test == test_preds
    })
    pred_path = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[INFO] Predictions saved to {pred_path}")

    return {
        'model_name': model_name,
        'model_path': str(model_path),
        'predictions_path': str(pred_path),
        'history': history,
        'test_metrics': {
            'accuracy': test_acc,
            'macro_f1': test_macro_f1,
            'weighted_f1': test_weighted_f1,
            'precision': test_prec,
            'recall': test_rec
        },
        'confusion_matrix': cm.tolist(),
        'test_samples': len(y_test),
        'val_samples': len(y_val),
        'train_samples': len(y_train)
    }

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 80)
    print("PHASE 3B: DISTILBERT CRISP vs FUZZY TRAINING")
    print("=" * 80)

    # Train Model A: Crisp
    print("\n>>> MODEL A: CRISP TRAINING")
    crisp_results = train_model(
        model_name="Model_A_Crisp",
        train_csv=str(PROJECT_ROOT / "weak_labeling_results" / "dataset_majority_train.csv"),
        val_csv=str(PROJECT_ROOT / "weak_labeling_results" / "dataset_majority_validation.csv"),
        test_csv=str(PROJECT_ROOT / "weak_labeling_results" / "dataset_majority_test.csv"),
        label_col='label',
        epochs=3
    )

    # Train Model B: Fuzzy
    print("\n>>> MODEL B: FUZZY TRAINING")
    fuzzy_results = train_model(
        model_name="Model_B_Fuzzy",
        train_csv=str(OUTPUT_DIR / "dataset_fuzzy_train_labeled.csv"),
        val_csv=str(OUTPUT_DIR / "dataset_fuzzy_validation_labeled.csv"),
        test_csv=str(OUTPUT_DIR / "dataset_fuzzy_test_labeled.csv"),
        label_col='fuzzy_label',
        epochs=3
    )

    # Save combined results
    combined_results = {
        "model_a_crisp": crisp_results,
        "model_b_fuzzy": fuzzy_results,
        "config": CONFIG
    }

    with open(OUTPUT_DIR / "distilbert_comparison_metrics.json", 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    # Per-class metrics CSV
    per_class_rows = []
    for model_name, results in [("Crisp", crisp_results), ("Fuzzy", fuzzy_results)]:
        # We'd need per-class from classification_report - simplified here
        pass

    print("\n" + "=" * 80)
    print("PHASE 3B TRAINING COMPLETE")
    print("=" * 80)

    # Summary comparison
    print("\n--- COMPARISON SUMMARY ---")
    for name, results in [("Crisp (Model A)", crisp_results), ("Fuzzy (Model B)", fuzzy_results)]:
        m = results['test_metrics']
        print(f"{name:20s}: Acc={m['accuracy']:.4f} | Macro-F1={m['macro_f1']:.4f} | Weighted-F1={m['weighted_f1']:.4f} | Train={results['train_samples']} | Test={results['test_samples']}")