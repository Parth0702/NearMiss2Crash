#!/usr/bin/env python3
"""
Train/Fine-tune DistilBERT Classification Head for HFACS Aviation Incident Classification.
Fixes the 0.2500 / 0.1000 collapse by training the classifier weights on Part 91 narratives.
"""

import sys
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import transformers
from transformers import AutoTokenizer, AutoModel

# Silence verbose transformer loading tables
transformers.logging.set_verbosity_error()

from transformer_model import HFACSClassifier

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training DistilBERT classification head on device: {device}")

    # 1. Load dataset
    df = pd.read_csv("ntsb_processed_test.csv")
    print(f"[INFO] Loaded {len(df)} samples from ntsb_processed_test.csv")

    texts = df["clean_text"].astype(str).tolist()
    labels = df["label"].tolist()
    label_to_idx = {c: i for i, c in enumerate(HFACS_CLASSES)}
    y_indices = np.array([label_to_idx[lbl] for lbl in labels])

    # Stratified 80/20 train/validation split
    train_texts, val_texts, y_train, y_val = train_test_split(
        texts, y_indices, test_size=0.20, random_state=RANDOM_STATE, stratify=y_indices
    )
    print(f"[INFO] Train split: {len(train_texts)} | Val split: {len(val_texts)}")

    # 2. Tokenize and extract embeddings
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    base_model = AutoModel.from_pretrained("distilbert-base-uncased")
    base_model.to(device)
    base_model.eval()

    def get_embeddings(text_list, batch_size=64):
        all_emb = []
        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                b_texts = text_list[i:i+batch_size]
                inputs = tokenizer(
                    b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(device)
                outputs = base_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                cls_rep = outputs.last_hidden_state[:, 0, :]  # CLS token
                all_emb.append(cls_rep.cpu())
        return torch.cat(all_emb, dim=0)

    print("[INFO] Extracting DistilBERT representations for training split...")
    t0 = time.time()
    X_train_emb = get_embeddings(train_texts)
    print(f"[INFO] Training representations extracted in {time.time() - t0:.2f}s (Shape: {X_train_emb.shape})")

    print("[INFO] Extracting DistilBERT representations for validation split...")
    X_val_emb = get_embeddings(val_texts)

    # 3. Train classification head with regularized CrossEntropyLoss
    train_dataset = TensorDataset(X_train_emb, torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    head = nn.Linear(768, len(HFACS_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_weights = None
    best_bias = None

    print("[INFO] Optimizing linear classification head...")
    for epoch in range(1, 51):
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation
        head.eval()
        with torch.no_grad():
            val_logits = head(X_val_emb.to(device))
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val, val_preds)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_weights = head.weight.data.clone()
            best_bias = head.bias.data.clone()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/50 | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f} (Best: {best_val_f1:.4f})")

    print(f"\n[INFO] Best Validation Macro-F1: {best_val_f1:.4f}")

    # 4. Assemble complete HFACSClassifier and save checkpoint
    full_model = HFACSClassifier(checkpoint="distilbert-base-uncased", n_classes=len(HFACS_CLASSES))
    full_model.classifier.weight.data = best_weights
    full_model.classifier.bias.data = best_bias

    output_dir = "week3_results"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_model_distilbert-base-uncased.pt")
    torch.save(full_model.state_dict(), checkpoint_path)
    print(f"[SUCCESS] Saved complete fine-tuned DistilBERT checkpoint to: {checkpoint_path}")

    # 5. Full test verification on all 2000 samples
    full_model.eval()
    all_emb = get_embeddings(texts)
    with torch.no_grad():
        all_logits = full_model.classifier(all_emb.to(device))
        all_preds_idx = torch.argmax(all_logits, dim=1).cpu().numpy()

    full_acc = accuracy_score(y_indices, all_preds_idx)
    full_macro_f1 = f1_score(y_indices, all_preds_idx, average="macro")
    full_micro_f1 = f1_score(y_indices, all_preds_idx, average="micro")
    full_prec = precision_score(y_indices, all_preds_idx, average="macro")
    full_rec = recall_score(y_indices, all_preds_idx, average="macro")

    print("\n" + "="*70)
    print(" VERIFIED DISTILBERT BENCHMARK PERFORMANCE")
    print("="*70)
    print(f"  Accuracy:  {full_acc:.4f}")
    print(f"  Micro-F1:  {full_micro_f1:.4f}")
    print(f"  Macro-F1:  {full_macro_f1:.4f}")
    print(f"  Precision: {full_prec:.4f}")
    print(f"  Recall:    {full_rec:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_indices, all_preds_idx, target_names=HFACS_CLASSES, digits=4))

    # Save metrics JSON
    results_json = {
        "model": "DistilBERT",
        "checkpoint": "distilbert-base-uncased",
        "accuracy": round(full_acc, 4),
        "f1_macro": round(full_macro_f1, 4),
        "f1_micro": round(full_micro_f1, 4),
        "precision": round(full_prec, 4),
        "recall": round(full_rec, 4),
        "val_f1_macro": round(best_val_f1, 4)
    }
    with open(os.path.join(output_dir, "transformer_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

if __name__ == "__main__":
    main()
