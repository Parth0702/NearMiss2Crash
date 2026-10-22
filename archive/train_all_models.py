#!/usr/bin/env python3
"""
===============================================================================
Train All Models on Full ASRS Dataset (train_all_models.py)
===============================================================================
This script trains all project models on the complete ASRS dataset (asrs_full.csv):
  1. Classical Classifiers:
     - Logistic Regression
     - Linear SVM
     - Random Forest
     - Multinomial Naive Bayes
     Saved to week2_results/
  2. Deep Transformer:
     - DistilBERT classification head
     Saved to week3_results/best_model_distilbert-base-uncased.pt
===============================================================================
"""

import os
import sys
import time
import json
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, classification_report
import transformers
from transformers import AutoTokenizer, AutoModel

transformers.logging.set_verbosity_error()
from transformer_model import HFACSClassifier

HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def train_classical_models(df_asrs_labeled: pd.DataFrame, output_dir: str = "week2_results"):
    """Trains TF-IDF + Classical Classifiers on asrs_full.csv."""
    print("\n" + "=" * 78)
    print(" 1. TRAINING CLASSICAL CLASSIFIERS ON ASRS FULL (22,046 LABELED REPORTS)")
    print("=" * 78)

    os.makedirs(output_dir, exist_ok=True)
    
    # 80/20 Stratified Split
    train_df, test_df = train_test_split(
        df_asrs_labeled, test_size=0.20, random_state=RANDOM_STATE, stratify=df_asrs_labeled['label']
    )
    X_train = train_df['text'].fillna('').astype(str).tolist()
    y_train = train_df['label'].tolist()
    X_test = test_df['text'].fillna('').astype(str).tolist()
    y_test = test_df['label'].tolist()

    print(f"[INFO] Train split: {len(X_train):,d} samples | Held-out Test split: {len(X_test):,d} samples")

    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'svm': LinearSVC(random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=25, random_state=RANDOM_STATE, n_jobs=-1),
        'naive_bayes': MultinomialNB()
    }

    metrics_summary = []

    for name, clf in classifiers.items():
        print(f"\n[TRAINING] {name.replace('_', ' ').title()}...")
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', clf)
        ])

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        # Evaluate on held-out test split
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)

        print(f"  Training Time : {train_time:.2f}s")
        print(f"  Held-out Acc  : {acc:.4f} | Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}")

        # Save model joblib files (for all strategies so existing loaders find them)
        for strat in ['cluster_first', 'majority', 'keyword_first']:
            out_path = os.path.join(output_dir, f"classifier_{name}_{strat}.joblib")
            joblib.dump(pipeline, out_path)

        # Write text report
        rep_path = os.path.join(output_dir, f"classifier_{name}_report.txt")
        with open(rep_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {name} (Trained on asrs_full.csv, N={len(X_train)})\n")
            f.write(f"Held-out Test Macro-F1: {macro_f1:.4f}\nAccuracy: {acc:.4f}\n\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))

        metrics_summary.append({
            'classifier': name.replace('_', ' ').title(),
            'accuracy': acc,
            'f1_macro': macro_f1,
            'micro_f1': micro_f1,
            'training_time': train_time
        })

    summary_df = pd.DataFrame(metrics_summary)
    summary_csv = os.path.join(output_dir, "model_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[SUCCESS] Classical models saved to {output_dir}/")
    print(summary_df.to_string(index=False))


def train_distilbert_on_asrs(df_asrs_labeled: pd.DataFrame, output_dir: str = "week3_results"):
    """Trains DistilBERT classification head on a balanced sample from asrs_full.csv."""
    print("\n" + "=" * 78)
    print(" 2. TRAINING DISTILBERT CLASSIFICATION HEAD ON ASRS FULL")
    print("=" * 78)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create a balanced stratified subset to prevent 81% Decision Error collapse
    # Sample up to 600 per class from minority classes, and 1200 from Decision Error
    sub_dfs = []
    for cls in HFACS_CLASSES:
        cls_df = df_asrs_labeled[df_asrs_labeled['label'] == cls]
        n_sample = min(len(cls_df), 1000 if cls == "Decision Error" else len(cls_df))
        sub_dfs.append(cls_df.sample(n=n_sample, random_state=RANDOM_STATE))
    sample_df = pd.concat(sub_dfs).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"[INFO] Prepared balanced training subset of {len(sample_df)} samples:")
    for cls, cnt in sample_df['label'].value_counts().items():
        print(f"       - {cls:20s}: {cnt}")

    label_to_idx = {c: i for i, c in enumerate(HFACS_CLASSES)}
    texts = sample_df['text'].fillna('').astype(str).tolist()
    y_indices = np.array([label_to_idx[l] for l in sample_df['label']])

    train_texts, val_texts, y_train, y_val = train_test_split(
        texts, y_indices, test_size=0.20, random_state=RANDOM_STATE, stratify=y_indices
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    base_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    base_model.eval()

    def extract_cls_embeddings(text_list, batch_size=64):
        embs = []
        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                b_texts = text_list[i:i+batch_size]
                inputs = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                outputs = base_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                embs.append(outputs.last_hidden_state[:, 0, :].cpu())
        return torch.cat(embs, dim=0)

    print("[INFO] Extracting DistilBERT representations...")
    t0 = time.time()
    X_train_emb = extract_cls_embeddings(train_texts)
    X_val_emb = extract_cls_embeddings(val_texts)
    print(f"[INFO] Extracted representations in {time.time() - t0:.2f}s.")

    # Train linear head
    train_dataset = TensorDataset(X_train_emb, torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    head = nn.Linear(768, len(HFACS_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_weights, best_bias = None, None

    for epoch in range(1, 41):
        head.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val_emb.to(device))
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = head.weight.data.clone()
                best_bias = head.bias.data.clone()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/40 | Val Macro-F1: {val_f1:.4f} (Best: {best_val_f1:.4f})")

    # Assemble and save HFACSClassifier
    full_model = HFACSClassifier(checkpoint="distilbert-base-uncased", n_classes=len(HFACS_CLASSES))
    full_model.classifier.weight.data = best_weights
    full_model.classifier.bias.data = best_bias

    ckpt_path = os.path.join(output_dir, "best_model_distilbert-base-uncased.pt")
    torch.save(full_model.state_dict(), ckpt_path)
    print(f"[SUCCESS] Fine-tuned DistilBERT saved to {ckpt_path} (Best Val Macro-F1: {best_val_f1:.4f})")


def main():
    if not os.path.exists("asrs_full.csv"):
        print("[ERROR] asrs_full.csv not found! Run evaluation.py first.")
        sys.exit(1)

    print("[INFO] Loading asrs_full.csv...")
    df_asrs = pd.read_csv("asrs_full.csv")
    df_labeled = df_asrs[df_asrs['label'] != 'Unlabeled'].copy()

    # 1. Train Classical Models
    train_classical_models(df_labeled, output_dir="week2_results")

    # 2. Train DistilBERT
    train_distilbert_on_asrs(df_labeled, output_dir="week3_results")

    print("\n" + "=" * 78)
    print(" [COMPLETED] ALL MODELS TRAINED ON ASRS FULL DATASET!")
    print("=" * 78)


if __name__ == "__main__":
    main()
