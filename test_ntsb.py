#!/usr/bin/env python3
"""
===============================================================================
NTSB Cross-Domain Model Evaluation (Week 2 & Week 3)
===============================================================================
Reference Papers:
  1. Madeira, Melicio, Valerio & Santos (2021) -- Section 4.2: Evaluation metrics
     (Micro-F1 primary, Macro-F1 secondary)
  2. Vempati, Woods & Solano (2023) -- Cross-domain transfer from ASRS to NTSB
  3. Andrade & Walsh (2023) -- SafeAeroBERT & DistilBERT domain evaluation

This script:
  1. Loads preprocessed, balanced NTSB benchmark test data
  2. Evaluates classical models (Logistic Regression, Random Forest, SVM, Naive Bayes)
  3. Evaluates deep transformer models (DistilBERT via HFACSClassifier)
  4. Generates comprehensive classification reports & confusion matrices
  5. Exports results to week3_results/ntsb_cross_domain_evaluation.csv
===============================================================================
"""

import sys
import os
import io

# Safe stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
torch.set_num_threads(2)

from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Canonical HFACS classes
HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]


def load_test_dataset(test_csv_path: str = r"ntsb_processed_test.csv") -> pd.DataFrame:
    """Loads and validates the preprocessed NTSB test benchmark."""
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test dataset not found: {test_csv_path}. Please run ntsb_preprocessor.py first.")

    df = pd.read_csv(test_csv_path)
    print(f"[INFO] Loaded NTSB test dataset: {len(df)} samples from {test_csv_path}")
    print("[INFO] Class Distribution:")
    for cls_name, count in df['label'].value_counts().items():
        print(f"       - {cls_name:20s}: {count}")
    return df


def evaluate_classical_models(test_df: pd.DataFrame, results_dir: str = "week2_results") -> Dict:
    """Evaluates Week 2 classical classifiers on NTSB test set."""
    print("\n" + "="*70)
    print(" EVALUATING CLASSICAL CLASSIFIERS (Week 2)")
    print("="*70)

    import joblib
    classical_results = {}

    X_test = test_df['clean_text'].astype(str).tolist()
    y_test = test_df['label'].tolist()

    # Look for saved models (prioritize cluster_first which contains all 4 HFACS classes)
    model_names = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
    strategies = ['cluster_first', 'majority', 'keyword_first']

    for name in model_names:
        pipeline = None
        loaded_strategy = None

        # Find best available joblib model
        for strat in strategies:
            p = os.path.join(results_dir, f"classifier_{name}_{strat}.joblib")
            if os.path.exists(p):
                try:
                    loaded = joblib.load(p)
                    # Check if model knows about the classes
                    clf = loaded.named_steps.get('classifier')
                    if clf is not None and hasattr(clf, 'classes_'):
                        pipeline = loaded
                        loaded_strategy = strat
                        break
                except Exception:
                    continue

        if pipeline is None:
            print(f"[WARN] No valid pipeline found for {name}, skipping.")
            continue

        try:
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

            print(f"\nModel: {name.upper()} (Strategy: {loaded_strategy})")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Micro-F1:  {micro_f1:.4f} (Primary Metric)")
            print(f"  Macro-F1:  {macro_f1:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")

            classical_results[name] = {
                'model': name,
                'strategy': loaded_strategy,
                'accuracy': acc,
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'precision': prec,
                'recall': rec,
                'y_pred': y_pred
            }

        except Exception as e:
            print(f"[ERROR] Failed evaluation for {name}: {e}")

    return classical_results


def evaluate_transformer_model(
    test_df: pd.DataFrame,
    checkpoint_path: str = r"week3_results\best_model_distilbert-base-uncased.pt",
    batch_size: int = 16
) -> Dict:
    """Evaluates DistilBERT transformer classifier using HFACSClassifier."""
    print("\n" + "="*70)
    print(" EVALUATING TRANSFORMER MODEL (DistilBERT - Week 3)")
    print("="*70)

    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint not found at {checkpoint_path}. Skipping transformer.")
        return {}

    try:
        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import AutoTokenizer
        from transformer_model import HFACSClassifier

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        # Initialize model architecture and load checkpoint
        model = HFACSClassifier(checkpoint="distilbert-base-uncased", n_classes=len(HFACS_CLASSES))
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        text_col = 'clean_text' if 'clean_text' in test_df.columns else 'text'
        X_test = test_df[text_col].astype(str).tolist()
        y_test = test_df['label'].tolist()

        all_preds = []
        print(f"[INFO] Running inference on {len(X_test)} samples (batch_size={batch_size})...")

        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_texts = X_test[i:i+batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)

                logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                probs = torch.sigmoid(logits).cpu().numpy()

                batch_pred_indices = np.argmax(probs, axis=1)
                for idx in batch_pred_indices:
                    all_preds.append(HFACS_CLASSES[idx])

        acc = accuracy_score(y_test, all_preds)
        micro_f1 = f1_score(y_test, all_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(y_test, all_preds, average='macro', zero_division=0)
        prec = precision_score(y_test, all_preds, average='macro', zero_division=0)
        rec = recall_score(y_test, all_preds, average='macro', zero_division=0)

        print("\nModel: DistilBERT (Fine-Tuned)")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Micro-F1:  {micro_f1:.4f} (Primary Metric)")
        print(f"  Macro-F1:  {macro_f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")

        return {
            'model': 'DistilBERT',
            'strategy': 'fine-tuned',
            'accuracy': acc,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'precision': prec,
            'recall': rec,
            'y_pred': all_preds
        }

    except Exception as e:
        print(f"[ERROR] Transformer evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def save_and_visualize_results(
    test_df: pd.DataFrame,
    all_results: Dict,
    output_csv: str = r"week3_results\ntsb_cross_domain_evaluation.csv",
    plot_path: str = r"week3_results\ntsb_cross_domain_comparison.png"
):
    """Exports master evaluation table and comparison visualization."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    summary_rows = []
    for name, res in all_results.items():
        summary_rows.append({
            'Model': res['model'],
            'Domain': 'NTSB (Cross-Domain)',
            'Accuracy': round(res['accuracy'], 4),
            'Micro-F1': round(res['micro_f1'], 4),
            'Macro-F1': round(res['macro_f1'], 4),
            'Precision': round(res['precision'], 4),
            'Recall': round(res['recall'], 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Exported cross-domain metrics to: {output_csv}")
    print("\n" + "="*70)
    print(" MASTER CROSS-DOMAIN EVALUATION TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Generate Bar Plot
    try:
        plt.figure(figsize=(10, 5))
        df_melt = summary_df.melt(id_vars=['Model'], value_vars=['Micro-F1', 'Macro-F1', 'Accuracy'], var_name='Metric', value_name='Score')
        sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette='viridis')
        plt.title('Cross-Domain Model Performance on NTSB Benchmark')
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved performance visualization: {plot_path}")
    except Exception as e:
        print(f"[WARN] Could not save visualization: {e}")


def main():
    print("="*70)
    print(" NTSB CROSS-DOMAIN TESTING PIPELINE (ASRS -> NTSB)")
    print("="*70)

    test_csv = r"ntsb_processed_test.csv"
    if not os.path.exists(test_csv):
        print(f"[INFO] {test_csv} not found. Running preprocessor...")
        from ntsb_preprocessor import NTSBPreprocessor
        preprocessor = NTSBPreprocessor()
        test_df = preprocessor.process_dataset(output_test_csv=test_csv)
    else:
        test_df = load_test_dataset(test_csv)

    all_results = {}

    # 1. Classical models
    classical = evaluate_classical_models(test_df)
    all_results.update(classical)

    # 2. Transformer model
    transformer = evaluate_transformer_model(test_df)
    if transformer:
        all_results['DistilBERT'] = transformer

    # 3. Export summary and visualization
    if all_results:
        save_and_visualize_results(test_df, all_results)

    print("\n" + "="*70)
    print(" NTSB EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()