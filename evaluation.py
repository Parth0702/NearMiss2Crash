#!/usr/bin/env python3
"""
===============================================================================
Step 0: Comprehensive Evaluation Framework (evaluation.py)
===============================================================================
Objective:
  Rigorous in-domain vs. cross-domain evaluation between ASRS and NTSB.
  
Key Methodological Corrections:
  1. Real ASRS Data: Ingests all 38,655 rows from elihoole/asrs-aviation-reports
     weakly labeled with KeywordWeakLabeler (saved to asrs_full.csv).
  2. Held-Out In-Domain Split: Eliminates 1.0000 memorization scores by using
     a stratified 80/20 train/test split (80% train, 20% held-out test).
  3. Consistent Labeling: Uses the identical KeywordWeakLabeler for both ASRS
     training and NTSB testing ground truth.
  4. Sensitivity Analysis: Separately compares against ntsb_processed_test.csv's
     preprocessor labels.
  5. Unlabeled Handling: Reports (a) labeled-only (1,413 rows) and (b) all 2,000
     NTSB rows with Unlabeled penalized as wrong.
  6. Per-Model Table: Prints held-out in-domain Macro-F1 (95% CI), NTSB Macro-F1
     (95% CI), absolute drop, and relative drop.
  7. Limitations Block: Explicitly documents keyword-derived label constraints.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, classification_report

from keyword_weak_labeler import KeywordWeakLabeler, SEED_KEYWORDS

HFACS_CLASSES = [
    "Decision Error",
    "Skill-Based Error",
    "Violation",
    "Perceptual Error"
]


def bootstrap_f1(y_true, y_pred, n: int = 1000, seed: int = 42) -> Tuple[float, np.ndarray]:
    """
    Computes bootstrap confidence intervals for macro-F1 score.
    
    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        n: Number of bootstrap iterations (default: 1000).
        seed: Random seed for reproducibility (default: 42).
        
    Returns:
        Tuple of (mean_macro_f1, np.array([2.5th_percentile, 97.5th_percentile])).
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return float(np.mean(scores)), np.percentile(scores, [2.5, 97.5])


def get_unified_keyword_labeler() -> KeywordWeakLabeler:
    """Returns KeywordWeakLabeler populated with all canonical seed keywords."""
    labeler = KeywordWeakLabeler()
    for cls, kws in SEED_KEYWORDS.items():
        for kw in kws:
            labeler.validated_keywords[kw] = cls
    return labeler


def ensure_asrs_full(csv_path: str = "asrs_full.csv") -> pd.DataFrame:
    """
    Loads asrs_full.csv if available. If missing, downloads elihoole/asrs-aviation-reports
    from HuggingFace, labels it with KeywordWeakLabeler, and persists it.
    """
    if os.path.exists(csv_path):
        print(f"[INFO] Ingesting existing real ASRS dataset from '{csv_path}'...")
        return pd.read_csv(csv_path)

    print(f"[INFO] '{csv_path}' not found. Fetching elihoole/asrs-aviation-reports from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("elihoole/asrs-aviation-reports", split="train")
    
    report_ids, texts = [], []
    for row in ds:
        rid = str(row.get("acn_num_ACN", ""))
        n1 = str(row.get("Report 1_Narrative", "") or "").strip()
        n2 = str(row.get("Report 2_Narrative", "") or "").strip()
        texts.append((n1 + " " + n2).strip() if n2 else n1)
        report_ids.append(rid)

    df_raw = pd.DataFrame({"report_id": report_ids, "source": "ASRS", "text": texts})
    
    labeler = get_unified_keyword_labeler()
    print(f"[INFO] Weak-labeling {len(df_raw)} records with KeywordWeakLabeler...")
    df_labeled = labeler.transform(df_raw, text_column="text")
    df_labeled["label"] = df_labeled["weak_label"]
    df_labeled.to_csv(csv_path, index=False)
    print(f"[INFO] Successfully saved {csv_path} ({os.path.getsize(csv_path)/1e6:.2f} MB)")
    return df_labeled


def run_evaluation_pipeline():
    print("=" * 82)
    print(" STEP 0: RIGOROUS CROSS-DOMAIN EVALUATION PIPELINE (ASRS -> NTSB)")
    print("=" * 82)

    # 1. Real ASRS Dataset
    df_asrs = ensure_asrs_full("asrs_full.csv")
    total_asrs = len(df_asrs)
    asrs_counts = df_asrs["label"].value_counts()
    n_asrs_unlabeled = asrs_counts.get("Unlabeled", 0)
    
    print("\n--- 1. Real ASRS Dataset Distribution (asrs_full.csv) ---")
    print(f"Total Reports Ingested : {total_asrs:,}")
    print(f"Unlabeled Reports      : {n_asrs_unlabeled:,} ({n_asrs_unlabeled/total_asrs*100:.2f}%)")
    print("Labeled Reports by HFACS Class:")
    for cls in HFACS_CLASSES:
        cnt = asrs_counts.get(cls, 0)
        print(f"  - {cls:20s}: {cnt:6,d} ({cnt/total_asrs*100:5.2f}%)")

    # Filter to labeled ASRS rows for 4-class training
    df_asrs_labeled = df_asrs[df_asrs["label"] != "Unlabeled"].copy()
    
    # Stratified 80/20 train/held-out test split
    train_df, held_out_test_df = train_test_split(
        df_asrs_labeled,
        test_size=0.20,
        random_state=42,
        stratify=df_asrs_labeled["label"]
    )
    print(f"\n[INFO] ASRS 80/20 Stratified Split:")
    print(f"       - Training Split  : {len(train_df):,d} samples")
    print(f"       - Held-Out Test   : {len(held_out_test_df):,d} samples")

    # 2. Real NTSB Test Benchmark & Weak Labeling
    ntsb_benchmark_path = "ntsb_processed_test.csv"
    if not os.path.exists(ntsb_benchmark_path):
        raise FileNotFoundError(f"Missing {ntsb_benchmark_path}")
    
    df_ntsb = pd.read_csv(ntsb_benchmark_path)
    total_ntsb = len(df_ntsb)
    
    # Label NTSB with the SAME weak labeler
    labeler = get_unified_keyword_labeler()
    df_ntsb_weak = labeler.transform(df_ntsb, text_column="text")
    ntsb_weak_counts = df_ntsb_weak["weak_label"].value_counts()
    n_ntsb_unlabeled = ntsb_weak_counts.get("Unlabeled", 0)
    n_ntsb_labeled = total_ntsb - n_ntsb_unlabeled

    print("\n--- 2. NTSB Benchmark Weak-Label Distribution (Same KeywordWeakLabeler) ---")
    print(f"Total NTSB Reports     : {total_ntsb:,}")
    print(f"Unlabeled Dropped      : {n_ntsb_unlabeled:,} ({n_ntsb_unlabeled/total_ntsb*100:.2f}%)")
    print(f"Retained for Eval      : {n_ntsb_labeled:,} ({n_ntsb_labeled/total_ntsb*100:.2f}%)")
    print("Class Counts (Consistent Weak Labels):")
    for cls in HFACS_CLASSES:
        cnt = ntsb_weak_counts.get(cls, 0)
        print(f"  - {cls:20s}: {cnt:4,d} ({cnt/total_ntsb*100:5.2f}%)")

    # 3. Vectorization (Fitted ONLY on ASRS training split)
    print("\n[INFO] Fitting TF-IDF Vectorizer (max_features=10,000, ngram_range=(1,2)) on ASRS Train...")
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train = vec.fit_transform(train_df["text"].fillna(""))
    y_train = train_df["label"].values

    # Held-out in-domain test features
    X_test_asrs = vec.transform(held_out_test_df["text"].fillna(""))
    y_test_asrs = held_out_test_df["label"].values

    # Cross-domain NTSB features (labeled-only)
    df_ntsb_labeled = df_ntsb_weak[df_ntsb_weak["weak_label"] != "Unlabeled"].copy()
    X_test_ntsb = vec.transform(df_ntsb_labeled["clean_text"].fillna(""))
    y_test_ntsb = df_ntsb_labeled["weak_label"].values

    # Cross-domain NTSB features (all 2,000 rows for Protocol B)
    X_test_ntsb_all = vec.transform(df_ntsb_weak["clean_text"].fillna(""))
    y_test_ntsb_all = df_ntsb_weak["weak_label"].values

    # Cross-domain NTSB features (preprocessor ground truth for sensitivity check)
    y_test_ntsb_preproc = df_ntsb["label"].values

    # 4. Train and Evaluate Classical Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        "Naive Bayes": MultinomialNB()
    }

    results_table = []
    sensitivity_table = []
    protocol_b_table = []

    print("\n--- 3. Model Training and Comprehensive Evaluation ---")
    for name, clf in models.items():
        print(f"[TRAINING] {name} on {len(train_df):,d} ASRS examples...")
        clf.fit(X_train, y_train)

        # In-domain held-out evaluation
        pred_asrs = clf.predict(X_test_asrs)
        macro_asrs = f1_score(y_test_asrs, pred_asrs, average="macro", zero_division=0)
        b_asrs_mean, b_asrs_ci = bootstrap_f1(y_test_asrs, pred_asrs)

        # Cross-domain NTSB evaluation (Protocol A: Labeled-Only, N=1,413)
        pred_ntsb = clf.predict(X_test_ntsb)
        macro_ntsb = f1_score(y_test_ntsb, pred_ntsb, average="macro", zero_division=0)
        b_ntsb_mean, b_ntsb_ci = bootstrap_f1(y_test_ntsb, pred_ntsb)

        abs_drop = macro_ntsb - macro_asrs
        rel_drop = (abs_drop / macro_asrs) * 100 if macro_asrs > 0 else 0.0

        results_table.append({
            "model": name,
            "asrs_f1": macro_asrs,
            "asrs_ci": b_asrs_ci,
            "ntsb_f1": macro_ntsb,
            "ntsb_ci": b_ntsb_ci,
            "abs_drop": abs_drop,
            "rel_drop": rel_drop
        })

        # Protocol B: All 2,000 NTSB rows (Unlabeled counted as wrong / penalized)
        pred_ntsb_all = clf.predict(X_test_ntsb_all)
        macro_penalized_4cls = f1_score(
            y_test_ntsb_all, pred_ntsb_all, labels=HFACS_CLASSES, average="macro", zero_division=0
        )
        b_all_mean, b_all_ci = bootstrap_f1(y_test_ntsb_all, pred_ntsb_all)
        
        protocol_b_table.append({
            "model": name,
            "macro_4cls_penalized": macro_penalized_4cls,
            "macro_5cls_all": b_all_mean,
            "ci_5cls_all": b_all_ci
        })

        # Sensitivity Check: Evaluated against ntsb_processed_test.csv original preprocessor labels
        macro_preproc = f1_score(y_test_ntsb_preproc, pred_ntsb_all, average="macro", zero_division=0)
        b_preproc_mean, b_preproc_ci = bootstrap_f1(y_test_ntsb_preproc, pred_ntsb_all)
        
        sensitivity_table.append({
            "model": name,
            "preproc_macro": macro_preproc,
            "preproc_ci": b_preproc_ci,
            "consistent_macro": macro_ntsb
        })

    # 5. Print Master Comparison Table (Protocol A: Headline Labeled-Only)
    print("\n" + "=" * 96)
    print(" 4. MASTER PER-MODEL TABLE: HELD-OUT IN-DOMAIN vs. CROSS-DOMAIN NTSB (SAME LABELER)")
    print("=" * 96)
    print(f"{'Model':22s} | {'Held-Out ASRS (95% CI)':26s} | {'NTSB Consistent (95% CI)':26s} | {'Abs Drop':10s} | {'Rel Drop':10s}")
    print("-" * 104)
    for r in results_table:
        asrs_str = f"{r['asrs_f1']:.4f} [{r['asrs_ci'][0]:.4f}, {r['asrs_ci'][1]:.4f}]"
        ntsb_str = f"{r['ntsb_f1']:.4f} [{r['ntsb_ci'][0]:.4f}, {r['ntsb_ci'][1]:.4f}]"
        abs_str = f"{r['abs_drop']:+.4f}"
        rel_str = f"{r['rel_drop']:+.2f}%"
        print(f"{r['model']:22s} | {asrs_str:26s} | {ntsb_str:26s} | {abs_str:10s} | {rel_str:10s}")

    # 6. Print Protocol B Table (All 2,000 Rows, Unlabeled Counted as Wrong)
    print("\n" + "=" * 96)
    print(" 5. PROTOCOL B: ALL 2,000 NTSB ROWS (UNLABELED ROWS COUNTED AS WRONG)")
    print("=" * 96)
    print(f"{'Model':22s} | {'4-Class Macro-F1 (FP Penalized)':32s} | {'5-Class Macro-F1 (Unlabeled F1=0)':34s}")
    print("-" * 96)
    for r in protocol_b_table:
        all_str = f"{r['macro_5cls_all']:.4f} [{r['ci_5cls_all'][0]:.4f}, {r['ci_5cls_all'][1]:.4f}]"
        print(f"{r['model']:22s} | {r['macro_4cls_penalized']:32.4f} | {all_str:34s}")

    # 7. Print Sensitivity Analysis Table (Disparate Preprocessor Labels)
    print("\n" + "=" * 96)
    print(" 6. SENSITIVITY CHECK: TESTED AGAINST PREPROCESSOR LABELS (DISPARATE PROCEDURES)")
    print("=" * 96)
    print(f"{'Model':22s} | {'Consistent NTSB Macro':23s} | {'Preprocessor NTSB Macro (95% CI)':36s} | {'Procedure Gap':15s}")
    print("-" * 96)
    for r in sensitivity_table:
        pre_str = f"{r['preproc_macro']:.4f} [{r['preproc_ci'][0]:.4f}, {r['preproc_ci'][1]:.4f}]"
        gap = r['consistent_macro'] - r['preproc_macro']
        print(f"{r['model']:22s} | {r['consistent_macro']:23.4f} | {pre_str:36s} | {gap:+15.4f}")

    # 8. Limitations Block
    print("\n" + "=" * 96)
    print(" LIMITATIONS & METHODOLOGICAL CONSTRAINTS")
    print("=" * 96)
    print("""  - Keyword-Derived Labels: All ground-truth annotations are generated via rule-based
    keyword matching (Madeira et al., 2021), not human expert consensus. Metrics reflect
    agreement with keyword heuristics rather than clinical psychological ground truth.
  - Lexical Inductive Bias: In-domain held-out scores benefit from lexical overlap between
    TF-IDF vocabulary and the deterministic seed keywords used for labeling.
  - Class Imbalance: In real ASRS data, Decision Error accounts for ~81% of labeled reports,
    while Violation and Perceptual Error represent <4% each, depressing Macro-F1 averages.
  - Unlabeled Discard: Dropping 587 (29.35%) Unlabeled NTSB samples in Protocol A selects for
    higher-confidence narratives; Protocol B provides the uncurated, penalty-inclusive bound.
""")


if __name__ == "__main__":
    run_evaluation_pipeline()
