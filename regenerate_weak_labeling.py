#!/usr/bin/env python3
"""
===============================================================================
Regenerate Weak Labeling Results from Real ASRS Data (regenerate_weak_labeling.py)
===============================================================================
Replaces the legacy 10-row demo artifacts in weak_labeling_results/ with
comprehensive results generated from real ASRS aviation incident narratives.

Pipeline:
  1. Loads asrs_full.csv (38,655 real reports)
  2. Runs KeywordWeakLabeler.fit() to produce real keyword_audit_log.csv
  3. Clusters a stratified multi-class sample (5,000 reports) with BERT+UMAP+HDBSCAN
  4. Generates:
     - cluster_keywords.json, cluster_config.json
     - label_comparison.csv, agreement_stats.json
     - combined_labels_majority.csv, combined_labels_keyword_first.csv, combined_labels_cluster_first.csv
  5. Splits datasets into train/validation/test sets with class balancing weights:
     - dataset_{strategy}_train.csv, dataset_{strategy}_validation.csv, dataset_{strategy}_test.csv
     - dataset_stats_{strategy}.json, weights_{strategy}.npy
===============================================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from keyword_weak_labeler import KeywordWeakLabeler, SEED_KEYWORDS
from weak_labeler_cluster import ClusterBasedWeakLabeler
from weak_labeling_framework import UnifiedWeakLabelingFramework, CONFIG
from week2_dataset_preparation import WeekendDatasetPreparation

OUTPUT_DIR = "weak_labeling_results"
HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    print("=" * 80)
    print(" REGENERATING WEAK LABELING RESULTS FROM REAL ASRS DATA")
    print("=" * 80)

    if not os.path.exists("asrs_full.csv"):
        print("[ERROR] asrs_full.csv not found!")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # 1. Load Real ASRS Dataset
    print("\n[STEP 1/5] Ingesting real ASRS dataset from asrs_full.csv...")
    df_asrs = pd.read_csv("asrs_full.csv")
    print(f"  Total records in asrs_full.csv: {len(df_asrs):,d}")

    # 2. Run Keyword Audit Log on Real Data
    print("\n[STEP 2/5] Running KeywordWeakLabeler consistency audit on real data...")
    kw_labeler = KeywordWeakLabeler(sample_size=15, threshold_ratio=0.80, random_state=RANDOM_STATE)
    audit_df = kw_labeler.fit(df_asrs, text_column='text', label_column='label')
    audit_path = os.path.join(OUTPUT_DIR, "keyword_audit_log.csv")
    audit_df.to_csv(audit_path, index=False)
    print(f"  Saved {len(audit_df)} keyword audit records to: {audit_path}")

    # 3. Stratified Representative Sample for Clustering & Combination
    # To keep git push lean and fast (<20MB total CSVs), sample 5,000 reports
    print("\n[STEP 3/5] Constructing balanced 5,000-report sample for cluster & ensemble...")
    sample_subsets = []
    for cls in HFACS_CLASSES + ['Unlabeled']:
        sub = df_asrs[df_asrs['label'] == cls]
        # allocate: 1500 Decision, 1500 Skill-Based, 680 Violation, 658 Perceptual, 662 Unlabeled
        target_n = 1500 if cls in ['Decision Error', 'Skill-Based Error'] else min(len(sub), 700)
        n_draw = min(len(sub), target_n)
        sample_subsets.append(sub.sample(n=n_draw, random_state=RANDOM_STATE))

    sample_df = pd.concat(sample_subsets).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Sampled {len(sample_df):,d} reports across classes:")
    for cls, cnt in sample_df['label'].value_counts().items():
        print(f"    - {cls:20s}: {cnt}")

    # Apply unified keyword labeling
    all_kw_labeler = KeywordWeakLabeler()
    for cls, kws in SEED_KEYWORDS.items():
        for kw in kws:
            all_kw_labeler.validated_keywords[kw] = cls
    keyword_labeled_df = all_kw_labeler.transform(sample_df, text_column='text')

    # Apply BERT + UMAP + HDBSCAN clustering
    print("\n[STEP 4/5] Applying BERT+UMAP+HDBSCAN cluster weak labeling...")
    cluster_labeler = ClusterBasedWeakLabeler(
        n_neighbors=15,
        min_dist=0.1,
        min_cluster_size=20,
        min_samples=5,
        random_state=RANDOM_STATE
    )

    cluster_labeled_df = cluster_labeler.transform(sample_df['text'].tolist())
    cluster_labeled_df['report_id'] = sample_df['report_id'].values

    # Map clusters to HFACS classes based on top TF-IDF words
    print("  Mapping discovered clusters to HFACS classes via semantic keywords...")
    cluster_hfacs_map = {}
    for cid, kws in cluster_labeler.cluster_keywords.items():
        kw_text = " ".join(kws).lower()
        matched_class = "Unlabeled"
        best_score = 0
        for cls, seed_list in SEED_KEYWORDS.items():
            score = sum(1 for sk in seed_list if sk.lower() in kw_text)
            if score > best_score:
                best_score = score
                matched_class = cls
        cluster_hfacs_map[cid] = matched_class

    # Apply semantic mapping
    cluster_labeled_df['weak_label'] = cluster_labeled_df['cluster_id'].map(cluster_hfacs_map).fillna("Unlabeled")

    # Save cluster keywords & config
    cluster_kw_path = os.path.join(OUTPUT_DIR, "cluster_keywords.json")
    with open(cluster_kw_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_labeler.cluster_keywords, f, indent=2)

    cluster_cfg_path = os.path.join(OUTPUT_DIR, "cluster_config.json")
    cfg_clusters = {
        str(cid): {
            "keywords": kws,
            "hfacs_class": cluster_hfacs_map.get(cid, "Unlabeled"),
            "description": f"Cluster {cid} automatically mapped via semantic seed keyword overlap"
        }
        for cid, kws in cluster_labeler.cluster_keywords.items()
    }
    with open(cluster_cfg_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "n_neighbors": 15, "min_dist": 0.1, "min_cluster_size": 20, "min_samples": 5, "random_state": 42
            },
            "clusters": cfg_clusters
        }, f, indent=2)

    # 4. Compare & Combine Labels
    framework = UnifiedWeakLabelingFramework(config=CONFIG)
    comparison_df, agreement_stats = framework.compare_results(keyword_labeled_df, cluster_labeled_df)

    strategies = ["majority", "keyword_first", "cluster_first"]
    for strat in strategies:
        comb_df = sample_df[['report_id', 'text']].copy()
        comb_df['keyword_label'] = keyword_labeled_df['weak_label']
        comb_df['cluster_label'] = cluster_labeled_df['weak_label']

        if strat == 'majority':
            comb_df['final_label'] = np.where(
                comb_df['keyword_label'] == comb_df['cluster_label'],
                comb_df['keyword_label'],
                np.where(comb_df['keyword_label'] != 'Unlabeled', comb_df['keyword_label'], comb_df['cluster_label'])
            )
        elif strat == 'keyword_first':
            comb_df['final_label'] = np.where(
                comb_df['keyword_label'] != 'Unlabeled',
                comb_df['keyword_label'],
                comb_df['cluster_label']
            )
        elif strat == 'cluster_first':
            comb_df['final_label'] = np.where(
                comb_df['cluster_label'] != 'Unlabeled',
                comb_df['cluster_label'],
                comb_df['keyword_label']
            )

        comb_path = os.path.join(OUTPUT_DIR, f"combined_labels_{strat}.csv")
        comb_df.to_csv(comb_path, index=False)
        print(f"  Generated {comb_path}: {len(comb_df)} rows")

    # 5. Dataset Preparation (Train/Val/Test Splits & Balancing Weights)
    print("\n[STEP 5/5] Splitting datasets and generating balancing weights...")
    prep = WeekendDatasetPreparation(
        weak_labeling_dir=OUTPUT_DIR,
        strategies=strategies,
        test_size=0.20,
        val_size=0.10,
        random_state=RANDOM_STATE
    )
    prep.prepare_all_strategies(raw_data_path="asrs_full.csv")

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f" [SUCCESS] weak_labeling_results/ REGENERATED IN {elapsed:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
