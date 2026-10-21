#!/usr/bin/env python3
"""
Phase 1: Working State Verification & Single-Reporter Crisp Baseline Generation
Extracts pure Report 1_Narrative from local cached HuggingFace Arrow file.
Generates crisp baseline using canonical KeywordWeakLabeler.
Outputs:
  - experiments/fuzzy_asrs/crisp_baseline_rows.csv
  - experiments/fuzzy_asrs/crisp_label_distribution.csv
  - experiments/fuzzy_asrs/phase1_run.log
"""

import os
import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from keyword_weak_labeler import KeywordWeakLabeler, SEED_KEYWORDS

OUTPUT_DIR = Path(__file__).resolve().parent
LOG_FILE = OUTPUT_DIR / "phase1_run.log"
CRISP_ROWS_CSV = OUTPUT_DIR / "crisp_baseline_rows.csv"
CRISP_DIST_CSV = OUTPUT_DIR / "crisp_label_distribution.csv"

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

ARROW_CACHE_PATH = Path(
    os.path.expanduser("~/.cache/huggingface/datasets/elihoole___asrs-aviation-reports/default/0.0.0/f1e681e92cddae20d01fc498d685f1cf6a052d34/asrs-aviation-reports-train.arrow")
)

def run_phase1():
    t0 = time.time()
    logging.info("=" * 80)
    logging.info("PHASE 1: SINGLE-REPORTER (REPORT 1 ONLY) CRISP BASELINE GENERATION")
    logging.info("=" * 80)
    logging.info(f"Project Root: {PROJECT_ROOT}")
    logging.info(f"Target Output Directory: {OUTPUT_DIR}")
    
    # 1. Check local source dataset
    if not ARROW_CACHE_PATH.exists():
        logging.error(f"FATAL: Cached Arrow dataset not found at {ARROW_CACHE_PATH}")
        sys.exit(1)
        
    logging.info(f"Loading local cached Arrow dataset from: {ARROW_CACHE_PATH}")
    ds = Dataset.from_file(str(ARROW_CACHE_PATH))
    total_raw = len(ds)
    logging.info(f"Loaded {total_raw:,} records from local Arrow cache.")

    # 2. Extract report_id and Report 1_Narrative only
    logging.info("Extracting report_id and Report 1_Narrative (single-reporter primary text)...")
    report_ids = []
    texts = []
    
    for row in ds:
        acn = str(row.get("acn_num_ACN", "") or "").strip()
        r1 = str(row.get("Report 1_Narrative", "") or "").strip()
        report_ids.append(f"ASRS_{acn}")
        texts.append(r1)

    df_report1 = pd.DataFrame({
        "report_id": report_ids,
        "input_source": "Report 1_Narrative",
        "text": texts
    })

    # 3. Text integrity diagnostics
    logging.info("\n--- TEXT INTEGRITY DIAGNOSTICS ---")
    total_rows = len(df_report1)
    unique_ids = df_report1["report_id"].nunique()
    null_count = df_report1["text"].isna().sum()
    empty_count = (df_report1["text"].str.strip() == "").sum()
    duplicate_narratives = df_report1["text"].duplicated().sum()
    char_lengths = df_report1["text"].str.len()
    word_lengths = df_report1["text"].str.split().str.len()

    logging.info(f"Total Rows              : {total_rows:,}")
    logging.info(f"Unique Report IDs       : {unique_ids:,}")
    logging.info(f"Duplicate Report IDs    : {total_rows - unique_ids:,}")
    logging.info(f"Null Text Count         : {null_count:,}")
    logging.info(f"Empty Text Count        : {empty_count:,}")
    logging.info(f"Duplicate Narratives    : {duplicate_narratives:,}")
    logging.info(f"Mean Character Length   : {char_lengths.mean():.1f} (Median: {char_lengths.median():.0f}, Min: {char_lengths.min()}, Max: {char_lengths.max()})")
    logging.info(f"Mean Word Count         : {word_lengths.mean():.1f} (Median: {word_lengths.median():.0f}, Min: {word_lengths.min()}, Max: {word_lengths.max()})")

    if total_rows != unique_ids:
        raise ValueError("FATAL: Duplicate report_ids detected in source dataset!")
    if empty_count > 0:
        logging.warning(f"WARNING: {empty_count} rows have empty Report 1 text!")

    # 4. Instantiate unchanged canonical KeywordWeakLabeler
    logging.info("\n--- INITIALIZING CANONICAL KEYWORD WEAK LABELER ---")
    labeler = KeywordWeakLabeler(random_state=config.RANDOM_SEED)
    total_kws = 0
    for cls_name, kw_list in SEED_KEYWORDS.items():
        logging.info(f"  Class: {cls_name:20s} | {len(kw_list):2d} keywords: {', '.join(kw_list)}")
        total_kws += len(kw_list)
        for kw in kw_list:
            labeler.validated_keywords[kw] = cls_name
    logging.info(f"Total validated seed keywords registered: {total_kws}")

    # 5. Execute weak labeling transform on Report 1 text
    logging.info("\nRunning KeywordWeakLabeler.transform() on Report 1 text...")
    t_transform_start = time.time()
    labeled_df = labeler.transform(df_report1, text_column="text")
    transform_elapsed = time.time() - t_transform_start
    logging.info(f"Weak labeling completed in {transform_elapsed:.2f} seconds ({len(labeled_df)/transform_elapsed:.1f} rows/s).")

    df_report1["crisp_label"] = labeled_df["weak_label"]
    df_report1["matched_keywords"] = labeled_df["matched_keywords"]

    # 6. Save crisp_baseline_rows.csv
    logging.info(f"Saving crisp baseline rows to: {CRISP_ROWS_CSV}")
    df_report1.to_csv(CRISP_ROWS_CSV, index=False, encoding="utf-8")
    logging.info(f"File size: {CRISP_ROWS_CSV.stat().st_size / (1024*1024):.2f} MB")

    # 7. Compute & Save Label Distribution
    logging.info("\n--- CRISP LABEL DISTRIBUTION (REPORT 1 ONLY) ---")
    dist_counts = df_report1["crisp_label"].value_counts(dropna=False)
    labeled_total = (df_report1["crisp_label"] != "Unlabeled").sum()
    unlabeled_total = (df_report1["crisp_label"] == "Unlabeled").sum()

    dist_records = []
    # Standard class ordering
    ordered_classes = config.HFACS_CORE_CLASSES + ["Unlabeled"]
    for cls_name in ordered_classes:
        cnt = int(dist_counts.get(cls_name, 0))
        pct_all = (cnt / total_rows) * 100
        pct_labeled = (cnt / labeled_total * 100) if (cls_name != "Unlabeled" and labeled_total > 0) else 0.0
        dist_records.append({
            "class": cls_name,
            "count": cnt,
            "percentage_all": round(pct_all, 2),
            "percentage_labeled": round(pct_labeled, 2) if cls_name != "Unlabeled" else None
        })
        logging.info(f"  {cls_name:20s}: {cnt:6,d} ({pct_all:5.2f}% of all | {pct_labeled:5.2f}% of labeled)")

    logging.info(f"\nTotal Labeled   : {labeled_total:,d} ({labeled_total/total_rows*100:.2f}%)")
    logging.info(f"Total Unlabeled : {unlabeled_total:,d} ({unlabeled_total/total_rows*100:.2f}%)")

    dist_df = pd.DataFrame(dist_records)
    dist_df.to_csv(CRISP_DIST_CSV, index=False)
    logging.info(f"Saved distribution summary to: {CRISP_DIST_CSV}")

    # 8. Compare against previous combined-text distribution
    logging.info("\n--- COMPARISON: REPORT 1 ALONE vs. COMBINED (REPORT 1 + 2) ---")
    df_combined = pd.read_csv(PROJECT_ROOT / "data/processed/asrs_master.csv", usecols=["report_id", "keyword_label"])
    comb_counts = df_combined["keyword_label"].value_counts()
    
    logging.info(f"{'Class':20s} | {'Report 1 Only':14s} | {'Combined (1+2)':14s} | {'Diff (R1 - Comb)':16s}")
    logging.info("-" * 70)
    for cls_name in ordered_classes:
        r1_c = int(dist_counts.get(cls_name, 0))
        cb_c = int(comb_counts.get(cls_name, 0))
        diff = r1_c - cb_c
        logging.info(f"{cls_name:20s} | {r1_c:6,d} ({r1_c/total_rows*100:5.2f}%) | {cb_c:6,d} ({cb_c/total_rows*100:5.2f}%) | {diff:+7,d} ({diff/total_rows*100:+5.2f}%)")

    agreed_labels = (df_report1["crisp_label"] == df_combined["keyword_label"]).sum()
    logging.info(f"\nExact label agreement between Report 1 and Combined: {agreed_labels:,} / {total_rows:,} ({agreed_labels/total_rows*100:.2f}%)")
    logging.info(f"Labels changed due to secondary report text: {total_rows - agreed_labels:,} ({(total_rows - agreed_labels)/total_rows*100:.2f}%)")

    total_elapsed = time.time() - t0
    logging.info("=" * 80)
    logging.info(f"PHASE 1 COMPLETE IN {total_elapsed:.2f} SECONDS")
    logging.info("=" * 80)

if __name__ == "__main__":
    run_phase1()
