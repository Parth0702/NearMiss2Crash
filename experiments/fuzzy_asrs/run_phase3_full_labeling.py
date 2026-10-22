"""
=================================================================================
Phase 3: Full-Dataset Fuzzy Weak Labeling
=================================================================================
Runs FuzzyWeakLabeler on the complete ASRS crisp_baseline_rows.csv dataset.
Exports results to experiments/fuzzy_asrs/results/fuzzy_full_results.csv
=================================================================================
"""

import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fuzzy_asrs.fuzzy_labeler import FuzzyWeakLabeler

# Setup logging
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_FILE = RESULTS_DIR / "phase3_full_labeling.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

CRISP_CSV = Path(__file__).resolve().parent / "crisp_baseline_rows.csv"
OUTPUT_CSV = RESULTS_DIR / "fuzzy_full_results.csv"

def main():
    t0 = time.time()
    logging.info("=" * 80)
    logging.info("PHASE 3: FULL-DATASET FUZZY WEAK LABELING")
    logging.info("=" * 80)

    # Load crisp baseline
    logging.info(f"Loading crisp baseline from {CRISP_CSV}")
    df = pd.read_csv(CRISP_CSV)
    total_rows = len(df)
    logging.info(f"Total narratives: {total_rows:,}")

    # Verify columns
    required_cols = ['report_id', 'text', 'crisp_label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Initialize fuzzy labeler with roadmap defaults
    logging.info("Initializing FuzzyWeakLabeler (τ=0.35, δ=0.15)...")
    fuzzy_labeler = FuzzyWeakLabeler(
        evidence_threshold=0.35,
        dominance_margin=0.15,
        count_occurrences=True
    )

    # Run fuzzy labeling
    logging.info("Running fuzzy weak labeling on full dataset...")
    t_transform = time.time()
    result_df = fuzzy_labeler.transform(df, text_column="text")
    transform_time = time.time() - t_transform
    logging.info(f"Transform completed in {transform_time:.2f}s ({total_rows/transform_time:.0f} rows/s)")

    # Verify output integrity
    logging.info("\n--- OUTPUT VERIFICATION ---")
    logging.info(f"Output rows: {len(result_df):,} (expected {total_rows:,})")
    assert len(result_df) == total_rows, "Row count mismatch!"
    logging.info("✓ Row count preserved")

    # Check for duplicates
    dup_ids = result_df['report_id'].duplicated().sum()
    logging.info(f"Duplicate report_ids: {dup_ids}")
    assert dup_ids == 0, "Duplicate report IDs found!"
    logging.info("✓ No duplicate report IDs")

    # Check confidence bounds
    conf_min = result_df['confidence'].min()
    conf_max = result_df['confidence'].max()
    logging.info(f"Confidence range: [{conf_min:.4f}, {conf_max:.4f}]")
    assert conf_min >= 0.0 and conf_max <= 1.0, "Confidence out of bounds!"
    logging.info("✓ Confidence values in [0, 1]")

    # Check for non-finite values
    non_finite = result_df['confidence'].isna().sum() + np.isinf(result_df['confidence']).sum()
    logging.info(f"Non-finite confidence values: {non_finite}")
    assert non_finite == 0, "Non-finite confidence values found!"
    logging.info("✓ All confidence values finite")

    # Verify no non-Uncertain with zero confidence
    non_uncertain_zero_conf = result_df[
        (result_df['fuzzy_label'] != 'Uncertain') & (result_df['confidence'] == 0.0)
    ]
    logging.info(f"Non-Uncertain with zero confidence: {len(non_uncertain_zero_conf)}")
    if len(non_uncertain_zero_conf) > 0:
        logging.warning(f"Found {len(non_uncertain_zero_conf)} non-Uncertain predictions with 0 confidence!")
    else:
        logging.info("✓ No non-Uncertain predictions with zero confidence")

    # Distribution
    logging.info("\n--- FUZZY LABEL DISTRIBUTION ---")
    dist = result_df['fuzzy_label'].value_counts()
    for label, count in dist.items():
        pct = count / total_rows * 100
        logging.info(f"  {label:25s}: {count:6,} ({pct:5.2f}%)")

    # Save results
    logging.info(f"\nSaving results to {OUTPUT_CSV}")
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    file_size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)
    logging.info(f"File size: {file_size_mb:.2f} MB")

    elapsed = time.time() - t0
    logging.info("=" * 80)
    logging.info(f"PHASE 3 FULL LABELING COMPLETE in {elapsed:.2f}s")
    logging.info("=" * 80)

    return result_df

if __name__ == "__main__":
    main()