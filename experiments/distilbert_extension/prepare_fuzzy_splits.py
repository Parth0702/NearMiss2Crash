"""
=================================================================================
Phase 3B: Create ASRS Fuzzy Labels and Prepare DistilBERT Training Data
=================================================================================
Applies FuzzyWeakLabeler to full ASRS dataset, creates train/val/test splits
matching the crisp label splits for fair comparison.
=================================================================================
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fuzzy_asrs.fuzzy_labeler import FuzzyWeakLabeler

RANDOM_SEED = 42
HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]

# Paths
CRISP_TRAIN = PROJECT_ROOT / "weak_labeling_results" / "dataset_majority_train.csv"
CRISP_VAL = PROJECT_ROOT / "weak_labeling_results" / "dataset_majority_validation.csv"
CRISP_TEST = PROJECT_ROOT / "weak_labeling_results" / "dataset_majority_test.csv"
FUZZY_FULL_RESULTS = PROJECT_ROOT / "experiments" / "fuzzy_asrs" / "results" / "fuzzy_full_results.csv"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "distilbert_extension"
OUTPUT_DIR.mkdir(exist_ok=True)

FUZZY_TRAIN = OUTPUT_DIR / "dataset_fuzzy_train.csv"
FUZZY_VAL = OUTPUT_DIR / "dataset_fuzzy_validation.csv"
FUZZY_TEST = OUTPUT_DIR / "dataset_fuzzy_test.csv"

def main():
    print("=" * 80)
    print("PHASE 3B: PREPARE FUZZY LABELS FOR DISTILBERT TRAINING")
    print("=" * 80)

    # Load fuzzy full results (already computed in Phase 3)
    print(f"Loading fuzzy full results from {FUZZY_FULL_RESULTS}")
    fuzzy_df = pd.read_csv(FUZZY_FULL_RESULTS)
    print(f"Loaded {len(fuzzy_df):,} rows with fuzzy labels")

    # Create a lookup from report_id to fuzzy_label
    fuzzy_lookup = fuzzy_df.set_index('report_id')['fuzzy_label'].to_dict()

    # Load crisp splits to get the exact same report_ids
    print("\nLoading crisp train/val/test splits...")
    crisp_train = pd.read_csv(CRISP_TRAIN)
    crisp_val = pd.read_csv(CRISP_VAL)
    crisp_test = pd.read_csv(CRISP_TEST)

    print(f"Crisp train: {len(crisp_train):,}")
    print(f"Crisp val:   {len(crisp_val):,}")
    print(f"Crisp test:  {len(crisp_test):,}")

    # Verify report_id column
    for name, df in [("train", crisp_train), ("val", crisp_val), ("test", crisp_test)]:
        if 'report_id' not in df.columns:
            raise ValueError(f"Missing report_id in crisp {name}")

    # Add ASRS_ prefix to crisp report_ids to match fuzzy results format
    for df in [crisp_train, crisp_val, crisp_test]:
        df['report_id'] = 'ASRS_' + df['report_id'].astype(str)

    # Create fuzzy splits by mapping fuzzy labels to the same report_ids
    def create_fuzzy_split(crisp_df, split_name):
        split_df = crisp_df[['report_id', 'text']].copy()
        split_df['fuzzy_label'] = split_df['report_id'].map(fuzzy_lookup)

        # Check for missing
        missing = split_df['fuzzy_label'].isna().sum()
        if missing > 0:
            print(f"  WARNING: {missing} missing fuzzy labels in {split_name}")

        # Exclude Uncertain for supervised training (keep for record)
        labeled = split_df[split_df['fuzzy_label'] != 'Uncertain'].copy()
        print(f"  {split_name}: {len(split_df):,} total, {len(labeled):,} labeled (excl. Uncertain)")

        return split_df, labeled

    fuzzy_train_full, fuzzy_train_labeled = create_fuzzy_split(crisp_train, "train")
    fuzzy_val_full, fuzzy_val_labeled = create_fuzzy_split(crisp_val, "val")
    fuzzy_test_full, fuzzy_test_labeled = create_fuzzy_split(crisp_test, "test")

    # Save full splits (with Uncertain)
    fuzzy_train_full.to_csv(FUZZY_TRAIN, index=False)
    fuzzy_val_full.to_csv(FUZZY_VAL, index=False)
    fuzzy_test_full.to_csv(FUZZY_TEST, index=False)
    print(f"\nSaved full splits (with Uncertain) to {OUTPUT_DIR}")

    # Also save labeled-only versions for training
    FUZZY_TRAIN_LABELED = OUTPUT_DIR / "dataset_fuzzy_train_labeled.csv"
    FUZZY_VAL_LABELED = OUTPUT_DIR / "dataset_fuzzy_validation_labeled.csv"
    FUZZY_TEST_LABELED = OUTPUT_DIR / "dataset_fuzzy_test_labeled.csv"

    fuzzy_train_labeled.to_csv(FUZZY_TRAIN_LABELED, index=False)
    fuzzy_val_labeled.to_csv(FUZZY_VAL_LABELED, index=False)
    fuzzy_test_labeled.to_csv(FUZZY_TEST_LABELED, index=False)
    print(f"Saved labeled-only splits to {OUTPUT_DIR}")

    # Distribution report
    print("\n--- FUZZY LABEL DISTRIBUTION (labeled only) ---")
    for name, df in [("train", fuzzy_train_labeled), ("val", fuzzy_val_labeled), ("test", fuzzy_test_labeled)]:
        dist = df['fuzzy_label'].value_counts()
        print(f"\n{name.upper()}:")
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {label:20s}: {count:5,} ({pct:5.2f}%)")

    # Save metadata
    metadata = {
        "random_seed": RANDOM_SEED,
        "hfacs_classes": HFACS_CLASSES,
        "train_total": int(len(fuzzy_train_full)),
        "train_labeled": int(len(fuzzy_train_labeled)),
        "val_total": int(len(fuzzy_val_full)),
        "val_labeled": int(len(fuzzy_val_labeled)),
        "test_total": int(len(fuzzy_test_full)),
        "test_labeled": int(len(fuzzy_test_labeled)),
        "train_uncertain": int(len(fuzzy_train_full) - len(fuzzy_train_labeled)),
        "val_uncertain": int(len(fuzzy_val_full) - len(fuzzy_val_labeled)),
        "test_uncertain": int(len(fuzzy_test_full) - len(fuzzy_test_labeled)),
        "source_fuzzy_results": str(FUZZY_FULL_RESULTS),
        "source_crisp_splits": {
            "train": str(CRISP_TRAIN),
            "val": str(CRISP_VAL),
            "test": str(CRISP_TEST)
        }
    }

    with open(OUTPUT_DIR / "fuzzy_splits_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {OUTPUT_DIR}/fuzzy_splits_metadata.json")

    print("\n" + "=" * 80)
    print("FUZZY SPLITS PREPARED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()