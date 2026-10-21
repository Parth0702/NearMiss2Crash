"""
=================================================================================
Run Fuzzy Labeler on Crisp Baseline Sample
=================================================================================
Loads a sample from crisp_baseline_rows.csv, applies FuzzyWeakLabeler,
and prints a comparison table of Crisp Label vs Fuzzy Label, Confidence,
Dominance, and Rule Fired.
=================================================================================
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fuzzy_asrs.fuzzy_labeler import FuzzyWeakLabeler

def main():
    csv_path = Path(__file__).resolve().parent / "crisp_baseline_rows.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run Phase 1 baseline first.")
        sys.exit(1)

    print(f"Loading sample from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=50)

    print(f"Initializing FuzzyWeakLabeler (evidence_threshold=0.35, dominance_margin=0.15)...")
    fuzzy_labeler = FuzzyWeakLabeler(evidence_threshold=0.35, dominance_margin=0.15)

    print("Running transform on sample...")
    result_df = fuzzy_labeler.transform(df, text_column="text")

    print("\n" + "=" * 100)
    print("FUZZY WEAK LABELER SAMPLE OUTPUT (First 15 Rows)")
    print("=" * 100)

    display_cols = ["report_id", "crisp_label", "fuzzy_label", "confidence", "dominance", "rule_fired", "evidence_counts", "matched_keywords"]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(result_df[display_cols].head(15).to_string(index=False))
    print("=" * 100)

    # Distribution on sample
    print("\nSample Agreement / Distribution:")
    print(result_df["fuzzy_label"].value_counts(dropna=False))

    print("\nCrisp vs Fuzzy cross-tabulation on sample:")
    print(pd.crosstab(result_df["crisp_label"], result_df["fuzzy_label"], margins=True))

if __name__ == "__main__":
    main()
