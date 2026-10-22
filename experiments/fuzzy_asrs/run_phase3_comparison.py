"""
=================================================================================
Phase 3: Crisp vs Fuzzy Comparison
=================================================================================
Compares Phase 1 crisp baseline with Phase 2 fuzzy labels on the full dataset.
Generates comparison tables and visualizations.
=================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FUZZY_CSV = RESULTS_DIR / "fuzzy_full_results.csv"

def main():
    print("=" * 80)
    print("PHASE 3: CRISP VS FUZZY COMPARISON")
    print("=" * 80)

    # Load fuzzy results - already include crisp_label from Phase 2 transform
    print("Loading fuzzy results (includes crisp labels)...")
    df = pd.read_csv(FUZZY_CSV)
    print(f"Rows: {len(df):,}")

    # Verify required columns
    required_cols = ['report_id', 'crisp_label', 'fuzzy_label', 'confidence', 'dominance', 'rule_fired']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    total = len(df)

    # 1. Coverage and abstention rates
    crisp_labeled = (df['crisp_label'] != 'Unlabeled').sum()
    crisp_unlabeled = (df['crisp_label'] == 'Unlabeled').sum()
    fuzzy_labeled = (df['fuzzy_label'] != 'Uncertain').sum()
    fuzzy_uncertain = (df['fuzzy_label'] == 'Uncertain').sum()

    print(f"\n--- COVERAGE ---")
    print(f"Crisp labeled:   {crisp_labeled:,} ({crisp_labeled/total*100:.2f}%)")
    print(f"Crisp unlabeled: {crisp_unlabeled:,} ({crisp_unlabeled/total*100:.2f}%)")
    print(f"Fuzzy labeled:   {fuzzy_labeled:,} ({fuzzy_labeled/total*100:.2f}%)")
    print(f"Fuzzy uncertain: {fuzzy_uncertain:,} ({fuzzy_uncertain/total*100:.2f}%)")

    # 2. Class distributions
    print(f"\n--- CLASS DISTRIBUTION ---")
    print("\nCrisp:")
    crisp_dist = df['crisp_label'].value_counts()
    for label, count in crisp_dist.items():
        print(f"  {label:20s}: {count:6,} ({count/total*100:5.2f}%)")

    print("\nFuzzy:")
    fuzzy_dist = df['fuzzy_label'].value_counts()
    for label, count in fuzzy_dist.items():
        print(f"  {label:20s}: {count:6,} ({count/total*100:5.2f}%)")

    # 3. Agreement where both labeled
    both_labeled = df[
        (df['crisp_label'] != 'Unlabeled') &
        (df['fuzzy_label'] != 'Uncertain')
    ]
    both_count = len(both_labeled)
    print(f"\n--- AGREEMENT (both labeled: {both_count:,}) ---")
    if both_count > 0:
        agree = (both_labeled['crisp_label'] == both_labeled['fuzzy_label']).sum()
        disagree = both_count - agree
        print(f"Agree:   {agree:,} ({agree/both_count*100:.2f}%)")
        print(f"Disagree: {disagree:,} ({disagree/both_count*100:.2f}%)")

        # Per-class agreement
        print("\nPer-class agreement:")
        for cls in ['Skill-Based Error', 'Decision Error', 'Violation', 'Perceptual Error']:
            crisp_cls = both_labeled[both_labeled['crisp_label'] == cls]
            fuzzy_cls = both_labeled[both_labeled['fuzzy_label'] == cls]
            if len(crisp_cls) > 0:
                cls_agree = (crisp_cls['fuzzy_label'] == cls).sum()
                print(f"  {cls:20s}: {cls_agree}/{len(crisp_cls)} ({cls_agree/len(crisp_cls)*100:.2f}%)")

    # 4. Transitions
    print(f"\n--- TRANSITION TABLE ---")
    transition = pd.crosstab(df['crisp_label'], df['fuzzy_label'], margins=True)
    print(transition.to_string())

    # Specific transitions of interest
    print(f"\n--- KEY TRANSITIONS ---")
    crisp_unlabeled_fuzzy_labeled = df[
        (df['crisp_label'] == 'Unlabeled') & (df['fuzzy_label'] != 'Uncertain')
    ]
    print(f"Crisp Unlabeled -> Fuzzy labeled: {len(crisp_unlabeled_fuzzy_labeled):,}")
    if len(crisp_unlabeled_fuzzy_labeled) > 0:
        print(crisp_unlabeled_fuzzy_labeled['fuzzy_label'].value_counts())

    crisp_labeled_fuzzy_uncertain = df[
        (df['crisp_label'] != 'Unlabeled') & (df['fuzzy_label'] == 'Uncertain')
    ]
    print(f"Crisp labeled -> Fuzzy Uncertain: {len(crisp_labeled_fuzzy_uncertain):,}")
    if len(crisp_labeled_fuzzy_uncertain) > 0:
        print(crisp_labeled_fuzzy_uncertain['crisp_label'].value_counts())

    # 5. Confidence stats for fuzzy labeled
    print(f"\n--- CONFIDENCE STATISTICS (fuzzy labeled only) ---")
    fuzzy_labeled_df = df[df['fuzzy_label'] != 'Uncertain']
    if len(fuzzy_labeled_df) > 0:
        print(f"Mean confidence: {fuzzy_labeled_df['confidence'].mean():.4f}")
        print(f"Median confidence: {fuzzy_labeled_df['confidence'].median():.4f}")
        print(f"Min confidence: {fuzzy_labeled_df['confidence'].min():.4f}")
        print(f"Max confidence: {fuzzy_labeled_df['confidence'].max():.4f}")
        print(f"Std confidence: {fuzzy_labeled_df['confidence'].std():.4f}")

    # Save comparison table
    print(f"\n--- SAVING RESULTS ---")
    transition_path = RESULTS_DIR / "crisp_vs_fuzzy_transition.csv"
    transition.to_csv(transition_path)
    print(f"Transition table saved to {transition_path}")

    # Save summary JSON
    summary = {
        "total_narratives": int(total),
        "crisp_labeled": int(crisp_labeled),
        "crisp_unlabeled": int(crisp_unlabeled),
        "crisp_coverage_pct": round(crisp_labeled/total*100, 2),
        "fuzzy_labeled": int(fuzzy_labeled),
        "fuzzy_uncertain": int(fuzzy_uncertain),
        "fuzzy_coverage_pct": round(fuzzy_labeled/total*100, 2),
        "both_labeled": int(both_count),
        "agreement_count": int(agree) if both_count > 0 else 0,
        "agreement_pct": round(agree/both_count*100, 2) if both_count > 0 else 0,
        "disagreement_count": int(disagree) if both_count > 0 else 0,
        "disagreement_pct": round(disagree/both_count*100, 2) if both_count > 0 else 0,
        "crisp_unlabeled_to_fuzzy_labeled": int(len(crisp_unlabeled_fuzzy_labeled)),
        "crisp_labeled_to_fuzzy_uncertain": int(len(crisp_labeled_fuzzy_uncertain))
    }

    summary_path = RESULTS_DIR / "comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Save detailed transitions
    transitions_detail = df[['report_id', 'crisp_label', 'fuzzy_label', 'confidence', 'dominance', 'rule_fired']].copy()
    transitions_detail_path = RESULTS_DIR / "detailed_transitions.csv"
    transitions_detail.to_csv(transitions_detail_path, index=False)
    print(f"Detailed transitions saved to {transitions_detail_path}")

    print(f"\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()