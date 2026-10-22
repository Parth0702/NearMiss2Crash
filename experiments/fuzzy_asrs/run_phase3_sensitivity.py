"""
=================================================================================
Phase 3: Threshold Sensitivity Analysis
=================================================================================
Runs fuzzy labeler with varying τ and δ thresholds.
=================================================================================
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fuzzy_asrs.fuzzy_labeler import FuzzyWeakLabeler

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SENSITIVITY_DIR = RESULTS_DIR / "sensitivity"
SENSITIVITY_DIR.mkdir(exist_ok=True)

CRISP_CSV = Path(__file__).resolve().parent / "crisp_baseline_rows.csv"

def run_labeler(df, evidence_threshold, dominance_margin):
    """Run fuzzy labeler and return key metrics."""
    labeler = FuzzyWeakLabeler(
        evidence_threshold=evidence_threshold,
        dominance_margin=dominance_margin,
        count_occurrences=True
    )
    result = labeler.transform(df, text_column="text")

    total = len(result)
    labeled = (result['fuzzy_label'] != 'Uncertain').sum()
    uncertain = (result['fuzzy_label'] == 'Uncertain').sum()
    dist = result['fuzzy_label'].value_counts().to_dict()

    # Agreement with crisp (where both labeled)
    both = result[(result['crisp_label'] != 'Unlabeled') & (result['fuzzy_label'] != 'Uncertain')]
    both_count = len(both)
    agree = (both['crisp_label'] == both['fuzzy_label']).sum() if both_count > 0 else 0

    return {
        "tau": evidence_threshold,
        "delta": dominance_margin,
        "total": int(total),
        "labeled": int(labeled),
        "uncertain": int(uncertain),
        "coverage_pct": round(labeled/total*100, 2),
        "abstention_pct": round(uncertain/total*100, 2),
        "both_labeled": int(both_count),
        "agreement_count": int(agree),
        "agreement_pct": round(agree/both_count*100, 2) if both_count > 0 else 0,
        "distribution": dist
    }

def main():
    print("=" * 80)
    print("PHASE 3: THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Load data
    print("Loading crisp baseline...")
    df = pd.read_csv(CRISP_CSV)
    print(f"Total narratives: {len(df):,}")

    # Baseline crisp distribution for reference
    crisp_dist = df['crisp_label'].value_counts().to_dict()
    print(f"Crisp distribution: {crisp_dist}")

    # 1. τ sensitivity (δ fixed at 0.15)
    print("\n" + "=" * 80)
    print("TAU SENSITIVITY (delta = 0.15)")
    print("=" * 80)

    tau_values = [0.25, 0.35, 0.50]
    tau_results = []

    for tau in tau_values:
        print(f"\nRunning tau={tau}...")
        t0 = time.time()
        metrics = run_labeler(df, tau, 0.15)
        elapsed = time.time() - t0
        print(f"  Labeled: {metrics['labeled']:,} ({metrics['coverage_pct']:.2f}%)")
        print(f"  Uncertain: {metrics['uncertain']:,} ({metrics['abstention_pct']:.2f}%)")
        print(f"  Agreement: {metrics['agreement_pct']:.2f}%")
        print(f"  Time: {elapsed:.1f}s")
        tau_results.append(metrics)

    # 2. δ sensitivity (τ fixed at 0.35)
    print("\n" + "=" * 80)
    print("DELTA SENSITIVITY (tau = 0.35)")
    print("=" * 80)

    delta_values = [0.10, 0.15, 0.20]
    delta_results = []

    for delta in delta_values:
        print(f"\nRunning delta={delta}...")
        t0 = time.time()
        metrics = run_labeler(df, 0.35, delta)
        elapsed = time.time() - t0
        print(f"  Labeled: {metrics['labeled']:,} ({metrics['coverage_pct']:.2f}%)")
        print(f"  Uncertain: {metrics['uncertain']:,} ({metrics['abstention_pct']:.2f}%)")
        print(f"  Agreement: {metrics['agreement_pct']:.2f}%")
        print(f"  Time: {elapsed:.1f}s")
        delta_results.append(metrics)

    # Save results
    print("\n--- SAVING RESULTS ---")

    tau_path = SENSITIVITY_DIR / "tau_sensitivity.json"
    with open(tau_path, 'w') as f:
        json.dump(tau_results, f, indent=2)
    print(f"τ sensitivity saved to {tau_path}")

    delta_path = SENSITIVITY_DIR / "delta_sensitivity.json"
    with open(delta_path, 'w') as f:
        json.dump(delta_results, f, indent=2)
    print(f"δ sensitivity saved to {delta_path}")

    # Check if τ values produce identical outputs (integer evidence counts)
    print("\n--- INTEGER EVIDENCE CHECK ---")
    if tau_results[0]['labeled'] == tau_results[1]['labeled'] == tau_results[2]['labeled']:
        print("⚠ WARNING: All τ values produced identical labeled counts!")
        print("   This is expected because evidence counts are integers.")
        print("   τ=0.25, 0.35, 0.50 all correspond to 'evidence >= 1'")
    else:
        print("τ values produced different outputs.")

    # Summary table
    print("\n" + "=" * 80)
    print("SENSITIVITY SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Setting':<20} {'Labeled':>10} {'Coverage':>10} {'Abstention':>12} {'Agreement':>10}")
    print("-" * 62)
    for r in tau_results:
        print(f"{'tau='+str(r['tau']):<20} {r['labeled']:>10,} {r['coverage_pct']:>9.2f}% {r['abstention_pct']:>11.2f}% {r['agreement_pct']:>9.2f}%")
    for r in delta_results:
        print(f"{'delta='+str(r['delta']):<20} {r['labeled']:>10,} {r['coverage_pct']:>9.2f}% {r['abstention_pct']:>11.2f}% {r['agreement_pct']:>9.2f}%")

    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()