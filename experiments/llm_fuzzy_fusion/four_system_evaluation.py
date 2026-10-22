#!/usr/bin/env python3
"""
NearMiss2Crash — Phase 3D: Four-System Evaluation

Compare on exactly the same 10 reports:
1. Crisp keyword labeling
2. Original fuzzy labeling
3. Gemini-only labeling
4. Gemini + fuzzy fusion

Records all required metrics without reporting classification accuracy
without independent reference labels.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# Import fusion labeler
from fusion_labeler import (
    FusionLabeler, FuzzyEvidence, GeminiEvidence,
    load_fuzzy_data, load_gemini_data, load_crisp_label,
    HFACS_CLASSES
)


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = PROJECT_ROOT / "experiments" / "llm_fuzzy_fusion" / "gemini_pilot_cache.json"
FUZZY_FULL_RESULTS = PROJECT_ROOT / "experiments" / "fuzzy_asrs" / "results" / "fuzzy_full_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "experiments" / "llm_fuzzy_fusion" / "four_system_evaluation.json"


def load_crisp_evidence(report_id: str, df_fuzzy: pd.DataFrame) -> Dict[str, Any]:
    """Get crisp keyword evidence for a report."""
    row = df_fuzzy[df_fuzzy['report_id'] == report_id]
    if len(row) == 0:
        return {"label": "Unknown", "matched_keywords": {}, "evidence_counts": {}}

    row = row.iloc[0]

    # Parse keyword matches from matched_keywords
    keyword_str = row.get('matched_keywords', '')
    matched_keywords = {}
    evidence_counts = {c: 0 for c in HFACS_CLASSES}

    for part in keyword_str.split('|'):
        if ':' in part:
            cls, keywords = part.split(':')
            if keywords:
                matched_keywords[cls] = keywords.split(',')
                evidence_counts[cls] = len(keywords.split(','))
            else:
                matched_keywords[cls] = []

    return {
        "label": row.get('crisp_label', 'Unlabeled'),
        "matched_keywords": matched_keywords,
        "evidence_counts": evidence_counts
    }


def evaluate_fusion_change(
    fusion_result,
    fuzzy_label: str,
    gemini_label: str,
    narrative: str
) -> Dict[str, Any]:
    """Evaluate whether fusion change from fuzzy is supported by narrative."""
    change_info = {
        "changed": fusion_result.change_from_fuzzy,
        "fuzzy_to_fusion": None,
        "gemini_agrees_with_fusion": False,
        "narrative_supports_change": None
    }

    if fusion_result.change_from_fuzzy:
        change_info["fuzzy_to_fusion"] = f"{fuzzy_label} -> {fusion_result.fused_label}"
        change_info["gemini_agrees_with_fusion"] = (gemini_label == fusion_result.fused_label)

        # Simple heuristic: if Gemini agrees with fusion and has high confidence,
        # the narrative likely supports the change
        if change_info["gemini_agrees_with_fusion"] and fusion_result.fused_confidence > 0.3:
            change_info["narrative_supports_change"] = "likely"
        elif change_info["gemini_agrees_with_fusion"]:
            change_info["narrative_supports_change"] = "possible"
        else:
            change_info["narrative_supports_change"] = "unclear"

    return change_info


def main():
    print("=" * 70)
    print("NearMiss2Crash — Phase 3D: Four-System Evaluation")
    print("=" * 70)

    # Load data
    print("\n[LOAD] Loading data...")
    df_fuzzy = pd.read_csv(FUZZY_FULL_RESULTS)

    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)

    # Get the 10 report IDs from cache
    report_ids = list(cache.keys())
    print(f"  Evaluating {len(report_ids)} reports")

    # Initialize fusion labeler
    fusion = FusionLabeler(alpha=0.5, min_support=0.25, min_dominance_gap=0.10)
    print(f"  Fusion: alpha=0.5, min_support=0.25, min_dominance_gap=0.10")

    results = []

    for report_id in report_ids:
        print(f"\n[EVAL] {report_id}...")

        # Get narrative text
        row = df_fuzzy[df_fuzzy['report_id'] == report_id]
        if len(row) == 0:
            print(f"  [WARN] Report not found in fuzzy data")
            continue
        narrative = row.iloc[0]['text']

        # Load evidence from all systems
        fuzzy_ev = load_fuzzy_data(report_id, df_fuzzy)
        gemini_ev = load_gemini_data(report_id, cache)
        crisp_ev = load_crisp_evidence(report_id, df_fuzzy)

        if fuzzy_ev is None or gemini_ev is None:
            print(f"  [WARN] Missing evidence")
            continue

        # 1. Crisp keyword labeling
        crisp_label = crisp_ev["label"]
        crisp_evidence_counts = crisp_ev["evidence_counts"]
        crisp_matched = crisp_ev["matched_keywords"]

        # 2. Original fuzzy labeling
        fuzzy_label = fuzzy_ev.fuzzy_label
        fuzzy_confidence = fuzzy_ev.confidence
        fuzzy_dominance = fuzzy_ev.dominance
        fuzzy_rule = fuzzy_ev.rule_fired

        # 3. Gemini-only labeling
        gemini_label = gemini_ev.proposed_label
        gemini_confidence = gemini_ev.confidence
        gemini_scores = gemini_ev.evidence_scores
        gemini_excerpts = gemini_ev.supporting_excerpts

        # 4. Fusion labeling
        fusion_result = fusion.fuse(fuzzy_ev, gemini_ev, report_id)
        fused_label = fusion_result.fused_label
        fused_confidence = fusion_result.fused_confidence
        fused_scores = fusion_result.fused_scores
        fused_abstained = fusion_result.abstained

        # Evaluate fusion change
        change_info = evaluate_fusion_change(
            fusion_result, fuzzy_label, gemini_label, narrative
        )

        # Build result record
        result = {
            "report_id": report_id,
            "narrative_preview": narrative[:200] + "..." if len(narrative) > 200 else narrative,

            # System 1: Crisp
            "crisp": {
                "label": crisp_label,
                "evidence_counts": crisp_evidence_counts,
                "matched_keywords": crisp_matched,
                "abstained": crisp_label == "Unlabeled"
            },

            # System 2: Fuzzy
            "fuzzy": {
                "label": fuzzy_label,
                "confidence": fuzzy_confidence,
                "dominance": fuzzy_dominance,
                "rule_fired": fuzzy_rule,
                "evidence_counts": fuzzy_ev.evidence_counts,
                "abstained": fuzzy_label == "Uncertain"
            },

            # System 3: Gemini
            "gemini": {
                "label": gemini_label,
                "confidence": gemini_confidence,
                "evidence_scores": gemini_scores,
                "supporting_excerpts": gemini_excerpts,
                "abstained": gemini_label == "Uncertain"
            },

            # System 4: Fusion
            "fusion": {
                "label": fused_label,
                "confidence": fused_confidence,
                "fused_scores": fused_scores,
                "alpha": fusion_result.alpha,
                "abstained": fused_abstained,
                "change_from_fuzzy": change_info["changed"],
                "change_detail": change_info["fuzzy_to_fusion"],
                "gemini_agrees_with_fusion": change_info["gemini_agrees_with_fusion"],
                "narrative_supports_change": change_info["narrative_supports_change"]
            }
        }

        results.append(result)

        # Print summary
        print(f"  Crisp:   {crisp_label}")
        print(f"  Fuzzy:   {fuzzy_label} (conf={fuzzy_confidence:.2f}, rule={fuzzy_rule})")
        print(f"  Gemini:  {gemini_label} (conf={gemini_confidence:.2f})")
        print(f"  Fusion:  {fused_label} (conf={fused_confidence:.2f})" +
              (" [ABSTAINED]" if fused_abstained else "") +
              (f" CHANGED: {change_info['fuzzy_to_fusion']}" if change_info["changed"] else ""))

    # Save results
    output = {
        "metadata": {
            "num_reports": len(results),
            "fusion_config": {
                "alpha": 0.5,
                "min_support": 0.25,
                "min_dominance_gap": 0.10
            },
            "note": "No classification accuracy reported without independent reference labels"
        },
        "results": results
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    # Summary statistics
    print("\n" + "=" * 70)
    print("FOUR-SYSTEM EVALUATION COMPLETE")
    print("=" * 70)

    # Agreement matrix
    print("\nLabel Agreement Summary:")
    labels = ["Crisp", "Fuzzy", "Gemini", "Fusion"]
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            agree = sum(1 for r in results if r[l1.lower()]["label"] == r[l2.lower()]["label"])
            total = len(results)
            print(f"  {l1} vs {l2}: {agree}/{total} ({agree/total*100:.1f}%)")

    # Fusion changes
    changes = [r for r in results if r["fusion"]["change_from_fuzzy"]]
    print(f"\nFusion changed Fuzzy prediction: {len(changes)}/{len(results)} reports")
    for r in changes:
        detail = r["fusion"]["change_detail"]
        gemini_agrees = r["fusion"]["gemini_agrees_with_fusion"]
        narrative_support = r["fusion"]["narrative_supports_change"]
        print(f"  {r['report_id']}: {detail} (Gemini agrees: {gemini_agrees}, Narrative support: {narrative_support})")

    # Abstentions
    for system in labels:
        key = system.lower()
        abstained = sum(1 for r in results if r[key]["abstained"])
        print(f"  {system} abstained: {abstained}/{len(results)}")

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()