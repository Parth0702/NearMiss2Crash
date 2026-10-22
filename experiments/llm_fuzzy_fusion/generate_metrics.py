import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

HFACS_CLASSES = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]

# Load data
cache = json.load(open("gemini_pilot_cache.json"))
df_fuzzy = pd.read_csv("../fuzzy_asrs/results/fuzzy_full_results.csv")

# Get the 10 report IDs from cache
report_ids = list(cache.keys())
print(f"Reports in cache: {len(report_ids)}")

# For each report, get all data
results = []

for report_id in report_ids:
    # Gemini data
    gemini_data = cache[report_id]
    gemini_resp = gemini_data['gemini_response']
    gemini_label = gemini_resp['proposed_label']
    gemini_conf = gemini_resp['confidence']
    gemini_evidence = gemini_resp['hfacs_evidence']

    # Find fuzzy data
    fuzzy_row = df_fuzzy[df_fuzzy['report_id'] == report_id]
    if len(fuzzy_row) > 0:
        fuzzy_row = fuzzy_row.iloc[0]
        fuzzy_label = fuzzy_row['fuzzy_label']
        fuzzy_conf = float(fuzzy_row['confidence'])
        fuzzy_dom = float(fuzzy_row['dominance'])
        fuzzy_rule = fuzzy_row['rule_fired']
        fuzzy_evidence_str = fuzzy_row['evidence_counts']
        # Parse evidence counts
        fuzzy_evidence = {}
        if isinstance(fuzzy_evidence_str, str):
            for part in fuzzy_evidence_str.split('|'):
                if ':' in part:
                    cls, cnt = part.split(':')
                    fuzzy_evidence[cls] = int(cnt)
    else:
        fuzzy_label = "Unknown"
        fuzzy_conf = 0.0
        fuzzy_dom = 0.0
        fuzzy_rule = "Unknown"
        fuzzy_evidence = {}

    # Crisp data
    crisp_row = df_fuzzy[df_fuzzy['report_id'] == report_id]
    if len(crisp_row) > 0:
        crisp_row = crisp_row.iloc[0]
        crisp_label = crisp_row['crisp_label']
        crisp_evidence_str = crisp_row['matched_keywords']
        crisp_evidence_counts = {c: 0 for c in HFACS_CLASSES}
        if isinstance(crisp_evidence_str, str):
            for part in crisp_evidence_str.split('|'):
                if ':' in part:
                    cls, kw = part.split(':')
                    if kw:
                        crisp_evidence_counts[cls] = len(kw.split(','))
    else:
        crisp_label = "Unknown"
        crisp_evidence_counts = {c: 0 for c in HFACS_CLASSES}

    # Fusion calculation
    # Normalize fuzzy evidence
    total_fuzzy = sum(fuzzy_evidence.values())
    if total_fuzzy > 0 and max(fuzzy_evidence.values()) > 0:
        max_fuzzy = max(fuzzy_evidence.values())
        F_scores = {c: fuzzy_evidence.get(c, 0) / max_fuzzy for c in HFACS_CLASSES}
    else:
        F_scores = {c: 0.0 for c in HFACS_CLASSES}

    # Gemini scores
    L_scores = {}
    for item in gemini_evidence:
        L_scores[item['class']] = item['evidence_score']
    # Fill missing with 0
    for c in HFACS_CLASSES:
        if c not in L_scores:
            L_scores[c] = 0.0

    # Fuse with alpha=0.5
    alpha = 0.5
    S_scores = {}
    for c in HFACS_CLASSES:
        F_c = F_scores.get(c, 0.0)
        L_c = L_scores.get(c, 0.0)
        S_c = alpha * F_c + (1 - alpha) * L_c
        S_scores[c] = S_c

    # Check thresholds
    ranked = sorted(HFACS_CLASSES, key=lambda c: S_scores[c], reverse=True)
    top_class = ranked[0]
    second_class = ranked[1] if len(ranked) > 1 else None
    top_score = S_scores[top_class]
    second_score = S_scores[second_class] if second_class else 0.0
    dominance_gap = top_score - second_score

    # Fusion decision
    abstained = False
    fused_label = top_class
    fused_confidence = top_score

    if top_score < 0.25:
        fused_label = "Uncertain"
        fused_confidence = 0.0
        abstained = True
    elif dominance_gap < 0.10:
        fused_label = "Uncertain"
        fused_confidence = 0.0
        abstained = True

    # Change from fuzzy
    change_from_fuzzy = (fused_label != fuzzy_label) if not abstained else False

    result = {
        "report_id": report_id,
        "narrative_preview": f"Report {report_id[:9]}",

        "crisp": {
            "label": crisp_label,
            "evidence_counts": crisp_evidence_counts,
            "abstained": crisp_label == "Unlabeled"
        },

        "fuzzy": {
            "label": fuzzy_label,
            "confidence": fuzzy_conf,
            "dominance": fuzzy_dom,
            "rule_fired": fuzzy_rule,
            "evidence_counts": fuzzy_evidence,
            "abstained": fuzzy_label == "Uncertain"
        },

        "gemini": {
            "label": gemini_label,
            "confidence": gemini_conf,
            "evidence_scores": {item['class']: item['evidence_score'] for item in gemini_evidence},
            "abstained": gemini_label == "Uncertain" if gemini_label != gemini_label else False
        },

        "fusion": {
            "label": fused_label,
            "confidence": fused_confidence,
            "fused_scores": S_scores,
            "alpha": alpha,
            "abstained": abstained,
            "change_from_fuzzy": change_from_fuzzy
        }
    }
    results.append(result)

# Calculate metrics
def calc_metrics(results, system):
    total = len(results)

    if system == "crisp":
        labeled = sum(1 for r in results if r["crisp"]["label"] != "Unknown" and r["crisp"]["label"] != "Unlabeled")
        abstentions = sum(1 for r in results if r["crisp"]["abstained"])
        coverage = labeled / total if total > 0 else 0
        class_dist = {}
        for r in results:
            if r["crisp"]["label"] != "Unknown" and r["crisp"]["label"] != "Unlabeled":
                class_dist[r["crisp"]["label"]] = class_dist.get(r["crisp"]["label"], 0) + 1

    elif system == "fuzzy":
        labeled = sum(1 for r in results if r["fuzzy"]["label"] != "Uncertain")
        abstentions = sum(1 for r in results if r["fuzzy"]["abstained"])
        coverage = labeled / total if total > 0 else 0
        class_dist = {}
        for r in results:
            if r["fuzzy"]["label"] != "Uncertain":
                class_dist[r["fuzzy"]["label"]] = class_dist.get(r["fuzzy"]["label"], 0) + 1

    elif system == "gemini":
        labeled = sum(1 for r in results if r["gemini"]["label"] != "Uncertain" and r["gemini"]["label"] != "Unknown")
        abstentions = sum(1 for r in results if r["gemini"]["label"] == "Uncertain")
        coverage = labeled / total if total > 0 else 0
        class_dist = {}
        for r in results:
            if r["gemini"]["label"] not in ["Uncertain", "Unknown"]:
                class_dist[r["gemini"]["label"]] = class_dist.get(r["gemini"]["label"], 0) + 1

    elif system == "fusion":
        labeled = sum(1 for r in results if not r["fusion"]["abstained"])
        abstentions = sum(1 for r in results if r["fusion"]["abstained"])
        coverage = labeled / total if total > 0 else 0
        class_dist = {}
        for r in results:
            if not r["fusion"]["abstained"]:
                class_dist[r["fusion"]["label"]] = class_dist.get(r["fusion"]["label"], 0) + 1

    return {
        "total": total,
        "definitive": labeled,
        "coverage": coverage,
        "abstentions": abstentions,
        "abstention_rate": abstentions / total if total > 0 else 0,
        "class_distribution": class_dist,
        "verified": labeled + abstentions == total
    }

metrics = {}
for sys in ["crisp", "fuzzy", "gemini", "fusion"]:
    metrics[sys] = calc_metrics(results, sys)
    print(f"{sys.upper():>10}: total={metrics[sys]['total']:3d} definitive={metrics[sys]['definitive']:3d} ({metrics[sys]['coverage']*100:5.1f}%) abstentions={metrics[sys]['abstentions']:3d} ({metrics[sys]['abstention_rate']*100:4.1f}%) verified={metrics[sys]['verified']}")

# Save corrected metrics
corrected_metrics = {
    "audit_date": "2026-10-22",
    "experiment_size": 10,
    "note": "Metrics recalculated from verified 10-report pilot data only. 148-report experiment unsubstantiated.",
    "metrics": metrics
}

with open("phase3d_corrected_metrics.json", "w") as f:
    json.dump(corrected_metrics, f, indent=2)

print("\nCorrected metrics saved to: phase3d_corrected_metrics.json")

# Save sample manifest
manifest = {
    "audit_date": "2026-10-22",
    "experiment_size": 10,
    "reports": results,
    "note": "Sample manifest from verified 10-report pilot. 148-report experiment unsubstantiated."
}

with open("phase3d_sample_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Sample manifest saved to: phase3d_sample_manifest.json")