# flight_phase_analysis.py
"""
Week 4 – Flight Phase Analysis

This script builds a weighted keyword list for flight phases (from the build plan),
assigns a phase to each incident report, and creates a heatmap of HFACS error
class vs. flight phase.  All results are saved under `week4_results/flight_phase/`.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Weighted phase keywords (from the build plan)
# ---------------------------------------------------------------------------
PHASE_KEYWORDS = {
    # Paper‑derived weights
    "runway": 2.96,
    "aircraft": 1.55,
    "landing": 1.18,
    "right": 1.18,
    "airport": 1.00,
    "approach": 0.96,
    "left": 0.96,
    "pattern": 0.86,
    "around": 0.73,
    "ctaf": 0.73,
    "tower": 0.73,
    "winds": 0.73,
    # Standard phase vocabulary (weight 1.0)
    "taxi": 1.0,
    "takeoff": 1.0,
    "climb": 1.0,
    "cruise": 1.0,
    "descent": 1.0,
    "final": 1.0,
    "go-around": 1.0,
    "touchdown": 1.0,
}

HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error",
]

# ---------------------------------------------------------------------------
# Load test split (same as used for evaluation)
# ---------------------------------------------------------------------------
TEST_CSV = "weak_labeling_results/dataset_majority_test.csv"
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Test split not found: {TEST_CSV}")

df = pd.read_csv(TEST_CSV)
df["text"] = df["text"].astype(str)

# ---------------------------------------------------------------------------
# Assign a phase to each report (simple keyword match with highest weight)
# ---------------------------------------------------------------------------
def assign_phase(text: str) -> str:
    text_low = text.lower()
    best_phase = "unknown"
    best_score = -1.0
    for kw, weight in PHASE_KEYWORDS.items():
        if kw in text_low:
            if weight > best_score:
                best_score = weight
                best_phase = kw
    return best_phase

df["phase"] = df["text"].apply(assign_phase)
df["label"] = df["label"].astype(str)

# ---------------------------------------------------------------------------
# Build contingency table (HFACS class vs. phase)
# ---------------------------------------------------------------------------
contingency = pd.crosstab(df["label"], df["phase"], normalize="index")
# Ensure all possible phases appear (even if zero)
contingency = contingency.reindex(columns=sorted(PHASE_KEYWORDS.keys()), fill_value=0)

# ---------------------------------------------------------------------------
# Plot heatmap
# ---------------------------------------------------------------------------
out_dir = "week4_results/flight_phase"
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(12, 6))
sns.heatmap(
    contingency,
    cmap="YlOrRd",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Row‑wise %"},
)
plt.title("HFACS Error Class vs. Flight Phase (row‑wise %)")
plt.ylabel("HFACS Class")
plt.xlabel("Flight Phase (keyword)")
plt.tight_layout()
heatmap_path = os.path.join(out_dir, "hfacs_phase_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

# Save the raw table for reference
contingency.to_csv(os.path.join(out_dir, "hfacs_phase_table.csv"))

print(f"Flight-phase analysis completed. Heatmap saved to {heatmap_path}")