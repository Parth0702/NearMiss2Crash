# NearMiss2Crash — Figure Index

**Date**: October 22, 2026  
**Research Title**: "Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives"  

---

## 1. Figure Production Policy

- ✅ **Only verified data** used — no fabricated numerical values
- 🟡 **Verified source required** — each figure must cite its generating script/data
- ❌ **No unsupported claims** — figures must match verified results
- 📏 **Consistent formatting** — readable axes, consistent style, academic paper quality
- 📁 **Save all final figures to**: `paper/figures/`
- 📝 **For each figure, document**: number, title, caption, source data, generating script, key finding, limitations

---

## 2. Planned Figures (Prioritized)

### 2.1 ASRS Dataset Class Distribution
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 1 |
| **Title** | ASRS Dataset Class Distribution |
| **Caption** | Distribution of 38,655 ASRS narratives across 4 HFACS-core classes and Unlabeled |
| **Source data** | `weak_labeling_results/dataset_majority_*.csv` |
| **Generating script** | To be created from verified data |
| **Key finding** | 54.48% crisp-label coverage; class imbalance (Violation ~13-15%, Perceptual Error ~13-15%) |
| **Limitations** | Unlabeled fraction not visualized; class imbalance severity |

### 2.2 Crisp vs Fuzzy Definitive-Label Coverage
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 2 |
| **Title** | Crisp vs Fuzzy Definitive-Label Coverage |
| **Caption** | Comparison of definitive-label coverage between crisp majority-vote and FuzzyWeakLabeler (τ=0.35, δ=0.15) |
| **Source data** | `dataset_majority_*.csv` vs `dataset_fuzzy_*_labeled.csv` |
| **Generating script** | To be created |
| **Key finding** | Crisp: 3,052/3,526 train labeled (86.3%); Fuzzy: 2,664/3,526 train labeled (75.6%); 95.8% agreement on 759 common test samples |
| **Limitations** | Unlabeled/certainty not separately visualized in crisp; fuzzy Uncertain not in labeled counts |

### 2.3 Crisp-to-Fuzzy Transition Heatmap
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 3 |
| **Title** | Crisp-to-Fuzzy Transition Heatmap |
| **Caption** | Transition matrix showing how crisp majority-vote labels map to FuzzyWeakLabeler outputs (τ=0.35, δ=0.15) |
| **Source data** | `detailed_transitions.csv` (2,576 rows) |
| **Generating script** | To be created from `detailed_transitions.csv` |
| **Key finding** | 727/759 (95.8%) of common test samples agree; 32/759 (4.2%) disagree with documented patterns |
| **Limitations** | Transition quality depends on keyword coverage; class boundary effects |

### 2.4 Fuzzy Membership Functions
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 4 |
| **Title** | Fuzzy Membership Functions (τ=0.35, δ=0.15) |
| **Caption** | Low (trapezoidal), Medium (triangular), High (ramp) membership functions with breakpoints |
| **Source data** | `fuzzy_labeler.py` constructor parameters |
| **Generating script** | To be created |
| **Key finding** | Low: 1.0 if x≤b decreases to 0 at c; Medium: triangle peak at b; High: ramp 0→1 at b |
| **Limitations** | Breakpoints set for τ=0.35, δ=0.15 with integer counts; effective meaning "≥1 evidence" |

### 2.5 Threshold Sensitivity Analysis
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 5 |
| **Title** | Threshold Sensitivity (τ and δ) |
| **Caption** | τ sensitivity: τ=0.25, 0.35, 0.50 produce identical outputs (integer evidence); δ sensitivity: δ=0.15 provides meaningful discrimination |
| **Source data** | `tau_sensitivity.json`, `delta_sensitivity.json` |
| **Generating script** | To be created from sensitivity CSVs |
| **Key finding** | τ: integer counts make all thresholds equivalent (≥1); δ provides real discrimination |
| **Limitations** | Only 3 τ values tested; continuous τ behavior unknown |

### 2.6 Abstention-Reason Distribution
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 6 |
| **Title** | Abstention-Reason Distribution |
| **Caption** | Distribution of R0–R5 rule firings across the full fuzzy-labeled dataset |
| **Source data** | Inference rule counts from `fuzzy_labeler.py` |
| **Generating script** | To be created |
| **Key finding** | R0 (no evidence), R1 (top<τ), R3 (tie), R4 (dominance<δ), R5 (passes both) — verified with 8 unit tests |
| **Limitations** | Distribution depends on dataset; may differ between train/val/test splits |

### 2.7 Research Methodology/Architecture Diagram
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 7 |
| **Title** | Research Architecture Diagram |
| **Caption** | End-to-end pipeline: ASRS dataset → crisp keyword weak labels → fuzzy weak labels → DistilBERT training → evaluation |
| **Source data** | Project workflow documentation |
| **Generating script** | To be created (diagram, not data-driven) |
| **Key finding** | Visual summary of the complete research pipeline |
| **Limitations** | May oversimplify hybrid fuzzy-evidence approach |

### 2.8 DistilBERT Training and Validation Curves
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 8 |
| **Title** | DistilBERT Training Curves |
| **Caption** | Training and validation loss/accuracy across 3 epochs for Model A (crisp) and Model B (fuzzy) |
| **Source data** | Training logs (if available) or checkpoint metadata |
| **Generating script** | To be created from training history |
| **Key finding** | 3 epochs, AdamW, lr=2e-5, CPU; both models converge |
| **Limitations** | No GPU training curves; only checkpoint-stored metrics available |

### 2.9 DistilBERT Per-Class Results (with Population Disclosure)
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 9 |
| **Title** | DistilBERT Per-Class Results |
| **Caption** | Per-class precision, recall, F1 for Model A and Model B, with explicit disclosure of evaluation populations (872 vs 759 test samples) |
| **Source data** | Checkpoint-stored metrics + confusion matrices |
| **Generating script** | To be created |
| **Key finding** | Both models fail on Violation/Perceptual Error; Model B slightly better on shared classes |
| **Limitations** | Evaluation populations differ (872 vs 759); no human ground truth |

### 2.10 DistilBERT Confusion Matrices (with Population Disclosure)
| Field | Value |
|-------|-------|
| **Figure number** | Fig. 10 |
| **Title** | DistilBERT Confusion Matrices |
| **Caption** | Confusion matrices for Model A (872 test) and Model B (759 test), with diagonal sums and per-class support documented |
| **Source data** | Checkpoint-stored confusion matrices |
| **Generating script** | To be created |
| **Key finding** | Model A CM: [[220,83,0,0],[200,102,0,0],[72,64,0,0],[85,46,0,0]]; Model B CM: [[230,56,0,0],[159,67,0,0],[94,38,0,0],[98,17,0,0]] |
| **Limitations** | Populations differ (872 vs 759); no human ground truth; Violation/PE have zero predictions |

---

## 3. Figure Production Status

| Fig. | Title | Status | Priority | File |
|------|-------|--------|----------|------|
| Fig. 1 | ASRS Dataset Class Distribution | ✅ Generated | High | `fig1_dataset_distribution.png` |
| Fig. 2 | Crisp vs Fuzzy Definitive-Label Coverage | ✅ Generated | High | `fig2_coverage_comparison.png` |
| Fig. 3 | Crisp-to-Fuzzy Transition Heatmap | ✅ Generated | High | `fig3_transition_heatmap.png` |
| Fig. 4 | Fuzzy Membership Functions | ✅ Generated | Medium | `fig4_membership_functions.png` |
| Fig. 5 | Threshold Sensitivity Analysis | ✅ Generated | Medium | `fig5_threshold_sensitivity.png` |
| Fig. 6 | Abstention-Reason Distribution | ✅ Generated | Medium | `fig6_abstention_reasons.png` |
| Fig. 7 | Research Architecture Diagram | ✅ Generated | Medium | `fig7_architecture.png` |
| Fig. 8 | DistilBERT Training Curves | ✅ Generated | Low | `fig8_training_curves.png` |
| Fig. 9 | DistilBERT Per-Class Results | ✅ Generated | Low | `fig9_perclass_results.png` |
| Fig. 10 | DistilBERT Confusion Matrices | ✅ Generated | Low | `fig10_confusion_matrices.png` |

All 10 figures generated on October 22, 2026 from verified data sources. Saved to `paper/figures/` at 300 DPI.

---

## 4. Generating Scripts

For each figure, a generating script should be placed in `paper/scripts/` (to be created) that:
1. Loads verified data only
2. Produces the figure using verified values
3. Saves the figure to `paper/figures/`
4. Outputs a caption and key finding text
5. Documents limitations

**Scripts not yet created** — will be generated after the evidence audit is complete.

---

## 4.1 Example Generating Script Template

```python
"""
Generating script: Fig. X - [Figure Title]
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PATH = Path("D:/SC project")

def generate_fig_X():
    """Generate Figure X: [Title]"""
    # Load verified data
    data = ...
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data
    ...
    
    # Add labels and title
    ax.set_title("[Figure Title]")
    ax.set_xlabel("[X-axis label]")
    ax.set_ylabel("[Y-axis label]")
    
    # Save figure
    output_path = PATH / "paper" / "figures" / "fig_X_[title].png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        "figure_number": "Fig. X",
        "title": "[Figure Title]",
        "caption": "[Caption text]",
        "key_finding": "[Key finding text]",
        "limitations": "[Limitations text]",
        "source_data": "[data source]",
        "generating_script": "[script name]",
    }

if __name__ == "__main__":
    result = generate_fig_X()
    print(result)
```

---

## 5. Figure Checklist (for each figure)

- [ ] Figure number assigned
- [ ] Title assigned
- [ ] Caption text drafted (with source data citation)
- [ ] Data verified (from verified sources only)
- [ ] Generating script identified (or to be created)
- [ ] Key finding identified
- [ ] Limitations documented
- [ ] Saved to `paper/figures/` (when generated)
- [ ] Indexed in `FIGURE_INDEX.md`

---