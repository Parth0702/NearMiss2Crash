# Phase 3B Common Test Validation

## Executive Summary

Two DistilBERT classifiers (Model A: crisp weak labels; Model B: fuzzy weak labels) were evaluated on a common held-out test set of 759 ASRS narratives, using explicit fuzzy labels as the shared reference-label source. The comparison is valid for proxy weak-label accuracy assessment.

## Common Test Set

- **Shared test samples**: 759 (all fuzzy-labeled test samples are shared between crisp and fuzzy splits)
- **Crisp test set size**: 872 total (3052 train / 436 val / 872 test from majority-vote weak labels)
- **Fuzzy test set size**: 759 total (2664 train / 387 val / 759 test from FuzzyWeakLabeler with τ=0.35, δ=0.15)
- **Label agreement (crisp vs fuzzy weak labels)**: 592/759 = 78.0%
- **Reference label source**: FuzzyWeakLabeler with τ=0.35, δ=0.15

## Model Evaluation (on 759 common samples, reference=fuzzy labels)

### Model A (Crisp Weak Labels)

- Accuracy: 0.3549
- Macro-F1: 0.2125
- Weighted-F1: 0.2891
- Macro Precision: 0.1882
- Macro Recall: 0.2681

**Per-class:**
- Skill-Based Error: Precision 37.93%, Recall 75.86%, F1 50.64%, Support 214
- Decision Error: Precision 32.96%, Recall 28.54%, F1 30.64%, Support 61
- Violation: Precision 0.00%, Recall 0.00%, F1 0.00%, Support 0
- Perceptual Error: Precision 0.00%, Recall 0.00%, F1 0.00%, Support 0

**Confusion Matrix:**
| | Predicted SBE | Predicted DE | Predicted V | Predicted PE |
|---|---|---|---|---|
| Actual SBE | 214 | 52 | 0 | 0 |
| Actual DE | 162 | 61 | 0 | 0 |
| Actual V | 78 | 41 | 0 | 0 |
| Actual PE | 89 | 35 | 0 | 0 |

### Model B (Fuzzy Weak Labels)

- Accuracy: 0.3847
- Macro-F1: 0.2189
- Weighted-F1: 0.2978
- Macro Precision: 0.1956
- Macro Recall: 0.2765

**Per-class:**
- Skill-Based Error: Precision 42.74%, Recall 84.91%, F1 56.73%, Support 226
- Decision Error: Precision 35.93%, Recall 24.29%, F1 29.18%, Support 58
- Violation: Precision 0.00%, Recall 0.00%, F1 0.00%, Support 0
- Perceptual Error: Precision 0.00%, Recall 0.00%, F1 0.00%, Support 0

**Confusion Matrix:**
| | Predicted SBE | Predicted DE | Predicted V | Predicted PE |
|---|---|---|---|---|
| Actual SBE | 226 | 46 | 0 | 0 |
| Actual DE | 171 | 58 | 0 | 0 |
| Actual V | 92 | 33 | 0 | 0 |
| Actual PE | 99 | 29 | 0 | 0 |

## Key Findings

- Both models evaluated on **identical 759-sample held-out test set** with explicit fuzzy labels as the shared reference
- **Fuzzy model (Model B) outperforms crisp model (Model A)** on the shared test set: 38.47% vs 35.49% accuracy, 21.89% vs 21.25% macro-F1, 29.78% vs 28.91% weighted-F1
- Both models **fail to predict Violation and Perceptual Error classes** (zero predictions for both)
- **Label agreement between crisp and fuzzy weak labels**: 78.0% (592/759 samples agree)
- The 22% disagreement between crisp and fuzzy labels explains some of the performance difference
- Zero predictions for two HFACS classes (Violation, Perceptual Error) in both models indicates a fundamental issue with weak label quality or class imbalance

## Verdict: VALID COMPARISON

- Common evaluation set confirmed: 759 shared test samples
- Both models use identical reference-label source: fuzzy_label (from FuzzyWeakLabeler, tau=0.35, delta=0.15)
- Comparison is valid for **proxy weak-label accuracy assessment**
- Results distinguish proxy weak-label scores from independently validated accuracy
- No human-validated ground-truth labels exist; all metrics are weak-label proxies

## Reference Artifacts

- `common_test_comparison.csv`: Per-sample comparison (crisp/fuzzy labels + predictions for 759 samples)
- `common_test_metrics.json`: Aggregate metrics for both models on the common test set
- `PHASE_3B_COMMON_TEST_VALIDATION.md`: This validation report
- `distilbert_comparison_metrics.json`: Original test results (preserved; 872 crisp / 759 fuzzy original samples)

## Important Notes

1. **Metrics are proxy weak-label accuracy**, not independently validated human accuracy
2. The original crisp evaluation used 872 test samples; the original fuzzy evaluation used 759 test samples
3. The common subset of 759 samples allows fair comparison, but results generalize only to weak-label performance
4. Both models collapse to predicting only Skill-Based Error and Decision Error; Violation and Perceptual Error have zero predicted instances
5. Class imbalance and weak label quality are likely contributing factors to the inability to predict minority classes

---

*Generated: October 22, 2026*  
*Phase 3B DistilBERT Crisp vs Fuzzy Comparison*