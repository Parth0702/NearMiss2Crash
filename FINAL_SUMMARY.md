# Phase 3B Common Test Comparison — Corrected

## Verdict: VALID COMPARISON

### Common Test Set
- **759 shared test samples** identified by intersecting crisp and fuzzy test `report_id`s (with ASRS_ prefix)
- All fuzzy-labeled test samples have matching crisp test counterparts
- **Label agreement**: 592/759 (78.0%) of crisp and fuzzy weak labels agree; 167/759 (22.0%) disagree
- **Reference label source**: `fuzzy_label` from `FuzzyWeakLabeler` with τ=0.35, δ=0.15

### Evaluation Results (on 759 common samples, reference=fuzzy labels)

| Metric | Model A (Crisp) | Model B (Fuzzy) | Δ |
|--------|-----------------|-----------------|---|
| Accuracy | 35.49% | **38.47%** | +2.98 pp |
| Macro-F1 | 21.25% | **21.89%** | +0.64 pp |
| Weighted-F1 | 28.91% | **29.78%** | +0.87 pp |

### Key Findings
- Both models evaluated on **identical 759-sample held-out test set** with explicit fuzzy-label reference
- **Fuzzy model (Model B) outperforms crisp model (Model A)** on the valid common test set
- Both models **fail to predict Violation and Perceptual Error** (zero predictions for both classes in both models)
- 78.0% label agreement between crisp/fuzzy weak labels shows substantial but not perfect overlap
- Comparison is valid for **proxy weak-label accuracy assessment** only

### Artifacts Generated
- `experiments/distilbert_extension/common_test_comparison.csv` — Per-sample comparison (759 samples)
- `experiments/distilbert_extension/common_test_metrics.json` — Aggregate metrics
- `PHASE_3B_COMMON_TEST_VALIDATION.md` — Full validation report
- `PHASE_3B_DISTILBERT_REPORT.md` — Corrected Phase 3B report

### Original vs Corrected
- **Original**: Model A on 872 samples vs Model B on 759 samples → **INCONCLUSIVE** (different populations)
- **Corrected**: Both models on 759 common samples with fuzzy reference → **VALID COMPARISON**

### Reference Artifacts (Preserved)
- `distilbert_comparison_metrics.json` — Original results (872 crisp / 759 fuzzy original populations)
- `models/model_a_crisp_head.pt` — Model A checkpoint
- `models/model_b_fuzzy_head.pt` — Model B checkpoint
- No models retrained; original checkpoints used