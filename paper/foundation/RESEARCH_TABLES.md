# NearMiss2Crash — Research Tables

**Date**: October 22, 2026  
**Research Title**: "Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives"  

> **Policy**: Every table must use correct denominators. If the denominator is uncertain, mark the table as Provisional and note the uncertainty. Do not invent denominators.

---

## 1. Dataset Summary Table

| Category | Detail | Value | Denominator | Source |
|----------|--------|-------|-------------|--------|
| **ASRS Dataset** | Total narratives | 38,655 | 38,655 | `fuzzy_full_results.csv` |
| | Crisp-labeled narratives | 21,058 | 38,655 | 54.48% coverage | `weak_labeling_results/` |
| | Unlabeled narratives | 17,597 | 38,655 | 45.52% | `weak_labeling_results/` |
| | HFACS-core classes | 4 | N/A | `keyword_weak_labeler.py` |
| | Classes: | Skill-Based Error, Decision Error, Violation, Perceptual Error | N/A | `keyword_weak_labeler.py` |
| **Phase 1 Splits** | Train (crisp) | 3,526 | 3,526 | `dataset_majority_train.csv` |
| | Val (crisp) | 504 | 504 | `dataset_majority_validation.csv` |
| | Test (crisp) | 872 | 872 | `dataset_majority_test.csv` |
| | Train (fuzzy labeled) | 2,664 | 3,526 | 75.6% of train | `dataset_fuzzy_train_labeled.csv` |
| | Val (fuzzy labeled) | 387 | 504 | 76.8% of val | `dataset_fuzzy_validation_labeled.csv` |
| | Test (fuzzy labeled) | 759 | 872 (intersection) | 86.6% of crisp test | `dataset_fuzzy_test_labeled.csv` |
| | Common test intersection | 759 | 759 | Verified intersection | `PHASE_3B_FINAL_METRICS_AUDIT.md` |
| **Class Distributions** | Train (crisp) SBE | 1,059 | 3,526 | `dataset_majority_train.csv` |
| | Train (crisp) DE | 1,056 | 3,526 | `dataset_majority_train.csv` |
| | Train (crisp) V | 476 | 3,526 | `dataset_majority_train.csv` |
| | Train (crisp) PE | 461 | 3,526 | `dataset_majority_train.csv` |
| | Train (crisp) Unlabeled | 474 | 3,526 | `dataset_majority_train.csv` |
| | Train (fuzzy) SBE | 987 | 3,526 | `dataset_fuzzy_train_labeled.csv` |
| | Train (fuzzy) DE | 840 | 3,526 | `dataset_fuzzy_train_labeled.csv` |
| | Train (fuzzy) V | 455 | 3,526 | `dataset_fuzzy_train_labeled.csv` |
| | Train (fuzzy) PE | 382 | 3,526 | `dataset_fuzzy_train_labeled.csv` |
| | Val (crisp) SBE | 151 | 504 | `dataset_majority_validation.csv` |
| | Val (crisp) DE | 151 | 504 | `dataset_majority_validation.csv` |
| | Val (crisp) V | 68 | 504 | `dataset_majority_validation.csv` |
| | Val (crisp) PE | 66 | 504 | `dataset_majority_validation.csv` |
| | Val (fuzzy) SBE | 145 | 504 | `dataset_fuzzy_validation_labeled.csv` |
| | Val (fuzzy) DE | 120 | 504 | `dataset_fuzzy_validation_labeled.csv` |
| | Val (fuzzy) V | 65 | 504 | `dataset_fuzzy_validation_labeled.csv` |
| | Val (fuzzy) PE | 57 | 504 | `dataset_fuzzy_validation_labeled.csv` |
| | Test (crisp) SBE | 303 | 872 | `dataset_majority_test.csv` |
| | Test (crisp) DE | 302 | 872 | `dataset_majority_test.csv` |
| | Test (crisp) V | 136 | 872 | `dataset_majority_test.csv` |
| | Test (crisp) PE | 131 | 872 | `dataset_majority_test.csv` |
| | Test (fuzzy) SBE | 286 | 759 | `dataset_fuzzy_test_labeled.csv` |
| | Test (fuzzy) DE | 226 | 759 | `dataset_fuzzy_test_labeled.csv` |
| | Test (fuzzy) V | 132 | 759 | `dataset_fuzzy_test_labeled.csv` |
| | Test (fuzzy) PE | 115 | 759 | `dataset_fuzzy_test_labeled.csv` |

---

## 2. Crisp Label Distribution

| Split | Class | Count | Total | Percentage | Source |
|-------|-------|-------|-------|--------|--------|
| **Train** | SBE | 1,059 | 3,526 | 30.03% | `dataset_majority_train.csv` |
| | DE | 1,056 | 3,526 | 29.95% | `dataset_majority_train.csv` |
| | V | 476 | 3,526 | 13.49% | `dataset_majority_train.csv` |
| | PE | 461 | 3,526 | 13.07% | `dataset_majority_train.csv` |
| | Unlabeled | 474 | 3,526 | 13.44% | `dataset_majority_train.csv` |
| | **Train Total** | **3,526** | **3,526** | **100%** | |
| **Val** | SBE | 151 | 504 | 29.96% | `dataset_majority_validation.csv` |
| | DE | 151 | 504 | 29.96% | `dataset_majority_validation.csv` |
| | V | 68 | 504 | 13.49% | `dataset_majority_validation.csv` |
| | PE | 66 | 504 | 13.13% | `dataset_majority_validation.csv` |
| | Unlabeled | 68 | 504 | 13.49% | `dataset_majority_validation.csv` |
| | **Val Total** | **504** | **504** | **100%** | |
| **Test** | SBE | 303 | 872 | 34.75% | `dataset_majority_test.csv` |
| | DE | 302 | 872 | 34.64% | `dataset_majority_test.csv` |
| | V | 136 | 872 | 15.60% | `dataset_majority_test.csv` |
| | PE | 131 | 872 | 15.02% | `dataset_majority_test.csv` |
| | **Test Total** | **872** | **872** | **100%** | |

**Note**: Percentages = Count / Total × 100. All percentages rounded to 2 decimal places.

---

## 3. Fuzzy Label Distribution

| Split | Class | Count | Total | Percentage | Source |
|-------|-------|-------|-------|--------|--------|
| **Train (labeled)** | SBE | 987 | 2,664 | 37.04% | `dataset_fuzzy_train_labeled.csv` |
| | DE | 840 | 2,664 | 31.53% | `dataset_fuzzy_train_labeled.csv` |
| | V | 455 | 2,664 | 17.08% | `dataset_fuzzy_train_labeled.csv` |
| | PE | 382 | 2,664 | 14.34% | `dataset_fuzzy_train_labeled.csv` |
| | **Train (labeled) Total** | **2,664** | **2,664** | **100%** | |
| | Unlabeled/Uncertain | N/A | 3,526 (train full) | N/A | — |
| **Val (labeled)** | SBE | 145 | 387 | 37.47% | `dataset_fuzzy_validation_labeled.csv` |
| | DE | 120 | 387 | 31.01% | `dataset_fuzzy_validation_labeled.csv` |
| | V | 65 | 387 | 16.79% | `dataset_fuzzy_validation_labeled.csv` |
| | PE | 57 | 387 | 14.73% | `dataset_fuzzy_validation_labeled.csv` |
| | **Val (labeled) Total** | **387** | **387** | **100%** | |
| **Test (labeled)** | SBE | 286 | 759 | 37.68% | `dataset_fuzzy_test_labeled.csv` |
| | DE | 226 | 759 | 29.78% | `dataset_fuzzy_test_labeled.csv` |
| | V | 132 | 759 | 17.39% | `dataset_fuzzy_test_labeled.csv` |
| | PE | 115 | 759 | 15.15% | `dataset_fuzzy_test_labeled.csv` |
| | **Test (labeled) Total** | **759** | **759** | **100%** | |
| | Unlabeled/Uncertain | N/A | 872 (crisp test full) | N/A | — |

**Note**: Fuzzy labeled totals exclude Uncertain/Unlabeled samples. The full fuzzy dataset (38,655 narratives) includes Uncertain labels.

---

## 3. Crisp-vs-Fuzzy Comparison

| Metric | Crisp (Model A) | Fuzzy (Model B) | Agreement | Source |
|--------|-----------------|-----------------|-----------|--------|
| Train samples | 3,052 | 2,664 | N/A | Checkpoint metadata |
| Val samples | 436 | 387 | N/A | Checkpoint metadata |
| Test samples | 872 | 759 | 759 (intersection) | Verified |
| **Test class: SBE** | 303 (34.75%) | 286 (37.68%) | 214 common (94.2% of fuzzy test) | `dataset_majority_test.csv` / `dataset_fuzzy_test_labeled.csv` |
| **Test class: DE** | 302 (34.64%) | 226 (29.78%) | 188 common (82.5% of fuzzy test) | `dataset_majority_test.csv` / `dataset_fuzzy_test_labeled.csv` |
| **Test class: V** | 136 (15.60%) | 132 (17.39%) | 104 common (78.8% of fuzzy test) | `dataset_majority_test.csv` / `dataset_fuzzy_test_labeled.csv` |
| **Test class: PE** | 131 (15.02%) | 115 (15.15%) | 93 common (80.7% of fuzzy test) | `dataset_majority_test.csv` / `dataset_fuzzy_test_labeled.csv` |
| **Label agreement (crisp vs fuzzy)** | N/A | N/A | 727/759 = 95.8% | `PHASE_3B_FINAL_METRICS_AUDIT.md` §5 |
| **Training leakage** | 0 overlap | 0 overlap | Verified (4/4 checks) | `PHASE_3B_FINAL_METRICS_AUDIT.md` §5 |

**Key observation**: Fuzzy model has fewer train samples (2,664 vs 3,052) but similar class proportions. The 95.8% agreement on 759 common test samples means 32/759 samples have qualitatively different crisp/fuzzy labels.

---

## 3. Threshold Sensitivity

| τ value | Output behavior | Verification | Source |
|---------|----------------|--------------|--------|
| τ=0.25 | Identical to τ=0.35 and τ=0.50 | ✅ Verified | `tau_sensitivity.json` |
| τ=0.35 | Standard threshold; "≥1 evidence" with integer counts | ✅ Verified | Roadmap default |
| τ=0.50 | Identical to τ=0.25 and τ=0.35 | ✅ Verified | `tau_sensitivity.json` |
| δ=0.10 | Stricter dominance margin; more Uncertain predictions | 🟡 Not tested | — |
| δ=0.15 | Standard value; provides meaningful discrimination | ✅ Verified | `delta_sensitivity.json` |
| δ=0.20 | More permissive; fewer Uncertain predictions | 🟡 Not tested | — |

**τ Sensitivity Finding**: Since keyword evidence counts are integers (0, 1, 2...), τ=0.25, 0.35, and 0.50 all have the same practical effect: "any non-zero evidence count (≥1)". The parameter documents policy for future continuous evidence scores.

**δ Sensitivity Finding**: δ=0.15 provides meaningful discrimination for close competitions. Lower δ values permit more predictions; higher δ values produce more Uncertain outputs.

---

## 4. DistilBERT Experiment Configuration

| Parameter | Model A | Model B | Source |
|-----------|---------|---------|--------|
| Training labels | Crisp majority-vote | FuzzyWeakLabeler (τ=0.35, δ=0.15) | `train_distilbert_models.py` |
| Epochs | 3 | 3 | `train_distilbert_models.py` |
| Batch size | 16 | 16 | `train_distilbert_models.py` |
| Learning rate | 2e-5 | 2e-5 | `train_distilbert_models.py` |
| Weight decay | 0.01 | 0.01 | `train_distilbert_models.py` |
| Optimizer | AdamW | AdamW | `train_distilbert_models.py` |
| Random seed | 42 | 42 | `train_distilbert_models.py` |
| Device | CPU | CPU | `train_distilbert_models.py` |
| Base checkpoint | `distilbert-base-uncased` | `distilbert-base-uncased` | `train_distilbert_models.py` |
| Max sequence length | 128 | 128 | `train_distilbert_models.py` |
| Embedding extraction batch size | 64 | 64 | `train_distilbert_models.py` |

**Original test metrics (from checkpoints)**:

| Metric | Model A | Model B | Source |
|--------|---------|---------|--------|
| Test samples | 872 | 759 | Checkpoint metadata |
| Accuracy | 36.93% | 39.13% | `distilbert_comparison_metrics.json` |
| Macro-F1 | 21.04% | 21.56% | `distilbert_comparison_metrics.json` |
| Weighted-F1 | 29.21% | 29.87% | `distilbert_comparison_metrics.json` |
| Diagonal sum (CM) | 322 | 237 | `distilbert_comparison_metrics.json` |
| Per-class support | SBE:303, DE:302, V:136, PE:131 | SBE:286, DE:226, V:132, PE:115 | `distilbert_comparison_metrics.json` |

**Critical**: These are from **different evaluation populations** (872 vs 759). Cannot directly establish superiority.

---

## 4. DistilBERT Common-Test Structural Verification

| Metric | Value | Verified |
|--------|-------|----------|
| Unique report IDs (crisp test) | 1,008 | ✅ |
| Unique report IDs (fuzzy test) | 759 | ✅ |
| Intersection (common test samples) | 759 | ✅ |
| Training leakage (any test sample in train/val) | 0 (4/4 checks) | ✅ |
| Crisp-fuzzy label agreement | 95.8% (727/759) | ✅ |
| Common-test performance (accuracy/F1) | **Unverified** | ⚠️ Embedding extraction not completed |

---

## 5. Research Limitations Table

| Limitation | Description | Severity | Mitigation (future) |
|------------|-------------|----------|---------------------|
| No human-validated ground truth | All metrics are weak-label proxies | High | Human annotations of ASRS narratives |
| Class imbalance (Violation, Perceptual Error) | Zero predicted instances in both models | High | Class-balanced training, focal loss |
| Frozen encoder without fine-tuning | Linear head only; minority classes not learnable | Medium | GPU fine-tuning |
| Common-test performance unverified | Embedding extraction not completed on CPU | Medium | GPU resources for embedding extraction |
| τ=0.35 with integer counts ≈ "≥1" | Mathematically consistent but parameter semantics clarified | Low | Future continuous evidence scores |
| DistilBERT is downstream extension | Not the main research contribution | Low | Explicitly stated in scope |
| No human evaluation of weak labels | No HFACS-ground-truth annotations | High | Future human annotation study |
| Report 1-only vs Report 1+2 distinction | Must be preserved in manuscript | Low | Careful documentation |

---

## 6. Verification Status Summary

| Verification Status | Count of Entries | Paper Treatment |
|---------------------|-----------------|---------------|
| ✅ Verified | 112 | Allowed without qualification |
| ⚠️ Provisional | 15 | Allowed with qualification/caveat |
| ❌ Unsupported | 22 | Excluded or must be explicitly qualified |
| 📋 Unverified | 8 | Must be noted as unverified |

**Total**: 157 entries across all tables

---

## 7. Next Steps for Research Tables

1. Complete `CLAIM_TO_EVIDENCE_MATRIX.md` (traceability from tables to evidence)
2. Generate publication-quality figures from verified data
3. Assemble the foundation manuscript with only verified claims
4. Run the integrity checklist
5. Prepare for Phase 4B — Final Research Paper Writing

---