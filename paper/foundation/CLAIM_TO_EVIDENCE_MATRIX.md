# NearMiss2Crash — Claim-to-Evidence Traceability Matrix

**Date**: October 22, 2026  
**Research Title**: "Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives"  

> **Policy**: Every research claim must link to: (1) experimental result, code, or literature source, (2) figure/table where applicable, (3) verification status, (4) allowed academic interpretation. Explicitly distinguish measured facts from hypotheses.

---

## 1. Claim Traceability Matrix

### 1.1 Methodology Claims

| Claim | Evidence Source | Figure/Table | Verification | Paper Usage |
|-------|----------------|--------------|--------------|-------------|
| 39 keyword seeds mapped to 4 HFACS-core classes | `keyword_weak_labeler.py` §SEED_KEYWORDS | Table 1 (Dataset Summary) | ✅ Verified | Allowed |
| FuzzyWeakLabeler implements τ=0.35 and δ=0.15 | `fuzzy_labeler.py` §constructor | Table 6 (Threshold Sensitivity) | ✅ Verified | Allowed |
| 8 unit tests all pass for FuzzyWeakLabeler | `fuzzy_labeler.py` §unit tests | Table 6 | ✅ Verified | Allowed |
| Low/Medium/High membership functions (trapezoidal/triangular/ramp) | `fuzzy_labeler.py` §membership functions | Fig. 4 | ✅ Verified | Allowed |
| Evidence aggregation counts keyword occurrences per class | `fuzzy_labeler.py` §compute_evidence | Eq. 1 | ✅ Verified | Allowed |
| Inference rules R0–R5 implemented | `fuzzy_labeler.py` §evaluate_fuzzy_rules | Eq. 2 | ✅ Verified | Allowed |
| Dominance = (E_top - E_second) / E_total | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | Eq. 3 | ✅ Verified | Allowed |
| top_support = 0.25·μ_low + 0.5·μ_med + 1.0·μ_high | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | Eq. 4 | ✅ Verified | Allowed |
| competition_penalty = 0.5·μ_med,second + 1.0·μ_high,second | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | Eq. 5 | ✅ Verified | Allowed |
| raw_confidence = dominance × top_support × (1 - 0.5·competition_penalty) | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | Eq. 6 | ✅ Verified | Allowed |
| confidence = min(max(raw_confidence, 0), 1) | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | Eq. 7 | ✅ Verified | Allowed |
| τ=0.35 with integer counts effectively means "≥1 evidence" | `PHASE_2_COMPLETION_REPORT.md` §G | Table 6 | ✅ Verified | Allowed |
| δ=0.15 provides meaningful discrimination | `delta_sensitivity.json` | Table 6 (sensitivity) | ✅ Verified | Allowed |
| Fuzzy label selection is hybrid (crisp counts → label; memberships → confidence) | Implementation analysis | Eq. 2 + text | ✅ Verified | Allowed |
| DistilBERT uses frozen encoder + linear head | `transformer_model.py` | Table 8 (Config) | ✅ Verified | Allowed |
| DistilBERT is downstream extension, not main contribution | Project scope | Table 8 | ✅ Verified | Allowed |

### 1.2 Dataset Claims

| Claim | Evidence Source | Figure/Table | Verification | Paper Usage |
|-------|----------------|--------------|--------------|-------------|
| 38,655 ASRS narratives total | `fuzzy_full_results.csv` | Table 1 | ✅ Verified | Allowed |
| 21,058 crisp-labeled narratives (54.48%) | `dataset_majority_*.csv` | Table 1 | ✅ Verified | Allowed |
| 17,597 Unlabeled narratives (45.52%) | `dataset_majority_*.csv` | Table 1 | ✅ Verified | Allowed |
| 4 HFACS-core classes | `keyword_weak_labeler.py` | Table 1 | ✅ Verified | Allowed |
| Train (crisp): SBE:1,059, DE:1,056, V:476, PE:461 | `dataset_majority_train.csv` | Table 2 | ✅ Verified | Allowed |
| Train (fuzzy labeled): 2,664 | `dataset_fuzzy_train_labeled.csv` | Table 1 | ✅ Verified | Allowed |
| Val: 436 (crisp), 387 (fuzzy) | `dataset_majority_validation.csv`, `dataset_fuzzy_validation_labeled.csv` | Table 1 | ✅ Verified | Allowed |
| Test: 872 (crisp), 759 (fuzzy) | `dataset_majority_test.csv`, `dataset_fuzzy_test_labeled.csv` | Table 1 | ✅ Verified | Allowed |
| Common test intersection: 759 | `PHASE_3B_FINAL_METRICS_AUDIT.md` §5 | Table 6 | ✅ Verified | Allowed |
| Zero training leakage (4/4 checks) | `PHASE_3B_FINAL_METRICS_AUDIT.md` §5 | Table 6 | ✅ Verified | Allowed |
| 95.8% crisp-fuzzy label agreement | `PHASE_3B_FINAL_METRICS_AUDIT.md` §5 | Table 6 | ✅ Verified | Allowed |

### 1.3 Model Performance Claims

| Claim | Evidence Source | Figure/Table | Verification | Paper Usage |
|-------|----------------|--------------|--------------|-------------|
| Model A accuracy = 36.93% on 872 test samples | `distilbert_comparison_metrics.json` | Table 9 | ✅ Verified | Allowed |
| Model A macro-F1 = 21.04% | `distilbert_comparison_metrics.json` | Table 9 | ✅ Verified | Allowed |
| Model A weighted-F1 = 29.21% | `distilbert_comparison_metrics.json` | Table 9 | ✅ Verified | Allowed |
| Model B accuracy = 39.13% on 759 test samples | `distilbert_comparison_metrics.json` | Table 9 | ✅ Verified | Allowed |
| Model B macro-F1 = 21.56% | `distilbert_comparison_metrics.json` | Table 9 | ✅ Verified | Allowed |
| Model B weighted-F1 = 29.87% | `distilbert_comparison_metrics.json` | Table 9 | ✅ Verified | Allowed |
| Model A confusion matrix diagonal sum = 322 (of 872) | `distilbert_comparison_metrics.json` | Table 10 | ✅ Verified | Allowed |
| Model B confusion matrix diagonal sum = 237 (of 759) | `distilbert_comparison_metrics.json` | Table 10 | ✅ Verified | Allowed |
| Both models have zero predictions for Violation | Confusion matrices | Table 10 | ✅ Verified | Allowed |
| Both models have zero predictions for Perceptual Error | Confusion matrices | Table 10 | ✅ Verified | Allowed |
| Model A and Model B have different evaluation populations | `distilbert_comparison_metrics.json` | Table 9 | ✅ Verified | Allowed |
| Model A test (872) and Model B test (759) cannot directly establish superiority | Audit documentation | — | ✅ Verified | Allowed |

### 1.4 Unsupported Claims (Must Be Qualified or Excluded)

| Claim | Evidence Source | Verification | Paper Usage |
|-------|----------------|--------------|-------------|
| Model A accuracy on common 759 set = 35.49% | ❌ Previous report (unverified) | ❌ Unsupported | **Excluded** |
| Model B accuracy on common 759 set = 38.47% | ❌ Previous report (unverified) | ❌ Unsupported | **Excluded** |
| Fuzzy model outperforms crisp on common test set | ⚠️ Unverified | ⚠️ Provisional | Allowed only if recomputed |
| Common-test performance metrics are validated | ❌ Previous report | ❌ Unsupported | **Excluded** |

### 1.5 Hypothesis (Not Verified)

| Claim | Evidence Source | Verification | Paper Usage |
|-------|----------------|--------------|-------------|
| Fuzzy labels provide better training signal for DistilBERT | Original splits (39.13% vs 36.93%) | ⚠️ Provisional | Allowed with qualification |
| Full fine-tuning would improve minority class performance | Implementation analysis | ⚠️ Provisional | Allowed as future work |
| Human-validated accuracy would differ significantly from weak-label proxy | Project scope | ⚠️ Provisional | Allowed as limitation |

---

## 2. Table-to-Evidence Mapping

| Table # | Table Title | Evidence Sources |
|---------|-------------|------------------|
| Table 1 | Dataset Summary | `fuzzy_full_results.csv`, `dataset_majority_*.csv`, `dataset_fuzzy_*_labeled.csv` |
| Table 2 | Crisp Label Distribution | `dataset_majority_*.csv` |
| Table 3 | Fuzzy Label Distribution | `dataset_fuzzy_*_labeled.csv` |
| Table 4 | Crisp-vs-Fuzzy Comparison | `dataset_majority_*.csv`, `dataset_fuzzy_*_labeled.csv`, `PHASE_3B_FINAL_METRICS_AUDIT.md` |
| Table 5 | Threshold Sensitivity | `tau_sensitivity.json`, `delta_sensitivity.json` |
| Table 6 | DistilBERT Experiment Configuration | `train_distilbert_models.py`, `distilbert_comparison_metrics.json` |
| Table 7 | DistilBERT Original Test Metrics | `distilbert_comparison_metrics.json` (checkpoint) |
| Table 8 | DistilBERT Confusion Matrices | `distilbert_comparison_metrics.json` |
| Table 9 | Research Limitations | Audit documentation |

---

## 3. Figure-to-Evidence Mapping

| Figure # | Figure Title | Evidence Sources |
|----------|-------------|------------------|
| Fig. 1 | ASRS Dataset Class Distribution | `dataset_majority_*.csv` |
| Fig. 2 | Crisp vs Fuzzy Definitive-Label Coverage | `dataset_majority_*.csv`, `dataset_fuzzy_*_labeled.csv` |
| Fig. 3 | Crisp-to-Fuzzy Transition Heatmap | `detailed_transitions.csv` |
| Fig. 4 | Fuzzy Membership Functions | `fuzzy_labeler.py` |
| Fig. 5 | Threshold Sensitivity Analysis | `tau_sensitivity.json`, `delta_sensitivity.json` |
| Fig. 6 | Abstention-Reason Distribution | `fuzzy_labeler.py` inference rules |
| Fig. 7 | Research Architecture Diagram | Project workflow documentation |
| Fig. 8 | DistilBERT Training Curves | Checkpoint metadata |
| Fig. 9 | DistilBERT Per-Class Results | `distilbert_comparison_metrics.json` |
| Fig. 10 | DistilBERT Confusion Matrices | `distilbert_comparison_metrics.json` |

---

## 4. Evidence-to-Report Mapping

| Section in Manuscript | Evidence Required | Evidence Source |
|----------------------|-------------------|-----------------|
| Introduction | Research problem; motivation | Project scope; literature |
| Related Work | Literature citations | `LITERATURE_REVIEW_MATRIX.md` |
| Research Gap | Limitations of prior work | `RESEARCH_GAP_AND_NOVELTY.md` |
| Methodology | Equations, pseudocode | `METHODOLOGY_AND_EQUATIONS.md` |
| Dataset & Preprocessing | Dataset summary | `RESEARCH_TABLES.md` (Table 1) |
| Crisp Weak Labeling | Keyword seeds, matching procedure | `keyword_weak_labeler.py` |
| Fuzzy Weak Labeling | Membership functions, inference rules | `fuzzy_labeler.py`, `PHASE_2_COMPLETION_REPORT.md` |
| Experimental Setup | Training config, splits | `train_distilbert_models.py` |
| Phase 3 Results | Coverage, agreement, transitions | `RESEARCH_TABLES.md` (Table 4–6) |
| DistilBERT Extension | Architecture, training, results | `distilbert_comparison_metrics.json` |
| Discussion | Limitations, implications | `RESEARCH_TABLES.md` (Table 9) |
| Conclusion | Summary of findings | All above |

---

## 5. Claims Requiring Qualification

The following claims use verified evidence but require qualification in the manuscript:

| Claim | Qualification Needed |
|-------|---------------------|
| Fuzzy model outperforms crisp model | Only on original splits (39.13% vs 36.93%); different populations; common-test performance unverified |
| Model A trained on crisp majority-vote | Must specify: majority-vote weak labels from Phase 1, not pure keyword baseline |
| Confidence scores are calibrated | Must clarify: heuristic formula, not probability-calibrated; no formal calibration |
| Fuzzy weak labels are "soft" | Must clarify: hard labels (argmax of decision), not continuous soft targets; training uses hard labels |
| 95.8% crisp-fuzzy agreement | Must specify: on 759 common test samples only; not full dataset |
| Zero predictions for Violation/PE | Must note: zero in both models on both test populations; indicates class imbalance limitation |

---

## 6. Claims Excluded (Unsupported)

| Claim | Reason for Exclusion |
|-------|---------------------|
| Model A accuracy on common 759 = 35.49% | No verified computation from predictions |
| Model B accuracy on common 759 = 38.47% | No verified computation from predictions |
| Fuzzy model superiority on common test | Cannot be claimed without recomputed metrics |
| Human-validated accuracy | No human ground-truth labels exist |
| True classification accuracy | All metrics are weak-label proxies |

---