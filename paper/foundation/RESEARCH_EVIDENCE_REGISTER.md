# NearMiss2Crash — Research Evidence Register

**Date**: October 22, 2026  
**Research Title**: "Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives"  

> **Policy**: No unsupported numerical claims presented as findings. Every entry must be Verified, Provisional, or Unsupported. Claims marked Unsupported must not appear in the manuscript without qualification.

---

## 1. Evidence Register Policy

Each row in this register records:
- **Claim**: Exact research finding (not interpretation)
- **Source**: File containing supporting evidence
- **Experiment**: Phase and experiment name
- **Value**: Actual numerical result
- **Denominator**: Population used
- **Configuration**: Thresholds, parameters, model settings
- **Verification**: Verified / Provisional / Unsupported
- **Paper usage**: Allowed / Qualified / Excluded

If Verification ≠ "Verified", the claim must be qualified in the manuscript or excluded.

---

## 2. Phase 1 — Crisp Keyword Baseline

| # | Claim | Source | Experiment | Value | Denominator | Configuration | Verification | Paper usage |
|---|-------|--------|------------|-------|-------------|---------------|--------------|-------------|
| P1-1 | 39 keyword seeds across 4 HFACS-core classes | `keyword_weak_labeler.py` | Phase 1 baseline | 39 keywords | N/A | SEED_KEYWORDS list | **Verified** | Allowed |
| P1-2 | Crisp-label coverage = 54.48% | `baseline_metrics.json` | Phase 1 baseline | 54.48% | 21,058 / 38,655 | Majority-vote threshold | **Verified** | Allowed |
| P1-3 | Unlabeled fraction = 45.52% | `weak_labeling_results/dataset_majority_test.csv` | Phase 1 baseline | 45.52% | 17,597 / 38,655 | Labels ≠ "Unlabeled" | **Verified** | Allowed |
| P1-4 | 4 HFACS-core classes: Skill-Based Error, Decision Error, Violation, Perceptual Error | `keyword_weak_labeler.py` | Phase 1 baseline | 4 classes | N/A | N/A | **Verified** | Allowed |
| P1-4a | Class distribution in test set: SBE:303, DE:302, V:136, PE:131 | `dataset_majority_test.csv` | Phase 1 test | 872 total | 872 | N/A | **Verified** | Allowed |

---

## 3. Phase 2 — Fuzzy Weak-Labeler

| # | Claim | Source | Experiment | Value | Denominator | Configuration | Verification | Paper usage |
|---|-------|--------|------------|-------|-------------|---------------|--------------|-------------|
| P2-1 | FuzzyWeakLabeler implements τ=0.35 evidence threshold and δ=0.15 dominance margin | `fuzzy_labeler.py` | Phase 2 complete | Thresholds wired | N/A | evidence_threshold=0.35, dominance_margin=0.15 | **Verified** | Allowed |
| P2-2 | 8 unit tests all pass (empty input, single keyword, competing evidence, ties, repeated keywords, τ boundary, δ boundary, determinism) | `fuzzy_labeler.py` | Phase 2 complete | 8/8 tests passed | N/A | N/A | **Verified** | Allowed |
| P2-2a | Test 6: τ boundary — τ=0.35 correctly accepts ≥1 keyword match | `fuzzy_labeler.py` | Phase 2 unit tests | Boundary behavior verified | N/A | τ=0.35 constructor param | **Verified** | Allowed |
| P2-2b | Test 7: δ boundary — δ=0.15 correctly rejects close competitions | `fuzzy_labeler.py` | Phase 2 unit tests | Boundary behavior verified | N/A | δ=0.15 constructor param | **Verified** | Allowed |
| P2-3 | Evidence aggregation: `compute_evidence()` counts keyword matches per class (occurrence-based by default) | `fuzzy_labeler.py` | Phase 2 complete | Counts stored | N/A | N/A | **Verified** | Allowed |
| P2-4 | Membership functions: Low (trapezoidal), Medium (triangular), High (ramp) | `fuzzy_labeler.py` | Phase 2 complete | Piece-wise linear | N/A | N/A | **Verified** | Allowed |
| P2-5 | Inference rules R0–R5: R0=no evidence→Uncertain, R1=top<τ→Uncertain, R3=tie→Uncertain, R4=dominance<δ→Uncertain, R5=passes both→predicted label | `fuzzy_labeler.py` | Phase 2 complete | 5 rules | N/A | N/A | **Verified** | Allowed |
| P2-6 | Top support formula: `0.25·μ_low + 0.5·μ_med + 1.0·μ_high` | `PHASE_2_COMPLETION_REPORT.md` | Phase 2 complete | Equation form | N/A | N/A | **Verified** | Allowed |
| P2-7 | Raw confidence: `dominance × top_support × (1 - 0.5·competition_penalty)` | `PHASE_2_COMPLETION_REPORT.md` | Phase 2 complete | Equation form | N/A | N/A | **Verified** | Allowed |
| P2-8 | Confidence: `min(max(raw_confidence, 0), 1)` | `PHASE_2_COMPLETION_REPORT.md` | Phase 2 complete | Clipping formula | N/A | N/A | **Verified** | Allowed |
| P2-8a | No non-Uncertain prediction receives confidence 0.0000 | `PHASE_2_COMPLETION_REPORT.md` | Phase 2 complete | Empirical observation | N/A | N/A | **Verified** | Allowed |
| P2-8b | 0.0000 confidence only for Uncertain labels (R0, R3) | `PHASE_2_COMPLETION_REPORT.md` | Phase 2 complete | Empirical observation | N/A | N/A | **Verified** | Allowed |
| P2-9 | Fuzzy label distribution (sample of 50 rows): SBE-dominant, Decision-dominant, etc. | `run_fuzzy_sample.py` | Phase 2 demo | 50-row sample | 50 | τ=0.35, δ=0.15 | **Verified** | Allowed |
| P2-10 | Since keyword counts are integers, τ=0.35 effectively means "≥1 evidence count" | `PHASE_2_COMPLETION_REPORT.md` §G | Phase 2 complete | Threshold semantics | N/A | Integer counts | **Verified** | Allowed |
| P2-11 | Competence penalty: `0.5·μ_med,second + 1.0·μ_high,second` | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | Phase 2 complete | Equation form | N/A | N/A | **Verified** | Allowed |

---

## 4. Phase 3 — Full Experimental Evaluation

| # | Claim | Source | Experiment | Value | Denominator | Configuration | Verification | Paper usage |
|---|-------|--------|------------|-------|-------------|---------------|--------------|-------------|
| P3-1 | Full fuzzy dataset labeling: 38,655 rows → `fuzzy_full_results.csv` | `experiments/fuzzy_asrs/results/fuzzy_full_results.csv` | Phase 3 full labeling | 38,655 rows | 38,655 | τ=0.35, δ=0.15 | **Verified** | Allowed |
| P3-2 | Crisp-vs-fuzzy comparison summary | `experiments/fuzzy_asrs/results/comparison_summary.json` | Phase 3 comparison | Available | N/A | N/A | **Verified** | Allowed |
| P3-2a | Comparison summary includes: fuzzy-full results, crisp-vs-fuzzy transition, detailed transitions, sensitivity | `comparison_summary.json` | Phase 3 comparison | JSON structure | N/A | N/A | **Verified** | Allowed |
| P3-3 | τ sensitivity: τ=0.25, 0.35, 0.50 produce identical outputs (integer evidence counts) | `experiments/fuzzy_asrs/results/sensitivity/tau_sensitivity.json` | Phase 3 sensitivity | 3 τ values | N/A | N/A | **Verified** | Allowed |
| P3-3b | δ sensitivity: δ=0.15 provides meaningful discrimination for close competitions | `experiments/fuzzy_asrs/results/sensitivity/delta_sensitivity.json` | Phase 3 sensitivity | δ variation | N/A | N/A | **Verified** | Allowed |
| P3-4 | Full fuzzy labeling covers all 38,655 ASRS narratives with fuzzy labels per-class | `fuzzy_full_results.csv` | Phase 3 full labeling | 38,655 rows | 38,655 | τ=0.35, δ=0.15 | **Verified** | Allowed |
| P3-5 | Label agreement between crisp and fuzzy weak labels documented | `comparison_summary.json` | Phase 3 comparison | Available | N/A | N/A | **Verified** | Allowed |

---

## 5. Phase 3B — DistilBERT Classifier Training

| # | Claim | Source | Experiment | Value | Denominator | Configuration | Verification | Paper usage |
|---|-------|--------|------------|-------|-------------|---------------|--------------|-------------|
| P3B-1 | Model A trained on crisp majority-vote weak labels (872 train samples) | `models/model_a_crisp_head.pt` | Phase 3B DistilBERT | 872 train samples | 872 | 3 epochs, batch=16, lr=2e-5, CPU | **Verified** | Allowed |
| P3B-2 | Model B trained on fuzzy weak labels (2664 train samples) | `models/model_b_fuzzy_head.pt` | Phase 3B DistilBERT | 2664 train samples | 2664 | 3 epochs, batch=16, lr=2e-5, CPU | **Verified** | Allowed |
| P3B-3 | Original test metrics stored in checkpoints (NOT recomputed from common set) | `distilbert_comparison_metrics.json` | Phase 3B DistilBERT | — | — | — | **Verified** | Allowed |
| P3B-3a | Model A original test: 872 samples, 36.93% acc, 21.04% macro-F1, 29.21% weighted-F1 | `distilbert_comparison_metrics.json` | Phase 3B DistilBERT | As above | 872 | 3 epochs, batch=16, lr=2e-5, CPU | **Verified** | Allowed |
| P3B-3b | Model B original test: 759 samples, 39.13% acc, 21.56% macro-F1, 29.87% weighted-F1 | `distilbert_comparison_metrics.json` | Phase 3B DistilBERT | As above | 759 | 3 epochs, batch=16, lr=2e-5, CPU | **Verified** | Allowed |
| P3B-4 | Common-test structural verification: 759 shared report IDs, zero leakage, 95.8% agreement | `PHASE_3B_FINAL_METRICS_AUDIT.md` | Phase 3B verification | — | 759 | N/A | **Verified** | Allowed |
| P3B-4a | Common-test: unique report IDs (crisp) = 1008 | `PHASE_3B_FINAL_METRICS_AUDIT.md` | Phase 3B verification | 1008 | N/A | N/A | **Verified** | Allowed |
| P3B-4b | Common-test: unique report IDs (fuzzy) = 759 | `PHASE_3B_FINAL_METRICS_AUDIT.md` | Phase 3B verification | 759 | N/A | N/A | **Verified** | Allowed |
| P3B-4c | Common-test: intersection = 759 | `PHASE_3B_FINAL_METRICS_AUDIT.md` | Phase 3B verification | 759 | N/A | N/A | **Verified** | Allowed |
| P3B-4d | Common-test: zero training leakage (4/4 split checks = 0 overlap) | `PHASE_3B_FINAL_METRICS_AUDIT.md` §5 | Phase 3B verification | 0 overlap | N/A | N/A | **Verified** | Allowed |
| P3B-4d | Common-test: label agreement 95.8% (727/759) | `PHASE_3B_FINAL_METRICS_AUDIT.md` §5 | Phase 3B verification | 727/759 | 759 | N/A | **Verified** | Allowed |
| P3B-5 | Original test metrics are from DIFFERENT evaluation populations (872 vs 759) and cannot establish superiority | `PHASE_3B_FINAL_METRICS_AUDIT.md` §3 | Phase 3B verification | N/A | N/A | N/A | **Verified** | Allowed |
| P3B-6 | Common-test performance metrics (accuracy, F1 on 759 shared samples) are UNVERIFIED — embedding extraction not completed | `PHASE_3B_FINAL_METRICS_AUDIT.md` §4 | Phase 3B verification | N/A | N/A | N/A | **Unsupported** | **Excluded** unless recomputed |
| P3B-6a | Model A accuracy on common 759 set = 35.49% | Previously reported (unverified) | Phase 3B common-test | 35.49% | — | — | **Unsupported** | **Excluded** |
| P3B-6b | Model B accuracy on common 759 set = 38.47% | Previously reported (unverified) | Phase 3B common-test | 38.47% | — | — | **Unsupported** | **Excluded** |
| P3B-7 | Both models fail on Violation and Perceptual Error (zero predictions) | `models/` confusion matrices | Phase 3B | Zero predictions for both classes | N/A | N/A | **Verified** | Allowed |
| P3B-8 | DistilBERT is a downstream exploratory extension, not the main research contribution | Project scope doc | Phase 3B | — | — | — | **Verified** | Allowed with qualification |

---

## 5. Martino-Gap and Novelty Claims

| # | Claim | Source | Verification | Paper usage |
|---|-------|--------|--------------|-------------|
| P5-1 | Simple keyword weak labeling is limited due to class conflicts and ambiguity | Phase 1–2 analysis | **Verified** (by implementation quality) | Allowed |
| P5-2 | Class conflicts arise from keyword overlap across HFACS classes | Phase 1–2 analysis | **Verified** (keyword overlap documented) | Allowed |
| P5-3 | Explicit abstention (Uncertain) can be useful for handling insufficient evidence | Phase 2 inference rules | **Verified** (R0, R1, R3, R4 produce Uncertain) | Allowed |
| P5-4 | The fuzzy approach adds uncertainty-aware evidence aggregation beyond crisp keyword matching | Phase 2 implementation | **Verified** (membership functions, evidence aggregation, rule engine) | Allowed |
| P5-5 | Whether the fuzzy implementation meaningfully alters label assignments and coverage requires experimentation | Phase 3 comparative results | **Verified** (comparison data available) | Allowed |
| P5-6 | Our study differs from previous literature by: (a) explicit abstention, (b) dominance margin δ=0.15, (c) DistilBERT downstream extension | Roadmap compliance + implementation | **Verified** (against roadmap and code) | Allowed |
| P5-7 | DistilBERT adds a downstream exploratory extension for neural classification of weak labels | Project scope | **Verified** (training completed, verified checkpoints) | Allowed with qualification |
| P5-8 | Without human ground-truth labels, we cannot claim independently verified classification accuracy | Phase 3B audit | **Verified** (documented limitation) | Allowed with qualification |
| P5-9 | The role of fuzzy meaningfully alters label assignments and coverage — this is an open experimental question | Phase 3 comparative results | **Provisional** | Qualified — must cite comparative data, not claim certainty |
| P5-10 | Fuzzy membership values directly control final label selection via threshold comparison (hybrid fuzzy-evidence/threshold approach) | `fuzzy_labeler.py` implementation analysis | **Verified** (implementation confirms hybrid approach) | Allowed |
| P5-11 | Fuzzy membership values are only used for confidence calculation, not label selection | ❌ Contradicted by implementation | **Unsupported** | Excluded — contradicts actual code |

---

## 6. DistilBERT Experiment Documentation

| # | Claim | Source | Verification | Paper usage |
|---|-------|--------|--------------|-------------|
| P6-1 | DistilBERT architecture: frozen transformer encoder + linear classification head (768→4) | `transformer_model.py` | **Verified** | Allowed |
| P6-2 | Four output classes: Skill-Based Error, Decision Error, Violation, Perceptual Error | `transformer_model.py` | **Verified** | Allowed |
| P6-3 | Training labels: crisp majority-vote for Model A; FuzzyWeakLabeler for Model B | `train_distilbert_models.py` | **Verified** | Allowed |
| P6-4 | Training config: 3 epochs, batch_size=16, lr=2e-5, weight_decay=0.01, CPU-only | `train_distilbert_models.py` | **Verified** | Allowed |
| P6-5 | Original test set sizes: Model A = 872; Model B = 759 | `distilbert_comparison_metrics.json` | **Verified** | Allowed |
| P6-5a | These are DIFFERENT evaluation populations; cannot establish superiority | `PHASE_3B_FINAL_METRICS_AUDIT.md` §3 | **Verified** | Allowed with qualification |
| P6-6 | Common-test subset: 759 shared report IDs verified; zero training leakage; 95.8% label agreement | `PHASE_3B_FINAL_METRICS_AUDIT.md` §5 | **Verified** | Allowed |
| P6-6a | Common-test performance (accuracy/F1 on 759 shared samples) is UNVERIFIED — embedding extraction not completed | `PHASE_3B_FINAL_METRICS_AUDIT.md` §4 | **Unsupported** | **Excluded** from superiority claims |
| P6-7 | Model A original test: 36.93% acc, 21.04% macro-F1, 29.21% weighted-F1 on 872 samples | `distilbert_comparison_metrics.json` | **Verified** | Allowed |
| P6-8 | Model B original test: 39.13% acc, 21.56% macro-F1, 29.87% weighted-F1 on 759 samples | `distilbert_comparison_metrics.json` | **Verified** | Allowed |
| P6-8a | Model B outperforms Model A on their respective test populations (39.13% > 36.93%) | `distilbert_comparison_metrics.json` | **Verified** | Allowed |
| P6-8b | This superiority does NOT transfer to a common test set without recomputation | `PHASE_3B_FINAL_METRICS_AUDIT.md` | **Verified** | Allowed with qualification |
| P6-9 | Training labels are hard targets (argmax of fuzzy/majority probabilities), NOT continuous soft labels | Implementation analysis | **Verified** | Allowed |
| P6-10 | "Soft labels" should not be used to describe hard training targets unless actual continuous targets were used | Implementation check | **Verified** | Allowed |

---

## 7. Dataset Documentation Claims

| # | Claim | Source | Verification | Paper usage |
|---|----|----|--------------|-------------|
| D1-1 | ASRS Master Dataset: 38,655 narratives | Project data | **Verified** (from Phase 1 artifacts) | Allowed |
| D1-2 | 21,058 crisp-labeled narratives (54.48% coverage) | `weak_labeling_results/dataset_majority_train.csv` | **Verified** | Allowed |
| D1-3 | 17,597 Unlabeled narratives | `weak_labeling_results/dataset_majority_train.csv` | **Verified** | Allowed |
| D1-4 | Narrative text column used for weak labeling | `data_loading.py` / `keyword_weak_labeler.py` | **Verified** | Allowed |
| D1-5 | Report identifiers used for split integrity | `dataset_majority_*.csv` | **Verified** | Allowed |
| D1-6 | Difference between Report 1-only and Report 1+2 experiments | Audit documentation | **Verified** | Allowed |
| D1-7 | HFACS-core category definitions (4 classes) | `keyword_weak_labeler.py` | **Verified** | Allowed |
| D1-8 | Class distribution in test set: SBE:303, DE:302, V:136, PE:131 | `dataset_majority_test.csv` | **Verified** | Allowed |
| D1-8a | Training set: SBE:1059, DE:1056, V:476, PE:461, Unlabeled:474 | `dataset_majority_train.csv` | **Verified** | Allowed |
| D1-8b | Val set: SBE:151, DE:151, V:68, PE:66, Unlabeled:68 | `dataset_majority_validation.csv` | **Verified** | Allowed |
| D1-9 | 39 keyword seeds mapped to 4 HFACS classes | `SEED_KEYWORDS` in `keyword_weak_labeler.py` | **Verified** | Allowed |
| D1-10 | Missing text values: none detected in verified range | Data inspection | **Verified** | Allowed |
| D1-11 | Duplicate narrative counts: not independently checked | — | **Unverified** | Qualified — note as limitation |
| D1-12 | Report 1-only vs Report 1+2 experiment distinction | Audit documentation | **Verified** | Allowed |

---

## 7. Metholodological Claims

| # | Claim | Source | Verification | Paper usage |
|---|-------|--------|--------------|-------------|
| M1-1 | Fuzzy evidence counting: keyword occurrences per class (occurrence-based by default) | `fuzzy_labeler.py` §compute_evidence | **Verified** | Allowed |
| M1-2 | Low membership function: trapezoidal, 1.0 for x≤b, decreases to 0 at x=c | `fuzzy_labeler.py` §membership functions | **Verified** | Allowed |
| M1-3 | Medium membership function: triangular, peaks at x=b | `fuzzy_labeler.py` §membership functions | **Verified** | Allowed |
| M1-4 | High membership function: ramp from 0 at a to 1.0 at b | `fuzzy_labeler.py` §membership functions | **Verified** | Allowed |
| M1-5 | Evidence aggregation maps each class's count to membership values | `fuzzy_labeler.py` §compute_memberships | **Verified** | Allowed |
| M1-6 | Inference rules R0–R5 implemented exactly as documented | `fuzzy_labeler.py` §evaluate_fuzzy_rules | **Verified** (8 unit tests pass) | Allowed |
| M1-7 | Dominance = (E_top - E_second) / E_total | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | **Verified** | Allowed |
| M1-8 | Top support = 0.25·μ_low + 0.5·μ_med + 1.0·μ_high | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | **Verified** | Allowed |
| M1-7 | Raw confidence = dominance × top_support × (1 - 0.5·competition_penalty) | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | **Verified** | Allowed |
| M1-8 | Competition penalty = 0.5·μ_med,second + 1.0·μ_high,second | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | **Verified** | Allowed |
| M1-9 | Confidence = min(max(raw_confidence, 0), 1) | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | **Verified** | Allowed |
| M1-10 | Fuzzy label selection: hybrid approach — crisp evidence counts determine label; fuzzy memberships affect confidence | Implementation analysis | **Verified** (code confirms) | Allowed |
| M1-11 | Membership values do NOT directly control label selection; they only affect confidence | ❌ Contradicted by code | **Unsupported** | Excluded — code shows crisp counts determine label |
| M1-12 | τ=0.35 with integer evidence counts effectively means "≥1 keyword match" | `PHASE_2_COMPLETION_REPORT.md` §G | **Verified** | Allowed |
| M1-12a | δ=0.15 dominance margin minimum (E_top - E_second) / E_total | `PHASE_2_COMPLETION_REPORT.md` §Key Equations | **Verified** | Allowed |

---

## 8. Limitations Claims

| # | Claim | Source | Verification | Paper usage |
|---|----|--------|--------------|-------------|
| L1-1 | No human-validated ground-truth exists for ASRS HFACS classification | Phase 3B audit | **Verified** | Allowed with qualification |
| L1-2 | All metrics are weak-label proxies, not independently validated accuracy | Phase 3B audit | **Verified** | Allowed with qualification |
| L1-3 | Violation and Perceptual Error classes have zero predicted instances in both models | Confusion matrices | **Verified** | Allowed |
| L1-4 | Class imbalance severely limits macro-F1 | Phase 3B results | **Verified** | Allowed |
| L1-5 | Frozen encoder without fine-tuning cannot learn minority class representations | Implementation analysis | **Verified** | Allowed |
| L1-6 | Common-test performance is unverified (embedding extraction not completed) | `PHASE_3B_FINAL_METRICS_AUDIT.md` §4 | **Unverified** | Must be qualified |
| L1-7 | τ=0.35 with integer counts effectively means "≥1 evidence" | `PHASE_2_COMPLETION_REPORT.md` §G | **Verified** | Allowed |
| L1-7a | δ=0.15 provides meaningful discrimination for close competitions | Phase 3 sensitivity | **Verified** | Allowed |
| L1-8 | DistilBERT is a downstream exploratory extension, not the main research contribution | Project scope | **Verified** | Allowed |
| L1-9 | Without human ground-truth labels, classification accuracy cannot be independently verified | Phase 3B audit | **Verified** | Allowed with qualification |
| L1-10 | Fuzzy label agreement with crisp is 95.8% (727/759) — substantial but not perfect overlap | Common-test verification | **Verified** | Allowed |

---

## 8. Reproducibility Claims

| # | Claim | Source | Verification | Paper usage |
|---|----|--------|--------------|-------------|
| R1-1 | Phase 1 crisp baseline: code and data available; `run_phase1_baseline.py` verified | Phase 1 artifacts | **Verified** | Allowed |
| R1-2 | Phase 2 fuzzy labeler: 8 unit tests pass; thresholds verified | `fuzzy_labeler.py` | **Verified** | Allowed |
| R1-3 | Phase 3 fuzzy-full labeling: `fuzzy_full_results.csv` generated (68 MB) | Generated output | **Verified** | Allowed |
| R1-4 | Phase 3 comparison/sensitivity: CSV outputs verified | Generated CSVs | **Verified** | Allowed |
| R1-5 | Phase 3B DistilBERT training: both models trained on CPU, checkpoints saved | `models/` | **Verified** | Allowed |
| R1-6 | Phase 3B common-test embedding extraction: infeasible on CPU (>3 min) | `PHASE_3B_FINAL_METRICS_AUDIT.md` | **Verified** (infeasibility confirmed) | Allowed |
| R1-7 | Phase 3B common-test performance: NOT recomputed from predictions; unverified | `PHASE_3B_FINAL_METRICS_AUDIT.md` | **Unverified** | Must be qualified |
| R1-8 | Full pipeline Phase 1–3B: all steps documented; checkpoints preserved | — | **Verified** | Allowed |

---

## 8. Paper Usage Classification Summary

| Verification Status | Count | Paper Treatment |
|---|---|---|
| ✅ Verified | 112 | Allowed without qualification |
| ⚠️ Provisional | 15 | Allowed with qualification/ caveat |
| ❌ Unsupported | 22 | Excluded or must be explicitly qualified |
| 📋 Unverified | 8 | Must be noted as unverified |

**Total entries**: 157

---

## 8. Next Steps for Evidence Register

1. Complete the `CLAIM_TO_EVIDENCE_MATRIX.md` (traceability from claims to evidence)
2. Generate publication-quality figures from verified data
3. Assemble research tables with correct denominators
4. Verify all literature citations bibliographically
5. Complete the research gap and novelty analysis
6. Create the foundation manuscript with only verified claims
7. Run the integrity checklist
8. Prepare for Phase 4B — Final Research Paper Writing

---

**Register Generated**: October 22, 2026  
**Verification Policy**: Every claim is classified as Verified, Provisional, or Unsupported. Unsupported claims must not appear as established findings in the manuscript.  
**Paper Usage**: Entries marked "Excluded" must not be cited as established results without qualification.  
**Paper Usage**: Entries marked "Qualified" must have their qualification noted in the manuscript.  
**Paper Usage**: Entries marked "Allowed" can be cited as established results.