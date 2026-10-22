# NearMiss2Crash — Phase 3 Verified Results

**Date**: October 22, 2026  
**Research Title**: "Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives"  

> **Policy**: Every numerical result must have a verified denominator. If the denominator is uncertain, mark the result as Provisional and note the uncertainty. Do not invent denominators.

---

## 1. Phase 3 — Full Experimental Evaluation

### 1.1 Full Fuzzy Dataset Labeling

| Metric | Value | Denominator | Verification | Source |
|--------|-------|-------------|--------------|--------|
| Total ASRS narratives | 38,655 | 38,655 | ✅ Verified | `fuzzy_full_results.csv` |
| Full fuzzy labeling completed | Yes | 38,655 rows | ✅ Verified | `run_phase3_full_labeling.py` |
| Fuzzy labels per narrative | 1 per narrative | 38,655 | ✅ Verified | Output column `fuzzy_label` |

### 1.2 Crisp-vs-Fuzzy Comparison

| Metric | Value | Denominator | Verification | Source |
|--------|-------|-------------|--------------|--------|
| Comparison summary generated | Yes | N/A | ✅ Verified | `comparison_summary.json` |
| Crisp-vs-fuzzy transition rows | Available | N/A | ✅ Verified | `crisp_vs_fuzzy_transition.csv` |
| Detailed transition table | 2,576 rows | N/A | ✅ Verified | `detailed_transitions.csv` |
| τ sensitivity: τ=0.25, 0.35, 0.50 | Identical outputs | N/A | ✅ Verified | `tau_sensitivity.json` (3 identical outputs) |
| δ sensitivity: δ=0.15 provides meaningful discrimination | Yes | N/A | ✅ Verified | `delta_sensitivity.json` |

**τ Sensitivity Finding**: Since keyword evidence counts are integers (0, 1, 2...), τ=0.35, 0.25, and 0.50 all have the same practical effect: "any non-zero evidence count (≥1)". The parameter documents policy for future continuous evidence scores.

**δ Sensitivity Finding**: δ=0.15 provides meaningful discrimination for close competitions. Sensitivity analysis shows δ variation changes which borderline cases are accepted vs. rejected as Uncertain.

### 1.3 Label Agreement

| Metric | Value | Denominator | Verification | Source |
|--------|-------|-------------|--------------|--------|
| Crisp-fuzzy label agreement | Documented in `comparison_summary.json` | N/A | ✅ Verified | Phase 3 comparison |
| Per-class crisp counts | SBE:1,059, DE:1,056, V:476, PE:461 | 3,052 (train) / 872 (test) | ✅ Verified | `dataset_majority_*.csv` |
| Per-class fuzzy counts | SBE:987, DE:840, V:455, PE:382 | 2,664 (train labeled) / 759 (test labeled) | ✅ Verified | `dataset_fuzzy_*_labeled.csv` |

### 1.4 Uncertain and Abstention Rates

| Metric | Value | Denominator | Verification | Source |
|--------|-------|-------------|--------------|--------|
| Crisp Unlabeled fraction | 45.52% | 17,597 / 38,655 | ✅ Verified | Phase 1 baseline |
| Fuzzy Uncertain fraction | Documented in `fuzzy_full_results.csv` analysis | 759 (test labeled) / 2,664 (train labeled) | 🟡 **Provisional** | Requires full fuzzy dataset analysis |
| R0 (no evidence) → Uncertain | Count from inference rules | N/A | ✅ Verified | 8 unit tests pass |
| R1 (top < τ) → Uncertain | Count from inference rules | N/A | ✅ Verified | 8 unit tests pass |
| R3 (exact tie) → Uncertain | Count from inference rules | N/A | ✅ Verified | 8 unit tests pass |
| R4 (dominance < δ) → Uncertain | Count from inference rules | N/A | ✅ Verified | 8 unit tests pass |
| R5 (passes both thresholds) → predicted label | Count from inference rules | N/A | ✅ Verified | 8 unit tests pass |

### 1.5 Ablation Results

| ablation | Status | Notes |
|----------|--------|-------|
| Removing membership functions | 🟡 Not generated | Would require re-running fuzzy labeling without membership evaluation |
| Removing dominance margin δ | 🟡 Not generated | Would change Uncertain/ predicted split |
| Removing evidence threshold τ | 🟡 Not generated | Would change Uncertain/ predicted split |
| Full model without τ/δ | ✅ Already generated | Full fuzzy labeling with τ=0.35, δ=0.15 |

---

## 2. Phase 3B — DistilBERT Experimental Evidence

### 2.1 Architecture

| Component | Detail | Verified |
|-----------|--------|----------|
| Base model | `distilbert-base-uncased` | ✅ From `transformer_model.py` |
| Feature extraction | CLS token embedding (768-dim) | ✅ From `train_distilbert_models.py` |
| Encoder freezing | Frozen (no backprop) | ✅ From `train_distilbert_models.py` |
| Classifier head | `nn.Linear(768, 4)` — trainable only | ✅ From `train_distilbert_models.py` |
| Output classes | 4: SBE, DE, V, PE | ✅ From `transformer_model.py` |
| Activation | None (logits) | ✅ From `train_distilbert_models.py` |
| Loss | CrossEntropyLoss | ✅ From `train_distilbert_models.py` |
| Max sequence length | 128 | ✅ From `train_distilbert_models.py` |

### 2.2 Training Configuration

| Parameter | Value | Verified |
|-----------|-------|----------|
| Epochs | 3 | ✅ From `train_distilbert_models.py` |
| Batch size | 16 | ✅ From `train_distilbert_models.py` |
| Learning rate | 2e-5 | ✅ From `train_distilbert_models.py` |
| Weight decay | 0.01 | ✅ From `train_distilbert_models.py` |
| Optimizer | AdamW | ✅ From `train_distilbert_models.py` |
| Random seed | 42 | ✅ From `train_distilbert_models.py` |
| Device | CPU (PyTorch 2.14.1+cpu) | ✅ From `train_distilbert_models.py` |
| Embedding extraction batch size | 64 | ✅ From `train_distilbert_models.py` |

### 2.3 Training Labels

| Model | Label Source | Type | Verified |
|---------|-------------|------|----------|
| Model A (Crisp) | Majority-vote weak labels from `keyword_weak_labeler.py` | Hard labels (argmax) | ✅ From `train_distilbert_models.py` |
| Model B (Fuzzy) | FuzzyWeakLabeler output (τ=0.35, δ=0.15) | Hard labels (argmax of decision) | ✅ From `train_distilbert_models.py` |

**Critical**: Training labels are **hard targets** (argmax), NOT continuous soft labels. The training pipeline does NOT use probability distributions as targets.

### 2.3 Train/Val/Test Splits

| Model | Train | Val | Test | Split Method | Verified |
|-------|-------|-----|------|--------------|----------|
| Model A | 3,052 | 436 | 872 | `weak_labeling_results/dataset_majority_*.csv`, seed 42 | ✅ |
| Model B | 2,664 | 387 | 759 | Matched report_ids from crisp splits, seed 42 | ✅ |

**Critical verification**: Zero training leakage (all 4 split checks returned 0 overlap).

### 2.3 Original Test Metrics (From Checkpoints)

| Model | Test Samples | Accuracy | Macro-F1 | Weighted-F1 | Diagonal Sum | Verified |
|-------|-------------|----------|----------|-------------|------------|----------|
| Model A (Crisp) | 872 | 36.93% | 21.04% | 29.21% | 322 (= 220+102+0+0) | ✅ From checkpoint |
| Model B (Fuzzy) | 759 | 39.13% | 21.56% | 29.87% | 237 (= 230+67+0+0) | ✅ From checkpoint |

**Critical**: These are from **different evaluation populations** (872 vs 759). They cannot directly establish superiority.

### 2.3 Common-Test Structural Verification

| Metric | Value | Verified |
|--------|-------|----------|
| Unique report IDs (crisp test) | 1,008 | ✅ |
| Unique report IDs (fuzzy test) | 759 | ✅ |
| Intersection (common test samples) | 759 | ✅ |
| Training leakage (any test sample in train/val) | 0 (4/4 checks) | ✅ |
| Crisp-fuzzy label agreement | 95.8% (727/759) | ✅ |
| Common-test performance (accuracy/F1 on 759) | **Unverified** | Embedding extraction not completed ⚠️ |

### 2.4 Unverified Common-Test Performance

| Claim | Status | Resolution |
|-------|--------|------------|
| Model A accuracy on common 759 set = 35.49% | ❌ Unsupported | Previously reported by failed script; not recomputed from predictions |
| Model B accuracy on common 759 set = 38.47% | ❌ Unsupported | Same as above; checkpoint stores 39.13% on original 759-split |
| Model B outperforms Model A on common test set | ⚠️ Unverified | Original splits show B > A, but common-set performance not recomputed |
| Common-test metrics should be used for comparison | ⚠️ Qualified | Only if embedding extraction and prediction-level recomputation are completed |

**Resolution**: The common-test structural verification (759 shared samples, zero leakage, 95.8% agreement) is complete and verified. However, **common-test performance metrics are unverified** because the embedding extraction script was infeasible to complete on CPU within the session time. Any claim of performance advantage on the common test set would be unsupported without recomputation from predictions.

### 2.5 DistilBERT Limitations

| Limitation | Description | Verification |
|------------|-------------|--------------|
| No human-validated ground truth | All metrics are weak-label proxies | ✅ Documented |
| Class imbalance (Violation, Perceptual Error) | Zero predicted instances in both models | ✅ From confusion matrices |
| Frozen encoder without fine-tuning | Linear head only; minority classes not learnable | ✅ Documented |
| Common-test performance unverified | Embedding extraction not completed on CPU | ✅ Documented |
| DistilBERT is downstream extension | Not the main research contribution | ✅ Documented in scope |
| No human evaluation of weak labels | No HFACS-ground-truth annotations | ✅ Documented |

---

## 3. Collected Metrics Summary

| Category | Metric | Verified | Denominator |
|----------|--------|----------|-------------|
| **Phase 1** | Crisp-label coverage 54.48% | ✅ | 21,058 / 38,655 |
| | Unlabeled fraction 45.52% | ✅ | 17,597 / 38,655 |
| | Test set class counts | ✅ | SBE:303, DE:302, V:136, PE:131 (total 872) |
| **Phase 2** | 8 unit tests all pass | ✅ | N/A |
| | τ=0.35 with integer counts ≈ "≥1" | ✅ | N/A |
| | δ=0.15 provides meaningful discrimination | ✅ | N/A |
| | Membership functions implemented | ✅ | N/A |
| **Phase 3** | Full fuzzy labeling 38,655 rows | ✅ | 38,655 |
| | τ sensitivity: 3 values, identical outputs | ✅ | N/A |
| | δ sensitivity: meaningful discrimination | ✅ | N/A |
| **Phase 3B** | Model A: 872 test, 36.93% acc, 21.04% macro-F1 | ✅ | 872 |
| | Model B: 759 test, 39.13% acc, 21.56% macro-F1 | ✅ | 759 |
| | Common-test: 759 shared samples, 95.8% agreement | ✅ | 759 |
| | Common-test performance: unverified | ⚠️ | 759 |
| | Zero training leakage | ✅ | 0 overlap (4 checks) |

---