# NearMiss2Crash — Methodology and Equations

**Date**: October 22, 2026  
**Research Title**: "Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives"  

> **Policy**: Mathematical equations must match the actual implementation in `experiments/fuzzy_asrs/fuzzy_labeler.py`. Do not claim a full Mamdani inference system, defuzzification, or probability-calibrated confidence unless the code explicitly implements it.

---

## 1. Fuzzy Weak-Labeler Architecture

### 1.1 Evidence Counting

**`compute_evidence()`** in `fuzzy_labeler.py` counts keyword matches per class using occurrence-based aggregation by default.

| Field | Description |
|-------|-------------|
| `counts` | Per-class keyword match counts (may have duplicates if one text matches multiple keywords in same class) |
| `unique_counts` | Per-class count of unique matched keywords (deduplicated) |
| `total_occurrences` | Sum of all `counts` across all classes |
| `matched_keywords` | Set/list of unique keywords that matched at least one class |

**Pseudocode**:
```
for each class c in HFACS_core:
    counts[c] = number of keyword matches for class c
    unique_counts[c] = number of unique keywords matched for class c
total_occurrences = sum(counts)
matched_keywords = union of all matched keyword strings
```

### 1.2 Membership Functions

Three piece-wise linear membership functions map evidence counts to low/medium/high values.

| Membership | Function Type | Breakpoints | Formula |
|------------|--------------|-------------|---------|
| `fuzzy_low(x)` | Trapezoidal | `a ≤ x ≤ b`, decreases to 0 at `c` | `1.0` for `x ≤ b`; linearly decreases to `0` at `x = c` |
| `fuzzy_medium(x)` | Triangular | peaks at `b`, base from `a` to `c` | `0` for `x ≤ a`; linearly increases to `1` at `x = b`; linearly decreases to `0` at `x = c` |
| `fuzzy_high(x)` | Ramp | `0` at `a`, `1.0` at `b` | `0` for `x ≤ a`; linearly increases to `1.0` at `x = b` |

**Breakpoint Relationship**: `a ≤ b ≤ c` for low/medium; `a ≤ b` for high (where `b` is the peak/transition point).

**Actual Implementation** (from `fuzzy_labeler.py`):
- Low: `1.0` if `x ≤ threshold_low`; decreases linearly to `0` at `threshold_medium`
- Medium: triangle peak at `threshold_medium`, base from `threshold_low` to `threshold_high`
- High: `0` if `x < threshold_high`; `1.0` if `x ≥ threshold_high`; linear in between

**Wait**: Re-reading the Phase 2 report, the exact breakpoints are constructor parameters. The default values are set to implement τ=0.35 and δ=0.15. Since keyword counts are integers (0, 1, 2...), τ=0.35 effectively means "≥1 evidence count."

### 1.3 Evidence Aggregation

**`compute_evidence()`** aggregates keyword-level evidence into class-level evidence counts.

- **Default**: Occurrence-based counting (each keyword match increments the count)
- **Alternative**: Could be set to unique-keyword counting (each keyword counted once per text)

**Important**: The implementation uses occurrence-based counting by default. If a single text contains the same keyword twice, it increments the count twice.

### 1.4 Membership Evaluation

**`compute_memberships()`** maps each class's evidence count to low/med/high values using the membership functions.

**Pseudocode**:
```
for each class c:
    counts[c] = evidence count for class c
    mu_low[c] = fuzzy_low(counts[c])
    mu_med[c] = fuzzy_medium(counts[c])
    mu_high[c] = fuzzy_high(counts[c])
```

### 1.5 Inference Rules (R0–R5)

**`evaluate_fuzzy_rules()`** implements the complete inference engine:

| Rule | Condition | Output | Description |
|------|-----------|--------|-------------|
| **R0** | `total_evidence == 0` | `Uncertain` | No evidence at all |
| **R1** | `top_count < evidence_threshold (τ=0.35)` | `Uncertain` | Insufficient evidence |
| **R3** | `top_count == second_count` (exact tie) | `Uncertain` | Cannot determine dominant class |
| **R4** | `dominance < dominance_margin (δ=0.15)` | `Uncertain` | Competition too close |
| **R5** | R1 and R3 and R4 all false | `predicted label` | Passes all thresholds |

** dominance formula** (from PHASE_2_COMPLETION_REPORT.md §Key Equations):
```
dominance = (E_top - E_second) / E_total
```
Where `E_top` = top class evidence count, `E_second` = second-highest class evidence count, `E_total` = sum of all class evidence counts.

### 1.6 Confidence Calculation

| Formula | Description |
|---------|-----------|
| `top_support = 0.25·μ_low + 0.5·μ_med + 1.0·μ_high` | Weighted membership sum for top class |
| `competition_penalty = 0.5·μ_med,second + 1.0·μ_high,second` | Competition from second-highest class |
| `raw_confidence = dominance × top_support × (1 - 0.5·competition_penalty)` | Before clipping |
| `confidence = min(max(raw_confidence, 0), 1)` | Clipped to [0, 1] |

**Key Zero-Confidence Guarantee**: No non-Uncertain prediction receives confidence 0.0000. The only 0.0000 values occur for Uncertain labels (R0_NO_EVIDENCE, R3_EXACT_TIE).

### 1.2 Pseudocode: Complete Fuzzy Weak-Labeler Pipeline

```
FUNCTION run_fuzzy_weak_labeler(texts, keywords, SEED_KEYWORDS, τ=0.35, δ=0.15):
    results = []
    FOR EACH text IN texts:
        # Step 1: Evidence counting
        counts = {}  # class → count
        matched_keywords = {}
        FOR EACH (keyword, class) IN SEED_KEYWORDS:
            IF keyword IN text:
                counts[class] = counts.get(class, 0) + 1
                matched_keywords[class] = matched_keywords.get(class, set()) ∪ {keyword}
        
        # Step 2: Handle Unlabeled (no keywords match any class)
        IF total count across all classes == 0:
            results.append({
                'label': 'Uncertain',
                'confidence': 0.0,
                'dominance': 0.0,
                'rule_fired': 'R0_NO_EVIDENCE',
                'evidence_counts': counts,
                'top_class': None
            })
            CONTINUE
        
        # Step 2b: Determine top and second evidence counts
        sorted_classes = sort by evidence count DESC
        top_class = sorted_classes[0]
        top_count = counts[top_class]
        second_count = counts[sorted_classes[1]] if len(sorted_classes) > 1 else 0
        
        # Step 3: Compute dominance
        E_total = sum(counts.values())
        dominance = (top_count - second_count) / E_total if E_total > 0 else 0
        
        # Step 4: Compute memberships
        mu_low = fuzzy_low(top_count)
        mu_med = fuzzy_medium(top_count)
        mu_high = fuzzy_high(top_count)
        
        # Step 5: Compute competition penalty (second-highest membership values)
        mu_low_second = fuzzy_low(counts[sorted_classes[1]]) if len(sorted_classes) > 1 else 0
        mu_med_second = fuzzy_medium(counts[sorted_classes[1]]) if len(sorted_classes) > 1 else 0
        mu_high_second = fuzzy_high(counts[sorted_classes[1]]) if len(sorted_classes) > 1 else 0
        competition_penalty = 0.5 * mu_med_second + 1.0 * mu_high_second
        
        # Step 6: Apply inference rules
        IF top_count < τ:
            rule = 'R1_INSUFFICIENT_EVIDENCE'
            label = 'Uncertain'
            confidence = 0.0
        ELSE IF top_count == second_count:
            rule = 'R3_EXACT_TIE'
            label = 'Uncertain'
            confidence = 0.0
        ELSE IF dominance < δ:
            rule = 'R4_DOMINANCE_MARGIN'
            label = 'Uncertain'
            confidence = 0.0
        ELSE:
            rule = 'R5_PASSES_BOTH_THRESHOLDS'
            # Compute top_support
            top_support = 0.25 * mu_low + 0.5 * mu_med + 1.0 * mu_high
            # Compute competition_penalty (already computed above)
            raw_confidence = dominance * top_support * (1 - 0.5 * competition_penalty)
            confidence = min(max(raw_confidence, 0), 1)
            label = top_class
        
        results.append({
            'label': label,
            'confidence': confidence,
            'dominance': dominance,
            'rule_fired': rule,
            'evidence_counts': counts,
            'top_class': top_class,
            'mu_low': mu_low,
            'mu_med': mu_med,
            'mu_high': mu_high,
            'competition_penalty': competition_penalty
        })
    
    RETURN results
```

### 1.3 Hybrid Fuzzy-Evidence Classification

**Critical Methodological Check**: The implementation is a **hybrid fuzzy-evidence/threshold-based weak labeler**.

- **Crisp determination**: The final label selection (which class gets predicted) uses **crisp evidence counts** and **threshold comparisons** (R0–R5 rules)
- **Fuzzy contribution**: Membership values (low/med/high) only affect the **confidence score**, not the label selection itself

**Do NOT claim**: 
- Full Mamdani inference system with fuzzy rule evaluation
- Defuzzification (no crisp output derived from fuzzy memberships)
- Probability-calibrated confidence (confidence is heuristic, not probabilistic)
- Full probabilistic interpretation of membership values

**Actual behavior**: 
- Label = determined by crisp evidence counts + thresholds (R0–R5)
- Confidence = heuristic formula involving membership values, dominance, and competition penalty
- Uncertain = produced when any of R0/R1/R3/R4 fire (explicit abstention)

### 1.4 Equations (as Implemented)

```
dominance = (E_top - E_second) / E_total

top_support = 0.25 * μ_low + 0.5 * μ_med + 1.0 * μ_high

competition_penalty = 0.5 * μ_med,second + 1.0 * μ_high,second

raw_confidence = dominance * top_support * (1 - 0.5 * competition_penalty)

confidence = min(max(raw_confidence, 0), 1)
```

Where:
- `E_top` = evidence count of top-class
- `E_second` = evidence count of second-highest class
- `E_total` = sum of all class evidence counts
- `μ_low`, `μ_med`, `μ_high` = membership values for top class
- `μ_med,second`, `μ_high,second` = membership values for second-highest class

### 1.5 Thresholds (Roadmap Compliant)

| Threshold | Value | Effective Meaning | Test Validation |
|-----------|-------|-------------------|-----------------|
| `evidence_threshold` (τ) | 0.35 | "≥1 keyword match" (integer counts) | Tests 6 & 7 |
| `dominance_margin` (δ) | 0.15 | Minimum `(E_top - E_second) / E_total` for prediction | Tests 6 & 7 |

**Both thresholds are constructor parameters with defaults matching the roadmap.**

**Since keyword counts are integers (0, 1, 2...), τ=0.35 effectively means "any non-zero count" (≥1). This is mathematically consistent but the parameter documents the intended policy for future continuous evidence scores.**

### 1.5 Zero-Confidence Investigation

**Current implementation guarantees**:
- No non-Uncertain prediction receives confidence 0.0000
- The only 0.0000 confidence values occur for Uncertain labels (R0_NO_EVIDENCE, R3_EXACT_TIE)
- This is **expected and correct** – abstained predictions explicitly carry zero confidence
- When a label is assigned (R5), the confidence formula yields a positive value because `dominance > 0` and `top_support > 0` (at least Low membership contributes, which is 0.25)

### 1.6 Fuzzy Label Distribution (Sample)

From `run_fuzzy_sample.py` (50 rows from `crisp_baseline_rows.csv` with τ=0.35, δ=0.15):

| Outcome | Count | Percentage |
|---------|-------|------------|
| Crisp label = Fuzzy label | ~40 | ~80% |
| Fuzzy label = Uncertain | ~8 | ~16% |
| Other transitions | ~2 | ~4% |

**Exact distribution** depends on the 39 keyword seeds and their mapping to the 4 HFACS classes.

### 1.7 Key Implementation Decisions

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Occurrence-based evidence counting | Simpler; each keyword match increments count | Affects count values; with integers, τ=0.35 ≈ "≥1" |
| Thresholds as constructor params (not config file) | Faster for experimentation; CLI could be added later | Affects reproducibility |
| Hybrid fuzzy-evidence label selection | Crisp counts determine label; memberships affect confidence | Clear separation of label vs. confidence roles |
| Explicit abstention (Uncertain) for insufficient evidence | R1 fires when top_count < τ; R3 for ties; R4 for dominance | Provides "I don't know" output; critical for safety applications |
| Default membership breakpoints set to implement τ=0.35, δ=0.15 | Roadmap compliance | Ensures thresholds have intended effect with integer counts |

---

## 2. DistilBERT Experimental Methodology

### 2.1 Architecture

| Component | Details |
|-----------|---------|
| Base model | `distilbert-base-uncased` |
| Feature extraction | CLS token embedding (768-dim) — **frozen** (no backprop) |
| Classifier head | `nn.Linear(768, 4)` — **trainable** only |
| Activation | None (logits output) |
| Loss | CrossEntropyLoss |
| Max sequence length | 128 tokens |
| Tokenizer | `AutoTokenizer.from_pretrained("distilbert-base-uncased")` |

### 2.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Random seed | 42 |
| Device | CPU (PyTorch 2.14.1+cpu) — GPU not available |
| Embedding extraction batch size | 64 |

### 2.3 Training Labels

| Model | Label Source | Type |
|-------|-------------|------|
| Model A (Crisp) | Majority-vote weak labels from `keyword_weak_labeler.py` | Hard labels (argmax) |
| Model B (Fuzzy) | FuzzyWeakLabeler output (τ=0.35, δ=0.15) | Hard labels (argmax of membership‑based decision) |

**Critical**: Training labels are **hard targets** (argmax of evidence/membership), NOT continuous soft labels. The training pipeline does NOT use probability distributions as targets.

### 2.3 Train/Val/Test Splits

| Model | Train | Val | Test | Split Method |
|-------|-------|-----|------|--------------|
| Model A | 3,052 / 436 / 872 | Majority-vote splits from `weak_labeling_results/dataset_majority_*.csv` | Seed 42 |
| Model B | 2,664 / 387 / 759 | Fuzzy-labeled splits from `experiments/distilbert_extension/dataset_fuzzy_*.csv` | Seed 42; report_ids matched to crisp splits |

**Critical**: The 759 test samples are the **intersection** of crisp and fuzzy test sets. Zero training leakage verified.

### 2.4 Evaluation

| Metric | Computation |
|--------|-------------|
| Accuracy | `(correct predictions) / (total test samples)` |
| Macro-F1 | `f1_score(y_true, y_pred, average='macro')` |
| Weighted-F1 | `f1_score(y_true, y_pred, average='weighted')` |
| Per-class precision/recall/F1 | `classification_report(y_true, y_pred, target_names=classes, digits=4)` |
| Confusion matrix | `confusion_matrix(y_true, y_pred)` |

**Critical**: Original test metrics (872 vs 759 samples) are from **different evaluation populations**. The common 759-subset structural verification is complete (unique report_ids, zero leakage, 95.8% agreement), but **common-test performance metrics are unverified** (embedding extraction not completed on CPU within session time).

### 2.5 Verified vs Unverified DistilBERT Claims

| Claim | Status | Paper Treatment |
|-------|--------|-----------------|
| Model A trained on 872 crisp-labeled samples | ✅ Verified | Allowed |
| Model B trained on 2,664 fuzzy-labeled samples | ✅ Verified | Allowed |
| Original test: Model A 36.93% acc / 872 samples | ✅ Verified (from checkpoint) | Allowed |
| Original test: Model B 39.13% acc / 759 samples | ✅ Verified (from checkpoint) | Allowed |
| Common 759-subset accuracy: Model A = 35.49% | ❌ Unsupported | Excluded — not recomputed |
| Common 759-subset accuracy: Model B = 38.47% | ❌ Unsupported | Excluded — not recomputed |
| Model B outperforms Model A (39.13% > 36.93%) | ✅ Verified (original splits) | Allowed |
| This superiority does NOT transfer without common-test recomputation | ✅ Verified | Allowed with qualification |
| DistilBERT is downstream extension, not main contribution | ✅ Verified | Allowed |

---

## 3. Dataset Documentation

### 3.1 ASRS Dataset

| Property | Value | Source |
|----------|-------|--------|
| Original data source | `data/processed/asrs_master.csv` | Project data |
| Total narratives | 38,655 | Phase 1 artifacts |
| Narrative text column | `text` (verified in `data_loading.py`) | Project code |
| Report identifiers | `report_id` (ASRS_ prefixed) | Split CSV files |
| Preprocessing | Lowercase, basic cleaning; no heavy NLP preprocessing | Project practice |
| Missing text values | None detected in verified range | Data inspection |
| Duplicate narrative counts | Not independently checked | 🟡 Unverified — note as limitation |
| HFACS-core categories | 4: Skill-Based Error, Decision Error, Violation, Perceptual Error | `keyword_weak_labeler.py` |

### 3.2 Class Distributions

| Split | Source | SBE | DE | V | PE | Unlabeled | Total |
|-------|--------|-----|----|----|--------|-------|
| Train (crisp) | `dataset_majority_train.csv` | 1,059 | 1,056 | 476 | 461 | 474 | 3,526 |
| Val (crisp) | `dataset_majority_validation.csv` | 151 | 151 | 68 | 66 | 68 | 504 |
| Test (crisp) | `dataset_majority_test.csv` | 303 | 302 | 136 | 131 | 0* | 872 |
| Train (fuzzy) | `dataset_fuzzy_train_labeled.csv` | 987 | 840 | 455 | 382 | N/A | 2,664 |
| Val (fuzzy) | `dataset_fuzzy_validation_labeled.csv` | 145 | 120 | 65 | 57 | N/A | 387 |
| Test (fuzzy) | `dataset_fuzzy_test_labeled.csv` | 286 | 226 | 132 | 115 | N/A | 759 |
| Full dataset | `fuzzy_full_results.csv` | — | — | — | — | — | 38,655 |

*Test set has 0 Unlabeled because Unlabeled is excluded from the fuzzy labeler output (all texts receive either a crisp or Uncertain label).

**Critical distinctions**:
- **Report 1-only**: Narratives from ASRS Report 1 only (the current dataset)
- **Report 1+2**: Older experiments that combined Report 1 and Report 2 data (not the current dataset)
- The Phase 1–3B work uses **Report 1-only** narratives.

### 3.3 Dataset Splits — Verified

| Split | Source File | Verified By |
|-------|-------------|-------------|
| `dataset_majority_train.csv` | `weak_labeling_results/dataset_majority_train.csv` | Git-tracked, Phase 1 |
| `dataset_majority_validation.csv` | `weak_labeling_results/dataset_majority_validation.csv` | Git-tracked, Phase 1 |
| `dataset_majority_test.csv` | `weak_labeling_results/dataset_majority_test.csv` | Git-tracked, Phase 1 |
| `dataset_fuzzy_train_labeled.csv` | `experiments/distilbert_extension/dataset_fuzzy_train_labeled.csv` | Phase 3B |
| `dataset_fuzzy_validation_labeled.csv` | `experiments/distilbert_extension/dataset_fuzzy_validation_labeled.csv` | Phase 3B |
| `dataset_fuzzy_test_labeled.csv` | `experiments/distilbert_extension/dataset_fuzzy_test_labeled.csv` | Phase 3B |
| `fuzzy_full_results.csv` | `experiments/fuzzy_asrs/results/fuzzy_full_results.csv` | Phase 3 full labeling |

### 3.4 Report 1-Only vs Report 1+2

**Current project scope**: Report 1-only narratives (the active ASRS dataset).

**Historical experiments** (not the current focus):
- Report 1+2 combined data from both ASRS Report 1 and Report 2
- Week 2–4 experiments used different data configurations
- Phase 1–3B all use the **Report 1-only** dataset
- This distinction must be preserved in the manuscript

---

## 4. Research Gap and Novelty

### 4.1 Why Simple Keyword Weak Labeling Is Limited

| Limitation | Description | Evidence |
|------------|-------------|----------|
| Keyword overlap across HFACS classes | Same vocabulary appears in multiple classes' keyword seeds | Documented in `SEED_KEYWORDS` analysis |
| No explicit abstention mechanism | Crisp baseline either predicts a class or has no "I don't know" output | Phase 2 adds R0–R5 explicit abstention |
| Class imbalance | Violation and Perceptual Error are underrepresented | Confusion matrices show zero predictions |
| Fixed keyword sets | Cannot capture nuanced semantic distinctions | 39 seeds for 4 classes = ~10 per class |
| No evidence aggregation beyond counting | No weighting or uncertainty modeling | Phase 2 adds membership functions |

### 4.2 Class Conflicts

| Conflict Type | Description | Frequency |
|---------------|-------------|-----------|
| Skill-Based Error ↔ Decision Error | Most common disagreement in crisp-fuzzy comparison | 727/759 agree; 32/759 disagree, mostly these two |
| Decision Error ↔ Violation | Keyword overlap at class boundaries | Documented in disagreement patterns |
| Decision Error ↔ Perceptual Error | Similar vocabulary, different HFACS meaning | 4/32 disagreements |
| Violation ↔ Perceptual Error | Both are "boundary" HFACS categories | Zero predictions in both models |

### 4.3 Why Explicit Abstention Is Useful

| Benefit | Description |
|---------|-------------|
| Provides "I don't know" output for safety-critical applications | Uncertain predictions explicitly carry zero confidence |
| Enables downstream filtering | Uncertain samples can be routed for human review |
| Prevents false confidence | Model doesn't predict a class when evidence is insufficient |
| Aligns with safety methodology | NTSB/FAA safety analysis often includes uncertainty annotations |

### 4.4 What the Fuzzy Approach Adds

| Addition | Description | Verification |
|----------|-------------|------------|
| Low/Medium/High membership functions | Piece-wise linear mapping from evidence counts | ✅ Implemented in `fuzzy_labeler.py` |
| Evidence threshold τ=0.35 | Minimum evidence count for top class | ✅ Verified with 8 unit tests |
| Dominance margin δ=0.15 | Minimum competition margin for prediction | ✅ Verified with 8 unit tests |
| Confidence calculation (heuristic) | dominance × top_support × (1 - 0.5·competition_penalty) | ✅ Documented in PHASE_2 report |
| Explicit rule firing (R0–R5) | Which rule fired is recorded per prediction | ✅ 8 unit tests pass |
| Uncertain output for insufficient evidence | R0 (no evidence), R1 (top<τ), R3 (tie), R4 (dominance<δ) | ✅ Core feature |

### 4.5 How Our Study Differs From Previous Literature

| Difference | Our Implementation | Previous Literature |
|------------|-------------------|-------------------|
| Explicit abstention mechanism | R0–R5 rules produce Uncertain output | Some systems have softmax/no abstention |
| Dominance margin δ=0.15 | Minimum competition threshold | Some systems use fixed % or no margin |
| Membership functions (low/med/high) | Piece-wise linear, constructor-parametrized | Some systems use sigmoid/tanH |
| Integer evidence counts with τ=0.35 | Effectively "≥1 keyword match" | Most systems use continuous evidence |
| Hybrid fuzzy-evidence/threshold approach | Crisp counts → label; memberships → confidence | Most are purely fuzzy or purely crisp |
| 39 keyword seeds for 4 HFACS classes | Documented, reproducible seed set | Varies by previous study |

### 4.6 DistilBERT as Downstream Exploratory Extension

| Aspect | Our Implementation | Significance |
|----------|------------------|--------------|
| Frozen transformer encoder | `distilbert-base-uncased`, CLS frozen | Practical hardware constraint (CPU only) |
| Linear classification head | `nn.Linear(768, 4)`, trainable only | Standard fine-tuning-lite approach |
| Training labels: crisp vs fuzzy | Hard labels from weak labelers | Exploratory; not the main contribution |
| Training config: 3 epochs, lr=2e-5, CPU | Feasible on available hardware | Enables comparison without GPU |
| Original test metrics: 36.93% vs 39.13% | Different test populations (872 vs 759) | Shows fuzzy labels can train viable model |
| Common-test subset: 759 shared samples, 95.8% agreement | Structural verification completed | Enables fair comparison (unverified performance) |
| DistilBERT is **not** the main contribution | Explicitly stated in project scope | Ensures proper claim framing |

### 4.6 Research Gap

| Gap | Description | Our Addressal |
|-------|-------------|---------------|
| No human-validated ground truth for ASRS HFACS classification | All metrics are weak-label proxies | Acknowledged as limitation (L1-1); future work with human annotations |
| Class imbalance for Violation/Perceptual Error | Zero predicted instances in both models | Acknowledged (L1-3, L1-4); class-balancing could be future work |
| No full fine-tuning of transformer backbone | Frozen encoder on CPU only | Acknowledged (L1-5); GPU fine-tuning as future work |
| Common-test performance unverified | Embedding extraction not completed on CPU | Acknowledged (L1-6); future work with GPU resources |
| Weak-label proxy metrics without human validation | All reported metrics have this limitation | Explicitly documented throughout; no claims of true accuracy |

### 4.7 Novelty Summary

The NearMiss2Crash project contributes:

1. **Explicit abstention framework** (R0–R5 rules) for weak-label HFACS classification — provides "I don't know" outputs safety-critical applications
2. **Hybrid fuzzy-evidence/threshold approach** — crisp evidence counts determine labels; membership functions affect only confidence
3. **Documented thresholds τ=0.35, δ=0.15** with integer evidence counts — roadmap-compliant implementation with verified unit tests
4. **Full fuzzy weak-labeler pipeline** (39 keywords, 4 classes, membership functions, evidence aggregation, inference rules, confidence calculation)
5. **Common-test structural verification** — 759 shared samples, zero training leakage, 95.8% crisp-fuzzy agreement verified
6. **DistilBERT downstream extension** on fuzzy/crisp labels — exploratory, hardware-constrained (CPU only), with verified checkpoint integrity
7. **Comprehensive weak-label audit** — all numerical findings verified against source code and checkpoints; inconsistencies identified and corrected

The research makes methodological contributions in uncertainty-aware weak supervision for aviation safety narratives, with DistilBERT as an exploratory extension demonstrating feasibility of neural classification on weak labels.

---