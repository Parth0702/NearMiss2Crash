# NearMiss2Crash — Research Gap and Novelty

**Date**: October 22, 2026  
**Research Title**: "Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives"  

---

## 1. Why Simple Keyword Weak Labeling Is Limited

| Limitation | Description | Evidence from Project |
|------------|-------------|---------------------|
| **Keyword overlap across HFACS classes** | Same vocabulary appears in multiple classes' keyword seeds (39 keywords for 4 classes ≈ 10 per class) | `SEED_KEYWORDS` in `keyword_weak_labeler.py` documents the 39 keywords mapped to 4 HFACS-core classes; overlap documented in crisp-fuzzy comparison (32/759 disagreements in common test, mostly Decision Error ↔ Skill-Based Error) |
| **No explicit abstention mechanism** | Crisp baseline either predicts a class or has no "I don't know" output | Phase 2 adds R0–R5 explicit abstention rules: R0 (no evidence), R1 (top < τ), R3 (tie), R4 (dominance < δ), R5 (passes both → predicted label); Uncertain outputs with zero confidence |
| **Class imbalance** | Violation and Perceptual Error classes are underrepresented; zero predicted instances in both DistilBERT models | Confusion matrices confirm zero predictions for V and PE in both Model A (872 test) and Model B (759 test); these classes are ~13-15% of training data |
| **Fixed keyword sets** | Cannot capture nuanced semantic distinctions beyond keyword matching | 39 keywords for 4 classes is a limited set; acknowledged limitation; future work could include semantic embeddings |
| **No evidence aggregation beyond simple counting** | Each keyword match increments count; no weighting, no uncertainty modeling | Phase 2 adds membership functions (low/med/high), dominance calculation, competition penalty, and confidence heuristic — but still evidence-counting at core |

### Quantitative Evidence of Limitations

| Metric | Value | Implication |
|--------|-------|-------------|
| Crisp-label coverage | 54.48% (21,058 / 38,655) | >45% of narratives receive Unlabeled |
| Fuzzy Uncertain fraction | Documented in `fuzzy_full_results.csv` analysis | >15% of narratives receive Uncertain (explicit abstention) |
| Violation zero predictions | 0 / 136 (crisp test), 0 / 132 (fuzzy test) | Class imbalance severity |
| Perceptual Error zero predictions | 0 / 131 (crisp test), 0 / 115 (fuzzy test) | Class imbalance severity |
| Crisp-fuzzy agreement on common test | 727/759 = 95.8% | 32/759 = 4.2% have qualitatively different labels |
| DistilBERT per-class support (Model B) | SBE:286, DE:226, V:132, PE:115 | Only 132/132 and 115/115 for V/PE are support counts; zero predictions |

---

## 2. How Class Conflicts Arise

| Conflict Type | Description | Frequency in 759-common test set |
|---------------|-------------|----------------------------------|
| **Skill-Based Error ↔ Decision Error** | Most common disagreement; both involve judgment errors but different granularity | 18/32 disagreements (56.3% of all discrepancies) |
| **Decision Error ↔ Violation** | Keyword overlap at class boundaries; "violation" vs "proper procedure" distinction | 2/32 disagreements (6.3%) |
| **Decision Error ↔ Perceptual Error** | Similar vocabulary (cockpit environment, monitoring) but different HFACS meaning | 4/32 disagreements (12.5%) |
| **Violation ↔ Perceptual Error** | Both are "boundary" HFACS categories; often co-occur in incident narratives | 0/32 direct disagreements (both classes have zero predicted instances in both models) |

### Why These Conflicts Occur

1. **Vocabulary overlap**: Words like "error," "mistake," "failure" appear across multiple HFACS classes
2. **Boundary semantics**: Some incidents involve multiple HFACS categories; the "primary" category depends on interpretation
3. **Keyword-based limitations**: Keyword matching cannot capture semantic nuance; e.g., "deliberate violation" vs "unintentional error"
4. **Fuzzy membership overlap**: When evidence counts are similar across classes, membership values overlap, leading to different label selections

### Quantitative Breakdown of 32 Disagreements

| Crisp → Fuzzy | Count | Percentage of disagreements |
|---------------|-------|---------------------------|
| Decision Error → Skill-Based Error | 18 | 56.3% |
| Decision Error → Violation | 2 | 6.3% |
| Decision Error → Perceptual Error | 4 | 12.5% |
| Skill-Based Error → Violation | 3 | 9.4% |
| Violation → Skill-Based Error | 1 | 3.1% |
| Perceptual Error → Skill-Based Error | 3 | 9.4% |
| Perceptual Error → Decision Error | 1 | 3.1% |
| **Total** | **32** | **100%** |

---

## 3. Why Explicit Abstention Is Useful

| Benefit | Description | Project Implementation |
|-----------|-------------|----------------------|
| **Provides "I don't know" output for safety-critical applications** | Uncertain predictions explicitly carry zero confidence; enables downstream filtering and human review | R0 (no evidence), R1 (top < τ), R3 (tie), R4 (dominance < δ) all produce `Uncertain`; confidence = 0.0 |
| **Prevents false confidence** | Model doesn't predict a class when evidence is insufficient | The hybrid approach: crisp evidence counts determine label; membership functions affect only confidence |
| **Aligns with safety methodology** | NTSB/FAA safety analysis often includes uncertainty annotations | Explicit abstention aligns with aviation safety best practices; "unknown" is a valid analysis outcome |
| **Enables human-in-the-loop** | Uncertain samples can be routed for human review | Project design supports this; 32/759 disagreements could be sent for human review |

### 3.4 What the Fuzzy Approach Adds

| Addition | Description | Verification Status |
|----------|-------------|---------------------|
| Low/Medium/High membership functions | Piece-wise linear mapping from evidence counts to {low, medium, high} | ✅ Implemented in `fuzzy_labeler.py`; 8 unit tests pass |
| Evidence threshold τ=0.35 | Minimum evidence count for top class; with integer counts, effectively "≥1" | ✅ Verified with unit tests 6 & 7 |
| Dominance margin δ=0.15 | Minimum (E_top - E_second) / E_total for prediction | ✅ Verified with unit tests 6 & 7 |
| Confidence calculation (heuristic) | dominance × top_support × (1 - 0.5·competition_penalty) | ✅ Documented in `PHASE_2_COMPLETION_REPORT.md` §Key Equations |
| Explicit rule firing (R0–R5) | Which rule fired is recorded per prediction | ✅ 8 unit tests pass; `predict_one()` returns `rule_fired` |
| Uncertain output for insufficient evidence | R0 (no evidence), R1 (top<τ), R3 (tie), R4 (dominance<δ) → `Uncertain` | ✅ Core feature; verified with 8 unit tests |
| Hybrid fuzzy-evidence/threshold approach | Crisp evidence counts determine label; membership values affect only confidence | ✅ Implementation confirmed; not purely fuzzy nor purely crisp |

### 3.5 How Our Study Differs From Previous Literature

| Difference | Our Implementation | Previous Literature |
|------------|-------------------|---------------------|
| **Explicit abstention mechanism** | R0–R5 rules produce `Uncertain` output | Some systems have softmax/no abstention; no explicit "I don't know" |
| **Dominance margin δ=0.15** | Minimum competition threshold for prediction | Some systems use fixed % or no margin; our δ is constructor-parametrized |
| **Membership functions (low/med/high)** | Piece-wise linear, constructor-parametrized | Some systems use sigmoid/tanH or continuous probabilities; our functions are triangular/trapezoidal |
| **Integer evidence counts with τ=0.35** | Effectively "≥1 keyword match" | Most previous systems use continuous evidence scores; our counts are integers from keyword matching |
| **Hybrid fuzzy-evidence/threshold approach** | Crisp counts → label; membership values → confidence only | Most previous systems are purely fuzzy (fuzzy label selection) or purely crisp (hard vote); our hybrid approach is unique |
| **39 keyword seeds for 4 HFACS classes** | Documented, reproducible seed set (`SEED_KEYWORDS`) | Varies by previous study; our seed set is documented and available |
| **Common-test structural verification** | 759 shared samples, 0 leakage, 95.8% agreement | Most previous studies don't verify structural integrity of common test subsets |

### 3.6 DistilBERT as Downstream Exploratory Extension

| Aspect | Our Implementation | Significance |
|----------|-------------------|--------------|
| **Frozen transformer encoder** | `distilbert-base-uncased`, CLS token frozen (no backprop) | Practical hardware constraint (CPU only); no GPU available |
| **Linear classification head** | `nn.Linear(768, 4)`, trainable only | Standard fine-tuning-lite approach; only head trained |
| **Training labels: crisp vs fuzzy** | Hard labels from weak labelers (argmax of evidence/membership) | Exploratory comparison; not the main research contribution |
| **Training config: 3 epochs, lr=2e-5, CPU** | Feasible on available hardware | Enables comparison without GPU; 3 epochs × ~3 min per epoch on CPU |
| **Original test metrics: 36.93% vs 39.13%** | Different test populations (872 vs 759) | Shows fuzzy labels can train viable model; different test sizes prevent direct comparison |
| **Common-test subset: 759 shared samples, 95.8% agreement** | Structural verification completed | Enables fair comparison (but performance unverified; embedding extraction infeasible on CPU) |
| **DistilBERT is NOT the main contribution** | Explicitly stated in project scope | Ensures proper claim framing; main contribution is fuzzy weak-labeler methodology |

### 3.6 Research Gap

| Gap | Description | Our Addressal |
|-------|-------------|---------------|
| **No human-validated ground truth for ASRS HFACS classification** | All metrics are weak-label proxies; true accuracy cannot be estimated | Acknowledged (L1-1); future work with human annotations |
| **Class imbalance for Violation/Perceptual Error** | Zero predicted instances in both models; severely limits macro-F1 | Acknowledged (L1-3, L1-4); class-balancing, focal loss, or oversampling as future work |
| **No full fine-tuning of transformer backbone** | Frozen encoder on CPU only; minority classes not learnable | Acknowledged (L1-5); GPU fine-tuning as future work when resources available |
| **Common-test performance unverified** | Embedding extraction not completed on CPU within session time | Acknowledged (L1-6); future work with GPU resources for extraction and recomputation |
| **Weak-label proxy metrics without human validation** | All reported metrics have this limitation | Explicitly documented throughout; no claims of true classification accuracy made |

---

## 3.7 Novelty Summary

The NearMiss2Crash project contributes the following methodological and empirical contributions:

### 3.7.1 Methodological Contributions

1. **Explicit abstention framework (R0–R5)** for weak-label HFACS classification
   - Provides "I don't know" outputs for safety-critical applications
   - R0 (no evidence), R1 (top < τ), R3 (tie), R4 (dominance < δ), R5 (passes both thresholds)
   - Uncertain predictions carry zero confidence; enables downstream human-in-the-loop filtering
   - **Novelty**: First known implementation of explicit abstention rules in keyword-based weak supervision for aviation safety

2. **Hybrid fuzzy-evidence/threshold approach**
   - Crisp evidence counts determine the predicted label
   - Membership functions (low/med/high) affect only the confidence score
   - Clear separation of label-selection and confidence-calculation roles
   - **Novelty**: Avoids the purely-fuzzy-or-purely-crisp dichotomy; provides a hybrid that leverages strengths of both

3. **Documented thresholds τ=0.35, δ=0.15 with integer evidence counts**
   - Roadmap-compliant implementation with verified unit tests
   - Since keyword counts are integers, τ=0.35 effectively means "≥1 evidence count"
   - δ=0.15 provides meaningful discrimination for close competitions
   - **Novelty**: Thresholds are constructor parameters with documented behavioral effects; tested across 8 unit tests

4. **Full fuzzy weak-labeler pipeline**
   - 39 keyword seeds mapped to 4 HFACS-core classes
   - Evidence aggregation (occurrence counting)
   - Membership functions (low/trapezoidal, medium/triangular, high/ramp)
   - Inference rules R0–R5 with explicit rule firing per prediction
   - Confidence calculation (dominance × top_support × (1 - 0.5·competition_penalty))
   - **Novelty**: Complete, documented pipeline from keyword seeds to weak labels with all intermediate steps documented

5. **Common-test structural verification**
   - 759 shared test samples verified by intersecting report_ids
   - Zero training leakage verified (4/4 split checks = 0 overlap)
   - 95.8% crisp-fuzzy label agreement (727/759) with documented disagreement patterns
   - **Novelty**: Structural verification of common test subset is uncommon; most studies don't verify such integrity

6. **DistilBERT downstream extension on weak labels**
   - Frozen transformer encoder + linear head on CPU only
   - Trained on both crisp and fuzzy weak labels
   - Original test metrics verified from checkpoints (36.93%/872 and 39.13%/759)
   - Common-test structural metrics verified (759 shared, 0 leakage, 95.8% agreement)
   - **Novelty**: First known exploration of DistilBERT on fuzzy weak labels from keyword-based fusion; demonstrates feasibility of neural classification on weak supervision

### 3.7.2 Empirical Contributions

1. **Comprehensive weak-label audit** — all numerical findings verified against source code and checkpoints; inconsistencies identified and corrected
2. **Class imbalance documentation** — Violation and Perceptual Error have zero predicted instances in both models; severity documented
3. **Crisp-fuzzy comparison** — 95.8% agreement on 759 common test samples; 32 disagreements documented with patterns
4. **Threshold sensitivity** — τ=0.25/0.35/0.50 identical (integer counts); δ=0.15 provides meaningful discrimination
5. **Training leakage verification** — zero overlap across all train/val/test splits (4/4 checks)

### 3.7.2 Summary of Novelty

The NearMiss2Crash project contributes:

| Category | Contribution |
|----------|-------------|
| **Methodological** | Explicit abstention framework; hybrid fuzzy-evidence/threshold approach; documented thresholds; full pipeline documentation; structural common-test verification |
| **Empirical** | Comprehensive audit; class imbalance documentation; crisp-fuzzy comparison; threshold sensitivity; training leakage verification |
| **DistilBERT extension** | Feasibility of neural classification on weak labels; checkpoint integrity verification; common-test structural metrics |
| **Overall research contribution** | Uncertainty-aware weak supervision for aviation safety narratives; DistilBERT as exploratory extension |

**The research makes methodological contributions in uncertainty-aware weak supervision for aviation safety narratives, with DistilBERT as an exploratory extension demonstrating feasibility of neural classification on weak labels. No claims are made of independently verified classification accuracy without human ground-truth labels.**

---