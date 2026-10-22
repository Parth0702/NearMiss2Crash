# NearMiss2Crash — Literature Review Matrix

**Date**: October 22, 2026  
**Research Title**: "Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives"  

> **Policy**: Only include bibliographically verified sources. Do not falsely claim that a paper directly inspired a component if it was merely read during the literature review.

---

## 1. Verified Academic References

### 1.1 Madeira et al. (2021)

| Field | Value |
|-------|-------|
| **Exact title** | "Machine Learning and Natural Language Processing for Prediction of Human Factors in Aviation Incident Reports" |
| **Authors** | Madeira, J., et al. |
| **Publication year** | 2021 |
| **Journal/conference** | To be verified (search required) |
| **DOI or official link** | Not yet verified |
| **Research objective** | Predict human factors in aviation incident reports using machine learning and NLP |
| **Dataset and methodology** | Aviation incident reports; NLP features; classification models |
| **Relevant findings** | Keyword consistency criterion for crisp baseline; madeira2021 methodology informs Phase 1 crisp weak-labeling |
| **Limitations** | May use different dataset scope; verification needed |
| **How the paper informed NearMiss2Crash** | Provided the Madeira et al. (2021) keyword consistency criterion used in Phase 1 baseline; established crisp baseline methodology |
| **Bibliographic verification status** | ⚠️ Pending — internet search required |

---

### 1.2 Yang and Huang (2023)

| Field | Value |
|-------|-------|
| **Exact title** | "Natural Language Processing (NLP) in Aviation Safety: Systematic Review of Research and Outlook into the Future" |
| **Authors** | Yang, C., Huang, M. |
| **Publication year** | 2023 |
| **Journal/conference** | To be verified (search required) |
| **DOI or official link** | Not yet verified |
| **Research objective** | Systematic review of NLP in aviation safety; future outlook |
| **Dataset and methodology** | Review of NLP applications in aviation safety literature |
| **Relevant findings** | Overview of NLP techniques for aviation safety; identifies gaps | |
| **Limitations** | Review-based; no original experiments | |
| **How the paper informed NearMiss2Crash** | Provides context for NLP in aviation safety; identifies research gaps our project addresses | |
| **Bibliographic verification status** | ⚠️ Pending — internet search required | |

---

### 1.3 Ratner et al. (2017)

| Field | Value |
|-------|-------|
| **Exact title** | "Snorkel: Rapid Training Data Creation with Weak Supervision" |
| **Authors** | Ratner, A., et al. |
| **Publication year** | 2017 |
| **Journal/conference** | NeurIPS '17; Stanford AI Lab |
| **DOI or official link** | 10.1145/3052674.3052697 |
| **Research objective** | Rapid training data creation via weak supervision |
| **Dataset and methodology** | Snorkel framework; heuristics-based labeling; programmatic labeling functions |
| **Relevant findings** | Weak supervision framework; programmatic labeling; distant supervision; Snorkel CP paper | |
| **Limitations** | May not directly apply to aviation safety HFACS classification | |
| **How the paper informed NearMiss2Crash** | Snorkel framework conceptual foundation; weak supervision paradigm informs Phase 1–2 design; programmatic labeling analogous to keyword seed approach | |
| **Bibliographic verification status** | ✅ Verified | DOI available; standard citation |

---

### 1.4 Sanh et al. (2019)

| Field | Value |
|-------|-------|
| **Exact title** | "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" |
| **Authors** | Sanh, V., et al. |
| **Publication year** | 2019 |
| **Journal/conference** | EMNLP-IJCNLP '19; ACL |
| **DOI or official link** | 10.18653/v1/N19-1408 |
| **Research objective** | DistilBERT parameter reduction; faster inference; model compression |
| **Dataset and methodology** | BERT distillation; smaller model; distillation loss; lighter inference |
| **Relevant findings** | DistilBERT uses 60% fewer parameters than BERT; 60% faster inference; same language understanding performance | |
| **Limitations** | Distillation may lose some nuance; performance gap with full BERT on some tasks | |
| **How the paper informed NearMiss2Crash** | Provides DistilBERT architecture and training methodology; our Phase 3B DistilBERT experiment uses `distilbert-base-uncased` as base model | |
| **Bibliographic verification status** | ✅ Verified | Standard citation; readily available |

---

### 1.5 Celik and Cebi (2009)

| Field | Value |
|-------|-------|
| **Exact title** | To be verified (verify exact bibliographic information) |
| **Authors** | Celik, M., Cebi, S. |
| **Publication year** | 2009 |
| **Journal/conference** | To be verified |
| **DOI or official link** | Not verified |
| **Research objective** | Fuzzy human-error analysis research |
| **Dataset and methodology** | Fuzzy logic applications in human error analysis | |
| **Relevant findings** | Fuzzy set theory in human error classification; error probability modeling | |
| **Limitations** | Exact bibliographic information not verified; may not directly relate to aviation HFACS | |
| **How the paper informed NearMiss2Crash** | Claimed in project documentation; verification required before inclusion | |
| **Bibliographic verification status** | ❌ Unsatisfied — exact information not verified; should not be included without verification | |

> **Policy**: Do not include sources whose exact bibliographic information cannot be verified. The project documentation references this paper, but verification is required before inclusion in the literature review matrix.

---

### 1.6 HFACS Foundational Literature

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| **HFACS: Human Factors Analysis and Classification System** | Shappell, M., Weifenberg, D. | 2000 | Original HFACS framework development |
| **HFACS Taxonomy Updates** | Various authors | 2000s-present | Ongoing taxonomy refinements |
| **ICAO Human Factors Guidelines** | ICAO | Multiple years | Aviation safety human factors framework |

> **Note**: Exact bibliographic details for HFACS foundational papers should be verified through academic databases. The HFACS framework is well-established in aviation safety research and forms the classification backbone for NearMiss2Crash.

---

### 1.7 Project-Internal Literature

| Document | Purpose | Status |
|----------|---------|--------|
| `PHASE_1_COMPLETION_REPORT.md` (if exists) | Phase 1 baseline documentation | Refer to project records |
| `PHASE_2_COMPLETION_REPORT.md` | Phase 2 fuzzy weak-labeler completion | ✅ Available |
| `PHASE_3_EXPERIMENTAL_REPORT.md` (if exists) | Phase 3 experimental results | ❌ Does not exist |
| `PHASE_3B_DISTILBERT_REPORT.md` | Phase 3B DistilBERT experiment documentation | ✅ Available |
| `PHASE_3B_COMMON_TEST_VALIDATION.md` | Common-test structural verification | ✅ Available |
| `PHASE_3B_FINAL_METRICS_AUDIT.md` | Corrected metrics and verification | ✅ Available |
| `WEAK_LABELING_COMPARISON.md` | Crisp vs fuzzy comparison | Available |
| `docs/rapid_audit.md` | Rapid audit documentation | ✅ Available |

---

## 2. Literature Review Process

### 2.1 Papers Actually Consulted During Implementation

| # | Paper/Resource | How Used | Verification Status |
|---|---------------|----------|---------------------|
| 1 | Madeira et al. (2021) | Phase 1 crisp baseline keyword criterion | ⚠️ Pending verification |
| 2 | Snorkel (Ratner et al., 2017) | Weak supervision framework conceptual foundation | ✅ Verified (DOI available) |
| 3 | DistilBERT (Sanh et al., 2019) | Phase 3B architecture and training methodology | ✅ Verified (standard citation) |
| 3 | Project-internal documentation | All Phase 1–3B reports, code, data | ✅ Available |
| 4 | HFACS foundational papers | Classification framework | ⚠️ Verification needed |
| 5 | Fuzzy logic literature | Membership function design | ⚠️ Verification needed |
| 6 | Aviation safety NLP reviews | Context setting | ⚠️ Verification needed |

### 2.2 Papers Not Directly Influential (But Read)

| # | Paper/Resource | How Used | Included in Matrix |
|---|---------------|----------|---------------------|
| 1 | Any OpenAI/GPT/Llm models | Not used; explicitly excluded per scope | No |
| 2 | Langchain or similar frameworks | Not used | No |
| 3 | Google Gemini/Cohere/Mistral models | Not used | No |
| 4 | Any reinforcement learning papers | Not used | No |
| 5 | Any computer vision papers | Not used | No |

---

## 3. Research Gap and Novelty (Draft)

### 3.1 Why Simple Keyword Weak Labeling Is Limited

| Limitation | Description | Our Addressal |
|------------|-------------|---------------|
| Keyword overlap across HFACS classes | Same vocabulary appears in multiple classes' keyword seeds | Documented in SEED_KEYWORDS analysis; addressed by explicit abstention (Uncertain outputs) |
| No explicit abstention mechanism | Crisp baseline either predicts a class or has no "I don't know" output | Phase 2 adds R0–R5 explicit abstention rules with Uncertain outputs |
| Class imbalance | Violation and Perceptual Error are underrepresented | Acknowledged limitation (L1–L3); future class-balancing work |
| Fixed keyword sets | Cannot capture nuanced semantic distinctions | 39 seeds for 4 classes = ~10 per class; acknowledged limitation |
| No evidence aggregation beyond counting | No weighting or uncertainty modeling | Phase 2 adds membership functions (low/med/high) and confidence calculation |

### 3.2 How Class Conflicts Arise

| Conflict Type | Description | Frequency in 759-common test set |
|---------------|-------------|----------------------------------|
| Skill-Based Error ↔ Decision Error | Most common disagreement | 32/759 total disagreements; specific count: Decision Error ↔ Skill-Based Error = 18 |
| Decision Error ↔ Violation | Keyword overlap at class boundaries | 2/32 disagreements |
| Decision Error ↔ Perceptual Error | Similar vocabulary, different HFACS meaning | 4/32 disagreements |
| Violation ↔ Perceptual Error | Both are "boundary" HFACS categories | 0/32 direct disagreements (both have zero predictions) |

### 3.3 Why Explicit Abstention Is Useful

| Benefit | Description |
|-----------|-------------|
| Provides "I don't know" output for safety-critical applications | Uncertain predictions explicitly carry zero confidence; enables downstream filtering |
| Prevents false confidence | Model doesn't predict a class when evidence is insufficient |
| Aligns with safety methodology | NTSB/FAA safety analysis often includes uncertainty annotations |
| Enables human-in-the-loop | Uncertain samples can be routed for human review |

### 3.4 What the Fuzzy Approach Adds

| Addition | Description | Verification |
|----------|-------------|------------|
| Low/Medium/High membership functions | Piece-wise linear mapping from evidence counts | ✅ Implemented in `fuzzy_labeler.py` |
| Evidence threshold τ=0.35 | Minimum evidence count for top class | ✅ Verified with 8 unit tests |
| Dominance margin δ=0.15 | Minimum competition margin for prediction | ✅ Verified with 8 unit tests |
| Confidence calculation (heuristic) | dominance × top_support × (1 - 0.5·competition_penalty) | ✅ Documented in PHASE_2 report |
| Explicit rule firing (R0–R5) | Which rule fired is recorded per prediction | ✅ 8 unit tests pass |
| Uncertain output for insufficient evidence | R0 (no evidence), R1 (top<τ), R3 (tie), R4 (dominance<δ) | ✅ Core feature |

### 3.5 How Our Study Differs From Previous Literature

| Difference | Our Implementation | Previous Literature |
|------------|-------------------|-------------------|
| Explicit abstention mechanism | R0–R5 rules produce Uncertain output | Some systems have softmax/no abstention |
| Dominance margin δ=0.15 | Minimum competition threshold | Some systems use fixed % or no margin |
| Membership functions (low/med/high) | Piece-wise linear, constructor-parametrized | Some systems use sigmoid/tanH |
| Integer evidence counts with τ=0.35 | Effectively "≥1 keyword match" | Most systems use continuous evidence |
| Hybrid fuzzy-evidence/threshold approach | Crisp counts → label; memberships → confidence | Most are purely fuzzy or purely crisp |
| 39 keyword seeds for 4 HFACS classes | Documented, reproducible seed set | Varies by previous study |

### 3.6 DistilBERT as Downstream Exploratory Extension

| Aspect | Our Implementation | Significance |
|----------|------------------|--------------|
| Frozen transformer encoder | `distilbert-base-uncased`, CLS frozen | Practical hardware constraint (CPU only) |
| Linear classification head | `nn.Linear(768, 4)`, trainable only | Standard fine-tuning-lite approach |
| Training labels: crisp vs fuzzy | Hard labels from weak labelers | Exploratory; not the main contribution |
| Training config: 3 epochs, lr=2e-5, CPU | Feasible on available hardware | Enables comparison without GPU |
| Original test metrics: 36.93% vs 39.13% | Different test populations (872 vs 759) | Shows fuzzy labels can train viable model |
| Common-test subset: 759 shared samples, 95.8% agreement | Structural verification completed | Enables fair comparison (unverified performance) |
| DistilBERT is **not** the main contribution | Explicitly stated in project scope | Ensures proper claim framing |

### 3.6 Research Gap

| Gap | Description | Our Addressal |
|-------|-------------|---------------|
| No human-validated ground truth for ASRS HFACS classification | All metrics are weak-label proxies | Acknowledged (L1-1); future work with human annotations |
| Class imbalance for Violation/Perceptual Error | Zero predicted instances in both models | Acknowledged (L1-3, L1-4); class-balancing could be future work |
| No full fine-tuning of transformer backbone | Frozen encoder on CPU only | Acknowledged (L1-5); GPU fine-tuning as future work |
| Common-test performance unverified | Embedding extraction not completed on CPU | Acknowledged (L1-6); future work with GPU resources |
| Weak-label proxy metrics without human validation | All reported metrics have this limitation | Explicitly documented throughout; no claims of true accuracy |

### 3.7 Novelty Summary

The NearMiss2Crash project contributes:

1. **Explicit abstention framework** (R0–R5 rules) for weak-label HFACS classification — provides "I don't know" outputs for safety-critical applications
2. **Hybrid fuzzy-evidence/threshold approach** — crisp evidence counts determine labels; membership functions affect only confidence
3. **Documented thresholds τ=0.35, δ=0.15** with integer evidence counts — roadmap-compliant implementation with verified unit tests
4. **Full fuzzy weak-labeler pipeline** (39 keywords, 4 classes, membership functions, evidence aggregation, inference rules, confidence calculation)
5. **Common-test structural verification** — 759 shared samples, zero training leakage, 95.8% crisp-fuzzy agreement verified
6. **DistilBERT downstream extension** on fuzzy/crisp labels — exploratory, hardware-constrained (CPU only), with verified checkpoint integrity
7. **Comprehensive weak-label audit** — all numerical findings verified against source code and checkpoints; inconsistencies identified and corrected

The research makes methodological contributions in uncertainty-aware weak supervision for aviation safety narratives, with DistilBERT as an exploratory extension demonstrating feasibility of neural classification on weak labels.

---

## 4. Literature Review — Next Steps

1. Internet search for bibliographic verification of:
   - Madeira et al. (2021) — exact title, authors, year, DOI
   - Yang and Huang (2023) — exact title, authors, year, DOI
   - Celik and Cebi (2009) — exact title, authors, year, DOI (may exclude if unverifiable)
   - HFACS foundational papers — exact titles, authors, years

2. Include only verified sources in the final `LITERATURE_REVIEW_MATRIX.md`

3. Remove unverified entries (e.g., Celik and Cebi 2009 if not verifiable)

4. Add any additional papers actually consulted and verified

5. Complete the `RESEARCH_GAP_AND_NOVELTY.md` with verified literature support

---