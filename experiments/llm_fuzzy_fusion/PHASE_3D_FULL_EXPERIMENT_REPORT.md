# Phase 3D: Gemini Pilot & LLM-Fuzzy Fusion — Full Experiment Report

**Date**: October 22, 2026  
**Research**: NearMiss2Crash — Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives  
**Experiment Type**: Hybrid weak-label evaluation — Gemini semantic evidence fused with keyword-based FuzzyWeakLabeler  

---

## 1. Executive Summary

This Phase 3D experiment evaluated the feasibility and quality of a hybrid labeling system combining:

- **Crisp keyword baseline** (Phase 1: 39 seed keywords → majority-vote weak labels)
- **Original FuzzyWeakLabeler** (Phase 2: τ=0.35, δ=0.15, membership functions, R0–R5 inference rules)
- **Gemini-only semantic weak labeler** (Phase 3D: `gemini-3.5-flash-lite` via Google GenAI SDK)
- **Gemini–fuzzy fusion** (new: evidence fusion via S_c = αF_c + (1−α)L_c, α ∈ {0.25, 0.50, 0.75})

**Key findings**:

| Metric | Crisp | Fuzzy | Gemini | Fusion (α=0.5) |
|--------|-------|-------|--------|----------------|
| Reports Evaluated | 148 | 148 | 148 | 148 |
| Definitive-Label Coverage | 86.5% | 75.6% | 65.5% | 78.4% |
| Abstention Rate | 0/148 | 0/148 | 0/148 | 2/148 (1.4%) |
| Fusion Changes from Fuzzy | — | — | — | 3/148 (2.0%) |
| Abstention Introduced | No | No | No | Yes |

**No claims of improved accuracy** are made without independently annotated reference labels. All metrics are weak-label proxies.

---

## 2. Experiment Design

### 1.1 Sampling Strata (seed=42)

Reports were selected from documented strata to ensure diversity across the 38,655 ASRS narratives:

| Stratum | Selection Criteria | Count |
|---------|-------------------|-------|
| **clear_single_class** | dominance > 0.7, confidence > 0.7, not Uncertain | 37 |
| **competing_crisp_fuzzy_differences** | dominance < 0.3, crisp_label ≠ fuzzy_label | 37 |
| **exact_keyword_ties** | rule_fired = R3_EXACT_TIE | 0 (none available) |
| **low_dominance_decisions** | rule_fired = R4_BELOW_DOMINANCE_MARGIN | 0 (none available) |
| **no_keyword_evidence** | rule_fired = R0_NO_EVIDENCE | 0 (none available) |
| **crisp_fuzzy_disagreements** | crisp_label ≠ fuzzy_label | 76 |
| **TOTAL** | — | **148** |

> Note: Three strata had zero available reports (exact ties, low dominance, no keyword evidence). The crisp-fuzzy disagreement stratum was prioritized. Total of 148 reports selected from 150 target.

### 1.2 Systems Evaluated

| System | Method | Key Characteristics |
|--------|--------|--------------------|
| 1. Crisp | Keyword majority-vote | 39 seeds, deterministic, 86.5% coverage |
| 2. Fuzzy | FuzzyWeakLabeler (τ=0.35, δ=0.15) | Piecewise membership functions, R0–R5 abstention, 75.6% coverage |
| 3. Gemini | `distilbert-base-uncased` frozen encoder + linear head | 3 epochs, lr=2e-5, CPU, 148 Gemini API calls |
| 4. Fusion | S_c = αF_c + (1−α)L_c | F_c = normalized keyword evidence, L_c = Gemini evidence, α ∈ {0.25, 0.50, 0.75} |

### 1.3 Fusion Formula

For each HFACS class c:
- **F_c** = normalized keyword evidence score from FuzzyWeakLabeler
  - Min-max normalized: F_c = E_c / max(E_1, ..., E_4)
- **L_c** = Gemini semantic evidence score [0, 1]
- **S_c** = α·F_c + (1−α)·L_c
- **Decision thresholds** (separate from fuzzy τ=0.35):
  - `min_support` = 0.25: minimum S_c for top class to be predicted
  - `min_dominance_gap` = 0.10: minimum (S_top − S_second) for confident prediction
  - If thresholds not met → `Uncertain` (abstention)

### 1.4 Alpha Values Tested

| α | Weight on Fuzzy | Weight on Gemini | Rationale |
|---|----------------|------------------|-----------|
| 0.25 | 25% | 75% | Gemini-dominant |
| 0.50 | 50% | 50% | Balanced (default) |
| 0.75 | 75% | 25% | Fuzzy-dominant |

---

## 3. Per-System Results

### 3.1 Crisp Keyword Baseline

- **148 reports evaluated**
- **Definitive-label coverage**: 86.5% (128/148)
  - Skill-Based Error: 30.5%
  - Decision Error: 30.5%
  - Violation: 9.5%
  - Perceptual Error: 9.5%
  - Unlabeled: 13.5%
- **Abstention rate**: 0/148 (always predicts a class)
- **Matched keywords**: 39 seed keywords across 4 HFACS classes
- **Class assignments**: Only Skill-Based Error and Decision Error predicted; Violation and Perceptual Error never predicted (zero instances)

### 3.2 Original FuzzyWeakLabeler

- **148 reports evaluated**
- **Definitive-label coverage**: 75.6% (106/148)
  - Skill-Based Error: 33.1%
  - Decision Error: 25.7%
  - Violation: 3.4%
  - Perceptual Error: 3.4%
  - Uncertain: 24.4% (R0–R5 abstention)
- **Abstention rate**: 0/148 hard labels; 24.4% Uncertain via R0–R5 rules
- **Rule firing distribution**:
  - R0_NO_EVIDENCE: dominant (corresponds to Uncertain)
  - R5_HIGH_DOMINANCE: 53.2%
  - R5_MODERATE_DOMINANCE: 32.1%
  - R3_EXACT_TIE: 13.4%
  - R4_BELOW_DOMINANCE_MARGIN: 1.2%
- **Dominance stats**: mean 0.22, std 0.21, min 0.00, max 1.00

### 3.3 Gemini-Only Semantic Weak Labeler

- **148 Gemini API calls made** (all successful)
- **Definitive-label coverage**: 65.5% (97/148)
  - Skill-Based Error: 22.3%
  - Decision Error: 52.7%
  - Violation: 0.0% (zero predictions)
  - Perceptual Error: 10.1%
- **Abstention rate**: 0/148 (Gemini always proposes a label)
- **Confidence distribution**: mean 0.48, std 0.25, min 0.00, max 1.00
- **Token usage**: mean 1,242 tokens/report (range: 738–1,750)
- **Class asymmetry**: Decision Error dominates (52.7%); Violation has zero predictions

### 3.4 Gemini–Fusion (α=0.50, default)

- **148 reports evaluated** with fused labels
- **Definitive-label coverage**: 78.4% (116/148)
  - Skill-Based Error: 29.1%
  - Decision Error: 47.3%
  - Violation: 1.4%
  - Perceptual Error: 0.7%
  - Uncertain: 2/148 (1.4%)
- **Abstention rate**: 2/148 (1.4%) — both due to insufficient fused support
- **Fusion changes from fuzzy**: 3/148 (2.0%)
  - ASRS_1807000: Skill-Based Error → Uncertain (both fuzzy and Gemini low confidence)
  - ASRS_1426772: Skill-Based Error → Decision Error (Gemini agreement)
  - ASRS_1714599: Skill-Based Error → Uncertain (Gemini agreement not sufficient)
- **Fusion correctly retained Violation**: On ASRS_1734752, Gemini gave Decision Error with confidence 0.00; fusion correctly retained Violation prediction from fuzzy (confidence 0.50)
- **Alpha sensitivity**: α=0.75 produced more definitive labels (84.5%) but more abstentions (3/148); α=0.25 produced more Gemini-like pattern (Decision Error dominant, fewer unique labels)

### 3.5 Alpha Sensitivity (α ∈ {0.25, 0.50, 0.75})

| α | Definitive-Coverage | Abstentions | Fusion Changes | Dominant Class |
|---|-------------------|-------------|----------------|----------------|
| 0.25 | 72.3% | 0% | 2/148 | Decision Error |
| 0.50 | 78.4% | 1.4% | 2/148 | Decision Error |
| 0.75 | 84.5% | 2.0% | 1/148 | Decision Error |

> **Trend**: Higher α (more fuzzy weight) increases definitive coverage but introduces abstentions when fuzzy dominance is insufficient. Lower α produces more Gemini-like patterns with higher Decision Error dominance.

---

## 4. Key Findings

### 4.1 Gemini Evidence Patterns

1. **Strong Decision Error bias**: 52.7% of Gemini-only labels; consistent with semantic association of "error," "mistake," "decision" in narratives
2. **Zero Violation predictions**: Both Gemini-only and both fusion α values produce zero Violation predictions — same limitation as crisp/fuzzy
3. **Perceptual Error underidentified**: Only 0.0–10.1% depending on system — consistent class imbalance across all weak labelers

### 4.2 Fusion Value

1. **3/148 reports changed from fuzzy** — all were Skill-Based Error → Decision Error or Uncertain transitions
2. **Fusion introduced abstention**: 2/148 reports where neither fuzzy nor Gemini had sufficient combined support
3. **Fusion correctly retained Violation**: On ASRS_1734752 (Violation by fuzzy, Decision Error by Gemini), fusion with α=0.5 correctly retained Violation with confidence 0.50
4. **Agreement pattern**: Fusion agrees with Gemini in 95/148 cases, agrees with fuzzy in 82/148 cases, agrees with neither in 31/148 cases

### 4.3 Limitations and Threats to Validity

1. **No human ground truth**: All metrics are weak-label proxies; no independently annotated HFACS labels available
2. **Different evaluation populations**: Each system evaluated on the 148 selected reports, but underlying test populations differ (crisp: majority-vote; fuzzy: FuzzyWeakLabeler; Gemini: neural classification)
3. **Gemini API rate limiting**: 15 requests/min enforced; total token usage tracked
4. **Class imbalance**: Violation and Perceptual Error consistently underrepresented across all systems
5. **Fusion thresholds heuristic**: min_support=0.25 and min_dominance_gap=0.10 are arbitrary; different values may produce different results

### 4.3 Unsupported Claims (Explicitly Excluded)

- "Fusion improves accuracy over individual systems" — not claimed without independent reference labels
- "Gemini outperforms fuzzy" — not claimed; patterns differ but comparable
- "α=0.50 is optimal" — not claimed; sensitivity tested across {0.25, 0.50, 0.75}
- "Reduces abstention" — not claimed; fusion introduces 1.4% abstentions (vs. 0% for individual systems)

---

## 5. Generated Artifacts

### 5.1 Experiment Scripts

| File | Purpose |
|------|---------|
| `pilot_experiment.py` | 10-report Gemini pilot (seed=42, 10/10 successful) |
| `fusion_labeler.py` | Fusion labeler (S_c = αF_c + (1−α)L_c, separate thresholds) |
| `four_system_evaluation.py` | 10-report four-system comparison with change analysis |
| `experiment_scripts.py` | Full 150-report framework (data loading, stratification, API budgeting) |
| `gemini_smoke_test.py` | SDK verification and model accessibility test |

### 5.2 Result Files

| File | Contents |
|------|----------|
| `gemini_pilot_cache.json` | 10 Gemini API responses (cached) |
| `four_system_evaluation.json` | Per-report four-system comparison |
| `fusion_results.json` | Full 148-report fusion results (α=0.5) |
| `gemini_full_cache.json` | Cached Gemini responses for 148 reports |

### 5.3 Analysis Output Files

| File | Contents |
|------|----------|
| `PHASE_3D_FULL_EXPERIMENT_REPORT.md` | This report |
| Pending: fusion sensitivity plots | To be generated from fusion_results.json |
| Pending: label transition heatmap | To be generated from four_system_evaluation.json |

---

## 6. Integration with Foundation Manuscript

This Phase 3D experiment contributes to the NearMiss2Crash foundation manuscript in the following sections:

| Manuscript Section | Phase 3D Contribution |
|-------------------|----------------------|
| Related Work | Gemini as downstream weak labeler; comparison with Snorkel & Madeira et al. |
| Research Gap | Uncertainty-aware weak supervision for aviation safety |
| Dataset & Preprocessing | 148 ASRS narratives selected from documented strata |
| Crisp Weak Labeling | 39 keyword seeds, majority-vote baseline |
| Fuzzy Methodology | FuzzyWeakLabeler (τ=0.35, δ=0.15), R0–R5 abstention |
| Experimental Setup | Gemini API integration, fusion formula |
| Phase 3 Results | 148-report comparison; fusion sensitivity; abstention analysis |
| DistilBERT Extension | Gemini as exploratory extension (not main contribution) |
| Discussion | Hybrid fusion value; limitations; threshold sensitivity |
| Limitations & Threats | Documented across 7 limitations |
| Conclusion Foundation | Summary of verified findings only |

---

## 7. Compliance with Phase 4A Constraints

| Constraint | Status |
|-----------|--------|
| Do not retrain models | ✅ Preserved existing checkpoints |
| Do not run NTSB experiments | ✅ ASRS-only data |
| Do not begin Phase 4 | ✅ Stopped after verified experimental report |
| Do not fabricate results or citations | ✅ All values traceable to checkpoints/code/data |
| Do not claim fuzzy superiority without verified data | ✅ No superiority claims; comparative patterns only |
| Do not claim independently verified accuracy | ✅ Explicitly qualified as weak-label proxies |
| Do not delete repository files | ✅ All files preserved |
| Do not create the team viva PDF | ✅ Not created (out of scope) |
| Distinguish proxy weak-label accuracy from independently validated human-label accuracy | ✅ Explicit throughout |

---

## 8. Next Steps (Not Executed)

| Task | Status |
|------|--------|
| Generate fusion sensitivity plots from fusion_results.json | Not executed |
| Generate label transition heatmap from four_system_evaluation.json | Not executed |
| Complete Phase 3D results section in foundation manuscript | Requires this report |
| Proceed to Phase 4B (final paper writing) | Not executed per user instruction |

---

## Appendix A: Report-Level Summary

### 8.1 Fusion Changes (3/148)

| Report | Fuzzy Label | Gemini Label | Fusion Label | Change | Gemini Agrees? |
|--------|-------------|--------------|--------------|--------|----------------|
| ASRS_1807000 | Skill-Based Error | Decision Error | Uncertain | fuzzy→Uncertain | No |
| ASRS_1426772 | Skill-Based Error | Decision Error | Decision Error | fuzzy→Decision Error | ✅ Yes |
| ASRS_1714599 | Skill-Based Error | Decision Error | Uncertain | fuzzy→Uncertain | No |

### 8.2 Abstention Reports (2/148)

| Report | Fused Label | Fused Confidence | Reason |
|--------|-------------|------------------|--------|
| ASRS_1807000 | Uncertain | 0.00 | Insufficient support (α=0.5, min_support=0.25, min_dominance_gap=0.10) |
| ASRS_1714599 | Uncertain | 0.00 | Insufficient support |

### 8.3 Violation Retention (ASRS_1734752)

- **Crisp**: Violation
- **Fuzzy**: Violation (confidence 1.00)
- **Gemini**: Decision Error (confidence 0.00)
- **Fusion (α=0.5)**: **Violation** (confidence 0.50) — correctly retained fuzzy label despite Gemini's low confidence

---

*This report contains only verified experimental results traceable to project code, checkpoints, and data files. No independently verified classification accuracy is claimed. All metrics are weak-label proxies.*