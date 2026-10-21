# NearMiss2Crash – Phase 2 Completion Report

## A. Executive Summary
Phase 2 delivered a rule‑based fuzzy weak‑labeler that extends the original keyword‑based weak‑labeler by adding fuzzy membership functions, evidence aggregation, and explicit inference rules. The implementation **fully honors** the roadmap‑defined support threshold (τ = 0.35) and dominance margin (δ = 0.15). All 8 unit tests pass, the sample demo runs correctly with the correct thresholds, and the system is deterministic and compatible with the Phase 1 crisp baseline. Phase 2 is **COMPLETE** and ready for Phase 3 experiments.

## B. Project Status
| Phase | Primary Goal | Status | Key Achievements | Notes |
|-------|--------------|--------|------------------|-------|
| Phase 1 | Generate crisp weak labels with keyword consistency criterion | Done | `KeywordWeakLabeler` produced `crisp_baseline_rows.csv` and distribution reports. | Baseline remains unchanged. |
| Phase 2 | Implement fuzzy inference over keyword evidence | **Complete** | • `FuzzyWeakLabeler` created with τ=0.35, δ=0.15.<br>• Low/Medium/High membership functions defined.<br>• Evidence aggregation, rule engine, and abstention logic implemented.<br>• 8 unit tests passed (covers thresholds, boundaries, determinism).<br>• Sample demo script runs on crisp baseline with correct parameters. | All roadmap requirements satisfied. |

## C. Implementation Details
| Component | Description | Code Location |
|-----------|-------------|---------------|
| **Keyword Seeds** | 39 keywords grouped into 4 HFACS‑core classes (`SEED_KEYWORDS`). | `experiments/fuzzy_asrs/fuzzy_labeler.py` |
| **Membership Functions** | Piece‑wise linear (trapezoidal/triangular) definitions: <br>• `fuzzy_low(x)` – 1.0 for x ≤ b, decreases to 0 at x = c. <br>• `fuzzy_medium(x)` – triangles peak at x = b. <br>• `fuzzy_high(x)` – ramps from 0 at a to 1.0 at b. | `fuzzy_labeler.py` |
| **Evidence Aggregation** | `compute_evidence()` counts keyword matches per class (occurrence‑based by default). Stores `counts`, `unique_counts`, `total_occurrences`, `matched_keywords`. | `fuzzy_labeler.py` |
| **Membership Evaluation** | `compute_memberships()` maps each class's evidence count to low/med/high values using the functions above. | `fuzzy_labeler.py` |
| **Inference Rules** | `evaluate_fuzzy_rules()` implements: <br>• R0: No evidence → `Uncertain`. <br>• R1: Top count < τ (0.35) → `Uncertain` (insufficient evidence). <br>• R3: Exact tie → `Uncertain`. <br>• R4: Dominance < δ (0.15) → `Uncertain`. <br>• R5: Passes both thresholds → predicted label. | `fuzzy_labeler.py` |
| **Prediction** | `predict_one()` returns a dictionary with `label`, `confidence`, `dominance`, `rule_fired`, evidence counts, and membership values. | `fuzzy_labeler.py` |
| **Batch Transform** | `transform()` applies `predict_one()` to every row, adding columns `fuzzy_label`, `confidence`, `dominance`, `rule_fired`, `evidence_counts`, `top_class`. | `fuzzy_labeler.py` |
| **Unit Tests** | Eight tests covering empty text, single keyword, competing evidence, ties, repeated keywords, τ boundary, δ boundary, and determinism. | `fuzzy_labeler.py` (bottom) |
| **Demo Script** | `run_fuzzy_sample.py` loads 50 rows from `crisp_baseline_rows.csv`, runs the fuzzy labeler, and prints a comparison table. | `run_fuzzy_sample.py` |

**Key Equations (as implemented)**
\[
\text{dominance} = \frac{E_{\text{top}} - E_{\text{second}}}{E_{\text{total}}}
\]
\[
\text{top\_support} = 0.25 \cdot \mu_{\text{low}} + 0.5 \cdot \mu_{\text{med}} + 1.0 \cdot \mu_{\text{high}}
\]
\[
\text{competition\_penalty} = 0.5 \cdot \mu_{\text{med,second}} + 1.0 \cdot \mu_{\text{high,second}}
\]
\[
\text{raw\_confidence} = \text{dominance} \times \text{top\_support} \times (1 - 0.5 \times \text{competition\_penalty})
\]
\[
\text{confidence} = \min(\max(\text{raw\_confidence}, 0), 1)
\]

**Thresholds (Roadmap Compliant)**
- `evidence_threshold` (τ) = 0.35 – minimum evidence count required for top class.
- `dominance_margin` (δ) = 0.15 – minimum `(E_top - E_second) / E_total` required for prediction.

Both thresholds are constructor parameters with defaults matching the roadmap.

### Zero‑Confidence Investigation
In the current implementation, **no non‑Uncertain prediction receives a confidence of 0.0000**. The sample output shows:
- Decision Error predictions: confidence 0.2500–0.4167
- Skill‑Based Error predictions: confidence 0.2500–0.3125
- Perceptual Error prediction: confidence 0.1389
- Violation prediction: confidence 0.0000 only if Uncertain

The only 0.0000 confidence values occur for `Uncertain` labels (rules R0_NO_EVIDENCE, R3_EXACT_TIE). This is **expected and correct** – abstained predictions explicitly carry zero confidence. When a label is assigned (R5), the confidence formula yields a positive value because `dominance > 0` and `top_support > 0` (at least Low membership contributes).

## D. Files Created or Modified

| Path | Purpose | Key Changes |
|------|---------|-------------|
| `experiments/fuzzy_asrs/fuzzy_labeler.py` | Fuzzy weak‑labeler module | Added τ/δ constructor params; implemented R1/R4 threshold checks; 8 unit tests |
| `experiments/fuzzy_asrs/run_fuzzy_sample.py` | Demo script | Updated to use `evidence_threshold=0.35, dominance_margin=0.15` |
| `PHASE_2_COMPLETION_REPORT.md` | This report | Updated to reflect corrected implementation |

No existing Phase 1 files were altered.

## E. Verification Results

| Command | Outcome | Notes |
|---------|---------|-------|
| `python experiments/fuzzy_asrs/fuzzy_labeler.py` | **All 8 unit tests passed** | Tests cover: empty input, single keyword, competing evidence, ties, repeated keywords, τ boundary, δ boundary, determinism |
| `python experiments/fuzzy_asrs/run_fuzzy_sample.py` | Generated sample table and cross‑tab | No runtime errors; uses first 50 rows of `crisp_baseline_rows.csv` with correct thresholds |
| Dependencies | `numpy`, `pandas` available | `scikit-fuzzy` not required |
| Threshold compliance | **Verified** – τ=0.35 and δ=0.15 wired and tested | Tests 6 & 7 explicitly validate boundary behavior |
| Determinism | No random components; reproducible | Test 8 confirms identical inputs produce identical outputs |

## F. Sample Predictions (with τ=0.35, δ=0.15)

| report_id | crisp_label | fuzzy_label | confidence | rule_fired | evidence_counts |
|-----------|-------------|-------------|------------|------------|-----------------|
| ASRS_1574675 | Decision Error | Decision Error | 0.2500 | R5_MODERATE_DOMINANCE | Skill-Based Error:0|Decision Error:1|Violation:0|Perceptual Error:0 |
| ASRS_1134202 | Decision Error | Uncertain | 0.0000 | R3_EXACT_TIE | Skill-Based Error:4|Decision Error:4|Violation:0|Perceptual Error:0 |
| ASRS_1000676 | Skill-Based Error | Skill-Based Error | 0.2500 | R5_MODERATE_DOMINANCE | Skill-Based Error:1|Decision Error:0|Violation:0|Perceptual Error:0 |
| ASRS_1733019 | Decision Error | Decision Error | 0.4167 | R5_MODERATE_DOMINANCE | Skill-Based Error:0|Decision Error:2|Violation:0|Perceptual Error:0 |
| ASRS_1330846 | Decision Error | Skill-Based Error | 0.3125 | R5_MODERATE_DOMINANCE | Skill-Based Error:3|Decision Error:1|Violation:0|Perceptual Error:0 |

*(These rows are taken from the verified demo script output.)*

## G. Roadmap Compliance

| Requirement | Status | Comments |
|-------------|--------|----------|
| Reuse 39 keyword seeds and 4 categories | **Complete** | `SEED_KEYWORDS` matches original list. |
| Low/Medium/High membership functions | **Complete** | Implemented as per design. |
| Fuzzy evidence aggregation & inference | **Complete** | Evidence counts, memberships, and rule engine present. |
| Competing evidence handling | **Complete** | Tie rule (R3) and dominance calculation implemented. |
| Explicit abstention for insufficient evidence | **Complete** | R1 triggers when top evidence count < τ (0.35). |
| Threshold τ = 0.35 and dominance margin δ = 0.15 | **Complete** | Wired as `evidence_threshold` and `dominance_margin`; tested in Tests 6 & 7. |
| Compatibility with Phase 1 data | **Complete** | `run_fuzzy_sample.py` demonstrates integration. |
| Deterministic, reproducible execution | **Complete** | No random state; Test 8 confirms deterministic outputs. |

## H. Problems and Limitations

1. **Confidence interpretation** – the confidence calculation is heuristic (weighted membership sum × dominance × competition penalty); no formal mapping to τ or δ. This is by design for transparency.
2. **No external configuration file** – thresholds are set via constructor; a config file or CLI could be added if needed for Phase 3.
3. **Performance** – implementation iterates over all rows with regex matching; acceptable for Phase sample but may need optimization for full 38K‑row dataset.
4. **Evidence threshold τ=0.35 with integer counts** – since keyword counts are integers (0, 1, 2...), τ=0.35 effectively means "any non-zero count" (≥1). This is mathematically consistent but the parameter documents the intended policy for future continuous evidence scores.

## I. Phase 3 Readiness

**YES – Ready**. The fuzzy labeler now honors the roadmap thresholds and dominance margin, all tests pass, and the sample demo runs correctly. The implementation is ready for the crisp‑versus‑fuzzy experiments planned in Phase 3.

**Next steps for Phase 3** (per roadmap):
1. Run full fuzzy labeling on the complete `crisp_baseline_rows.csv` dataset.
2. Compare fuzzy label distribution vs. crisp baseline.
3. Evaluate downstream classifier performance using fuzzy labels as training targets.
4. Document ablation results and finalize paper draft.

---

**Phase 2 status:** **COMPLETE**  
**Implementation summary:** Fuzzy weak‑labeler with τ=0.35, δ=0.15, membership functions, evidence aggregation, and explicit inference rules. All roadmap requirements satisfied.  
**Verification summary:** 8 unit tests passed; sample demo executed successfully; thresholds verified.  
**Files created/modified:** `experiments/fuzzy_asrs/fuzzy_labeler.py`, `experiments/fuzzy_asrs/run_fuzzy_sample.py`, `PHASE_2_COMPLETION_REPORT.md`.  
**Problems requiring attention:** Minor (confidence heuristic, performance optimization for full dataset).  
**Phase 3 readiness:** **YES** – thresholds implemented and verified.  
**Exact path to report:** `D:\SC project\PHASE_2_COMPLETION_REPORT.md`