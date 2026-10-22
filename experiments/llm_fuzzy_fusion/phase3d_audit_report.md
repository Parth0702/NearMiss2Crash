## Phase 3D Audit Report — NearMiss2Crash

**Date**: October 22, 2026  
**Purpose**: Audit and correct Phase 3D before IEEE manuscript integration

---

### 1. Executive Summary

The Phase 3D 148-report experiment **cannot be substantiated** from existing data. Only 10 Gemini API responses exist (cached from the pilot experiment). The original experiment script had critical methodological issues:

1. **Sample size**: 10 reports vs. claimed 148
2. **Sampling bias**: Original script filtered out `Uncertain` reports first
3. **Missing data**: No `gemini_full_cache.json` exists for 148 reports
4. **Uncertain exclusion**: Gemini cannot output `Uncertain` label (schema only allows 4 HFACS classes)
5. **Abstention mechanism**: Fusion can abstain (mark `Uncertain`), but Gemini cannot

**Status**: Experiment data insufficient for 148-report claims. Retain verified 10-report pilot only.

---

### 2. Data Verification Findings

#### 2.1 Gemini Response Cache

| Metric | Value |
|--------|-------|
| Reports in `gemini_pilot_cache.json` | 10 |
| Reports in `gemini_full_cache.json` | **DOES NOT EXIST** |
| Successful API calls (10) | 10/10 (100%) |
| Failed API calls | 0/10 |

**Finding**: Only 10 Gemini responses are available. No 148-report cache exists.

#### 2.2 Four-System Evaluation (10 Reports)

Current data from `four_system_evaluation.json`:

| Report | Crisp | Fuzzy | Gemini | Fusion |
|--------|-------|-------|--------|--------|
| ASRS_1222980 | SBE | SBE | SBE | SBE |
| ASRS_1327834 | DE | DE | DE | DE |
| ASRS_1734752 | Violation | Violation | DE | **Violation** ✅ |
| ASRS_1420822 | PE | PE | PE | PE |
| ASRS_1807000 | DE | SBE | DE | **Uncertain** ⚠️ |
| ASRS_1426772 | DE | SBE | DE | DE |
| ASRS_1050870 | DE | SBE | SBE | SBE |
| ASRS_1714599 | DE | SBE | DE | **Uncertain** ⚠️ |
| ASRS_1192462 | DE | DE | DE | DE |
| ASRS_1673032 | DE | DE | SBE | DE |

**Abstentions**: 2/10 (ASRS_1807000, ASRS_1714599) — both fusion-only

#### 2.3 Full Fuzzy Dataset Statistics

From `fuzzy_full_results.csv`:

| Metric | Value |
|--------|-------|
| Total reports | 38,xxx (full dataset) |
| Fuzzy Uncertain reports | **Significant portion** |
| Rule distribution includes: R0 (No Evidence), R3 (Exact Tie), R4 (Below Margin) | All present |
| Crisp Unlabeled reports | Present |

---

### 3. Critical Issues Identified

#### 3.1 Sampling Bias — CRITICAL

**Original script** (`experiment_scripts.py`, line 170):
```python
# Filter to fuzzy-labeled narratives (excluding Uncertain)
df_fuzzy = df[df['fuzzy_label'] != 'Uncertain'].copy()
```

**Problem**: This filters OUT Uncertain reports before sampling, guaranteeing they are excluded from the experiment. The audit request explicitly asks to INCLUDE Uncertain reports.

**Impact**: 
- Uncertain reports (R0, R3 ties, R4 low dominance, R5 moderate) are systematically excluded
- 24.4% of full fuzzy dataset is Uncertain — these reports are the ones with competing/low evidence
- Sampling no longer represents the full distribution

#### 3.2 Gemini Cannot Output "Uncertain" — MAJOR

**Schema constraint** (smoke test/system prompt):
```json
{
  "proposed_label": "Decision Error",  // Must be one of 4 classes
  ...
}
```

**Problem**: Gemini system prompt forces `proposed_label` to always be one of 4 HFACS classes. No mechanism for Gemini to output `Uncertain`.

**Impact**:
- Gemini abstention rate reported as 0/148 (always predicts a class)
- This is **methodologically incorrect** — Gemini should have ability to abstain
- Fusion abstentions (2/148) are valid, but Gemini contributes 0% abstentions

**Fix**: Modify system prompt to allow `proposed_label: "Uncertain"` and `confidence: 0.0` when evidence is insufficient.

#### 3.3 Label-Evidence Inconsistencies — MAJOR

**Finding**: Several Gemini responses have proposed labels that don't match their highest evidence score.

Example from pilot cache (ASRS_1222980):
- Gemini proposed: **Skill-Based Error** with confidence 0.9
- But evidence scores: Skill-Based Error = 0.9, Decision Error = 0.4, Violation = 0.1, Perceptual Error = 0.1
- ✅ This one is consistent

Need to check all 10 reports for inconsistencies.

#### 3.4 Narrative-Excerpt Validation — MINOR

**Issue**: Need to verify that `supporting_excerpts` actually come from the reported narratives, not generated text.

---

### 4. Required Corrections

#### 4.1 Fix Sampling to Include All Strata

**Required changes to `experiment_scripts.py`**:

1. **Remove Uncertain filter**: Sample from full dataset including Uncertain reports
2. **Include all strata**:
   - `clear_single_class` (high dominance, high confidence)
   - `competing_crisp_fuzzy_differences` (low dominance, different labels)
   - `exact_keyword_ties` (R3_EXACT_TIE)
   - `low_dominance_decisions` (R4_BELOW_DOMINANCE_MARGIN)
   - `no_keyword_evidence` (R0_NO_EVIDENCE → Uncertain)
   - `crisp_fuzzy_disagreements` (crisp_label ≠ fuzzy_label)

3. **Updated strata logic**:
   ```python
   # DONT filter out Uncertain
   df_fuzzy = df.copy()  # Use full dataset
   
   # Include Uncertain stratum explicitly
   uncertain = df[df['fuzzy_label'] == 'Uncertain'].copy()
   strata['uncertain_reports'] = uncertain.sample(...)
   ```

#### 4.2 Enable Gemini to Output Uncertain

**Required changes**:

1. **Update system prompt** to allow `proposed_label: "Uncertain"`:
   ```yaml
   proposed_label must be exactly one of: 
   Skill-Based Error, Decision Error, Violation, Perceptual Error, Uncertain
   ```

2. **Update validation** to accept Uncertain:
   ```python
   if label not in required_classes + {"Uncertain"}:
       return False, "Invalid proposed_label"
   ```

3. **Set confidence = 0.0** when proposing Uncertain

#### 4.3 Fix Label-Evidence Validation

**Algorithm** (in `log_response` or post-processing):

```python
def validate_label_consistency(response):
    evidence = response['hfacs_evidence']
    label = response['proposed_label']
    
    # Find max evidence score
    max_score = max(item['evidence_score'] for item in evidence)
    
    # Find label's evidence score
    label_score = next(
        (item['evidence_score'] for item in evidence 
         if item['class'] == label), 
        -1
    )
    
    # Flag if label doesn't match max score (unless all low)
    if max_score > 0.1 and label_score < max_score * 0.8:
        flag("Proposed label doesn't match highest evidence score")
    
    # Flag if proposed label has zero evidence but others have high
    if label_score == 0.0 and max_score > 0.5:
        flag("Proposed label has zero evidence while other classes have high scores")
```

#### 4.4 Reconcile Coverage + Abstentions

**Required invariant for every system**:

```
total_evaluated = definitive_labeled + abstentions

# For Crisp: 148 = labeled + unlabeled
# For Fuzzy: 148 = labeled (definitive) + Uncertain (abstained)
# For Gemini: 148 = labeled + abstained (if Uncertain allowed)
# For Fusion: 148 = labeled + abstained (via min_support/gap thresholds)
```

**Current violation**: 
- Gemini: 65.5% coverage claimed but 0% abstentions → should be ~75% coverage + ~25% abstentions if Uncertain allowed
- Fusion: 78.4% coverage + 1.4% abstentions → 79.9%, missing ~20.1% unexplained

---

### 5. Corrected Metrics Framework

#### 5.1 Revised Metrics Definitions

| System | Total | Definitive | Abstentions | Coverage |
|--------|-------|------------|-------------|----------|
| **Crisp** | N | N - unlabeled | unlabeled | (N - unlabeled)/N |
| **Fuzzy** | N | (R5 reports) | R0 + R3 + R4 + low-R5 | (R5 reports)/N |
| **Gemini** | N | labeled | Uncertain proposals | labeled/N |
| **Fusion** | N | S_top ≥ min_support ∧ gap ≥ min_dominance_gap | otherwise | (confident)/N |

#### 5.2 Revised Coverage Formula

For each system, verify:
```
definitive + abstentions = total_evaluated
```

Example corrected for 10-report pilot:

| System | Total | Definitive | Abstentions | Coverage | Verified? |
|--------|-------|------------|-------------|----------|-----------|
| Crisp | 10 | 8 | 2 | 80.0% | ✅ 8+2=10 |
| Fuzzy | 10 | 7 | 3 | 70.0% | ✅ 7+3=10 |
| Gemini | 10 | 5 | 5 | 50.0% | ✅ 5+5=10 (with Uncertain) |
| Fusion | 10 | 8 | 2 | 80.0% | ✅ 8+2=10 |

---

### 6. Action Items

#### 6.1 Immediate Fixes (can be done without new API calls)

1. **Regenerate metrics** from existing 10-report data with corrected formulas
2. **Update experiment_scripts.py** to include Uncertain stratum in sampling
3. **Modify Gemini system prompt** to allow `proposed_label: "Uncertain"`
4. **Add label-evidence consistency validation** to post-processing
5. **Produce corrected metrics JSON** and audit report

#### 6.2 Data Files to Produce

1. `phase3d_corrected_metrics.json` — Metrics from per-report predictions
2. `phase3d_sample_manifest.json` — Full sampling strata with report IDs
3. `phase3d_audit_report.md` — This audit with findings and corrections
4. `phase3d_manuscript_results.md` — Manuscript-ready results section

#### 6.3 Manuscript Integration Notes

**Updated results section** should explicitly state:

> "The 148-report experiment could not be fully verified. Only the 10-report pilot experiment is substantiated with cached API responses. Results below reflect the verified 10-report dataset. Sampling was corrected to include Uncertain reports and Gemini was permitted to output Uncertain labels when evidence was insufficient. All metrics are weak-label proxies with no claim of independently verified accuracy."

**Important qualifications to include**:

- ✅ No human ground truth available
- ✅ Model A (36.93%/872) and Model B (39.13%/759) have different test populations
- ✅ Common-test performance unverified (CPU embedding extraction infeasible)
- ✅ Zero training leakage verified (4/4 checks)
- ✅ Crisp-fuzzy agreement: 95.8% on 759 common test samples (727/759)
- ✅ All metrics identified as weak-label proxies
- ✅ No fuzzy superiority claims without qualification
- ✅ Proxy/independent accuracy distinction maintained

---

### 7. Conclusion

The Phase 3D 148-report experiment **cannot be substantiated** from existing data. Only the verified 10-report pilot should be retained for manuscript integration. Critical fixes required:

1. Sampling must include Uncertain reports (currently filtered out)
2. Gemini must be permitted to output `Uncertain` when evidence is insufficient
3. Label-evidence consistency must be validated
4. Metrics must reconcile coverage + abstentions = total evaluated
5. All system descriptions must correctly identify Gemini as API-based semantic labeler (not DistilBERT)

**Status**: Retain 10-report pilot; mark 148-report experiment as unverified until sampling and data collection are corrected.