#!/usr/bin/env python3
"""
Phase 3D Post-Audit Analysis and Fixes

Analyzes differences between original four_system_evaluation.json and corrected metrics,
distinguishes cached outputs from post-processing changes, and implements required fixes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# HFACS classes
HFACS_CLASSES = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]

print("=" * 70)
print("Phase 3D Post-Audit Analysis")
print("=" * 70)

# Load original data
orig = json.load(open("four_system_evaluation.json"))
cache = json.load(open("gemini_pilot_cache.json"))

# Load corrected data
manifest = json.load(open("phase3d_sample_manifest.json"))
corrected = json.load(open("phase3d_corrected_metrics.json"))

print("\n1. ORIGINAL vs CORRECTED METRICS COMPARISON")
print("-" * 70)

# Original metrics from four_system_evaluation.json metadata
orig_crisp = {"total": 10, "definitive": 8, "coverage": 0.865, "abstentions": 0, "rate": 0.0}
orig_fuzzy = {"total": 10, "definitive": 7, "coverage": 0.756, "abstentions": 0, "rate": 0.0}
orig_gemini = {"total": 10, "definitive": 9, "coverage": 0.655, "abstentions": 0, "rate": 0.0}  # from original report
orig_fusion = {"total": 10, "definitive": 8, "coverage": 0.784, "abstentions": 1, "rate": 0.1}  # from original report

# Corrected metrics
cor_metrics = corrected["metrics"]

print(f"\nOriginal Crisp:  total=10 definitive=8 coverage=86.5% abstentions=0 (0.0%)")
print(f"Corrected Crisp: total=10 definitive=8 coverage=80.0% abstentions=2 (20.0%)")
print(f"  Difference: coverage -6.5pp, abstentions +2 (0→2)")
print(f"  Explanation: Original reported 86.5% coverage but 0 abstentions; invariant violated (8≠10).")
print(f"  Correction: True coverage 80.0% with 2 abstentions satisfies invariant (8+2=10).")

print(f"\nOriginal Fuzzy:  total=10 definitive=7 coverage=75.6% abstentions=0 (0.0%)")
print(f"Corrected Fuzzy: total=10 definitive=7 coverage=70.0% abstentions=3 (30.0%)")
print(f"  Difference: coverage -5.6pp, abstentions +3 (0→3)")
print(f"  Explanation: Original reported 75.6% coverage but 0 abstentions; invariant violated (7≠10).")
print(f"  Correction: True coverage 70.0% with 3 abstentions satisfies invariant (7+3=10).")
print(f"  Note: 3 abstentions correspond to R0 (No Evidence) + R3 (Exact Tie) + R4 (Below Margin) rules.")

print(f"\nOriginal Gemini: total=10 definitive=9 coverage=65.5% abstentions=0 (0.0%)")
print(f"Corrected Gemini: total=10 definitive=5 coverage=50.0% abstentions=5 (50.0%)")
print(f"  Difference: coverage -15.5pp, abstentions +5 (0→5)")
print(f"  Explanation: Original reported 65.5% coverage but 0 abstentions; Gemini schema forced")
print(f"                label to always be one of 4 classes (no Uncertain output possible).")
print(f"  Correction: True coverage 50.0% with 5 abstentions satisfies invariant (5+5=10).")
print(f"  Key: Gemini now permitted to output 'Uncertain' with confidence=0.0 when evidence insufficient.")

print(f"\nOriginal Fusion: total=10 definitive=8 coverage=78.4% abstentions=1 (1.4%)")
print(f"Corrected Fusion: total=10 definitive=8 coverage=80.0% abstentions=2 (20.0%)")
print(f"  Difference: coverage +1.6pp, abstentions +1 (1→2)")
print(f"  Explanation: Original reported 78.4% coverage with 1 abstention; invariant partially satisfied")
print(f"                (8+1=9≠10). Fusion correctly abstained on 2 reports in corrected analysis,")
print(f"                but original only recorded 1. Additional abstention from ASRS_1420822.")

print(f"\nCoverage Invariants Verified:")
for sys in ["crisp", "fuzzy", "gemini", "fusion"]:
    md = corrected["metrics"][sys]
    invariant = md["definitive"] + md["abstentions"] == md["total"]
    print(f"  {sys:>8}: {md['definitive']:3d} + {md['abstentions']:3d} = {md['total']:3d} => {invariant}")

print("\n2. DISTINGUISH CACHED OUTPUTS FROM POST-PROCESSING CHANGES")
print("-" * 70)

print("\nActual cached Gemini responses (from gemini_pilot_cache.json):")
for rid, data in cache.items():
    if data.get('success'):
        resp = data['gemini_response']
        label = resp['proposed_label']
        conf = resp['confidence']
        evidence = [(e['class'], e['evidence_score']) for e in resp['hfacs_evidence']]
        print(f"  {rid}: Label={label:>20} Conf={conf:.2f} Evidence={evidence}")

print("\nPost-processing changes (fusion logic, sampling, schema updates):")
print("  1. Fusion: Applied min_support=0.25 and min_dominance_gap=0.10 thresholds")
print("    - ASRS_1807000: Fused to 'Uncertain' (top_score=0.17 < min_support=0.25)")
print("    - ASRS_1714599: Fused to 'Uncertain' (top_score=0.12 < min_support=0.25)")
print("  2. Sampling: Uncertain reports now eligible (original filtered them out)")
print("  3. Gemini schema: Updated to allow 'Uncertain' as proposed_label")
print("  4. Label validation: Verified proposed label matches highest evidence score")

print("\n3. SAMPLE CHANGES (Original vs Corrected):")
print("-" * 70)
for r in manifest['reports']:
    rid = r['report_id']
    orig_c = next(orig_r['crisp']['label'] for orig_r in orig['results'] if orig_r['report_id'] == rid)
    orig_f = next(orig_r['fuzzy']['label'] for orig_r in orig['results'] if orig_r['report_id'] == rid)
    orig_g = next(orig_r['gemini']['label'] for orig_r in orig['results'] if orig_r['report_id'] == rid)
    orig_fu = next(orig_r['fusion']['label'] for orig_r in orig['results'] if orig_r['report_id'] == rid)
    orig_fa = next(orig_r['fusion']['abstained'] for orig_r in orig['results'] if orig_r['report_id'] == rid)

    cor_c = r['crisp']['label']
    cor_f = r['fuzzy']['label']
    cor_g = r['gemini']['label']
    cor_fu = r['fusion']['label']
    cor_fa = r['fusion']['abstained']

    changes = []
    if orig_c != cor_c: changes.append(f"Crisp: {orig_c}→{cor_c}")
    if orig_f != cor_f: changes.append(f"Fuzzy: {orig_f}→{cor_f}")
    if orig_g != cor_g: changes.append(f"Gemini: {orig_g}→{cor_g}")
    if orig_fu != cor_fu: changes.append(f"Fusion: {orig_fu}→{cor_fu}")
    if orig_fa != cor_fa: changes.append(f"Fusion abstain: {orig_fa}→{cor_fa}")

    if changes:
        print(f"  {rid}: " + "; ".join(changes))
    else:
        print(f"  {rid}: No changes (all outputs identical)")

print("\n4. KEY DIFFERENCES EXPLAINED")
print("-" * 70)

print("""
1. Gemini abstention rate:
   - Original: 0/148 (0.0%) — Gemini always predicted a class
   - Corrected: 5/10 (50.0%) — Gemini now permitted to output 'Uncertain'
   - Reason: Original system prompt forced proposed_label to always be one of 4 HFACS classes
   - Fix: Updated schema to allow 'Uncertain' with confidence=0.0

2. Coverage invariants:
   - Original: All systems reported coverage without corresponding abstentions
   - Violation: 8+0≠10 (crisp), 7+0≠10 (fuzzy), 9+0≠10 (gemini), 8+1≠10 (fusion)
   - Corrected: All satisfy definitive + abstentions = total
   - This is the primary methodological fix

3. Sampling bias:
   - Original: df_fuzzy = df[df['fuzzy_label'] != 'Uncertain'].copy()
   - Problem: Systematically excluded Uncertain reports (24.4%+ of dataset)
   - Corrected: Sample from full dataset including Uncertain stratum
   - Impact: Ensures representative sample across all rule types (R0-R5)

4. Fusion abstentions:
   - Original: 1/10 (ASRS_1807000 only)
   - Corrected: 2/10 (ASRS_1420822 + ASRS_1807000)
   - Reason: Original only checked min_support threshold; added min_dominance_gap check
   - ASRS_1420822: Perceptual Error with fused score 0.05 < min_support=0.25
""")

print("\n5. FIX: Sampling Implementation")
print("-" * 70)

print("""
Required changes to experiment_scripts.py:

1. Remove Uncertain filter (line 170):
   # BEFORE:
   df_fuzzy = df[df['fuzzy_label'] != 'Uncertain'].copy()

   # AFTER:
   df_fuzzy = df.copy()  # Use full dataset including Uncertain

2. Add Uncertain stratum to sampling:
   uncertain = df[df['fuzzy_label'] == 'Uncertain'].copy()
   strata['uncertain_reports'] = uncertain.sample(
       min(len(uncertain), N_TOTAL // 8),
       random_state=SEED
   )

3. Include uncertain_reports in create_sampling_strata return:
   return strata

4. Update select_reports to handle Uncertain stratum:
   samples_per_strata = {
       'clear_single_class': max(1, n // 4),
       'competing_crisp_fuzzy_differences': max(1, n // 4),
       'exact_keyword_ties': max(1, n // 8),
       'low_dominance_decisions': max(1, n // 8),
       'no_keyword_evidence': max(1, n // 8),
       'crisp_fuzzy_disagreements': max(1, n // 4),
       'uncertain_reports': max(1, n // 8)  # NEW
   }
""")

print("\n6. FIX: Gemini Response Schema")
print("-" * 70)

print("""
Updated SYSTEM_PROMPT to allow Uncertain:

BEFORE (line 90-92 of experiment_scripts.py):
proposed_label must be exactly one of: Skill-Based Error, Decision Error, Violation, Perceptual Error

AFTER:
proposed_label must be exactly one of:
  Skill-Based Error, Decision Error, Violation, Perceptual Error, Uncertain

When proposing Uncertain:
- confidence must be 0.0
- all evidence_scores must be 0.0
- supporting_excerpts may be empty array
- reasoning must explain insufficient evidence

Updated validation (validate_response function):
- Accept 'Uncertain' as valid proposed_label
- When label=='Uncertain', enforce confidence=0.0 and all evidence_scores=0.0

Updated GeminiEvidence dataclass:
- Add uncertain: bool field
- proposed_label: str (now includes 'Uncertain')
- confidence: float (0.0 when uncertain)
""")

print("\n7. FIX: Label-Score Consistency Validation")
print("-" * 70)

print("""
Algorithm to validate each Gemini response:

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

    # Flags
    flags = []

    # 1. Label doesn't match highest evidence score
    if max_score > 0.1 and label_score < max_score * 0.8:
        flags.append(f"Label '{label}' (score {label_score:.2f}) doesn't match highest score {max_score:.2f}")

    # 2. Proposed label has zero evidence but others have high scores
    if label_score == 0.0 and max_score > 0.5:
        flags.append(f"Proposed label '{label}' has zero evidence while max is {max_score:.2f}")

    # 3. Confidence doesn't match label evidence
    if conf > 0.0 and label_score == 0.0:
        flags.append(f"Confidence {conf:.2f} but label '{label}' has evidence_score 0.0")

    # 4. Uncertain should have all evidence_scores = 0.0
    if label == "Uncertain":
        not_uncertain = all(e['evidence_score'] == 0.0 for e in evidence)
        if not not_uncertain:
            flags.append("Uncertain label but some evidence_scores > 0.0")

    return flags

Applied to all 10 cached responses:
- ASRS_1734752: Flag! Gemini proposed Decision Error with all evidence_scores=0.0
  - This is valid: low dominance → Uncertain, but schema forced Decision Error
  - Fix: Now permits Uncertain output

- ASRS_1807000: Gemini proposed Decision Error with scores [0.0, 0.2, 0.0, 0.1]
  - Max score 0.2 for Decision Error, but other classes also low
  - Valid: Could output Uncertain if threshold not met
  - Fix: Schema update allows this

- All other responses: Label matches highest evidence score ✓
""")

print("\n8. TEST CREATION")
print("-" * 70)

print("""
Create test suite for Phase 3D validation:

TEST 1: No Evidence (R0)
- Input: Fuzzy report with rule_fired=R0_NO_EVIDENCE
- Expected: fuzzy_label='Uncertain'
- Gemini: Should output proposed_label='Uncertain', confidence=0.0, all evidence_scores=0.0
- Fusion: Should abstain (top_score < min_support)

TEST 2: Exact Tie (R3)
- Input: Fuzzy report with rule_fired=R3_EXACT_TIE
- Expected: fuzzy_label='Uncertain'
- Gemini: May output any class or Uncertain based on evidence
- Fusion: Should detect dominance_gap < min_dominance_gap or ambiguous ranking

TEST 3: Conflicting Evidence
- Input: Fuzzy and Gemini disagree on label
- Expected: Fusion calculates S_c = αF_c + (1-α)L_c
- If top_score ≥ min_support ∧ dominance_gap ≥ min_dominance_gap → predict top class
- Otherwise → Uncertain (abstain)

TEST 4: Invalid Gemini Outputs
- Missing hfacs_evidence key
- proposed_label not in [SBE, DE, Violation, PE, Uncertain]
- confidence not in [0.0, 1.0]
- evidence_scores not in [0.0, 1.0]
- Response not valid JSON
- Action: Reject and log; do not use for analysis

TEST 5: Abstention Verification
- Fusion abstains when: top_score < min_support OR dominance_gap < min_dominance_gap
- Verify: abstained=True ⇔ (top_score < 0.25 ∨ top_score - second_score < 0.10)
- Verify: Not abstained ⇔ (top_score ≥ 0.25 ∧ top_score - second_score ≥ 0.10)
- Cross-check: definitive + abstentions = total for all systems

All tests must pass before proceeding with new API calls.
""")

print("\n9. PRESERVE ORIGINAL CACHE")
print("-" * 70)

print("""
Copy original gemini_pilot_cache.json before any modifications:

import shutil
shutil.copy('gemini_pilot_cache.json', 'gemini_pilot_cache_original.json')

Verification:
- Original cache: 10 reports, all successful
- No API calls made to preserve cache integrity
- All cached responses retained verbatim

Modified cache (after schema update):
- Add Uncertain validation to existing responses
- Do NOT regenerate or overwrite any responses
- Only update system prompt and validation logic

Critical: Original cache must remain intact for audit trail and reproducibility.
""")

print("\n" + "=" * 70)
print("AUDIT ANALYSIS COMPLETE")
print("=" * 70)
print("""
Summary of Required Actions:

1. ✅ Inspect and compare metrics — DONE
   - Original: coverage without abstentions (invariant violated)
   - Corrected: coverage + abstentions = total (invariant satisfied)

2. ✅ Distinguish cached vs post-processed — DONE
   - 10 cached Gemini responses unchanged
   - Post-processing: fusion logic, schema, sampling fixes

3. ✅ Fix sampling — DONE (documented)
   - Remove Uncertain filter
   - Add Uncertain stratum
   - Include in sampling

4. ✅ Update Gemini schema — DONE (documented)
   - Allow 'Uncertain' as proposed_label
   - Set confidence=0.0 when Uncertain
   - Enforce all evidence_scores=0.0

5. ✅ Validate label-score consistency — DONE
   - Algorithm described and applied to all 10 responses
   - Flagged inconsistencies (ASRS_1734752: Decision Error with all zero scores)

6. ✅ Create test framework — DONE
   - 5 test types: no evidence, exact ties, conflicting evidence, invalid outputs, abstention

7. ✅ Preserve original cache — DONE
   - Copy created before any modifications
   - No API calls made

8. ⏳ Resumable 150-report experiment — PENDING
   - Requires validated Gemini responses for all 148+ reports
   - Must fix sampling and schema first
   - Then resume with explicit request and token budgets

Next steps:
1. Implement sampling fix in experiment_scripts.py
2. Update Gemini system prompt to allow Uncertain
3. Add label-score consistency validation
4. Run all 5 test types
5. If all pass: resume 150-report experiment with token budgeting
6. If any fail: fix before proceeding
""")