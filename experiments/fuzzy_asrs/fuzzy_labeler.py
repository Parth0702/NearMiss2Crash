"""
=================================================================================
Fuzzy Weak Labeler Module (Phase 2 - NearMiss2Crash)
=================================================================================
Reference & Context:
  - Phase 1 Baseline: KeywordWeakLabeler (Madeira et al. 2021 consistency criterion)
    in `keyword_weak_labeler.py`.
  - Phase 2 Purpose: Implement a transparent, principled Fuzzy Inference System (FIS)
    for weak labeling ASRS narrative texts into HFACS unsafe-act categories.
  - Unlike crisp majority voting which silently breaks ties by dictionary order,
    the Fuzzy Weak Labeler:
      1. Accumulates raw lexical evidence counts per class (accounting for repetitions).
      2. Evaluates continuous Low / Medium / High fuzzy membership degrees per class.
      3. Applies explicit fuzzy inference rules across class evidence.
      4. Calculates a continuous dominance / confidence score.
      5. Explicitly emits 'Uncertain' when evidence is absent, competing, or below
         a configurable abstention threshold.

Canonical HFACS Core Classes:
  1. Skill-Based Error
  2. Decision Error
  3. Violation
  4. Perceptual Error
=================================================================================
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union


# -----------------------------------------------------------------------------
# Canonical 39 Seed Keyword-to-Class Mappings (identical to keyword_weak_labeler)
# -----------------------------------------------------------------------------
SEED_KEYWORDS: Dict[str, List[str]] = {
    "Skill-Based Error": [
        "checklist", "forgot", "omitted", "lapsed", "routine", "habit",
        "inadvertent", "slip", "memory", "missed call", "misread"
    ],
    "Decision Error": [
        "decided", "elected", "chose", "continued", "judgment", "risk",
        "weather", "visual approach", "divert", "below minimums"
    ],
    "Violation": [
        "knowingly", "intentionally", "skipped", "unauthorized",
        "rule", "sop", "regulation", "sterile cockpit", "exceeded"
    ],
    "Perceptual Error": [
        "disorientation", "spatial", "illusion", "misjudged distance",
        "visibility", "flare", "glare", "featureless terrain", "depth perception"
    ]
}

HFACS_CLASSES: List[str] = list(SEED_KEYWORDS.keys())


# -----------------------------------------------------------------------------
# Membership Functions (Piecewise Linear / Triangular / Trapezoidal)
# -----------------------------------------------------------------------------
def fuzzy_low(x: float, a: float = 0.0, b: float = 1.0, c: float = 2.5) -> float:
    """
    Trapezoidal/Ramp membership for 'Low' evidence:
      - 1.0 for x <= b
      - linear decrease from 1.0 to 0.0 for b < x < c
      - 0.0 for x >= c
    Default: [0, 1.0, 2.5] -> 1 match has high Low membership; >= 3 matches have 0 Low membership.
    """
    if x <= b:
        return 1.0
    elif x < c:
        return float((c - x) / (c - b))
    else:
        return 0.0


def fuzzy_medium(x: float, a: float = 1.0, b: float = 2.5, c: float = 4.5) -> float:
    """
    Triangular membership for 'Medium' evidence:
      - 0.0 for x <= a or x >= c
      - linear increase from 0.0 to 1.0 for a < x <= b
      - linear decrease from 1.0 to 0.0 for b < x < c
    Default: [1.0, 2.5, 4.5] -> peaks around 2-3 matches.
    """
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return float((x - a) / (b - a))
    else:
        return float((c - x) / (c - b))


def fuzzy_high(x: float, a: float = 2.5, b: float = 4.5) -> float:
    """
    Ramp membership for 'High' evidence:
      - 0.0 for x <= a
      - linear increase from 0.0 to 1.0 for a < x < b
      - 1.0 for x >= b
    Default: [2.5, 4.5] -> starts rising after 2 matches, fully High at >= 5 matches.
    """
    if x <= a:
        return 0.0
    elif x < b:
        return float((x - a) / (b - a))
    else:
        return 1.0


class FuzzyWeakLabeler:
    """
    Fuzzy rule-based weak labeler for aviation incident narratives.

    Attributes:
        seed_keywords: Dict mapping class names to keyword lists.
        classes: Ordered list of target class names.
        abstain_threshold: Minimum dominance / confidence required to assign
            a crisp label instead of 'Uncertain'. Range [0.0, 1.0].
            Default 0.20 (requires top class to have meaningful margin over competition).
        count_occurrences: If True, counts every occurrence of matched keywords
            (e.g., repeated mentions indicate stronger evidence). If False, counts
            unique matching keywords. Default: True.
    """

    def __init__(
        self,
        seed_keywords: Optional[Dict[str, List[str]]] = None,
        evidence_threshold: float = 0.35,
        dominance_margin: float = 0.15,
        count_occurrences: bool = True,
        low_params: Tuple[float, float, float] = (0.0, 1.0, 2.5),
        med_params: Tuple[float, float, float] = (1.0, 2.5, 4.5),
        high_params: Tuple[float, float] = (2.5, 4.5)
    ):
        """
        Args:
            seed_keywords: Optional override of SEED_KEYWORDS dict
            evidence_threshold: Minimum evidence count τ required for a class to be
                considered (default: 0.35). In practice, evidence counts are integers
                0, 1, 2... so this serves as documentation of the policy; the actual
                check uses raw evidence counts >= 1 for any meaningful signal.
            dominance_margin: Minimum dominance margin δ before making a prediction
                (default: 0.15), representing (E_top - E_second) / E_total >= δ.
            count_occurrences: If True, counts every occurrence of matched keywords
                (e.g., repeated mentions indicate stronger evidence). If False, counts
                unique matching keywords. Default: True.
        """
        self.seed_keywords = seed_keywords or SEED_KEYWORDS
        self.classes = list(self.seed_keywords.keys())
        self.evidence_threshold = float(evidence_threshold)
        self.dominance_margin = float(dominance_margin)
        self.count_occurrences = count_occurrences
        self.low_params = low_params
        self.med_params = med_params
        self.high_params = high_params

        # Compile regex patterns once for efficiency
        self._compiled_patterns: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        for cls_name, kw_list in self.seed_keywords.items():
            patterns = []
            for kw in kw_list:
                # Word boundary match, case-insensitive
                p = re.compile(r'\b' + re.escape(kw.lower()) + r'\b')
                patterns.append((kw, p))
            self._compiled_patterns[cls_name] = patterns

    def compute_evidence(self, text: str) -> Dict[str, Any]:
        """
        Calculates raw keyword evidence for each HFACS class.

        Returns a dictionary with:
          - 'counts': Dict[class_name, int] (total occurrences or unique hits)
          - 'unique_counts': Dict[class_name, int] (unique keywords matched)
          - 'total_occurrences': Dict[class_name, int]
          - 'matched_keywords': Dict[class_name, List[str]]
          - 'all_matched': List[str]
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "counts": {c: 0 for c in self.classes},
                "unique_counts": {c: 0 for c in self.classes},
                "total_occurrences": {c: 0 for c in self.classes},
                "matched_keywords": {c: [] for c in self.classes},
                "all_matched": []
            }

        text_lower = text.lower()
        counts: Dict[str, int] = {}
        unique_counts: Dict[str, int] = {}
        total_occurrences: Dict[str, int] = {}
        matched_keywords: Dict[str, List[str]] = {}
        all_matched: List[str] = []

        for cls_name, patterns in self._compiled_patterns.items():
            matched_cls_kws = []
            occ_count = 0
            for kw, pat in patterns:
                matches = pat.findall(text_lower)
                if matches:
                    matched_cls_kws.append(kw)
                    occ_count += len(matches)
                    all_matched.extend([kw] * len(matches))

            unique_counts[cls_name] = len(matched_cls_kws)
            total_occurrences[cls_name] = occ_count
            matched_keywords[cls_name] = matched_cls_kws
            counts[cls_name] = occ_count if self.count_occurrences else len(matched_cls_kws)

        return {
            "counts": counts,
            "unique_counts": unique_counts,
            "total_occurrences": total_occurrences,
            "matched_keywords": matched_keywords,
            "all_matched": all_matched
        }

    def compute_memberships(self, evidence_counts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Evaluates continuous Low / Medium / High fuzzy membership for each class.

        Returns:
            Dict[class_name, Dict['low'|'med'|'high', float]]
        """
        memberships = {}
        for cls_name in self.classes:
            x = float(evidence_counts.get(cls_name, 0))
            memberships[cls_name] = {
                "low": fuzzy_low(x, *self.low_params),
                "med": fuzzy_medium(x, *self.med_params),
                "high": fuzzy_high(x, *self.high_params)
            }
        return memberships

    def evaluate_fuzzy_rules(
        self,
        evidence_counts: Dict[str, int],
        memberships: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Executes explicit fuzzy inference rules with roadmap thresholds:

        Rules:
          R0 (Null Evidence): If all classes have count == 0 -> 'Uncertain', score = 0.0.
          R1 (Insufficient Evidence): If top class count < evidence_threshold (τ) -> 'Uncertain'.
          R3 (Tie at Top): If top two classes have equal counts > 0 -> 'Uncertain'.
          R4 (Dominance Margin): dominance = (E_top - E_second) / E_total.
             If dominance < dominance_margin (δ) -> 'Uncertain'.
          R5 (Accept): Top class passes both thresholds -> predicted label.

        Returns:
            Dict containing predicted_label, confidence, dominance, top_class, runner_up_class
        """
        counts = [evidence_counts[c] for c in self.classes]
        total_evidence = sum(counts)

        # Rule 0: No evidence
        if total_evidence == 0:
            return {
                "predicted_label": "Uncertain",
                "confidence": 0.0,
                "dominance": 0.0,
                "top_class": None,
                "runner_up_class": None,
                "rule_fired": "R0_NO_EVIDENCE"
            }

        # Sort classes by evidence count descending
        sorted_classes = sorted(self.classes, key=lambda c: evidence_counts[c], reverse=True)
        c_top = sorted_classes[0]
        c_second = sorted_classes[1]
        top_count = evidence_counts[c_top]
        second_count = evidence_counts[c_second]

        # Rule 1: Insufficient evidence for top class (τ check)
        if top_count < self.evidence_threshold:
            return {
                "predicted_label": "Uncertain",
                "confidence": 0.0,
                "dominance": 0.0,
                "top_class": c_top,
                "runner_up_class": c_second,
                "rule_fired": "R1_INSUFFICIENT_EVIDENCE"
            }

        # Rule 3: Exact tie at top (two or more classes share top_count > 0)
        if top_count == second_count:
            return {
                "predicted_label": "Uncertain",
                "confidence": 0.0,
                "dominance": 0.0,
                "top_class": c_top,
                "runner_up_class": c_second,
                "rule_fired": "R3_EXACT_TIE"
            }

        # Calculate relative dominance score in [0.0, 1.0]
        # dominance = (E_top - E_runner_up) / E_total
        dominance = float((top_count - second_count) / total_evidence)

        # Confidence: weighted by top class membership level and penalized by competition
        top_m = memberships[c_top]
        second_m = memberships[c_second]

        # Degree of evidence support using weighted sum of Low, Medium, High memberships
        # Weights: Low (25%), Medium (50%), High (100%) - strong evidence receives highest weight
        # This ensures non-zero confidence for any positive evidence level
        top_support = 0.25 * top_m["low"] + 0.5 * top_m["med"] + 1.0 * top_m["high"]
        # Penalty from competing class
        competition_penalty = 0.5 * second_m["med"] + 1.0 * second_m["high"]

        raw_confidence = dominance * top_support * (1.0 - 0.5 * competition_penalty)
        confidence = float(np.clip(raw_confidence, 0.0, 1.0))

        # Rule 4: Dominance margin check (δ check)
        if dominance < self.dominance_margin:
            return {
                "predicted_label": "Uncertain",
                "confidence": round(confidence, 4),
                "dominance": round(dominance, 4),
                "top_class": c_top,
                "runner_up_class": c_second,
                "rule_fired": "R4_BELOW_DOMINANCE_MARGIN"
            }

        # Rule 5: Class is accepted
        rule_fired = "R5_HIGH_DOMINANCE" if top_m["high"] > 0.5 else "R5_MODERATE_DOMINANCE"
        return {
            "predicted_label": c_top,
            "confidence": round(confidence, 4),
            "dominance": round(dominance, 4),
            "top_class": c_top,
            "runner_up_class": c_second,
            "rule_fired": rule_fired
        }

    def predict_one(self, text: str) -> Dict[str, Any]:
        """
        Labels a single narrative string.
        Returns a rich evaluation dictionary.
        """
        ev = self.compute_evidence(text)
        mem = self.compute_memberships(ev["counts"])
        rule_res = self.evaluate_fuzzy_rules(ev["counts"], mem)

        return {
            "label": rule_res["predicted_label"],
            "confidence": rule_res["confidence"],
            "dominance": rule_res["dominance"],
            "rule_fired": rule_res["rule_fired"],
            "top_class": rule_res["top_class"],
            "runner_up_class": rule_res["runner_up_class"],
            "counts": ev["counts"],
            "unique_counts": ev["unique_counts"],
            "total_occurrences": ev["total_occurrences"],
            "matched_keywords": ev["matched_keywords"],
            "memberships": mem
        }

    def transform(
        self,
        df: pd.DataFrame,
        text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Applies fuzzy weak labeling to a DataFrame.

        Produces:
          - fuzzy_label: predicted class or 'Uncertain'
          - confidence: float score in [0.0, 1.0]
          - dominance: float margin in [0.0, 1.0]
          - rule_fired: identifier of rule triggered
          - evidence_counts: formatted string of counts per class
          - top_class: top candidate class
        """
        df_out = df.copy()
        fuzzy_labels = []
        confidences = []
        dominances = []
        rules = []
        evidence_strs = []
        top_classes = []

        for _, row in df_out.iterrows():
            text = str(row[text_column]) if pd.notna(row.get(text_column)) else ""
            res = self.predict_one(text)

            fuzzy_labels.append(res["label"])
            confidences.append(res["confidence"])
            dominances.append(res["dominance"])
            rules.append(res["rule_fired"])
            top_classes.append(res["top_class"])

            # Formatted evidence string: e.g. "Skill:2|Decision:0|Violation:0|Perceptual:0"
            c_str = "|".join([f"{k}:{v}" for k, v in res["counts"].items()])
            evidence_strs.append(c_str)

        df_out["fuzzy_label"] = fuzzy_labels
        df_out["confidence"] = confidences
        df_out["dominance"] = dominances
        df_out["rule_fired"] = rules
        df_out["top_class"] = top_classes
        df_out["evidence_counts"] = evidence_strs

        return df_out


# -----------------------------------------------------------------------------
# Standalone Unit Test Suite
# -----------------------------------------------------------------------------
def run_unit_tests():
    """
    Minimal unit tests verifying required behavior:
      1. Empty text
      2. Single supported class
      3. Competing class evidence
      4. Ties between classes
      5. Repeated keyword occurrences
      6. Evidence threshold (τ) boundary
      7. Dominance margin (δ) boundary
      8. Deterministic predictions
    """
    print("=" * 80)
    print("RUNNING FUZZY WEAK LABELER UNIT TESTS")
    print("=" * 80)

    # Default thresholds per roadmap: τ = 0.35, δ = 0.15
    labeler = FuzzyWeakLabeler(evidence_threshold=0.35, dominance_margin=0.15, count_occurrences=True)

    # 1. Empty Text
    print("[Test 1] Empty text handling...")
    for empty_input in ["", "   ", None, "\n\t"]:
        res = labeler.predict_one(empty_input)
        assert res["label"] == "Uncertain", f"Expected Uncertain for empty input, got {res['label']}"
        assert res["confidence"] == 0.0
        assert res["dominance"] == 0.0
        assert res["rule_fired"] == "R0_NO_EVIDENCE"
    print("  -> PASS: All empty/whitespace inputs safely return 'Uncertain' with 0.0 confidence.")

    # 2. Single Supported Class (strong evidence - 2 keywords)
    print("\n[Test 2] Single supported class (strong evidence)...")
    # 'checklist' and 'forgot' are both Skill-Based Error (count = 2)
    text_single = "I reviewed the checklist and realized I forgot the landing gear procedure."
    res_single = labeler.predict_one(text_single)
    assert res_single["label"] == "Skill-Based Error", f"Expected Skill-Based Error, got {res_single['label']}"
    assert res_single["counts"]["Skill-Based Error"] == 2
    assert res_single["counts"]["Decision Error"] == 0
    assert res_single["dominance"] == 1.0, f"Expected 1.0 dominance for unopposed class, got {res_single['dominance']}"
    assert res_single["confidence"] > 0.0
    assert res_single["rule_fired"] in ("R5_HIGH_DOMINANCE", "R5_MODERATE_DOMINANCE")
    print(f"  -> PASS: Unopposed class correctly labeled '{res_single['label']}' with dominance={res_single['dominance']}, rule={res_single['rule_fired']}.")

    # 3. Competing Class Evidence (strong dominance)
    print("\n[Test 3] Competing class evidence (strong dominance)...")
    # 'decided' (Decision Error) vs 'checklist' (Skill-Based Error)
    # 3 Decision Error keywords vs 1 Skill-Based Error keyword
    text_competing = "We decided and elected to divert immediately even though I was running the checklist."
    res_competing = labeler.predict_one(text_competing)
    assert res_competing["counts"]["Decision Error"] == 3  # decided, elected, divert
    assert res_competing["counts"]["Skill-Based Error"] == 1  # checklist
    # Dominance = (3 - 1) / (3 + 1) = 2/4 = 0.50 >= δ=0.15
    assert res_competing["dominance"] == 0.50, f"Expected dominance 0.50, got {res_competing['dominance']}"
    assert res_competing["label"] == "Decision Error"
    assert res_competing["rule_fired"] in ("R5_HIGH_DOMINANCE", "R5_MODERATE_DOMINANCE")
    print(f"  -> PASS: Competing evidence correctly resolved in favor of dominant class '{res_competing['label']}' (dominance={res_competing['dominance']}).")

    # 4. Ties Between Classes
    print("\n[Test 4] Ties between classes...")
    # 1 keyword for Skill-Based ('forgot') and 1 keyword for Violation ('unauthorized')
    text_tie = "The pilot forgot the procedure and made an unauthorized entry."
    res_tie = labeler.predict_one(text_tie)
    assert res_tie["counts"]["Skill-Based Error"] == 1
    assert res_tie["counts"]["Violation"] == 1
    assert res_tie["label"] == "Uncertain", f"Expected Uncertain for tie, got {res_tie['label']}"
    assert res_tie["dominance"] == 0.0
    assert res_tie["rule_fired"] == "R3_EXACT_TIE"
    print(f"  -> PASS: Exact tie between top classes (1 vs 1) safely abstains with 'Uncertain' (no biased tie-breaking).")

    # 5. Repeated Keyword Occurrences
    print("\n[Test 5] Repeated keyword occurrences...")
    # 'checklist' repeated 3 times
    text_repeat = "I ran the checklist before departure. During cruise I re-read the checklist. Before landing the checklist was verified."
    res_repeat = labeler.predict_one(text_repeat)
    assert res_repeat["counts"]["Skill-Based Error"] == 3, f"Expected 3 occurrences of 'checklist', got {res_repeat['counts']['Skill-Based Error']}"
    assert res_repeat["unique_counts"]["Skill-Based Error"] == 1, "Expected 1 unique keyword"
    assert res_repeat["label"] == "Skill-Based Error"
    assert res_repeat["dominance"] == 1.0

    # Contrast with count_occurrences=False
    labeler_unique = FuzzyWeakLabeler(evidence_threshold=0.35, dominance_margin=0.15, count_occurrences=False)
    res_unique = labeler_unique.predict_one(text_repeat)
    assert res_unique["counts"]["Skill-Based Error"] == 1, f"Expected count 1 when count_occurrences=False, got {res_unique['counts']['Skill-Based Error']}"
    print(f"  -> PASS: Repeated keyword occurrences correctly counted as 3 occurrences (or 1 when count_occurrences=False).")

    # 6. Evidence Threshold (tau) Boundary Test
    print("\n[Test 6] Evidence threshold (tau=0.35) boundary...")
    # With 1 keyword match, evidence count = 1, which is >= 0.35, so should pass
    text_one_kw = "I forgot the procedure."  # 'forgot' -> Skill-Based Error
    res_one = labeler.predict_one(text_one_kw)
    # With only 1 keyword, dominance = 1.0, evidence count = 1 >= τ
    assert res_one["counts"]["Skill-Based Error"] == 1
    assert res_one["dominance"] == 1.0
    # Since top_count=1 >= 0.35 and dominance=1.0 >= 0.15, should predict
    assert res_one["label"] == "Skill-Based Error"
    print(f"  -> PASS: Single keyword (count=1 >= tau) correctly predicts with dominance=1.0.")

    # 7. Dominance Margin (delta) Boundary Test
    print("\n[Test 7] Dominance margin (delta=0.15) boundary...")
    # Need a case where dominance < 0.15 -> should abstain
    # Example: 3 vs 2 evidence gives dominance = (3-2)/(3+2) = 0.20 >= 0.15 -> predicts
    # 6 vs 5 gives dominance = 1/11 ≈ 0.09 < 0.15 -> abstains
    # But our keywords are limited, let's use a controlled scenario
    labeler_test = FuzzyWeakLabeler(evidence_threshold=0.35, dominance_margin=0.15)
    # Create a mock scenario by directly testing the evaluate_fuzzy_rules
    # For 6 vs 5 counts:
    mock_counts = {c: 0 for c in labeler_test.classes}
    mock_counts["Skill-Based Error"] = 6
    mock_counts["Decision Error"] = 5
    # Mock memberships
    mock_memberships = labeler_test.compute_memberships(mock_counts)
    rule_res = labeler_test.evaluate_fuzzy_rules(mock_counts, mock_memberships)
    assert abs(rule_res["dominance"] - (6-5)/11) < 0.001
    assert rule_res["dominance"] < 0.15
    assert rule_res["predicted_label"] == "Uncertain"
    assert rule_res["rule_fired"] == "R4_BELOW_DOMINANCE_MARGIN"
    print(f"  -> PASS: Dominance {rule_res['dominance']:.4f} < delta=0.15 correctly abstains.")

    # 8. Deterministic Predictions
    print("\n[Test 8] Deterministic predictions...")
    text_deterministic = "We decided and elected to divert and chose to continue."
    res1 = labeler.predict_one(text_deterministic)
    res2 = labeler.predict_one(text_deterministic)
    res3 = labeler.predict_one(text_deterministic)
    assert res1["label"] == res2["label"] == res3["label"]
    assert res1["confidence"] == res2["confidence"] == res3["confidence"]
    assert res1["dominance"] == res2["dominance"] == res3["dominance"]
    print(f"  -> PASS: Three identical inputs produce identical outputs (label={res1['label']}).")

    print("\n" + "=" * 80)
    print("ALL 8 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
