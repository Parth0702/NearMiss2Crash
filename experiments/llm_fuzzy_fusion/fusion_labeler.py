#!/usr/bin/env python3
"""
NearMiss2Crash — Phase 3D: Fusion Labeler

Implements evidence fusion between FuzzyWeakLabeler (keyword-based)
and Gemini (semantic) for HFACS classification.

Fusion formula: S_c = αF_c + (1-α)L_c

Where:
- F_c = normalized keyword evidence score for class c
- L_c = Gemini semantic evidence score for class c
- α = weighting parameter (default 0.5)
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


# HFACS classes
HFACS_CLASSES = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(HFACS_CLASSES)}


@dataclass
class FuzzyEvidence:
    """Evidence from FuzzyWeakLabeler."""
    evidence_counts: Dict[str, int]  # raw keyword counts per class
    fuzzy_label: str                 # predicted label from fuzzy
    confidence: float                # fuzzy confidence
    dominance: float                 # dominance score
    rule_fired: str                  # which rule fired

    def normalized_scores(self) -> Dict[str, float]:
        """Normalize evidence counts to [0, 1] range using min-max normalization."""
        total = sum(self.evidence_counts.values())
        if total == 0:
            return {c: 0.0 for c in HFACS_CLASSES}
        max_count = max(self.evidence_counts.values())
        if max_count == 0:
            return {c: 0.0 for c in HFACS_CLASSES}
        return {c: self.evidence_counts[c] / max_count for c in HFACS_CLASSES}


@dataclass
class GeminiEvidence:
    """Evidence from Gemini semantic analysis."""
    evidence_scores: Dict[str, float]  # per-class scores [0, 1]
    proposed_label: str                # Gemini's proposed label
    confidence: float                  # Gemini's confidence
    supporting_excerpts: Dict[str, List[str]]  # per-class excerpts
    reasoning: Dict[str, str]          # per-class reasoning


@dataclass
class FusionResult:
    """Result of fusion labeling."""
    report_id: str
    fused_scores: Dict[str, float]       # S_c = αF_c + (1-α)L_c
    fused_label: str                     # predicted label
    fused_confidence: float              # confidence in fused label
    original_fuzzy_label: str
    original_fuzzy_confidence: float
    gemini_label: str
    gemini_confidence: float
    alpha: float
    abstained: bool
    change_from_fuzzy: bool
    change_supported: Optional[bool] = None


class FusionLabeler:
    """
    Hybrid labeling system combining FuzzyWeakLabeler and Gemini.

    For each HFACS class:
    - F_c = normalized keyword evidence score (from fuzzy)
    - L_c = Gemini semantic evidence score
    - S_c = αF_c + (1-α)L_c

    Decision thresholds (separate from original fuzzy τ=0.35):
    - min_support: minimum S_c for top class to be considered
    - min_dominance_gap: minimum (S_top - S_second) for confident prediction
    - If thresholds not met: Uncertain
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_support: float = 0.25,
        min_dominance_gap: float = 0.10
    ):
        """
        Args:
            alpha: Weight for fuzzy evidence (1-alpha for Gemini)
            min_support: Minimum fused score for top class
            min_dominance_gap: Minimum gap between top and second class
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if not 0 <= min_support <= 1:
            raise ValueError("min_support must be in [0, 1]")
        if not 0 <= min_dominance_gap <= 1:
            raise ValueError("min_dominance_gap must be in [0, 1]")

        self.alpha = alpha
        self.min_support = min_support
        self.min_dominance_gap = min_dominance_gap

    def fuse(
        self,
        fuzzy: FuzzyEvidence,
        gemini: GeminiEvidence,
        report_id: str
    ) -> FusionResult:
        """
        Fuse fuzzy and Gemini evidence.

        Returns FusionResult with fused label, confidence, and metadata.
        """
        # Get normalized fuzzy scores
        F_scores = fuzzy.normalized_scores()

        # Get Gemini scores (already 0-1)
        L_scores = gemini.evidence_scores

        # Compute fused scores
        S_scores = {}
        for c in HFACS_CLASSES:
            F_c = F_scores.get(c, 0.0)
            L_c = L_scores.get(c, 0.0)
            S_c = self.alpha * F_c + (1 - self.alpha) * L_c
            S_scores[c] = S_c

        # Rank classes by fused score
        ranked = sorted(HFACS_CLASSES, key=lambda c: S_scores[c], reverse=True)
        top_class = ranked[0]
        second_class = ranked[1] if len(ranked) > 1 else None

        top_score = S_scores[top_class]
        second_score = S_scores[second_class] if second_class else 0.0

        dominance_gap = top_score - second_score

        # Decision logic with separate thresholds
        if top_score < self.min_support:
            # Insufficient evidence
            return FusionResult(
                report_id=report_id,
                fused_scores=S_scores,
                fused_label="Uncertain",
                fused_confidence=0.0,
                original_fuzzy_label=fuzzy.fuzzy_label,
                original_fuzzy_confidence=fuzzy.confidence,
                gemini_label=gemini.proposed_label,
                gemini_confidence=gemini.confidence,
                alpha=self.alpha,
                abstained=True,
                change_from_fuzzy=fuzzy.fuzzy_label != "Uncertain"
            )

        if dominance_gap < self.min_dominance_gap:
            # Close competition
            return FusionResult(
                report_id=report_id,
                fused_scores=S_scores,
                fused_label="Uncertain",
                fused_confidence=0.0,
                original_fuzzy_label=fuzzy.fuzzy_label,
                original_fuzzy_confidence=fuzzy.confidence,
                gemini_label=gemini.proposed_label,
                gemini_confidence=gemini.confidence,
                alpha=self.alpha,
                abstained=True,
                change_from_fuzzy=fuzzy.fuzzy_label != "Uncertain"
            )

        # Confident prediction
        change = (top_class != fuzzy.fuzzy_label)

        # Confidence = top_score * (1 - competition_penalty)
        # competition_penalty = second_score / top_score if top_score > 0 else 0
        competition_penalty = second_score / top_score if top_score > 0 else 0
        fused_confidence = top_score * (1 - 0.5 * competition_penalty)
        fused_confidence = max(0.0, min(1.0, fused_confidence))

        return FusionResult(
            report_id=report_id,
            fused_scores=S_scores,
            fused_label=top_class,
            fused_confidence=fused_confidence,
            original_fuzzy_label=fuzzy.fuzzy_label,
            original_fuzzy_confidence=fuzzy.confidence,
            gemini_label=gemini.proposed_label,
            gemini_confidence=gemini.confidence,
            alpha=self.alpha,
            abstained=False,
            change_from_fuzzy=change
        )


def load_fuzzy_data(report_id: str, df_fuzzy: pd.DataFrame) -> Optional[FuzzyEvidence]:
    """Extract fuzzy evidence for a report."""
    row = df_fuzzy[df_fuzzy['report_id'] == report_id]
    if len(row) == 0:
        return None
    row = row.iloc[0]

    # Parse evidence_counts string like "Skill-Based Error:0|Decision Error:1|Violation:0|Perceptual Error:0"
    evidence_str = row.get('evidence_counts', '')
    evidence_counts = {}
    for part in evidence_str.split('|'):
        if ':' in part:
            cls, count = part.split(':')
            evidence_counts[cls] = int(count)

    # Ensure all classes present
    for c in HFACS_CLASSES:
        if c not in evidence_counts:
            evidence_counts[c] = 0

    return FuzzyEvidence(
        evidence_counts=evidence_counts,
        fuzzy_label=row.get('fuzzy_label', 'Uncertain'),
        confidence=row.get('confidence', 0.0),
        dominance=row.get('dominance', 0.0),
        rule_fired=row.get('rule_fired', 'R0_NO_EVIDENCE')
    )


def load_gemini_data(report_id: str, cache: Dict[str, Any]) -> Optional[GeminiEvidence]:
    """Extract Gemini evidence from cache."""
    if report_id not in cache:
        return None
    entry = cache[report_id]
    if not entry.get('success', False):
        return None

    resp = entry['gemini_response']

    evidence_scores = {}
    supporting_excerpts = {}
    reasoning = {}

    for item in resp['hfacs_evidence']:
        cls = item['class']
        evidence_scores[cls] = item['evidence_score']
        supporting_excerpts[cls] = item['supporting_excerpts']
        reasoning[cls] = item['reasoning']

    # Ensure all classes present
    for c in HFACS_CLASSES:
        if c not in evidence_scores:
            evidence_scores[c] = 0.0
            supporting_excerpts[c] = []
            reasoning[c] = ""

    return GeminiEvidence(
        evidence_scores=evidence_scores,
        proposed_label=resp['proposed_label'],
        confidence=resp['confidence'],
        supporting_excerpts=supporting_excerpts,
        reasoning=reasoning
    )


def load_crisp_label(report_id: str, df_fuzzy: pd.DataFrame) -> str:
    """Get crisp label for a report."""
    row = df_fuzzy[df_fuzzy['report_id'] == report_id]
    if len(row) == 0:
        return "Unknown"
    return row.iloc[0].get('crisp_label', 'Unlabeled')


if __name__ == "__main__":
    import pandas as pd

    # Quick test
    print("FusionLabeler module loaded successfully")
    print(f"HFACS classes: {HFACS_CLASSES}")
    print(f"Default alpha: 0.5")
    print(f"Default min_support: 0.25")
    print(f"Default min_dominance_gap: 0.10")