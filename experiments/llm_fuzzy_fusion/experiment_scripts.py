#!/usr/bin/env python3
"""
NearMiss2Crash — Phase 3D: Full 150-Report Gemini Experiment

Reproducible experiment with documented sampling strata and comprehensive analysis.
"""

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import fusion components
from fusion_labeler import FusionLabeler

# Load environment
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUSION_CONFIG = {
    "alpha": 0.5,
    "min_support": 0.25,
    "min_dominance_gap": 0.10
}

# Experiment parameters
SEED = 42
N_TOTAL = 150
SENSITIVITY_TEST_N = 20  # Number of reports for sensitivity testing

# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
CRASH_DATA = PROJECT_ROOT / "experiments" / "fuzzy_asrs" / "results" / "fuzzy_full_results.csv"
ASRS_COLUMNS = PROJECT_ROOT / "asrs_columns_report.txt"

# Gemini configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY")
MAX_REQUESTS_PER_MIN = 15  # For rate limiting
REQUEST_TIMEOUT = 120  # seconds per request
MAX_TOTAL_TOKENS = 8000  # Daily quota warning threshold

# System prompt for Gemini (same as smoke test)
SYSTEM_PROMPT = """You are an aviation safety analyst expert in HFACS (Human Factors Analysis and Classification System).
Analyze the provided aviation incident narrative and provide structured evidence for all four HFACS-core classes.

RETURN ONLY THIS EXACT JSON FORMAT (no other keys, no nesting):

{
  "hfacs_evidence": [
    {
      "class": "Skill-Based Error",
      "evidence_score": 0.0,
      "supporting_excerpts": ["excerpt from text"],
      "reasoning": "explanation"
    },
    {
      "class": "Decision Error",
      "evidence_score": 0.0,
      "supporting_excerpts": ["excerpt from text"],
      "reasoning": "explanation"
    },
    {
      "class": "Violation",
      "evidence_score": 0.0,
      "supporting_excerpts": ["excerpt from text"],
      "reasoning": "explanation"
    },
    {
      "class": "Perceptual Error",
      "evidence_score": 0.0,
      "supporting_excerpts": ["excerpt from text"],
      "reasoning": "explanation"
    }
  ],
  "proposed_label": "Decision Error",
  "confidence": 0.0
}

Rules:
- evidence_score must be a number 0.0 to 1.0
- supporting_excerpts must be an array of strings (direct quotes from narrative)
- reasoning must be a string
- proposed_label must be exactly one of: Skill-Based Error, Decision Error, Violation, Perceptual Error
- confidence must be a number 0.0 to 1.0
- Do NOT include any other keys like "hfacs_analysis", "unsafe_acts", etc."""


def validate_response(response_json: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate the JSON response against the HFACS schema."""
    try:
        if "hfacs_evidence" not in response_json:
            return False, "Missing 'hfacs_evidence' key"
        if "proposed_label" not in response_json:
            return False, "Missing 'proposed_label' key"
        if "confidence" not in response_json:
            return False, "Missing 'confidence' key"

        evidence = response_json["hfacs_evidence"]
        if not isinstance(evidence, list) or len(evidence) != 4:
            return False, f"'hfacs_evidence' must be a list of 4 items"

        required_classes = {"Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"}
        found_classes = set()

        for item in evidence:
            if not isinstance(item, dict):
                return False, "Each evidence item must be an object"

            if "class" not in item:
                return False, "Missing 'class' in evidence item"
            cls = item["class"]
            if cls not in required_classes:
                return False, f"Invalid class: {cls}"
            found_classes.add(cls)

            if "evidence_score" not in item:
                return False, f"Missing 'evidence_score' for {cls}"
            score = item["evidence_score"]
            if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
                return False, f"Invalid evidence_score for {cls}: {score}"

            if "supporting_excerpts" not in item:
                return False, f"Missing 'supporting_excerpts' for {cls}"
            excerpts = item["supporting_excerpts"]
            if not isinstance(excerpts, list) or not all(isinstance(e, str) for e in excerpts):
                return False, f"'supporting_excerpts' for {cls} must be list of strings"

            if "reasoning" not in item:
                return False, f"Missing 'reasoning' for {cls}"
            if not isinstance(item["reasoning"], str):
                return False, f"'reasoning' for {cls} must be a string"

        if found_classes != required_classes:
            return False, f"Missing classes. Found: {found_classes}"

        label = response_json["proposed_label"]
        if label not in required_classes:
            return False, f"Invalid proposed_label: {label}"

        conf = response_json["confidence"]
        if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
            return False, f"Invalid confidence: {conf}"

        return True, "Valid"

    except Exception as e:
        return False, f"Validation error: {e}"


def load_data() -> pd.DataFrame:
    """Load and prepare ASRS data."""
    df = pd.read_csv(CRASH_DATA)

    # Filter to fuzzy-labeled narratives (excluding Uncertain)
    df_fuzzy = df[df['fuzzy_label'] != 'Uncertain'].copy()

    # Add derived features for sampling
    df_fuzzy['has_clever_evidence'] = df_fuzzy['dominance'] > 0.7
    df_fuzzy['has_competing_evidence'] = df_fuzzy['dominance'] < 0.3
    df_fuzzy['is_tie'] = df_fuzzy['rule_fired'] == 'R3_EXACT_TIE'
    df_fuzzy['is_low_dominance'] = df_fuzzy['rule_fired'] == 'R4_BELOW_DOMINANCE_MARGIN'
    df_fuzzy['is_no_evidence'] = df_fuzzy['rule_fired'] == 'R0_NO_EVIDENCE'

    return df_fuzzy


def create_sampling_strata(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Create documented sampling strata for diverse representation.
    Returns report_ids (not indices) for each stratum.
    """
    strata = {}

    # Stratification 1: Clear single-class evidence (high confidence, high dominance)
    clear = df[
        (df['dominance'] > 0.7) &
        (df['confidence'] > 0.7) &
        (df['fuzzy_label'] != 'Uncertain')
    ].copy()

    strata['clear_single_class'] = clear.sample(
        min(len(clear), N_TOTAL // 4),
        random_state=SEED
    )['report_id'].tolist()

    # Stratification 2: Competing evidence (low dominance, different crisp/fuzzy)
    competing = df[
        (df['dominance'] < 0.3) &
        (df['fuzzy_label'] != 'Uncertain') &
        (df['crisp_label'] != df['fuzzy_label'])
    ].copy()

    strata['competing_crisp_fuzzy_differences'] = competing.sample(
        min(len(competing), N_TOTAL // 4),
        random_state=SEED
    )['report_id'].tolist()

    # Stratification 3: Exact keyword ties
    ties = df[df['rule_fired'] == 'R3_EXACT_TIE'].copy()
    strata['exact_keyword_ties'] = ties.sample(
        min(len(ties), N_TOTAL // 8),
        random_state=SEED
    )['report_id'].tolist()

    # Stratification 4: Low dominance decisions
    low_dom = df[df['rule_fired'] == 'R4_BELOW_DOMINANCE_MARGIN'].copy()
    strata['low_dominance_decisions'] = low_dom.sample(
        min(len(low_dom), N_TOTAL // 8),
        random_state=SEED
    )['report_id'].tolist()

    # Stratification 5: No keyword evidence
    no_evidence = df[df['rule_fired'] == 'R0_NO_EVIDENCE'].copy()
    strata['no_keyword_evidence'] = no_evidence.sample(
        min(len(no_evidence), N_TOTAL // 8),
        random_state=SEED
    )['report_id'].tolist()

    # Stratification 6: Crisp-Fuzzy disagreements (wide distribution)
    disagreements = df[df['crisp_label'] != df['fuzzy_label']].copy()
    strata['crisp_fuzzy_disagreements'] = disagreements.sample(
        min(len(disagreements), N_TOTAL - len([idx for strat in strata.values() for idx in strat])),
        random_state=SEED
    )['report_id'].tolist()

    return strata


def select_reports(strata: Dict[str, List[str]], n: int = N_TOTAL) -> List[str]:
    """Select reports from strata ensuring diversity."""
    selected_indices = []
    samples_per_strata = {
        'clear_single_class': max(1, n // 4),
        'competing_crisp_fuzzy_differences': max(1, n // 4),
        'exact_keyword_ties': max(1, n // 8),
        'low_dominance_decisions': max(1, n // 8),
        'no_keyword_evidence': max(1, n // 8),
        'crisp_fuzzy_disagreements': max(1, n // 4)  # Fill remaining
    }

    for stratum, indices in strata.items():
        available = samples_per_strata[stratum]
        to_select = min(available, len(indices))
        if to_select > 0:
            selected_indices.extend(indices[:to_select])

    # If we don't have enough, add more from disagreements
    if len(selected_indices) < n:
        remaining_indices = [idx for idx in strata['crisp_fuzzy_disagreements']
                            if idx not in selected_indices]
        additional = min(n - len(selected_indices), len(remaining_indices))
        selected_indices.extend(remaining_indices[:additional])

    # Shuffle and return
    random.seed(SEED)
    selected_indices = random.sample(selected_indices, min(n, len(selected_indices)))

    print(f"Selected {len(selected_indices)} reports:")
    print("Distribution by stratum:")
    for stratum in strata.keys():
        stratum_indices = [idx for idx in selected_indices if idx in strata[stratum]]
        print(f"  {stratum}: {len(stratum_indices)} reports")

    return selected_indices


def log_response(client: genai.Client, narrative: str, model_name: str, attempt_num: int = 0) -> Dict[str, Any]:
    """Log API response with metadata."""
    start_time = time.time()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=narrative,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )

        elapsed = time.time() - start_time

        result = {
            "success": True,
            "response_text": response.text,
            "elapsed_time": elapsed,
            "attempt": attempt_num,
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            result["token_usage"] = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', None),
                "candidates_tokens": getattr(response.usage_metadata, 'candidates_token_count', None),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', None)
            }

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed,
            "attempt": attempt_num,
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


def run_experiment():
    """Run the full Phase 3D experiment."""
    print("=" * 80)
    print("NearMiss2Crash — Phase 3D: Full 150-Report Gemini Experiment")
    print("=" * 80)

    if not API_KEY:
        print("[FAIL] GEMINI_API_KEY not found in .env")
        sys.exit(1)

    print(f"[CONFIG] Model: {GEMINI_MODEL}")
    print(f"[CONFIG] Total reports: {N_TOTAL}")
    print(f"[CONFIG] Sensitivity test reports: {SENSITIVITY_TEST_N}")
    print(f"[CONFIG] Max requests per minute: {MAX_REQUESTS_PER_MIN}")
    print(f"[CONFIG] Request timeout: {REQUEST_TIMEOUT} seconds")
    print(f"[CONFIG] Max total tokens: {MAX_TOTAL_TOKENS}")

    # Load data
    print("\n[LOAD] Loading ASRS data...")
    df_fuzzy = load_data()
    print(f"  Total fuzzy-labeled narratives: {len(df_fuzzy)}")

    # Create sampling strata
    print("\n[STRATIFY] Creating sampling strata...")
    strata = create_sampling_strata(df_fuzzy)

    # Select reports
    print("\n[SELECT] Selecting reports...")
    selected_indices = select_reports(strata, N_TOTAL)

    # Initialize clients
    print("\n[INIT] Initializing clients...")
    gemini_client = genai.Client(api_key=API_KEY)

    fusion = FusionLabeler(**FUSION_CONFIG)

    # Prepare results containers
    gemini_cache = {}
    fusion_results = []
    sensitivity_results = []

    print(f"\n[RUN] Starting Gemini labeling...")

    # Process reports
    for i, report_id in enumerate(selected_indices):
        print(f"\n[{i+1}/{N_TOTAL}] Processing {report_id}")

        # Get narrative
        row = df_fuzzy.loc[report_id]
        narrative = row['text']
        fuzzy_label = row['fuzzy_label']
        crisp_label = row['crisp_label']
        rule_fired = row['rule_fired']
        dominance = row['dominance']
        confidence = row['confidence']

        # Get Gemini evidence (from cache or API)
        gemini_result = None
        if report_id in gemini_cache:
            print(f"  [CACHE] Using cached Gemini response")
            gemini_result = gemini_cache[report_id]
        else:
            print(f"  [API] Making Gemini API call")

            # Rate limiting
            if i > 0 and i % MAX_REQUESTS_PER_MIN == 0:
                print(f"  [RATE] Waiting 60 seconds for rate limit...")
                time.sleep(60)

            # Make API call with retries
            for attempt in range(3):
                result = log_response(gemini_client, narrative, GEMINI_MODEL, attempt)

                if result["success"]:
                    gemini_result = result
                    break
                elif attempt < 2:
                    print(f"  [RETRY] Attempt {attempt + 1} failed: {result['error']}")
                    time.sleep(10)
                else:
                    print(f"  [FAIL] All attempts failed: {result['error']}")
                    gemini_result = result

        if gemini_result is None:
            print(f"  [SKIP] Skipping report {report_id}")
            continue

        # Cache Gemini result
        gemini_cache[report_id] = gemini_result

        # Parse Gemini response
        gemini_parsed = None
        if gemini_result["success"]:
            try:
                gemini_parsed = json.loads(gemini_result["response_text"])
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Failed to parse JSON: {e}")
                print(f"  Raw response: {gemini_result['response_text'][:500]}")
                continue

            # Validate
            valid, msg = validate_response(gemini_parsed)
            if not valid:
                print(f"  [ERROR] Validation failed: {msg}")
                continue

            # Create GeminiEvidence
            gemini_evidence = type('GeminiEvidence', (), {})()
            gemini_evidence.evidence_scores = {item['class']: item['evidence_score'] for item in gemini_parsed['hfacs_evidence']}
            gemini_evidence.proposed_label = gemini_parsed['proposed_label']
            gemini_evidence.confidence = gemini_parsed['confidence']
            gemini_evidence.supporting_excerpts = {item['class']: item['supporting_excerpts'] for item in gemini_parsed['hfacs_evidence']}
            gemini_evidence.reasoning = {item['class']: item['reasoning'] for item in gemini_parsed['hfacs_evidence']}
        else:
            print(f"  [ERROR] Gemini API failed: {gemini_result['error']}")
            continue

        # Create fuzzy evidence
        fuzzy_evidence = type('FuzzyEvidence', (), {})()
        fuzzy_evidence.evidence_counts = {
            cls: int(row.get(f'keyword_{cls}', 0))
            for cls in ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]
        }
        fuzzy_evidence.fuzzy_label = fuzzy_label
        fuzzy_evidence.confidence = confidence
        fuzzy_evidence.dominance = dominance
        fuzzy_evidence.rule_fired = rule_fired

        # Compute fused result
        fusion_result = fusion.fuse(fuzzy_evidence, gemini_evidence, report_id)

        # Record fusion result
        fusion_results.append({
            'report_id': report_id,
            'report_category': next((k for k, v in strata.items() if report_id in v), 'other'),
            'crisp_label': crisp_label,
            'fuzzy_label': fuzzy_label,
            'gemini_label': gemini_parsed['proposed_label'] if gemini_parsed else None,
            'fusion_label': fusion_result.fused_label,
            'fusion_confidence': fusion_result.fused_confidence,
            'fusion_abstained': fusion_result.abstained,
            'fusion_changed_from_fuzzy': fusion_result.change_from_fuzzy,
            'gemini_confidence': gemini_parsed['confidence'] if gemini_parsed else None,
            'fuzzy_confidence': confidence,
            'fuzzy_dominance': dominance,
            'rule_fired': rule_fired,
            'alpha_used': FUSION_CONFIG['alpha'],
            'token_usage': gemini_result.get('token_usage', {}),
            'api_response_time': gemini_result.get('elapsed_time', 0)
        })

        # Sensitivity test on first N reports
        if i < SENSITIVITY_TEST_N:
            sensitivity_results.append({
                'report_id': report_id,
                'alpha': FUSION_CONFIG['alpha'],
                'fused_label': fusion_result.fused_label,
                'fused_confidence': fusion_result.fused_confidence,
                'fused_scores': fusion_result.fused_scores
            })

        # Progress indicator
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/{N_TOTAL} reports processed")

    # Save results
    print(f"\n[SAVE] Saving results...")

    # Save Gemini cache
    cache_file = PROJECT_ROOT / "experiments" / "llm_fuzzy_fusion" / "gemini_full_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, 'w') as f:
        json.dump(gemini_cache, f, indent=2)
    print(f"  Gemini cache saved to {cache_file}")

    # Save fusion results
    fusion_file = PROJECT_ROOT / "experiments" / "llm_fuzzy_fusion" / "fusion_results.json"
    with open(fusion_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_reports': N_TOTAL,
                'sensitivity_test_n': SENSITIVITY_TEST_N,
                'fusion_config': FUSION_CONFIG,
                'model': GEMINI_MODEL,
                'seed': SEED,
                'sampling_strata': strata.keys()
            },
            'results': fusion_results
        }, f, indent=2)
    print(f"  Fusion results saved to {fusion_file}")

    # Print summary
    print(f"\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    # Statistics
    total_reports = len(fusion_results)
    abstained = sum(1 for r in fusion_results if r['fusion_abstained'])
    changed = sum(1 for r in fusion_results if r['fusion_changed_from_fuzzy'])

    print(f"\nSummary Statistics:")
    print(f"  Total reports processed: {total_reports}")
    print(f"  Gemini successful responses: {len(gemini_cache)}")
    print(f"  Fusion abstentions: {abstained}/{total_reports} ({abstained/total_reports*100:.1f}%)")
    print(f"  Fusion changes from fuzzy: {changed}/{total_reports} ({changed/total_reports*100:.1f}%)")

    # Distribution by stratum
    print(f"\nDistribution by sampling stratum:")
    for stratum in strata.keys():
        count = sum(1 for r in fusion_results if r['report_category'] == stratum)
        print(f"  {stratum}: {count} reports ({count/total_reports*100:.1f}%)")

    # Fusion sensitivity analysis
    if sensitivity_results:
        print(f"\nFusion Sensitivity Analysis (α={FUSION_CONFIG['alpha']}):")

        # Analyze confidence distribution
        confidences = [r['fusion_confidence'] for r in fusion_results if not r['fusion_abstained']]
        if confidences:
            print(f"  Fused confidence - mean: {np.mean(confidences):.3f}, std: {np.std(confidences):.3f}")

        # Analyze class distribution
        fusion_labels = [r['fusion_label'] for r in fusion_results if not r['fusion_abstained']]
        label_dist = pd.Series(fusion_labels).value_counts()
        print(f"  Fusion label distribution:")
        for label, count in label_dist.items():
            print(f"    {label}: {count} ({count/total_reports*100:.1f}%)")

    # Safety warnings
    total_tokens = sum(r.get('token_usage', {}).get('total_tokens', 0)
                     for r in fusion_results if r.get('token_usage'))

    if total_tokens > MAX_TOTAL_TOKENS * 0.9:
        print(f"\n[WARNING] Approaching token limit: {total_tokens} tokens")
        print(f"  Daily quota: {MAX_TOTAL_TOKENS} tokens")

    # Save detailed report
    report = {
        'metadata': {
            'experiment': 'Phase 3D: Gemini Pilot & LLM-Fuzzy Fusion',
            'total_reports': N_TOTAL,
            'sensitivity_test_n': SENSITIVITY_TEST_N,
            'sampling_seed': SEED,
            'fusion_config': FUSION_CONFIG,
            'gemini_model': GEMINI_MODEL,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'notes': 'All metrics are weak-label proxies. No claims of independently verified classification accuracy.'
        },
        'sampling_strata': strata,
        'results': fusion_results
    }

    report_file = PROJECT_ROOT / "experiments" / "llm_fuzzy_fusion" / "PHASE_3D_FULL_EXPERIMENT_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(f"# Phase 3D: Gemini Pilot & LLM-Fuzzy Fusion — Full Experiment Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Reports**: {N_TOTAL}\n")
        f.write(f"**Gemini Model**: {GEMINI_MODEL}\n")
        f.write(f"**Fusion Configuration**: α={FUSION_CONFIG['alpha']}, min_support={FUSION_CONFIG['min_support']}, min_dominance_gap={FUSION_CONFIG['min_dominance_gap']}\n\n")

        f.write(f"## Abstract\n")
        f.write(f"This experiment continues Phase 3D by running a comprehensive 150-report Gemini labeling pilot followed by LLM-fuzzy evidence fusion. "
                f"The study evaluates how Gemini semantic evidence complements fuzzy weak labels in hybrid fusion, maintaining all methodological constraints.\n\n")

        f.write(f"## Sampling Strategy\n")
        f.write(f"Reports were selected from documented strata to ensure diversity:\n")
        for stratum in strata.keys():
            count = len(strata[stratum])
            f.write(f"- **{stratum}**: {count} reports\n")

        f.write(f"\n## Methods\n")
        f.write(f"- **Data Source**: ASRS narratives with fuzzy weak labels")
        f.write(f"- **Gemini Model**: {GEMINI_MODEL}")
        f.write(f"- **Fusion Formula**: S_c = αF_c + (1-α)L_c")
        f.write(f"- **Decision Thresholds**: min_support={FUSION_CONFIG['min_support']}, min_dominance_gap={FUSION_CONFIG['min_dominance_gap']}")
        f.write(f"\n\n## Results Summary\n")
        f.write(f"- **Reports Processed**: {total_reports}")
        f.write(f"- **Gemini Success Rate**: {len(gemini_cache)}/{total_reports} ({len(gemini_cache)/total_reports*100:.1f}%)")
        f.write(f"- **Fusion Abstentions**: {abstained}/{total_reports} ({abstained/total_reports*100:.1f}%)")
        f.write(f"- **Fusion Changes from Fuzzy**: {changed}/{total_reports} ({changed/total_reports*100:.1f}%)")

        f.write(f"\n\n## Detailed Results\n")
        f.write(f"Full results are available in `experiments/llm_fuzzy_fusion/fusion_results.json`.")

    print(f"\nDetailed report saved to: {report_file}")

    return {
        'gemini_cache': gemini_cache,
        'fusion_results': fusion_results,
        'sensitivity_results': sensitivity_results,
        'report_file': report_file
    }


if __name__ == "__main__":
    results = run_experiment()
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to {results['report_file']}")