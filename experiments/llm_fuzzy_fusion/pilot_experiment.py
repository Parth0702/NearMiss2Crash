#!/usr/bin/env python3
"""
NearMiss2Crash — Phase 3D: Gemini Pilot Experiment

Select 10 diverse ASRS narratives and run Gemini labeling on each.
Cache all responses for later fusion experiments.
"""

import os
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv


# Load environment
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# Configuration
SEED = 42
N_SAMPLES = 10
CACHE_FILE = PROJECT_ROOT / "experiments" / "llm_fuzzy_fusion" / "gemini_pilot_cache.json"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY")

# ASRS data
FUZZY_FULL_RESULTS = PROJECT_ROOT / "experiments" / "fuzzy_asrs" / "results" / "fuzzy_full_results.csv"

# HFACS classes
HFACS_CLASSES = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]

# System prompt for Gemini
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


def validate_response(response_json: Dict[str, Any]) -> tuple[bool, str]:
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

        required_classes = set(HFACS_CLASSES)
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


def select_diverse_narratives(df: pd.DataFrame, n: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """Select diverse narratives for pilot experiment."""
    random.seed(seed)

    # Get fuzzy-labeled narratives (exclude Uncertain)
    labeled = df[df['fuzzy_label'] != 'Uncertain'].copy()

    # Get crisp-labeled narratives (exclude Unlabeled)
    crisp_labeled = df[(df['crisp_label'] != 'Unlabeled') & (df['fuzzy_label'] != 'Uncertain')].copy()

    samples = []

    # 1. Clear single-class evidence (high fuzzy confidence)
    high_conf = labeled[labeled['confidence'] > 0.7].copy()
    if len(high_conf) > 0:
        for cls in HFACS_CLASSES:
            cls_samples = high_conf[high_conf['fuzzy_label'] == cls]
            if len(cls_samples) > 0:
                samples.append(cls_samples.sample(1, random_state=seed).iloc[0].to_dict())
                if len(samples) >= n:
                    break

    # 2. Competing evidence (fuzzy dominance < 0.3 but labeled)
    competing = labeled[(labeled['dominance'] > 0) & (labeled['dominance'] < 0.3)].copy()
    if len(competing) > 0 and len(samples) < n:
        for _, row in competing.head(n - len(samples)).iterrows():
            samples.append(row.to_dict())

    # 3. Exact keyword ties (rule R3_EXACT_TIE)
    ties = df[df['rule_fired'] == 'R3_EXACT_TIE'].copy()
    if len(ties) > 0 and len(samples) < n:
        for _, row in ties.head(n - len(samples)).iterrows():
            samples.append(row.to_dict())

    # 4. Low dominance decisions (R4_BELOW_DOMINANCE_MARGIN)
    low_dom = df[df['rule_fired'] == 'R4_BELOW_DOMINANCE_MARGIN'].copy()
    if len(low_dom) > 0 and len(samples) < n:
        for _, row in low_dom.head(n - len(samples)).iterrows():
            samples.append(row.to_dict())

    # 5. No keyword evidence (R0_NO_EVIDENCE)
    no_evidence = df[df['rule_fired'] == 'R0_NO_EVIDENCE'].copy()
    if len(no_evidence) > 0 and len(samples) < n:
        for _, row in no_evidence.head(n - len(samples)).iterrows():
            samples.append(row.to_dict())

    # Fill remaining with random labeled samples
    remaining = n - len(samples)
    if remaining > 0:
        remaining_df = labeled[~labeled['report_id'].isin([s['report_id'] for s in samples])]
        if len(remaining_df) > 0:
            for _, row in remaining_df.sample(min(remaining, len(remaining_df)), random_state=seed).iterrows():
                samples.append(row.to_dict())

    # Ensure we have exactly n samples
    samples = samples[:n]

    # Print selection summary
    print(f"Selected {len(samples)} narratives:")
    for i, s in enumerate(samples):
        print(f"  {i+1}. {s['report_id']} | fuzzy={s.get('fuzzy_label', 'N/A')} | crisp={s.get('crisp_label', 'N/A')} | rule={s.get('rule_fired', 'N/A')} | conf={s.get('confidence', 0):.2f} | dom={s.get('dominance', 0):.2f}")

    return samples


def run_gemini_on_narrative(client: genai.Client, model_name: str, narrative: str, report_id: str) -> Dict[str, Any]:
    """Run Gemini on a single narrative and return structured result."""
    combined_prompt = f"{SYSTEM_PROMPT}\n\nNarrative:\n{narrative}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=combined_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )

        if not response.text:
            return {"report_id": report_id, "success": False, "error": "Empty response"}

        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as e:
            return {"report_id": report_id, "success": False, "error": f"JSON decode error: {e}", "raw": response.text[:500]}

        valid, msg = validate_response(parsed)
        if not valid:
            return {"report_id": report_id, "success": False, "error": f"Validation failed: {msg}", "raw": response.text[:500]}

        # Extract token usage if available
        token_usage = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', None),
                "candidates_tokens": getattr(response.usage_metadata, 'candidates_token_count', None),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', None)
            }

        return {
            "report_id": report_id,
            "success": True,
            "gemini_response": parsed,
            "token_usage": token_usage
        }

    except Exception as e:
        return {"report_id": report_id, "success": False, "error": str(e)}


def load_existing_cache() -> Dict[str, Any]:
    """Load existing cache if present."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any]):
    """Save cache to file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def main():
    print("=" * 70)
    print("NearMiss2Crash — Phase 3D: Gemini Pilot Experiment")
    print("=" * 70)

    if not API_KEY:
        print("[FAIL] GEMINI_API_KEY not found in .env")
        sys.exit(1)

    print(f"[OK] API key loaded")
    print(f"[INFO] Model: {GEMINI_MODEL}")
    print(f"[INFO] Seed: {SEED}")
    print(f"[INFO] Samples: {N_SAMPLES}")
    print(f"[INFO] Cache: {CACHE_FILE}")

    # Load ASRS data
    print("\n[LOAD] Loading ASRS data...")
    df = pd.read_csv(FUZZY_FULL_RESULTS)
    print(f"  Total narratives: {len(df)}")
    print(f"  Fuzzy labeled: {(df['fuzzy_label'] != 'Uncertain').sum()}")
    print(f"  Crisp labeled: {(df['crisp_label'] != 'Unlabeled').sum()}")

    # Select diverse narratives
    print("\n[SELECT] Selecting diverse narratives...")
    selected = select_diverse_narratives(df, N_SAMPLES, SEED)

    # Load existing cache
    cache = load_existing_cache()
    print(f"\n[CACHE] Loaded {len(cache)} existing cached responses")

    # Initialize Gemini client
    client = genai.Client(api_key=API_KEY)
    print("[OK] Gemini client initialized")

    # Process each narrative
    new_results = 0
    for i, sample in enumerate(selected):
        report_id = str(sample['report_id'])
        narrative = sample['text']

        print(f"\n[{i+1}/{N_SAMPLES}] Processing {report_id}...")

        # Check cache
        if report_id in cache:
            print(f"  [CACHE] Using cached response")
            continue

        # Run Gemini
        result = run_gemini_on_narrative(client, GEMINI_MODEL, narrative, report_id)

        if result["success"]:
            print(f"  [OK] Label: {result['gemini_response']['proposed_label']} (conf: {result['gemini_response']['confidence']:.2f})")
            new_results += 1
        else:
            print(f"  [FAIL] {result.get('error', 'Unknown error')}")

        cache[report_id] = result

        # Save cache incrementally
        save_cache(cache)

    # Summary
    print("\n" + "=" * 70)
    print("PILOT EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total selected:     {N_SAMPLES}")
    print(f"New API calls:      {new_results}")
    print(f"Cached responses:   {len(cache)}")
    print(f"Cache file:         {CACHE_FILE}")

    # Print summary of results
    success_count = sum(1 for v in cache.values() if v.get('success', False))
    print(f"Successful:         {success_count}")

    if success_count > 0:
        print("\nGemini Labels:")
        for report_id, result in cache.items():
            if result.get('success'):
                r = result['gemini_response']
                tokens = result.get('token_usage', {})
                token_str = f" (tokens: {tokens.get('total_tokens', 'N/A')})" if tokens else ""
                print(f"  {report_id}: {r['proposed_label']} (conf={r['confidence']:.2f}){token_str}")


if __name__ == "__main__":
    main()