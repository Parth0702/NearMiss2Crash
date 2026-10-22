#!/usr/bin/env python3
"""
NearMiss2Crash — Gemini Smoke Test

Validates that the configured Gemini model is accessible via API key,
and can produce structured HFACS evidence for a fictional aviation narrative.

Uses Google's official google-genai Python SDK.
Loads GEMINI_API_KEY and GEMINI_MODEL securely from .env file.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv


# Load environment variables from .env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


# Fictional aviation incident narrative for testing
FICTIONAL_NARRATIVE = """
Cessna 172 N12345, during a VFR cross-country flight from KPAO to KSQL,
encountered unexpected deteriorating weather. The pilot, a private pilot
with 150 hours total time, continued VFR into IMC conditions despite
not holding an instrument rating. The pilot descended below MVA while
attempting to maintain visual contact with terrain. ATC issued multiple
warnings about terrain proximity. The aircraft impacted rising terrain
approximately 3 NM from KSQL. The pilot stated post-incident: 'I thought
I could get under the clouds. I didn't want to divert because of fuel
concerns, but I had 45 minutes of fuel remaining.'
"""


# Structured JSON schema for HFACS evidence
HFACS_SCHEMA = {
    "type": "object",
    "properties": {
        "hfacs_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class": {
                        "type": "string",
                        "enum": ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]
                    },
                    "evidence_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "supporting_excerpts": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["class", "evidence_score", "supporting_excerpts", "reasoning"]
            }
        },
        "proposed_label": {
            "type": "string",
            "enum": ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["hfacs_evidence", "proposed_label", "confidence"]
}


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
        # Check required top-level keys
        if "hfacs_evidence" not in response_json:
            return False, "Missing 'hfacs_evidence' key"
        if "proposed_label" not in response_json:
            return False, "Missing 'proposed_label' key"
        if "confidence" not in response_json:
            return False, "Missing 'confidence' key"

        evidence = response_json["hfacs_evidence"]
        if not isinstance(evidence, list) or len(evidence) != 4:
            return False, f"'hfacs_evidence' must be a list of 4 items, got {len(evidence) if isinstance(evidence, list) else 'non-list'}"

        required_classes = {"Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"}
        found_classes = set()

        for item in evidence:
            if not isinstance(item, dict):
                return False, "Each evidence item must be an object"

            if "class" not in item:
                return False, "Missing 'class' in evidence item"
            cls = item["class"]
            if cls not in required_classes:
                return False, f"Invalid class: {cls}. Must be one of {required_classes}"
            found_classes.add(cls)

            if "evidence_score" not in item:
                return False, f"Missing 'evidence_score' for {cls}"
            score = item["evidence_score"]
            if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
                return False, f"Invalid evidence_score for {cls}: {score} (must be 0.0-1.0)"

            if "supporting_excerpts" not in item:
                return False, f"Missing 'supporting_excerpts' for {cls}"
            excerpts = item["supporting_excerpts"]
            if not isinstance(excerpts, list) or not all(isinstance(e, str) for e in excerpts):
                return False, f"'supporting_excerpts' for {cls} must be a list of strings"

            if "reasoning" not in item:
                return False, f"Missing 'reasoning' for {cls}"
            if not isinstance(item["reasoning"], str):
                return False, f"'reasoning' for {cls} must be a string"

        if found_classes != required_classes:
            return False, f"Missing classes. Found: {found_classes}, Required: {required_classes}"

        # Validate proposed_label
        label = response_json["proposed_label"]
        if label not in required_classes:
            return False, f"Invalid proposed_label: {label}"

        # Validate confidence
        conf = response_json["confidence"]
        if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
            return False, f"Invalid confidence: {conf} (must be 0.0-1.0)"

        return True, "Valid"

    except Exception as e:
        return False, f"Validation error: {e}"


def test_model(client: genai.Client, model_name: str, debug: bool = False) -> Optional[Dict[str, Any]]:
    """Test a specific model with the smoke test narrative."""
    # Combine system prompt and narrative into a single user message
    combined_prompt = f"{SYSTEM_PROMPT}\n\nNarrative:\n{FICTIONAL_NARRATIVE}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=combined_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )

        if response.text:
            if debug:
                print(f"  [DEBUG] Raw response from {model_name}:")
                print(f"  {response.text[:1000]}")
            try:
                parsed = json.loads(response.text)
                return parsed
            except json.JSONDecodeError as e:
                print(f"  [FAIL] Invalid JSON from {model_name}: {e}")
                print(f"  Raw response: {response.text[:500]}")
                return None
        else:
            print(f"  [FAIL] Empty response from {model_name}")
            return None

    except Exception as e:
        print(f"  [FAIL] Error with {model_name}: {e}")
        return None


def main():
    print("=" * 70)
    print("NearMiss2Crash — Gemini Smoke Test")
    print("=" * 70)

    # Load API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[FAIL] GEMINI_API_KEY not found in .env")
        sys.exit(1)

    print(f"[OK] API key loaded (ends with: ...{api_key[-8:]})")

    # Load configured model
    configured_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    print(f"[INFO] Configured model: {configured_model}")

    # Initialize client
    try:
        client = genai.Client(api_key=api_key)
        print("[OK] Google GenAI client initialized")
    except Exception as e:
        print(f"[FAIL] Failed to initialize client: {e}")
        sys.exit(1)

    # Test configured model
    print(f"\n[TEST] Testing configured model: {configured_model}")
    result = test_model(client, configured_model, debug=False)

    if result is not None:
        # Validate response
        valid, msg = validate_response(result)
        if valid:
            print(f"[OK] Response validation: PASSED")

            # Print results
            print("\n" + "=" * 70)
            print("SMOKE TEST PASSED")
            print("=" * 70)
            print(f"Model:          {configured_model}")
            print(f"Proposed Label: {result['proposed_label']}")
            print(f"Confidence:     {result['confidence']:.2f}")
            print("\nHFACS Evidence:")
            for item in result['hfacs_evidence']:
                print(f"  {item['class']}:")
                print(f"    Score:      {item['evidence_score']:.2f}")
                print(f"    Excerpts:   {item['supporting_excerpts']}")
                print(f"    Reasoning:  {item['reasoning']}")

            # Token usage (if available)
            print(f"\n[WARN]  Token usage not directly available in google-genai v2 response")
            print("   (Use response.usage_metadata if available in your version)")

            sys.exit(0)
        else:
            print(f"[FAIL] Response validation FAILED: {msg}")
    else:
        print(f"[FAIL] Configured model '{configured_model}' not accessible or failed")

    # Quick fallback: just test a few known working models
    print("\n[RETRY] Trying minimal fallback models...")
    fallback_models = [
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-2.5-flash-lite",
    ]

    for model_name in fallback_models:
        print(f"\n[TEST] Testing fallback: {model_name}")
        result = test_model(client, model_name, debug=False)

        if result is not None:
            valid, msg = validate_response(result)
            if valid:
                print(f"[OK] Fallback model '{model_name}' works!")
                print("\n" + "=" * 70)
                print("SMOKE TEST PASSED (with fallback model)")
                print("=" * 70)
                print(f"Configured model:     {configured_model} (FAILED)")
                print(f"Working model:        {model_name}")
                print(f"Proposed Label:       {result['proposed_label']}")
                print(f"Confidence:           {result['confidence']:.2f}")
                print("\nHFACS Evidence:")
                for item in result['hfacs_evidence']:
                    print(f"  {item['class']}: {item['evidence_score']:.2f} - {item['reasoning'][:80]}...")

                print(f"\n[NOTE] REQUIRED .env CHANGE:")
                print(f"   GEMINI_MODEL={model_name}")
                print(f"   (Change from: GEMINI_MODEL={configured_model})")
                sys.exit(0)
            else:
                print(f"   Validation failed: {msg}")

    # All models failed
    print("\n" + "=" * 70)
    print("SMOKE TEST FAILED — No accessible models found")
    print("=" * 70)
    print(f"Configured model: {configured_model} — not accessible")
    print(f"Tried fallbacks:  {', '.join(fallback_models)}")
    print("\nPossible issues:")
    print("  • API key may not have access to Gemini models")
    print("  • Model name format may be incorrect")
    print("  • Regional restrictions on model access")
    sys.exit(1)


if __name__ == "__main__":
    main()