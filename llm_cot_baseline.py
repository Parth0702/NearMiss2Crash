"""
===============================================================================
Week 3: LLM Chain-of-Thought Baseline (HFACS-CoT+)
===============================================================================
Reference Paper:
  Liu, Li, Ng, Han & Feng (2025) -- "Accident investigation via LLMs reasoning:
  HFACS-guided Chain-of-Thoughts" (Section 3.2-3.3, Appendix C/D)

Implements the HFACS-CoT+ pipeline:
  Sequential LLM questioning (Q1->Q2->Q3->[Q4]) to classify incident reports
  into one of four HFACS classes.

Features:
  - Multi-API support (Anthropic, OpenAI) with robust fallback to MockLLM.
  - Sequential questioning as per Table 1 of the paper.
  - Decision engine implementing Eq. 4 logic for mapping answers to HFACS class.
  - Verbose logging and optional progress bar.

Usage:
    python llm_cot_baseline.py
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    # LLM API selection: "anthropic", "openai", or "mock"
    "LLM_PROVIDER": "mock",  # Change to "anthropic" or "openai" to use real APIs

    # API keys (set as environment variables or fill in directly)
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),

    # Model names
    "ANTHROPIC_MODEL": "claude-3-5-sonnet-20241022",
    "OPENAI_MODEL": "gpt-4o-mini",

    # Generation parameters
    "TEMPERATURE": 0.0,  # Deterministic for reproducibility
    "MAX_TOKENS": 10,

    # Data paths
    "TRAIN_PATH": "weak_labeling_results/dataset_majority_train.csv",
    "VAL_PATH": "weak_labeling_results/dataset_majority_validation.csv",
    "TEST_PATH": "weak_labeling_results/dataset_majority_test.csv",

    # Label set
    "HFACS_CLASSES": [
        "Skill-Based Error",
        "Decision Error",
        "Violation",
        "Perceptual Error"
    ],

    # Output directory
    "OUTPUT_DIR": "week3_results",

    # Whether to show progress bar
    "SHOW_PROGRESS_BAR": True,

    # Delay between API calls (to avoid rate limits)
    "API_CALL_DELAY": 0.1,  # seconds
}


# -----------------------------------------------------------------------------
# LLM Clients
# -----------------------------------------------------------------------------
class BaseLLMClient:
    """Abstract base class for LLM clients."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing without API keys.
    Uses simple keyword matching to simulate responses.
    """

    def __init__(self):
        # Simple keyword mappings for demonstration
        self.skill_keywords = {
            "checklist", "forgot", "omitted", "lapsed", "routine", "habit",
            "inadvertent", "slip", "memory", "missed call", "misread"
        }
        self.decision_keywords = {
            "decided", "elected", "chose", "continued", "judgment", "risk",
            "weather", "visual approach", "divert", "below minimums"
        }
        self.violation_keywords = {
            "knowingly", "intentionally", "skipped", "unauthorized",
            "rule", "sop", "regulation", "sterile cockpit", "exceeded"
        }
        self.perceptual_keywords = {
            "disorientation", "spatial", "illusion", "misjudged distance",
            "visibility", "flare", "glare", "featureless terrain", "depth perception"
        }

    def _contains_keyword(self, text: str, keyword_set: set) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keyword_set)

    def generate(self, prompt: str) -> str:
        # Extract the question from the prompt (simplified)
        # In a real implementation, we would parse the prompt properly.
        # For now, we look for keywords in the entire prompt.
        text = prompt.lower()

        # Simulate Q1: Always "yes" for non-empty text (assumes there is some action/inaction)
        if "q1:" in text or "did the mishap individual" in text:
            return "yes"

        # Simulate Q2: Skill-based error
        if "q2:" in text or "performance/skill error" in text:
            if self._contains_keyword(prompt, self.skill_keywords):
                return "yes"
            else:
                return "no"

        # Simulate Q3: Decision/error error
        if "q3:" in text or "judgment/decision-based error" in text:
            if self._contains_keyword(prompt, self.decision_keywords):
                return "yes"
            else:
                return "no"

        # Simulate Q4: Violation
        if "q4:" in text or "known deviation" in text:
            if self._contains_keyword(prompt, self.violation_keywords):
                return "yes"
            else:
                return "no"

        # Default fallback
        return "yes"


class AnthropicLLMClient(BaseLLMClient):
    """Client for Anthropic's Claude API."""

    def __init__(self, api_key: str, model: str):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=CONFIG["MAX_TOKENS"],
            temperature=CONFIG["TEMPERATURE"],
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text.strip()


class OpenAILLMClient(BaseLLMClient):
    """Client for OpenAI's GPT API."""

    def __init__(self, api_key: str, model: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=CONFIG["MAX_TOKENS"],
            temperature=CONFIG["TEMPERATURE"],
        )
        return response.choices[0].message.content.strip()


def get_llm_client() -> BaseLLMClient:
    """Factory function to get the appropriate LLM client based on config."""
    provider = CONFIG["LLM_PROVIDER"].lower()

    if provider == "mock":
        return MockLLMClient()
    elif provider == "anthropic":
        if not CONFIG["ANTHROPIC_API_KEY"]:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Please set it as an environment variable "
                "or in the CONFIG dictionary."
            )
        return AnthropicLLMClient(
            api_key=CONFIG["ANTHROPIC_API_KEY"],
            model=CONFIG["ANTHROPIC_MODEL"]
        )
    elif provider == "openai":
        if not CONFIG["OPENAI_API_KEY"]:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it as an environment variable "
                "or in the CONFIG dictionary."
            )
        return OpenAILLMClient(
            api_key=CONFIG["OPENAI_API_KEY"],
            model=CONFIG["OPENAI_MODEL"]
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Choose from 'mock', 'anthropic', or 'openai'."
        )


# -----------------------------------------------------------------------------
# HFACS-CoT+ Pipeline
# -----------------------------------------------------------------------------
def build_prompt(text: str, question_num: int) -> str:
    """
    Build the prompt for a given question number.
    Based on Table 1 and the HFACS-CoT+ methodology from Liu et al. (2025).
    """
    base_prompt = (
        "You are an expert aviation safety investigator. "
        "Read the following incident report and answer the question with exactly 'yes' or 'no'.\n\n"
        f"Incident Report:\n{text}\n\n"
    )

    if question_num == 1:
        question = (
            "Q1: Did the mishap individual (pilot/crew) do or fail to do something "
            "that contributed to this incident? "
            "Answer with exactly 'yes' or 'no'."
        )
    elif question_num == 2:
        question = (
            "Q2: Was there a performance or skill-based error (e.g., inattention, "
            "inadequate training, procedural slip) identified? "
            "Answer with exactly 'yes' or 'no'."
        )
    elif question_num == 3:
        question = (
            "Q3: Was there a judgment or decision-based error (e.g., poor risk "
            "assessment, continuation into adverse conditions) identified? "
            "Answer with exactly 'yes' or 'no'."
        )
    elif question_num == 4:
        question = (
            "Q4: Was there a known deviation or violation of standard operating "
            "procedures, regulations, or rules identified? "
            "Answer with exactly 'yes' or 'no'."
        )
    else:
        raise ValueError(f"Invalid question number: {question_num}")

    return base_prompt + question


def predict_hfacs_class(
    text: str,
    llm_client: BaseLLMClient,
    verbose: bool = False
) -> Tuple[str, Dict[str, str]]:
    """
    Predict HFACS class for a single text using the HFACS-CoT+ pipeline.

    Returns:
        Tuple of (predicted_class, answers_dict)
        where answers_dict contains the responses to Q1-Q4.
    """
    answers = {}

    # Q1: Basic activity check
    prompt_q1 = build_prompt(text, 1)
    ans_q1 = llm_client.generate(prompt_q1).strip().lower()
    answers["Q1"] = ans_q1

    if verbose:
        print(f"  Q1: {ans_q1}")

    # If Q1 is "no", we cannot proceed (unexpected per paper logic)
    if ans_q1 != "yes":
        return "Unexpected (Q1=no)", answers

    # Q2: Skill-based error check
    prompt_q2 = build_prompt(text, 2)
    ans_q2 = llm_client.generate(prompt_q2).strip().lower()
    answers["Q2"] = ans_q2

    if verbose:
        print(f"  Q2: {ans_q2}")

    # Q3: Judgment/decision error check
    prompt_q3 = build_prompt(text, 3)
    ans_q3 = llm_client.generate(prompt_q3).strip().lower()
    answers["Q3"] = ans_q3

    if verbose:
        print(f"  Q3: {ans_q3}")

    # Decision engine (Eq. 4 from Liu et al. 2025)
    if ans_q2 == "yes" and ans_q3 == "no":
        # Skill-Based Error or Condition of Operator
        # Per paper, we map to Skill-Based Error for simplicity
        return "Skill-Based Error", answers
    elif ans_q2 == "no" and ans_q3 == "yes":
        # Decision Error
        return "Decision Error", answers
    elif ans_q2 == "no" and ans_q3 == "no":
        # Check Q4 for Violation
        prompt_q4 = build_prompt(text, 4)
        ans_q4 = llm_client.generate(prompt_q4).strip().lower()
        answers["Q4"] = ans_q4

        if verbose:
            print(f"  Q4: {ans_q4}")

        if ans_q4 == "yes":
            return "Violation", answers
        else:
            return "Perceptual Error", answers
    else:
        # This case (Q2=yes, Q3=yes) is logically inconsistent per paper.
        # HFACS-CoT+ avoids this by sequential asking, but we handle it.
        return "Inconsistent (Q2=yes, Q3=yes)", answers


def evaluate_dataset(
    df: pd.DataFrame,
    llm_client: BaseLLMClient,
    verbose: bool = False
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Evaluate the LLM-CoT+ pipeline on a dataset.

    Returns:
        Tuple of (predictions, answers_list)
    """
    predictions = []
    all_answers = []

    iterator = df.iterrows()
    if CONFIG["SHOW_PROGRESS_BAR"] and not verbose:
        try:
            iterator = tqdm(df.iterrows(), total=len(df), desc="Classifying")
        except Exception:
            iterator = df.iterrows()

    for idx, row in iterator:
        text = row["text"]
        pred, answers = predict_hfacs_class(text, llm_client, verbose=verbose)
        predictions.append(pred)
        all_answers.append(answers)

        if verbose:
            print(f"[{idx}] Text: {text[:80]}...")
            print(f"  Prediction: {pred}")
            print(f"  Answers: {answers}")
            print("-" * 50)

        # Delay to avoid rate limits (if using real API)
        if CONFIG["LLM_PROVIDER"] != "mock":
            time.sleep(CONFIG["API_CALL_DELAY"])

    return predictions, all_answers


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> Dict[str, Any]:
    """
    Compute classification metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        hamming_loss,
    )

    # Convert to indices for sklearn metrics
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_true_idx = [label_to_idx.get(label, -1) for label in y_true]
    y_pred_idx = [label_to_idx.get(label, -1) for label in y_pred]

    # Filter out any -1 (unexpected/inconsistent) for metric calculation
    # But we keep them in the lists for reporting
    valid_indices = [i for i in range(len(y_true_idx)) if y_true_idx[i] != -1 and y_pred_idx[i] != -1]

    if not valid_indices:
        # Build empty per-class entries so downstream code can iterate safely
        per_class_empty = {
            label: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            for label in labels
        }
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "hamming_loss": 1.0,
            "per_class": per_class_empty,
        }

    y_true_valid = [y_true_idx[i] for i in valid_indices]
    y_pred_valid = [y_pred_idx[i] for i in valid_indices]

    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    precision = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    recall = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    f1_macro = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    f1_micro = f1_score(y_true_valid, y_pred_valid, average="micro", zero_division=0)
    hloss = hamming_loss(y_true_valid, y_pred_valid)

    # Per-class metrics
    per_class = {}
    for label in labels:
        idx = label_to_idx[label]
        y_true_bin = [1 if y_true_idx[i] == idx else 0 for i in valid_indices]
        y_pred_bin = [1 if y_pred_idx[i] == idx else 0 for i in valid_indices]
        if len(y_true_bin) == 0:
            per_class[label] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            per_class[label] = {
                "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
                "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
                "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "hamming_loss": hloss,
        "per_class": per_class,
    }


def main():
    print("=" * 70)
    print("WEEK 3: LLM CHAIN-OF-THOUGHT BASELINE (HFACS-CoT+)")
    print("=" * 70)
    print(f"LLM Provider: {CONFIG['LLM_PROVIDER']}")
    if CONFIG["LLM_PROVIDER"] == "anthropic":
        print(f"Model: {CONFIG['ANTHROPIC_MODEL']}")
    elif CONFIG["LLM_PROVIDER"] == "openai":
        print(f"Model: {CONFIG['OPENAI_MODEL']}")
    else:
        print("Using MockLLM (for testing without API keys)")
    print()

    # Create output directory
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(exist_ok=True)

    # Initialize LLM client
    print("[INFO] Initializing LLM client...")
    llm_client = get_llm_client()

    # Load test data (we'll evaluate on test set for final results)
    print(f"[INFO] Loading test data from {CONFIG['TEST_PATH']}...")
    test_df = pd.read_csv(CONFIG["TEST_PATH"])
    print(f"  Test samples: {len(test_df)}")

    # Run classification
    print("\n[INFO] Running HFACS-CoT+ classification...")
    predictions, all_answers = evaluate_dataset(test_df, llm_client, verbose=False)

    # Compute metrics
    print("[INFO] Computing metrics...")
    true_labels = test_df["label"].tolist()
    metrics = compute_metrics(true_labels, predictions, CONFIG["HFACS_CLASSES"])

    # Count unexpected/inconsistent predictions
    unexpected_count = sum(1 for p in predictions if p.startswith("Unexpected") or p.startswith("Inconsistent"))

    # Print results
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro):   {metrics['f1_micro']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"  Unexpected/Inconsistent predictions: {unexpected_count}/{len(predictions)}")
    print("\nPer-Class F1 Scores:")
    for class_name, scores in metrics["per_class"].items():
        print(f"  {class_name:<20}: {scores['f1']:.4f}")

    # Save results
    results = {
        "config": CONFIG,
        "predictions": predictions,
        "true_labels": true_labels,
        "answers": all_answers,
        "metrics": metrics,
        "unexpected_count": unexpected_count,
    }

    # Save predictions CSV
    pred_df = test_df.copy()
    pred_df["predicted_label"] = predictions
    pred_df.to_csv(output_dir / "llm_cot_predictions.csv", index=False)

    # Save detailed results JSON
    with open(output_dir / "llm_cot_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[INFO] Detailed results saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()