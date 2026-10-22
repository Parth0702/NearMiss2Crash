"""
===============================================================================
Week 3: Unified Model Comparison
===============================================================================
Aggregates results from all model families (Week 1-3) into a master comparison table.

Models Included:
  Week 1-2 Baselines:
    - Logistic Regression
    - SVM (LinearSVC)
    - Random Forest
    - Naive Bayes

  Week 3 Advanced:
    - DistilBERT (transformer_model.py)
    - SafeAeroBERT (transformer_model.py with domain checkpoint)
    - LLM-CoT+ (llm_cot_baseline.py)

Outputs:
  - week3_results/model_comparison.csv
  - week3_results/model_comparison_visualization.png (optional)
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    # Directories where results are stored
    "WEEK2_RESULTS_DIR": "week2_results",
    "WEEK3_RESULTS_DIR": "week3_results",

    # Output directory for comparison
    "OUTPUT_DIR": "week3_results",

    # Model families to compare
    "MODEL_FAMILIES": [
        # Week 2 classical baselines
        {
            "name": "Logistic Regression",
            "week": 2,
            "type": "classical",
            "results_file": "classifier_logistic_regression_report.txt",
            # We'll extract metrics from the report or from saved JSON if available
        },
        {
            "name": "Random Forest",
            "week": 2,
            "type": "classical",
            "results_file": "classifier_random_forest_report.txt",
        },
        {
            "name": "SVM",
            "week": 2,
            "type": "classical",
            "results_file": "classifier_svm_report.txt",
        },
        {
            "name": "Naive Bayes",
            "week": 2,
            "type": "classical",
            "results_file": "classifier_naive_bayes_report.txt",
        },
        # Week 3 transformers
        {
            "name": "DistilBERT",
            "week": 3,
            "type": "transformer",
            "results_file": "transformer_results.json",
            "checkpoint": "distilbert-base-uncased",
        },
        {
            "name": "SafeAeroBERT",
            "week": 3,
            "type": "transformer",
            "results_file": "transformer_results.json",
            "checkpoint": "andradewalsh/SafeAeroBERT",  # or local path
        },
        # Week 3 LLM
        {
            "name": "LLM-CoT+",
            "week": 3,
            "type": "llm",
            "results_file": "llm_cot_results.json",
        },
    ],

    # Domains for comparison (in-domain vs cross-domain)
    "DOMAINS": ["ASRS", "NTSB"],  # We'll simulate or note if cross-domain results missing

    # HFACS classes
    "HFACS_CLASSES": [
        "Skill-Based Error",
        "Decision Error",
        "Violation",
        "Perceptual Error"
    ],
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def extract_metrics_from_classical_report(report_path: str) -> Optional[Dict[str, float]]:
    """
    Parse metrics from a classical model report (txt file).
    Expected format: lines like "Accuracy: 0.XXXX", "F1 Score (macro): 0.XXXX", etc.
    """
    if not os.path.exists(report_path):
        return None

    metrics = {}
    try:
        with open(report_path, 'r') as f:
            content = f.read()

        # Extract accuracy
        if "Accuracy:" in content:
            for line in content.split('\n'):
                if line.strip().startswith("Accuracy:"):
                    val = line.split(":")[1].strip()
                    metrics["accuracy"] = float(val)
                    break

        # Extract F1 macro
        if "F1 Score (macro):" in content:
            for line in content.split('\n'):
                if line.strip().startswith("F1 Score (macro):"):
                    val = line.split(":")[1].strip()
                    metrics["f1_macro"] = float(val)
                    break

        # Extract precision macro
        if "Precision (macro):" in content:
            for line in content.split('\n'):
                if line.strip().startswith("Precision (macro):"):
                    val = line.split(":")[1].strip()
                    metrics["precision"] = float(val)
                    break

        # Extract recall macro
        if "Recall (macro):" in content:
            for line in content.split('\n'):
                if line.strip().startswith("Recall (macro):"):
                    val = line.split(":")[1].strip()
                    metrics["recall"] = float(val)
                    break

        # Extract hamming loss / loss
        if "Hamming Loss:" in content:
            for line in content.split('\n'):
                if line.strip().startswith("Hamming Loss:"):
                    val = line.split(":")[1].strip()
                    metrics["loss"] = float(val)
                    break

        # If we have at least accuracy and f1, return
        if "accuracy" in metrics and "f1_macro" in metrics:
            return metrics
    except Exception as e:
        print(f"[WARNING] Could not parse {report_path}: {e}")

    return None


def load_json_results(json_path: str) -> Optional[Dict[str, float]]:
    """
    Load metrics from a JSON results file.
    """
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract test metrics
        if "test_metrics" in data:
            metrics = data["test_metrics"]
        elif "metrics" in data:
            metrics = data["metrics"]
        else:
            # Try to find metrics directly
            metrics = data

        # Standardize keys
        standardized = {}
        if "accuracy" in metrics:
            standardized["accuracy"] = float(metrics["accuracy"])
        if "f1_macro" in metrics:
            standardized["f1_macro"] = float(metrics["f1_macro"])
        if "precision" in metrics:
            standardized["precision"] = float(metrics["precision"])
        if "recall" in metrics:
            standardized["recall"] = float(metrics["recall"])
        if "loss" in metrics:
            standardized["loss"] = float(metrics["loss"])
        elif "hamming_loss" in metrics:
            standardized["loss"] = float(metrics["hamming_loss"])

        return standardized if standardized else None
    except Exception as e:
        print(f"[WARNING] Could not parse JSON {json_path}: {e}")
        return None


def get_model_results(model_info: Dict) -> Optional[Dict[str, float]]:
    """
    Get results for a given model from its results file.
    """
    week = model_info["week"]
    results_file = model_info["results_file"]

    # Determine directory
    if week == 2:
        base_dir = CONFIG["WEEK2_RESULTS_DIR"]
    elif week == 3:
        base_dir = CONFIG["WEEK3_RESULTS_DIR"]
    else:
        return None

    results_path = os.path.join(base_dir, results_file)

    # For transformer models, we might have different checkpoints
    # For now, we assume the results file contains the latest run
    # In a more sophisticated version, we'd filter by checkpoint

    if results_file.endswith(".txt"):
        return extract_metrics_from_classical_report(results_path)
    elif results_file.endswith(".json"):
        return load_json_results(results_path)
    else:
        return None


def main():
    print("=" * 70)
    print("WEEK 3: UNIFIED MODEL COMPARISON")
    print("=" * 70)

    # Create output directory
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(exist_ok=True)

    # Collect results for each model family
    comparison_rows = []

    for model_info in CONFIG["MODEL_FAMILIES"]:
        model_name = model_info["name"]
        print(f"\n[INFO] Processing {model_name}...")

        results = get_model_results(model_info)

        if results is None:
            print(f"  [WARNING] No results found for {model_name}. Skipping.")
            # Still add a row with NaN/zeros to keep the table complete
            results = {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "loss": 1.0,
            }

        # Check if real NTSB cross-domain evaluation results exist
        ntsb_eval_path = os.path.join(CONFIG["WEEK3_RESULTS_DIR"], "ntsb_cross_domain_evaluation.csv")
        ntsb_dict = {}
        if os.path.exists(ntsb_eval_path):
            try:
                ntsb_df = pd.read_csv(ntsb_eval_path)
                for _, r in ntsb_df.iterrows():
                    key = r['Model'].lower().replace('_', ' ').replace('-', ' ').strip()
                    ntsb_dict[key] = r.to_dict()
            except Exception:
                pass

        for domain in CONFIG["DOMAINS"]:
            if domain == "NTSB":
                mkey = model_name.lower().replace('_', ' ').replace('-', ' ').strip()
                if mkey in ntsb_dict:
                    nr = ntsb_dict[mkey]
                    row = {
                        "Model": model_name,
                        "Domain": "NTSB",
                        "Micro-F1": nr.get("Micro-F1", 0.0),
                        "Macro-F1": nr.get("Macro-F1", 0.0),
                        "Precision": nr.get("Precision", 0.0),
                        "Recall": nr.get("Recall", 0.0),
                        "Accuracy": nr.get("Accuracy", 0.0),
                        "Loss": nr.get("Loss", 0.0),
                    }
                else:
                    row = {
                        "Model": model_name,
                        "Domain": "NTSB",
                        "Micro-F1": 0.0,
                        "Macro-F1": 0.0,
                        "Precision": 0.0,
                        "Recall": 0.0,
                        "Accuracy": 0.0,
                        "Loss": 1.0,
                    }
            else:
                row = {
                    "Model": model_name,
                    "Domain": domain,
                    "Micro-F1": results.get("f1_micro", results.get("f1_macro", 0.0)),
                    "Macro-F1": results.get("f1_macro", 0.0),
                    "Precision": results.get("precision", 0.0),
                    "Recall": results.get("recall", 0.0),
                    "Accuracy": results.get("accuracy", 0.0),
                    "Loss": results.get("loss", 0.0),
                }
            comparison_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_rows)

    # Reorder columns for clarity
    column_order = ["Model", "Domain", "Accuracy", "Micro-F1", "Macro-F1", "Precision", "Recall", "Loss"]
    df = df[column_order]

    # Save to CSV
    output_path = output_dir / "model_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Comparison table saved to: {output_path}")

    # Print the table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False, float_format="%.4f"))

    # Generate simple visualization (bar chart)
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Prepare data for plotting: Macro-F1 by Model and Domain
        fig, ax = plt.subplots(figsize=(12, 6))

        # Pivot for grouped bar chart
        pivot_df = df.pivot(index="Model", columns="Domain", values="Macro-F1")

        # Plot
        pivot_df.plot(kind="bar", ax=ax)
        ax.set_title("Model Performance Comparison (Macro-F1)")
        ax.set_ylabel("Macro-F1 Score")
        ax.set_xlabel("Model")
        ax.legend(title="Domain")
        ax.tick_params(axis='x', rotation=45)

        # Tight layout and save
        plt.tight_layout()
        viz_path = output_dir / "model_comparison_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[INFO] Visualization saved to: {viz_path}")
    except ImportError:
        print("\n[INFO] matplotlib not installed, skipping visualization.")
    except Exception as e:
        print(f"\n[INFO] Could not generate visualization: {e}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()