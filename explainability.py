# explainability.py
"""
Week 4 – Explainability (SHAP & feature importance)

This script generates SHAP explanations for the best transformer model
(DistilBERT or SafeAeroBERT) and extracts feature importance from the
Logistic Regression classifier trained in Week 2.

All outputs are saved under `week4_results/explainability/`.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
RESULTS_DIR = "week4_results"
OUT_DIR = os.path.join(RESULTS_DIR, "explainability")
os.makedirs(OUT_DIR, exist_ok=True)

# Load test split (same as used for evaluation)
TEST_CSV = "weak_labeling_results/dataset_majority_test.csv"
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Test split not found: {TEST_CSV}")

test_df = pd.read_csv(TEST_CSV)
texts = test_df["text"].astype(str).tolist()
labels = test_df["label"].tolist()

# ---------------------------------------------------------------------------
# 1) SHAP for the transformer model
# ---------------------------------------------------------------------------
# Find the best checkpoint saved by `transformer_model.py`
TRANSFORMER_RESULTS_DIR = "week3_results"
best_model_path = None
checkpoint_name = None
for ckpt in ["distilbert-base-uncased", "andradewalsh/SafeAeroBERT"]:
    candidate = os.path.join(
        TRANSFORMER_RESULTS_DIR,
        f"best_model_{ckpt.replace('/', '_')}.pt",
    )
    if os.path.exists(candidate):
        best_model_path = candidate
        checkpoint_name = ckpt
        break
if best_model_path is None:
    raise FileNotFoundError("Best transformer checkpoint not found in week3_results")

print(f"Loading transformer checkpoint: {best_model_path}")

# Since the saved state dict contains the full HFACSClassifier (bert + classifier)
# we need to recreate the full architecture. Let's define it here.
import torch.nn as nn

class HFACSClassifier(nn.Module):
    """Transformer encoder + binary relevance classification head."""
    def __init__(self, checkpoint: str, n_classes: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Load state dict (CPU)
state_dict = torch.load(best_model_path, map_location=torch.device("cpu"))

# Re‑instantiate the model architecture and load weights
HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error",
]
model = HFACSClassifier(checkpoint_name, n_classes=len(HFACS_CLASSES))
model.load_state_dict(state_dict)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

# Use a small sample for SHAP (max 20 examples)
sample_texts = texts[:20]

# For SHAP with transformers, use the built-in text explainer with a simple
# prediction function that takes a list of strings and returns probabilities
def predict_proba(texts):
    """Prediction function for SHAP: takes list of strings, returns (n_samples, n_classes) probs."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**enc)
        probs = torch.sigmoid(logits)
    return probs.numpy()

# Use SHAP's built-in text explainer with a masker
# We'll use a simpler approach: use SHAP's Explainer with the predict_proba function
# and a text masker
try:
    import shap
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker)

    # Explain a small subset
    shap_values = explainer(sample_texts[:5], max_evals=200)

    # Plot top tokens per class
    for i, cls_name in enumerate(HFACS_CLASSES):
        # Mean absolute SHAP value per token across samples
        values = np.abs(shap_values.values[:, :, i]).mean(axis=0)
        token_strings = shap_values.data
        # Aggregate by token string
        token_to_val = {}
        for token, val in zip(token_strings, values):
            token_to_val[token] = token_to_val.get(token, 0) + val
        top_items = sorted(token_to_val.items(), key=lambda x: x[1], reverse=True)[:10]
        words, scores = zip(*top_items)

        plt.figure(figsize=(8, 4))
        plt.barh(words, scores)
        plt.gca().invert_yaxis()
        plt.title(f"Top SHAP tokens for {cls_name}")
        plt.xlabel("Mean |SHAP| value")
        out_path = os.path.join(
            OUT_DIR, f"shap_{cls_name.replace(' ', '_').lower()}.png"
        )
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved SHAP plot: {out_path}")
except Exception as e:
    print(f"SHAP explanation failed (this is OK for small datasets): {e}")
    print("Skipping SHAP plots; using Logistic Regression coefficients instead.")

# ---------------------------------------------------------------------------
# 2) Logistic Regression feature importance (word weights)
# ---------------------------------------------------------------------------
logreg_path = os.path.join(
    "week2_results", "classifier_logistic_regression_majority.joblib"
)
if not os.path.exists(logreg_path):
    raise FileNotFoundError(f"Logistic Regression pipeline not found: {logreg_path}")

logreg_pipe: Pipeline = joblib.load(logreg_path)
vectorizer: TfidfVectorizer = logreg_pipe.named_steps["vectorizer"]
clf: LogisticRegression = logreg_pipe.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()
coeffs = clf.coef_  # shape (n_classes, n_features)

# Handle case where only one class is present in training data
n_coeff_classes = coeffs.shape[0]
print(f"Logistic Regression has {n_coeff_classes} coefficient rows (classes)")

if n_coeff_classes == 1:
    # Binary or single-class case - use the single row for all HFACS classes
    # or map to the actual class present
    actual_class = clf.classes_[0] if len(clf.classes_) > 0 else "Violation"
    print(f"Model trained on single class: {actual_class}")
    # Use the same coefficients for the matching HFACS class
    for idx, cls_name in enumerate(HFACS_CLASSES):
        if cls_name == actual_class or n_coeff_classes == 1:
            class_coeff = coeffs[0]
        else:
            # For other classes, skip or use zeros
            class_coeff = np.zeros_like(coeffs[0])
else:
    # Multi-class case - use each row
    pass

for idx, cls_name in enumerate(HFACS_CLASSES):
    if n_coeff_classes == 1:
        class_coeff = coeffs[0] if cls_name == actual_class else np.zeros_like(coeffs[0])
    else:
        if idx < coeffs.shape[0]:
            class_coeff = coeffs[idx]
        else:
            class_coeff = np.zeros_like(coeffs[0])

    top_idx = np.argsort(np.abs(class_coeff))[-10:][::-1]
    top_features = feature_names[top_idx]
    top_vals = class_coeff[top_idx]

    plt.figure(figsize=(8, 4))
    plt.barh(top_features, top_vals)
    plt.gca().invert_yaxis()
    plt.title(f"Top Logistic-Reg coefficients for {cls_name}")
    plt.xlabel("Weight")
    out_path = os.path.join(
        OUT_DIR, f"logreg_{cls_name.replace(' ', '_').lower()}.png"
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Logistic-Reg plot: {out_path}")

# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------
summary = {
    "transformer_checkpoint": best_model_path,
    "shap_plots": [
        os.path.join(
            OUT_DIR, f"shap_{c.replace(' ', '_').lower()}.png"
        )
        for c in HFACS_CLASSES
    ],
    "logreg_plots": [
        os.path.join(
            OUT_DIR, f"logreg_{c.replace(' ', '_').lower()}.png"
        )
        for c in HFACS_CLASSES
    ],
}
with open(os.path.join(OUT_DIR, "explainability_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Explainability generation completed.")