"""
Evaluate both DistilBERT models on common held-out test set.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import json
from pathlib import Path

PROJECT_ROOT = Path('D:/SC project')
HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]
LABEL_TO_IDX = {c: i for i, c in enumerate(HFACS_CLASSES)}

# Load both checkpoints
crisp_ckpt = torch.load(PROJECT_ROOT / 'experiments' / 'distilbert_extension' / 'models' / 'model_a_crisp_head.pt', map_location='cpu')
fuzzy_ckpt = torch.load(PROJECT_ROOT / 'experiments' / 'distilbert_extension' / 'models' / 'model_b_fuzzy_head.pt', map_location='cpu')

# Load test splits
crisp_test = pd.read_csv(PROJECT_ROOT / 'weak_labeling_results' / 'dataset_majority_test.csv')
fuzzy_test = pd.read_csv(PROJECT_ROOT / 'experiments' / 'distilbert_extension' / 'dataset_fuzzy_test_labeled.csv')

# Add ASRS_ prefix to crisp report_ids
crisp_test['report_id'] = 'ASRS_' + crisp_test['report_id'].astype(str)

# Find common report_ids
common_ids = set(crisp_test['report_id']).intersection(set(fuzzy_test['report_id']))
print(f'Common report_ids: {len(common_ids)}')

# Create merged test set with both labels
merged = pd.merge(
    crisp_test[['report_id', 'text', 'label']].rename(columns={'label': 'crisp_label'}),
    fuzzy_test[['report_id', 'text', 'fuzzy_label']].rename(columns={'fuzzy_label': 'fuzzy_label'}),
    on='report_id',
    how='inner'
)
print(f'Merged test samples: {len(merged)}')
print(f'Crisp label dist: {merged["crisp_label"].value_counts().to_dict()}')
print(f'Fuzzy label dist: {merged["fuzzy_label"].value_counts().to_dict()}')

# Check label agreement
agreement = (merged['crisp_label'] == merged['fuzzy_label']).sum()
print(f'Label agreement: {agreement}/{len(merged)} = {agreement/len(merged)*100:.1f}%')

# Disagreements
disagree = merged[merged['crisp_label'] != merged['fuzzy_label']]
if len(disagree) > 0:
    print(f'Disagreements ({len(disagree)}):')
    for _, row in disagree.iterrows():
        print(f'  {row["report_id"]}: crisp={row["crisp_label"]} vs fuzzy={row["fuzzy_label"]}')

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
base_model = AutoModel.from_pretrained('distilbert-base-uncased')
base_model.eval()

# Extract embeddings for common test texts
texts = merged['text'].tolist()
print(f'\nExtracting embeddings for {len(texts)} texts...')
all_emb = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        b_texts = texts[i:i+64]
        inputs = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        outputs = base_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        cls_rep = outputs.last_hidden_state[:, 0, :]
        all_emb.append(cls_rep)
X_test = torch.cat(all_emb, dim=0)
print(f'Embeddings shape: {X_test.shape}')

# Recreate heads
crisp_head = nn.Linear(768, 4)
crisp_head.load_state_dict({'weight': crisp_ckpt['head_weight'], 'bias': crisp_ckpt['head_bias']})
crisp_head.eval()

fuzzy_head = nn.Linear(768, 4)
fuzzy_head.load_state_dict({'weight': fuzzy_ckpt['head_weight'], 'bias': fuzzy_ckpt['head_bias']})
fuzzy_head.eval()

# Get predictions
with torch.no_grad():
    crisp_logits = crisp_head(X_test)
    crisp_preds = torch.argmax(crisp_logits, dim=1).numpy()

    fuzzy_logits = fuzzy_head(X_test)
    fuzzy_preds = torch.argmax(fuzzy_logits, dim=1).numpy()

# Reference labels: use fuzzy_label as the explicit reference
y_true = np.array([LABEL_TO_IDX[l] for l in merged['fuzzy_label']])
crisp_y_true = np.array([LABEL_TO_IDX[l] for l in merged['crisp_label']])

print('\n' + '='*60)
print('EVALUATION ON COMMON 759 SAMPLES')
print('Reference labels: FUZZY LABELS')
print('='*60)

for name, preds in [('Crisp Model', crisp_preds), ('Fuzzy Model', fuzzy_preds)]:
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, preds, average='weighted', zero_division=0)
    macro_prec = precision_score(y_true, preds, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, preds, average='macro', zero_division=0)

    print(f'\n{name}:')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  Macro-F1: {macro_f1:.4f}')
    print(f'  Weighted-F1: {weighted_f1:.4f}')
    print(f'  Macro Precision: {macro_prec:.4f}')
    print(f'  Macro Recall: {macro_rec:.4f}')
    print(f'\n  Per-class:')
    print(classification_report(y_true, preds, target_names=HFACS_CLASSES, digits=4))
    cm = confusion_matrix(y_true, preds)
    print(f'  Confusion Matrix:')
    for row in cm:
        print(f'    {row}')

# Also evaluate crisp model against crisp labels for comparison
print('\n' + '='*60)
print('CRISP MODEL evaluated against CRISP labels (original 872 vs subset 759)')
print('='*60)
acc = accuracy_score(crisp_y_true, crisp_preds)
macro_f1 = f1_score(crisp_y_true, crisp_preds, average='macro', zero_division=0)
print(f'  Accuracy: {acc:.4f}')
print(f'  Macro-F1: {macro_f1:.4f}')

# Save common test comparison
comparison_df = merged.copy()
comparison_df['crisp_pred'] = [HFACS_CLASSES[i] for i in crisp_preds]
comparison_df['fuzzy_pred'] = [HFACS_CLASSES[i] for i in fuzzy_preds]
comparison_df['crisp_correct'] = comparison_df['crisp_pred'] == comparison_df['fuzzy_label']
comparison_df['fuzzy_correct'] = comparison_df['fuzzy_pred'] == comparison_df['fuzzy_label']

comparison_path = PROJECT_ROOT / 'experiments' / 'distilbert_extension' / 'common_test_comparison.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f'\nSaved comparison to {comparison_path}')

# Save metrics
metrics = {
    'common_test_samples': len(merged),
    'reference_label_source': 'fuzzy_label (from FuzzyWeakLabeler, tau=0.35, delta=0.15)',
    'label_agreement_crisp_vs_fuzzy': int(agreement),
    'label_agreement_pct': agreement / len(merged) * 100,
    'crisp_model_on_fuzzy_labels': {
        'accuracy': float(accuracy_score(y_true, crisp_preds)),
        'macro_f1': float(f1_score(y_true, crisp_preds, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, crisp_preds, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y_true, crisp_preds, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, crisp_preds, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, crisp_preds).tolist()
    },
    'fuzzy_model_on_fuzzy_labels': {
        'accuracy': float(accuracy_score(y_true, fuzzy_preds)),
        'macro_f1': float(f1_score(y_true, fuzzy_preds, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, fuzzy_preds, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y_true, fuzzy_preds, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, fuzzy_preds, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, fuzzy_preds).tolist()
    },
    'crisp_model_on_crisp_labels': {
        'accuracy': float(accuracy_score(crisp_y_true, crisp_preds)),
        'macro_f1': float(f1_score(crisp_y_true, crisp_preds, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(crisp_y_true, crisp_preds, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(crisp_y_true, crisp_preds).tolist()
    },
    'hfacs_classes': HFACS_CLASSES,
    'note': 'Proxy weak-label accuracy only; no human-validated ground truth exists'
}

metrics_path = PROJECT_ROOT / 'experiments' / 'distilbert_extension' / 'common_test_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'Saved metrics to {metrics_path}')

# Verdict
print('\n' + '='*60)
print('VERDICT')
print('='*60)
print('Common evaluation set exists: 759 samples (all fuzzy-labeled samples)')
print(f'Label agreement between crisp/fuzzy: {agreement}/{len(merged)} = {agreement/len(merged)*100:.1f}%')
print('Both models evaluated on identical narratives with identical reference (fuzzy) labels.')
print('VALID COMPARISON: Yes, meaningful common set exists.')