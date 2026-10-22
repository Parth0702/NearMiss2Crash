import torch
import json

# Load both checkpoints
crisp = torch.load('D:/SC project/experiments/distilbert_extension/models/model_a_crisp_head.pt', map_location='cpu')
fuzzy = torch.load('D:/SC project/experiments/distilbert_extension/models/model_b_fuzzy_head.pt', map_location='cpu')

# Load saved metrics
with open('D:/SC project/experiments/distilbert_extension/distilbert_comparison_metrics.json', 'r') as f:
    metrics = json.load(f)

print('=== CRISP ===')
print('Test metrics:', json.dumps(metrics['model_a_crisp']['test_metrics'], indent=2))
print('Test samples:', metrics['model_a_crisp']['test_samples'])
print('Confusion matrix shape:', len(metrics['model_a_crisp']['confusion_matrix']), 'x', len(metrics['model_a_crisp']['confusion_matrix'][0]))
print('Classes:', metrics['config']['HFACS_CLASSES'])

print()
print('=== FUZZY ===')
print('Test metrics:', json.dumps(metrics['model_b_fuzzy']['test_metrics'], indent=2))
print('Test samples:', metrics['model_b_fuzzy']['test_samples'])
print('Confusion matrix shape:', len(metrics['model_b_fuzzy']['confusion_matrix']), 'x', len(metrics['model_b_fuzzy']['confusion_matrix'][0]))
print('Classes:', metrics['config']['HFACS_CLASSES'])

print()
print('HFACS classes match:', metrics['config']['HFACS_CLASSES'] == metrics['config']['HFACS_CLASSES'])

# The key issue: different test samples (872 vs 759), different evaluation populations
# Need to find common held-out subset

# Let's examine the actual report IDs in the split files
import pandas as pd

# Load crisp test split
crisp_test = pd.read_csv('D:/SC project/weak_labeling_results/dataset_majority_test.csv')
print()
print('=== CRISP TEST SPLIT ===')
print('Total rows:', len(crisp_test))
print('Label distribution:', crisp_test['label'].value_counts().to_dict())

# Load fuzzy test split (labeled only)
fuzzy_test = pd.read_csv('D:/SC project/experiments/distilbert_extension/dataset_fuzzy_test_labeled.csv')
print()
print('=== FUZZY TEST SPLIT ===')
print('Total rows:', len(fuzzy_test))
print('Label distribution:', fuzzy_test['fuzzy_label'].value_counts().to_dict())

# Check report_ids - crisp has ASRS_ prefix now
crisp_test['report_id'] = 'ASRS_' + crisp_test['report_id'].astype(str)
print()
print('Crisp report_id sample:', crisp_test['report_id'].iloc[0])
print('Fuzzy report_id sample:', fuzzy_test['report_id'].iloc[0])

# Find common report_ids
common_ids = set(crisp_test['report_id']).intersection(set(fuzzy_test['report_id']))
print(f'Common report_ids: {len(common_ids)}')
print(f'Crisp only: {len(set(crisp_test[\"report_id\"]) - set(fuzzy_test[\"report_id\"]))}')
print(f'Fuzzy only: {len(set(fuzzy_test[\"report_id\"]) - set(crisp_test[\"report_id\"]))}')