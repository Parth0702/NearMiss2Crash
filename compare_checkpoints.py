import torch
import json

# Load both checkpoints
crisp = torch.load('D:\\SC project\\experiments\\distilbert_extension\\models\\model_a_crisp_head.pt', map_location='cpu')
fuzzy = torch.load('D:\\SC project\\experiments\\distilbert_extension\\models\\model_b_fuzzy_head.pt', map_location='cpu')

print('=== CRISP CHECKPOINT ===')
print('Config:', json.dumps(crisp['config'], indent=2, default=str))
print('Test metrics:', json.dumps(crisp['test_metrics'], indent=2))
print('Test samples:', crisp['test_samples'])
print('Confusion matrix shape:', len(crisp['confusion_matrix']), 'x', len(crisp['confusion_matrix'][0]))

print()
print('=== FUZZY CHECKPOINT ===')
print('Config:', json.dumps(fuzzy['config'], indent=2, default=str))
print('Test metrics:', json.dumps(fuzzy['test_metrics'], indent=2))
print('Test samples:', fuzzy['test_samples'])
print('Confusion matrix shape:', len(fuzzy['confusion_matrix']), 'x', len(fuzzy['confusion_matrix'][0]))

# Check HFACS classes match
print()
print('HFACS classes crisp:', crisp['config']['HFACS_CLASSES'])
print('HFACS classes fuzzy:', fuzzy['config']['HFACS_CLASSES'])
print('Classes match:', crisp['config']['HFACS_CLASSES'] == fuzzy['config']['HFACS_CLASSES'])

# Verify confusion matrices are 4x4
print()
print('Crisp confusion matrix:')
for row in crisp['confusion_matrix']:
    print(' ', row)
print('Fuzzy confusion matrix:')
for row in fuzzy['confusion_matrix']:
    print(' ', row)