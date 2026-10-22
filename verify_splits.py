import pandas as pd

# Check fuzzy splits
for f in ['train', 'val', 'test']:
    df = pd.read_csv(f'D:/SC project/experiments/distilbert_extension/dataset_fuzzy_{f}_labeled.csv')
    print(f'{f}: {len(df)} rows, cols={list(df.columns)}')
    print(f'  Labels: {df["fuzzy_label"].value_counts().to_dict()}')
print()

# Check crisp splits
for f in ['train', 'val', 'test']:
    df = pd.read_csv(f'D:/SC project/weak_labeling_results/dataset_majority_{f}.csv')
    print(f'{f}: {len(df)} rows, cols={list(df.columns)}')
    print(f'  Labels: {df["label"].value_counts().to_dict()}')