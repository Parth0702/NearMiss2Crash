import pandas as pd

# Check crisp splits for Unlabeled counts
for f in ['train', 'validation', 'test']:
    path = f'D:\\SC project\\weak_labeling_results\\dataset_majority_{f}.csv'
    df = pd.read_csv(path)
    total = len(df)
    labeled = df[df['label'] != 'Unlabeled']
    unlabeled = total - len(labeled)
    print(f'{f}: total={total}, labeled={len(labeled)}, unlabeled={unlabeled}')
    print(f'  Labels: {df["label"].value_counts().to_dict()}')