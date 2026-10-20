import os
import pandas as pd
from weak_labeling_framework import UnifiedWeakLabelingFramework, CONFIG

framework = UnifiedWeakLabelingFramework(config=CONFIG)
asrs_path = 'asrs_full.csv' if os.path.exists('asrs_full.csv') else 'asrs_data.csv'
df = framework.load_data(asrs_path)
keyword_df, keyword_audit = framework.run_keyword_labeling(df)
cluster_df = framework.run_cluster_labeling(df)

for method in ['majority', 'keyword_first', 'cluster_first']:
    framework.combine_labels(keyword_df, cluster_df, method=method)
    print(f"Generated combined_labels_{method}.csv")
