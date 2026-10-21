"""
Generate comprehensive ASRS column audit report:
- Distinguishes narrative vs ID/date vs analyst-coded columns
- Correctly treats empty strings/whitespace as missing values
- Reports true non-empty count, true % non-null, and top 20 most frequent values for each analyst-coded column
- Writes report to asrs_columns_report.txt
"""

import sys
from datasets import load_dataset
import pandas as pd
import numpy as np

import config

def generate_report():
    print(f"[INFO] Loading HuggingFace dataset '{config.ASRS_HF_DATASET_NAME}'...")
    ds = load_dataset(config.ASRS_HF_DATASET_NAME, split="train")
    total_rows = len(ds)
    print(f"[INFO] Dataset loaded with {total_rows:,} rows and {len(ds.column_names)} columns.")
    
    df = ds.to_pandas()
    
    narrative_cols = [
        'Report 1_Narrative',
        'Report 1.1_Callback',
        'Report 1.2_Synopsis',
        'Report 2_Narrative',
        'Report 2.1_Callback'
    ]
    id_date_cols = [
        'acn_num_ACN',
        'Time_Date',
        'Time.1_Local Time Of Day'
    ]
    
    all_cols = list(df.columns)
    analyst_cols = [c for c in all_cols if c not in narrative_cols and c not in id_date_cols]
    
    # Pre-clean series: strip whitespace and treat empty strings as null
    cleaned_series = {}
    stats = {}
    for c in all_cols:
        s = df[c].fillna('').astype(str).str.strip()
        s_valid = s[s != '']
        cleaned_series[c] = s_valid
        stats[c] = {
            'count': len(s_valid),
            'pct': (len(s_valid) / total_rows) * 100,
            'unique': s_valid.nunique()
        }
        
    lines = []
    lines.append("=" * 90)
    lines.append("ASRS AVIATION REPORTS (elihoole/asrs-aviation-reports) COLUMN AUDIT REPORT")
    lines.append("=" * 90)
    lines.append(f"Total Rows: {total_rows:,}")
    lines.append(f"Total Columns: {len(all_cols)}")
    lines.append(f"Narrative Columns: {len(narrative_cols)}")
    lines.append(f"Identifier/Date Columns: {len(id_date_cols)}")
    lines.append(f"Analyst-Coded Columns: {len(analyst_cols)}")
    lines.append("")
    lines.append("Note: In the raw dataset, missing values are represented as empty strings ('').")
    lines.append("Statistics below reflect TRUE non-empty, populated values.")
    lines.append("")
    
    lines.append("=" * 90)
    lines.append("--- 1. MASTER LIST OF ALL 111 COLUMNS ---")
    lines.append("=" * 90)
    lines.append(f"{'#':>3s} | {'Category':<13s} | {'Column Name':<60s} | {'Non-Empty':>9s} | {'% Non-Empty':>11s}")
    lines.append("-" * 104)
    for i, col in enumerate(all_cols, 1):
        category = "NARRATIVE" if col in narrative_cols else ("ID/DATE" if col in id_date_cols else "ANALYST-CODED")
        c_stat = stats[col]
        lines.append(f"{i:3d} | {category:<13s} | {col:<60s} | {c_stat['count']:9,d} | {c_stat['pct']:10.2f}%")
        
    lines.append("")
    lines.append("=" * 90)
    lines.append("--- 2. DETAILED BREAKDOWN OF ALL 103 ANALYST-CODED COLUMNS ---")
    lines.append("(Ranked by % non-empty descending; includes % populated and 20 most frequent values)")
    lines.append("=" * 90)
    
    # Sort analyst columns by non-empty count descending
    analyst_cols_sorted = sorted(analyst_cols, key=lambda c: stats[c]['count'], reverse=True)
    
    for rank_idx, col in enumerate(analyst_cols_sorted, 1):
        s_valid = cleaned_series[col]
        c_stat = stats[col]
        non_empty_cnt = c_stat['count']
        pct_non_empty = c_stat['pct']
        unique_cnt = c_stat['unique']
        
        lines.append("")
        lines.append("-" * 90)
        lines.append(f"[{rank_idx:3d}/103] ANALYST COLUMN: {col}")
        lines.append(f"Populated Count: {non_empty_cnt:,} / {total_rows:,} ({pct_non_empty:.2f}%) | Unique Distinct Values: {unique_cnt:,}")
        lines.append("Top 20 Most Frequent Values:")
        
        if non_empty_cnt == 0:
            lines.append("  (Column is completely empty / 0% populated)")
            continue
            
        top20 = s_valid.value_counts().head(20)
        for v_rank, (val, cnt) in enumerate(top20.items(), 1):
            val_clean = val.replace('\n', ' ').strip()
            if len(val_clean) > 65:
                val_clean = val_clean[:62] + "..."
            pct_of_col = (cnt / non_empty_cnt) * 100
            pct_of_total = (cnt / total_rows) * 100
            lines.append(f"  {v_rank:2d}. {val_clean:<65s} | {cnt:6,d} ({pct_of_col:5.1f}% of populated, {pct_of_total:5.1f}% of total)")

    report_str = "\n".join(lines)
    
    with open(config.ASRS_COLUMNS_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_str)
        
    print(f"[SUCCESS] Wrote comprehensive audit report to {config.ASRS_COLUMNS_REPORT_TXT} ({len(report_str):,} chars)")
    return report_str

if __name__ == "__main__":
    generate_report()
