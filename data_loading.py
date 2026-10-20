"""
===============================================================================
Data Loading & Schema Standardization Module (Week 1, Day 1-2)
===============================================================================
Reference Paper:
  Liu, Li, Ng, Han & Feng (2025) -- "Accident investigation via LLMs reasoning:
  HFACS-guided Chain-of-Thoughts" (Section 4, GAHFACS dataset construction)

This module handles:
  1. Loading raw NTSB CAROL/avall data exports (ev-id, narr_accp, narr_accf, findings)
  2. Loading raw ASRS report exports
  3. Standardizing any input schema into canonical format:
     [report_id, source, text, label]
===============================================================================
"""

import os
import pandas as pd
from typing import Optional, Union, Dict


CANONICAL_COLUMNS = ['report_id', 'source', 'text', 'label']

HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]


def load_ntsb_raw(
    narratives_filepath: Optional[str] = None,
    findings_filepath: Optional[str] = None,
    combined_filepath: Optional[str] = None,
    far_part_filter: str = "091"
) -> pd.DataFrame:
    """
    Loads raw NTSB CAROL / avall dataset based on Liu et al. (2025) Section 4 GAHFACS methodology.

    GAHFACS Construction logic:
      - Filter NTSB records by far_part == '091' (General Aviation Part 91)
      - Join accident narratives (narr_accp = primary, narr_accf = factual) with findings table via 'ev_id'
      - Combine primary + factual narratives into a single text body
      - Map NTSB finding/category codes into HFACS labels

    Returns a DataFrame with columns: [report_id, source, text, label]
    """
    if combined_filepath and os.path.exists(combined_filepath):
        df = pd.read_csv(combined_filepath)
        return standardize_schema(df, source_name="NTSB")

    if not narratives_filepath or not os.path.exists(narratives_filepath):
        raise FileNotFoundError(f"NTSB narratives file not found: {narratives_filepath}")

    narr_df = pd.read_csv(narratives_filepath, low_memory=False)

    # Standardize column casing
    narr_df.columns = [c.lower().strip() for c in narr_df.columns]

    # Filter by Far-Part 091 (General Aviation) if column exists
    if 'far_part' in narr_df.columns and far_part_filter:
        narr_df['far_part'] = narr_df['far_part'].astype(str).str.zfill(3)
        narr_df = narr_df[narr_df['far_part'] == far_part_filter].copy()

    # EV-ID identifier
    ev_col = 'ev_id' if 'ev_id' in narr_df.columns else ('ev-id' if 'ev-id' in narr_df.columns else 'report_id')

    # Combine narr_accp (primary narrative) and narr_accf (factual narrative)
    narr_accp = narr_df.get('narr_accp', pd.Series([''] * len(narr_df))).fillna('')
    narr_accf = narr_df.get('narr_accf', pd.Series([''] * len(narr_df))).fillna('')

    combined_text = (narr_accp.astype(str) + " " + narr_accf.astype(str)).str.strip()

    # Fallback to general 'text' or 'narrative' column if specific NTSB cols missing
    if (combined_text == "").all():
        for alt_col in ['text', 'narrative', 'narr_cause', 'prob_cause']:
            if alt_col in narr_df.columns:
                combined_text = narr_df[alt_col].fillna('').astype(str)
                break

    result_df = pd.DataFrame({
        'report_id': narr_df[ev_col].astype(str) if ev_col in narr_df.columns else [f"NTSB_{i}" for i in range(len(narr_df))],
        'source': 'NTSB',
        'text': combined_text,
        'label': narr_df.get('label', pd.Series(['Unlabeled'] * len(narr_df)))
    })

    # Join findings table if available
    if findings_filepath and os.path.exists(findings_filepath):
        find_df = pd.read_csv(findings_filepath, low_memory=False)
        find_df.columns = [c.lower().strip() for c in find_df.columns]
        find_ev = 'ev_id' if 'ev_id' in find_df.columns else ('ev-id' if 'ev-id' in find_df.columns else 'report_id')

        if find_ev in find_df.columns and 'hfacs_category' in find_df.columns:
            # Map findings to hfacs label
            finding_map = find_df.groupby(find_ev)['hfacs_category'].first().to_dict()
            result_df['label'] = result_df['report_id'].map(finding_map).fillna(result_df['label'])

    return result_df


def load_asrs_raw(filepath: str) -> pd.DataFrame:
    """
    Loads raw ASRS export CSV.
    Handles standard ASRS export field names (ACN, Narrative, Text, Callback, Synopsis).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ASRS file not found: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    return standardize_schema(df, source_name="ASRS")


def standardize_schema(df: pd.DataFrame, source_name: str = "UNKNOWN") -> pd.DataFrame:
    """
    Standardizes any input DataFrame into canonical schema:
    [report_id, source, text, label]

    Maps column name variations across raw ASRS, raw NTSB, and preprocessed datasets.
    """
    df = df.copy()
    col_lower = {str(c).lower().strip(): c for c in df.columns}

    # 1. Map report_id
    id_col = None
    for cand in ['report_id', 'acn', 'ev_id', 'ev-id', 'id', 'report_number']:
        if cand in col_lower:
            id_col = col_lower[cand]
            break

    if id_col:
        report_ids = df[id_col].astype(str)
    else:
        report_ids = [f"{source_name}_{i+1}" for i in range(len(df))]

    # 2. Map source
    if 'source' in col_lower:
        sources = df[col_lower['source']]
    else:
        sources = [source_name] * len(df)

    # 3. Map narrative text
    text_col = None
    for cand in ['text', 'narrative', 'synopsis', 'narr_accp', 'description', 'report']:
        if cand in col_lower:
            text_col = col_lower[cand]
            break

    if text_col:
        texts = df[text_col].fillna('').astype(str)
    else:
        # Check if multiple narrative columns exist and combine them
        narr_cols = [c for c in df.columns if 'narr' in c.lower() or 'text' in c.lower()]
        if narr_cols:
            texts = df[narr_cols].fillna('').astype(str).agg(' '.join, axis=1)
        else:
            raise ValueError(f"Could not locate a text/narrative column in dataset. Columns: {list(df.columns)}")

    # 4. Map HFACS label
    label_col = None
    for cand in ['label', 'hfacs_category', 'category', 'unsafe_act', 'hfacs']:
        if cand in col_lower:
            label_col = col_lower[cand]
            break

    if label_col:
        labels = df[label_col].fillna('Unlabeled').astype(str)
    else:
        labels = ['Unlabeled'] * len(df)

    out_df = pd.DataFrame({
        'report_id': report_ids,
        'source': sources,
        'text': texts,
        'label': labels
    })

    # Drop rows with empty text
    out_df = out_df[out_df['text'].str.strip() != ''].reset_index(drop=True)
    return out_df[CANONICAL_COLUMNS]


if __name__ == "__main__":
    print("[TEST] Running data_loading.py tests...")

    # Test on existing local CSV files
    for fname, src in [("asrs_full.csv", "ASRS"), ("asrs_data.csv", "ASRS"), ("ntsb_data.csv", "NTSB")]:
        if os.path.exists(fname):
            loaded_df = standardize_schema(pd.read_csv(fname), source_name=src)
            print(f"\n[OK] Standardized {fname} ({src}):")
            print(f"  Shape: {loaded_df.shape}")
            print(f"  Columns: {list(loaded_df.columns)}")
            print(f"  Sample:\n{loaded_df.head(2)}")
