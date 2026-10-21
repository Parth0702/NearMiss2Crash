"""
Build data/processed/asrs_master.csv from HuggingFace dataset elihoole/asrs-aviation-reports.
Strictly adheres to project schema and canonical KeywordWeakLabeler.
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datasets import load_dataset

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from keyword_weak_labeler import KeywordWeakLabeler, SEED_KEYWORDS

def get_canonical_labeler() -> KeywordWeakLabeler:
    """Returns canonical KeywordWeakLabeler loaded with all seed keywords."""
    labeler = KeywordWeakLabeler(random_state=config.RANDOM_SEED)
    for cls_name, kw_list in SEED_KEYWORDS.items():
        for kw in kw_list:
            labeler.validated_keywords[kw] = cls_name
    return labeler

def format_event_date(val: str) -> str:
    """Format YYYYMM date string into YYYY-MM."""
    s = str(val or "").strip()
    if len(s) == 6 and s.isdigit():
        return f"{s[:4]}-{s[4:]}"
    return s

def build_asrs_master() -> pd.DataFrame:
    print("=" * 80)
    print(" TASK 1: BUILDING ASRS MASTER DATASET (asrs_master.csv)")
    print("=" * 80)

    print(f"[INFO] Ingesting HuggingFace dataset '{config.ASRS_HF_DATASET_NAME}'...")
    ds = load_dataset(config.ASRS_HF_DATASET_NAME, split="train")
    total_raw = len(ds)
    print(f"[INFO] Ingested {total_raw:,} records from HuggingFace.")

    narrative_cols = {
        'Report 1_Narrative',
        'Report 1.1_Callback',
        'Report 1.2_Synopsis',
        'Report 2_Narrative',
        'Report 2.1_Callback'
    }
    id_date_cols = {
        'acn_num_ACN',
        'Time_Date',
        'Time.1_Local Time Of Day'
    }
    analyst_cols = [c for c in ds.column_names if c not in narrative_cols and c not in id_date_cols]

    report_ids = []
    sources = []
    texts = []
    text_origins = []
    event_dates = []
    flight_phases = []
    raw_codes_list = []
    cause_texts = []
    text_may_contain_cause_list = []
    hfacs_codes = []
    llm_labels = []
    gold_labels = []
    label_sources = []

    print("[INFO] Processing rows, narratives, and packaging raw_codes JSON...")
    for row in ds:
        acn = str(row.get("acn_num_ACN", "") or "").strip()
        rid = f"ASRS_{acn}"
        report_ids.append(rid)
        sources.append("ASRS")

        # Narrative concatenation
        n1 = str(row.get("Report 1_Narrative", "") or "").strip()
        n2 = str(row.get("Report 2_Narrative", "") or "").strip()
        synopsis = str(row.get("Report 1.2_Synopsis", "") or "").strip()

        if n1 and n2:
            combined_text = n1 + " " + n2
        elif n1:
            combined_text = n1
        elif n2:
            combined_text = n2
        elif synopsis:
            combined_text = synopsis
        else:
            combined_text = ""
        texts.append(combined_text)

        text_origins.append("pilot_first_person")
        event_dates.append(format_event_date(row.get("Time_Date", "")))

        # Flight phase
        phase = str(row.get("Aircraft 1.9_Flight Phase", "") or "").strip()
        flight_phases.append(phase if phase else "Unknown")

        # Raw codes packaging: all non-empty analyst fields
        row_codes = {}
        for c in analyst_cols:
            val = str(row.get(c, "") or "").strip()
            if val:
                row_codes[c] = val
        raw_codes_list.append(json.dumps(row_codes, ensure_ascii=False))

        cause_texts.append("")
        text_may_contain_cause_list.append(False)
        hfacs_codes.append("")
        llm_labels.append("")
        gold_labels.append("")
        label_sources.append("keyword")

    # Fail loudly on duplicate report_ids
    if len(report_ids) != len(set(report_ids)):
        dup_count = len(report_ids) - len(set(report_ids))
        raise ValueError(f"FATAL: Detected {dup_count} duplicate report_ids in ASRS dataset!")
    print(f"[SUCCESS] All {len(report_ids):,} ASRS report_ids are unique and prefixed.")

    df_asrs = pd.DataFrame({
        "report_id": report_ids,
        "source": sources,
        "text": texts,
        "text_origin": text_origins,
        "event_date": event_dates,
        "flight_phase": flight_phases,
        "raw_codes": raw_codes_list,
        "cause_text": cause_texts,
        "text_may_contain_cause": text_may_contain_cause_list,
        "hfacs_code": hfacs_codes,
        "llm_label": llm_labels,
        "gold_label": gold_labels,
        "label_source": label_sources
    })

    # Run canonical keyword weak labeler
    print("[INFO] Applying canonical KeywordWeakLabeler to ASRS narratives...")
    labeler = get_canonical_labeler()
    labeled_df = labeler.transform(df_asrs, text_column="text")
    df_asrs["keyword_label"] = labeled_df["weak_label"]

    # Reorder strictly to MASTER_SCHEMA_COLUMNS
    df_asrs = df_asrs[config.MASTER_SCHEMA_COLUMNS]

    # Verify allowed classes
    invalid_classes = set(df_asrs["keyword_label"].unique()) - set(config.ALLOWED_CLASSES)
    if invalid_classes:
        raise ValueError(f"FATAL: Found invalid keyword_label classes: {invalid_classes}")

    # Ensure output directory exists and save
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.ASRS_MASTER_CSV
    df_asrs.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved asrs_master.csv to: {output_path}")
    print(f"          Rows: {len(df_asrs):,}, File size: {output_path.stat().st_size / 1e6:.2f} MB")
    print("\n--- Keyword Label Distribution ---")
    print(df_asrs["keyword_label"].value_counts(dropna=False))

    return df_asrs

if __name__ == "__main__":
    build_asrs_master()
