"""
Build data/processed/ntsb_master.csv following Liu et al. (2025) GAHFACS construction:
- Filter Part 91 General Aviation and event dates from 2008-01-01 onwards.
- Separate narratives: Analysis -> cause_text; Factual Information -> text.
- If only Analysis exists: strip probable cause sentences and set text_origin = 'investigator_narrative'.
- Compute text_may_contain_cause flag.
- Package investigation/finding metadata into raw_codes as JSON.
- Run canonical KeywordWeakLabeler on text.
"""

import sys
import os
import re
from pathlib import Path
import json
import pandas as pd
import numpy as np

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

CAUSE_SPLIT_REGEX = re.compile(
    r'(?i)(?:the\s+national\s+transportation\s+safety\s+board\s+determines|'
    r'probable\s+cause(?:\s+and\s+findings)?[:\s]|'
    r'the\s+pilot(?:\'s|\s+in\s+command\'s)\s+failure|'
    r'pilot\'s\s+failure|'
    r'findings?:)',
    re.IGNORECASE
)

CAUSE_CHECK_REGEX = re.compile(
    r'(?i)\b(?:probable\s+cause|the\s+pilot(?:\'s|\s+in\s+command\'s)\s+failure|'
    r'pilot\'s\s+failure|the\s+national\s+transportation\s+safety\s+board\s+determines|'
    r'national\s+transportation\s+safety\s+board\s+determines)\b',
    re.IGNORECASE
)

def strip_probable_cause_sentences(text: str) -> str:
    """
    Strips sentences or trailing sections containing probable cause determinations
    from investigator analysis text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Split into sentences or lines
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        # Split sentences roughly by period followed by space
        sentences = re.split(r'(?<=[.!?])\s+', line_str)
        kept_sentences = [s for s in sentences if not CAUSE_SPLIT_REGEX.search(s)]
        if kept_sentences:
            clean_lines.append(" ".join(kept_sentences))

    return "\n".join(clean_lines).strip()

def check_text_may_contain_cause(text: str) -> bool:
    """Check whether narrative text contains cause-leaking terminology."""
    if not text or not isinstance(text, str):
        return False
    return bool(CAUSE_CHECK_REGEX.search(text))

def build_ntsb_master() -> pd.DataFrame:
    print("=" * 80)
    print(" TASK 2: BUILDING NTSB MASTER DATASET (ntsb_master.csv)")
    print("=" * 80)

    avdata_path = config.NTSB_RAW_DATASET_DIR / "AviationData.csv"
    an_path = config.NTSB_RAW_DATASET_DIR / "archive" / "Analysis Avation Reports.csv"

    if not avdata_path.exists() or not an_path.exists():
        raise FileNotFoundError(f"Missing required NTSB source tables at {avdata_path} or {an_path}")

    print(f"[INFO] Ingesting NTSB metadata from: {avdata_path}")
    df_av = pd.read_csv(avdata_path, encoding='latin-1', low_memory=False)
    print(f"       Loaded {len(df_av):,} total records.")

    print(f"[INFO] Ingesting NTSB narrative reports from: {an_path}")
    df_an = pd.read_csv(an_path, encoding='latin-1', low_memory=False)
    print(f"       Loaded {len(df_an):,} narrative records.")

    # Clean IDs
    df_av['Event.Id'] = df_av['Event.Id'].astype(str).str.strip()
    df_an['Event ID'] = df_an['Event ID'].astype(str).str.strip()

    # Date conversion & filter >= 2008-01-01
    df_av['Event.Date'] = pd.to_datetime(df_av['Event.Date'], errors='coerce')
    date_mask = df_av['Event.Date'] >= '2008-01-01'
    p91_mask = df_av['FAR.Description'].astype(str).str.contains('91', case=False, na=False)

    df_filtered = df_av[date_mask & p91_mask].copy()
    print(f"[INFO] Filtered to Part 91 & Event.Date >= 2008-01-01: {len(df_filtered):,} records.")

    # Deduplicate Event.Id in AviationData (if multi-aircraft accident)
    df_unique = df_filtered.drop_duplicates(subset=['Event.Id']).copy()
    print(f"[INFO] Unique Event.Ids in filtered subset: {len(df_unique):,}")

    # Merge with Analysis & Factual Information
    merged = df_unique.merge(df_an, left_on='Event.Id', right_on='Event ID', how='left')
    print(f"[INFO] Merged with narrative table: {len(merged):,} rows.")

    # Metadata fields to package into raw_codes
    finding_cols = [
        'Investigation.Type', 'Accident.Number', 'Injury.Severity', 'Aircraft.damage',
        'Aircraft.Category', 'Registration.Number', 'Make', 'Model', 'Amateur.Built',
        'Number.of.Engines', 'Engine.Type', 'FAR.Description', 'Schedule',
        'Purpose.of.flight', 'Air.carrier', 'Total.Fatal.Injuries',
        'Total.Serious.Injuries', 'Total.Minor.Injuries', 'Total.Uninjured',
        'Weather.Condition', 'Broad.phase.of.flight', 'Report.Status'
    ]

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

    print("[INFO] Processing narratives, separating causes, and building rows...")
    for _, row in merged.iterrows():
        ev_id = str(row['Event.Id']).strip()
        rid = f"NTSB_{ev_id}"
        report_ids.append(rid)
        sources.append("NTSB")

        # Event date (YYYY-MM-DD)
        dt = row['Event.Date']
        dt_str = dt.strftime('%Y-%m-%d') if pd.notna(dt) else ""
        event_dates.append(dt_str)

        # Flight phase
        phase = str(row.get('Broad.phase.of.flight', '') or '').strip()
        if not phase or phase.lower() in ('nan', 'unk', 'unknown'):
            phase = "Unknown"
        flight_phases.append(phase)

        # Narratives
        factual = str(row.get('Factual Information', '') or '').strip()
        analysis = str(row.get('Analysis', '') or '').strip()

        # Clean "nan" representations
        if factual.lower() == 'nan':
            factual = ""
        if analysis.lower() == 'nan':
            analysis = ""

        # Put analysis content (and probable cause) into cause_text
        cause_texts.append(analysis)

        # Separate text and text_origin
        if len(factual) > 30:
            assigned_text = factual
            origin = "witness_factual"
        elif len(analysis) > 30:
            assigned_text = strip_probable_cause_sentences(analysis)
            origin = "investigator_narrative"
        else:
            assigned_text = ""
            origin = "unknown"

        texts.append(assigned_text)
        text_origins.append(origin)

        # Flag text_may_contain_cause
        may_contain_cause = check_text_may_contain_cause(assigned_text)
        text_may_contain_cause_list.append(may_contain_cause)

        # Package findings/metadata into raw_codes JSON
        row_codes = {}
        for c in finding_cols:
            val = str(row.get(c, '') or '').strip()
            if val and val.lower() != 'nan':
                row_codes[c] = val
        raw_codes_list.append(json.dumps(row_codes, ensure_ascii=False))

        hfacs_codes.append("")
        llm_labels.append("")
        gold_labels.append("")
        label_sources.append("keyword")

    # Fail loudly on duplicate report_ids
    if len(report_ids) != len(set(report_ids)):
        dup_count = len(report_ids) - len(set(report_ids))
        raise ValueError(f"FATAL: Detected {dup_count} duplicate report_ids in NTSB dataset!")
    print(f"[SUCCESS] All {len(report_ids):,} NTSB report_ids are unique and prefixed.")

    df_ntsb = pd.DataFrame({
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
    print("[INFO] Applying canonical KeywordWeakLabeler to NTSB narratives...")
    labeler = get_canonical_labeler()
    labeled_df = labeler.transform(df_ntsb, text_column="text")
    df_ntsb["keyword_label"] = labeled_df["weak_label"]

    # Reorder strictly to MASTER_SCHEMA_COLUMNS
    df_ntsb = df_ntsb[config.MASTER_SCHEMA_COLUMNS]

    # Verify allowed classes
    invalid_classes = set(df_ntsb["keyword_label"].unique()) - set(config.ALLOWED_CLASSES)
    if invalid_classes:
        raise ValueError(f"FATAL: Found invalid keyword_label classes: {invalid_classes}")

    # Ensure output directory exists and save
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.NTSB_MASTER_CSV
    df_ntsb.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved ntsb_master.csv to: {output_path}")
    print(f"          Rows: {len(df_ntsb):,}, File size: {output_path.stat().st_size / 1e6:.2f} MB")
    print("\n--- Keyword Label Distribution ---")
    print(df_ntsb["keyword_label"].value_counts(dropna=False))

    pct_cause = (df_ntsb["text_may_contain_cause"].sum() / len(df_ntsb)) * 100
    print(f"\n--- Cause Leakage Flag ---")
    print(f"text_may_contain_cause = True: {df_ntsb['text_may_contain_cause'].sum():,} ({pct_cause:.2f}%)")

    return df_ntsb

if __name__ == "__main__":
    build_ntsb_master()
