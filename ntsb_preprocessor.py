"""
===============================================================================
NTSB Data Preprocessing & Cleaning Pipeline (Week 2, Day 8-9)
===============================================================================
Reference Papers:
  1. Madeira, Melicio, Valerio & Santos (2021) -- "Machine Learning and NLP for
     Prediction of Human Factors in Aviation Incident Reports" (Section 2.2-2.3)
  2. Shi, Zurada, Cao & Guan (2022) -- "Aligning Aviation Safety Reporting System
     Anomaly Codes for Crew" (Section 3.1)
  3. Liu, Li, Ng, Han & Feng (2025) -- "Accident investigation via LLMs reasoning:
     HFACS-guided Chain-of-Thoughts" (Section 4, GAHFACS Part 91 filtering)

Implementation:
  1. Scope Filtering: Part 91 General Aviation records from NTSB_database.csv
  2. Cleaning & Normalization:
     - Unicode normalization & removal of Windows-1252 non-breaking characters
     - Lowercasing, punctuation stripping, alphabetic token parsing
     - Dual stopword removal (Standard English + Aviation domain stopwords)
     - Lemmatization via NLTK WordNetLemmatizer
  3. Consistent HFACS Weak Labeling:
     - Madeira et al. (2021) Table 1 seed keywords
     - Ambiguity filtering (assign label ONLY when single HFACS class matches)
  4. Benchmark Test Split Export:
     - Canonical schema: [report_id, source, text, clean_text, label, flight_phase]
===============================================================================
"""

import os
import re
import string
import unicodedata
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources
for res in ['stopwords', 'wordnet', 'punkt']:
    try:
        nltk.data.find(f'corpora/{res}' if res != 'punkt' else f'tokenizers/{res}')
    except LookupError:
        nltk.download(res, quiet=True)

# -----------------------------------------------------------------------------
# Domain Stopwords & Seed Keywords
# -----------------------------------------------------------------------------
STANDARD_STOPWORDS = set(stopwords.words('english'))

# Shi et al. (2022) & Madeira et al. (2021) non-informative domain stopwords
DOMAIN_STOPWORDS = {
    'summary', 'probable', 'cause', 'causes', 'accident', 'accidents',
    'incident', 'incidents', 'contribute', 'contributed', 'contributing',
    'factor', 'factors', 'find', 'finding', 'findings', 'conclusion',
    'conclusions', 'investigation', 'investigations', 'ntsb', 'occurred',
    'airplane', 'aircraft', 'pilot', 'copilot', 'crew', 'flight', 'report',
    'reported', 'faa', 'federal', 'aviation', 'administration', 'board',
    'safety', 'national', 'transportation', 'day', 'time', 'date', 'year',
    'approximately', 'feet', 'miles', 'airport', 'runway', 'located'
}

COMBINED_STOPWORDS = STANDARD_STOPWORDS.union(DOMAIN_STOPWORDS)

# Seed keywords adapted from Madeira et al. (2021) Table 1
SEED_KEYWORDS: Dict[str, List[str]] = {
    "Skill-Based Error": [
        "checklist", "forgot", "omitted", "lapsed", "routine", "habit",
        "inadvertent", "inadvertently", "slip", "memory", "misread",
        "improper flare", "failure to maintain airspeed", "failure to maintain control",
        "loss of control", "improper compensation", "overcontrol", "undercorrection"
    ],
    "Decision Error": [
        "decided", "elected", "chose", "continued flight", "decision", "judgment",
        "divert", "alternate airport", "fuel exhaustion", "vfr into imc",
        "deteriorating weather", "improper planning", "inadequate preflight"
    ],
    "Violation": [
        "knowingly", "intentionally", "unauthorized", "regulation", "regulations",
        "noncompliance", "exceeded limitations", "failed to comply", "without authorization",
        "sop", "sterile cockpit", "aerobatic flight", "low altitude maneuver"
    ],
    "Perceptual Error": [
        "spatial disorientation", "disorientation", "illusion", "visual illusion",
        "misjudged distance", "depth perception", "featureless terrain", "whiteout",
        "glare", "dark night", "black hole", "false horizon"
    ]
}

# Precompile regexes for fast matching
COMPILED_KEYWORD_REGEX = {
    cls: [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in kws]
    for cls, kws in SEED_KEYWORDS.items()
}


class NTSBPreprocessor:
    """End-to-end preprocessor for NTSB accident reports."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = COMBINED_STOPWORDS

    def clean_text(self, text: str) -> str:
        """
        Executes Madeira's 3-stage preprocessing:
        1. Normalization (Unicode decomposition, Windows-1252 artifact removal)
        2. Alphabetic tokenization & lowercasing
        3. Dual stopword filtering & lemmatization
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Normalize unicode (decompose non-breaking hyphens, special spaces, smart quotes)
        text = unicodedata.normalize('NFKD', text)
        text = text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\xa0', ' ')

        # Lowercase
        text = text.lower()

        # Remove punctuation by replacing with space
        for char in string.punctuation:
            text = text.replace(char, ' ')

        # Tokenize alphabetic-only words (strip numbers and symbols)
        tokens = [w for w in text.split() if w.isalpha() and len(w) > 2]

        # Dual stopword removal + lemmatization
        clean_tokens = [
            self.lemmatizer.lemmatize(w)
            for w in tokens
            if w not in self.stopwords
        ]

        return " ".join(clean_tokens)

    def assign_label(self, raw_text: str) -> str:
        """
        Assigns an HFACS category using Madeira et al. (2021) consistent keyword matching.
        Strict rule: Returns a label ONLY if exactly one class matches (unambiguous).
        """
        if not isinstance(raw_text, str):
            return "Unlabeled"

        matched_classes = []
        for cls_name, regex_list in COMPILED_KEYWORD_REGEX.items():
            if any(rgx.search(raw_text) for rgx in regex_list):
                matched_classes.append(cls_name)

        if len(matched_classes) == 1:
            return matched_classes[0]
        return "Unlabeled"

    def process_dataset(
        self,
        input_csv: str = r"ntsb dataset\archive\NTSB_database.csv",
        output_test_csv: str = r"ntsb_processed_test.csv",
        output_full_csv: Optional[str] = r"ntsb_all_part91.csv",
        max_samples_per_class: int = 500
    ) -> pd.DataFrame:
        """
        Processes raw NTSB dataset:
        - Filters Part 91 General Aviation
        - Cleans text narratives
        - Generates HFACS labels
        - Samples balanced benchmark test dataset
        """
        print(f"[INFO] Ingesting NTSB database from {input_csv}...")
        df_raw = pd.read_csv(input_csv, encoding='latin-1', low_memory=False)
        print(f"       Loaded {len(df_raw)} total rows.")

        # 1. Scope filter: Part 91 General Aviation
        far_col = 'Far Description' if 'Far Description' in df_raw.columns else 'far_description'
        if far_col in df_raw.columns:
            mask_p91 = df_raw[far_col].astype(str).str.contains('91', case=False, na=False)
            df = df_raw[mask_p91].copy()
            print(f"[INFO] Filtered to Part 91 General Aviation: {len(df)} records.")
        else:
            df = df_raw.copy()

        # 2. Extract valid Analysis narratives
        analysis_col = 'Analysis' if 'Analysis' in df.columns else 'analysis'
        df = df[df[analysis_col].dropna().astype(str).str.strip().str.len() > 50].copy()
        print(f"[INFO] Valid narrative records (>50 chars): {len(df)}.")

        # Extract columns
        event_id_col = 'Event Id' if 'Event Id' in df.columns else ('Event ID' if 'Event ID' in df.columns else 'event_id')
        phase_col = 'Broad Phase Of Flight' if 'Broad Phase Of Flight' in df.columns else 'broad_phase_of_flight'

        report_ids = df[event_id_col].astype(str).tolist()
        raw_texts = df[analysis_col].astype(str).tolist()
        phases = df[phase_col].fillna('Unknown').astype(str).tolist() if phase_col in df.columns else ['Unknown'] * len(df)

        # 3. Labeling and cleaning
        print("[INFO] Cleaning narratives and assigning HFACS labels...")
        clean_texts = []
        labels = []

        for i, text in enumerate(raw_texts):
            clean_t = self.clean_text(text)
            clean_texts.append(clean_t)
            lbl = self.assign_label(text)
            labels.append(lbl)

        # Construct dataframe
        processed_df = pd.DataFrame({
            'report_id': report_ids,
            'source': 'NTSB',
            'text': raw_texts,
            'clean_text': clean_texts,
            'label': labels,
            'flight_phase': phases
        })

        # Drop any records where clean_text became too short (< 5 words)
        processed_df = processed_df[processed_df['clean_text'].str.split().str.len() >= 5].reset_index(drop=True)
        print(f"[INFO] Total cleaned Part 91 records: {len(processed_df)}")
        print("[INFO] Full label distribution:")
        print(processed_df['label'].value_counts())

        # Save all Part 91 if requested
        if output_full_csv:
            processed_df.to_csv(output_full_csv, index=False, encoding='utf-8')
            print(f"[INFO] Saved full Part 91 dataset ({len(processed_df)} rows) to: {output_full_csv}")

        # 4. Construct balanced benchmark test set from labeled records
        labeled_df = processed_df[processed_df['label'] != 'Unlabeled'].copy()
        print(f"\n[INFO] Total unambiguously labeled records: {len(labeled_df)}")
        print(labeled_df['label'].value_counts())

        # Sample stratified per class (up to max_samples_per_class per class)
        balanced_dfs = []
        for cls_name in SEED_KEYWORDS.keys():
            cls_pool = labeled_df[labeled_df['label'] == cls_name]
            n_take = min(len(cls_pool), max_samples_per_class)
            if n_take > 0:
                sampled = cls_pool.sample(n=n_take, random_state=42)
                balanced_dfs.append(sampled)
                print(f"       Class '{cls_name}': selected {n_take} samples.")

        test_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"\n[SUCCESS] Final Balanced NTSB Benchmark Test Set: {len(test_df)} samples.")
        print(test_df['label'].value_counts())

        # Save benchmark test dataset
        test_df.to_csv(output_test_csv, index=False, encoding='utf-8')
        print(f"[SUCCESS] Exported benchmark test set to: {output_test_csv}")

        return test_df


if __name__ == "__main__":
    preprocessor = NTSBPreprocessor()
    test_data = preprocessor.process_dataset()
