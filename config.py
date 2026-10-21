"""
Configuration settings for ASRS-HFACS / NearMiss2Crash project.
Centralizes random seeds, allowed classes, schema definitions, and file paths.
"""

from pathlib import Path

# Random Seed
RANDOM_SEED = 42

# Canonical HFACS and Label Classes
HFACS_CORE_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]

ALLOWED_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error",
    "Other/Unclear",
    "Unlabeled"
]

# Canonical Master Schema Columns
MASTER_SCHEMA_COLUMNS = [
    "report_id",
    "source",
    "text",
    "text_origin",
    "event_date",
    "flight_phase",
    "raw_codes",
    "cause_text",
    "text_may_contain_cause",
    "hfacs_code",
    "keyword_label",
    "llm_label",
    "gold_label",
    "label_source"
]

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLD_DATA_DIR = DATA_DIR / "gold"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"
ARCHIVE_DIR = BASE_DIR / "archive"

# Key Data File Paths
ASRS_HF_DATASET_NAME = "elihoole/asrs-aviation-reports"
ASRS_FULL_CSV = BASE_DIR / "asrs_full.csv"
ASRS_MASTER_CSV = PROCESSED_DATA_DIR / "asrs_master.csv"
NTSB_MASTER_CSV = PROCESSED_DATA_DIR / "ntsb_master.csv"
NTSB_ALL_PART91_CSV = BASE_DIR / "ntsb_all_part91.csv"
NTSB_PROCESSED_TEST_CSV = BASE_DIR / "ntsb_processed_test.csv"
NTSB_RAW_DATASET_DIR = BASE_DIR / "ntsb dataset"

# Output Report Paths
ASRS_COLUMNS_REPORT_TXT = BASE_DIR / "asrs_columns_report.txt"
DATASET_REPORT_MD = BASE_DIR / "dataset_report.md"

# NTSB Cause-detection keywords/phrases
NTSB_CAUSE_PHRASES = [
    "probable cause",
    "the pilot's failure",
    "the pilot in command's failure",
    "the national transportation safety board determines",
    "the national transportation safety board determines the probable cause",
    "national transportation safety board determines",
    "pilot's failure",
    "findings:",
    "finding:"
]
