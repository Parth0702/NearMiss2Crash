# Repository Cleanup Audit

**Project**: NearMiss2Crash — Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives  
**Date**: October 22, 2026  
**Total repository size**: ~380 MB (excluding .venv)

---

## A. File Classification Summary

| Category | Count | Size | Action |
|----------|-------|------|--------|
| **KEEP** | 35 files + dirs | ~380 MB | Essential for current research |
| **ARCHIVE** | 8 dirs + 15 files | ~180 MB | Historical experiments, move to archive/ |
| **DELETE CANDIDATE** | 6 files | ~45 MB | Temporary/cached/redundant files |
| **REVIEW NEEDED** | 3 files | ~36 MB | Personal/untracked files |

---

## B. Detailed Classification Table

### KEEP — Essential for Current Research

| Path | Size | Purpose | Dependencies |
|------|------|---------|--------------|
| `.git/` | — | Version control | Git |
| `.gitignore` | 450 B | Git ignore rules | Git |
| `.venv/` | — | Python environment | All Python scripts |
| `PHASE_2_COMPLETION_REPORT.md` | 10.5 KB | Phase 2 documentation | — |
| `PHASE_3B_DISTILBERT_ASSET_AUDIT.md` | 5.3 KB | Phase 3B asset audit | — |
| `README_WEAK_LABELING.md` | 8.6 KB | Weak labeling docs | — |
| `config.py` | 2 KB | Central config | All scripts |
| `data_loading.py` | 7.4 KB | Data utilities | Baseline scripts |
| `keyword_weak_labeler.py` | 8.4 KB | Phase 1 crisp labeler | Phase 1, 2, 3 |
| `evaluation.py` | 13.8 KB | Evaluation framework | Phase 1, 3 |
| `transformer_model.py` | 13.5 KB | DistilBERT training | Phase 3B |
| `train_distilbert_head.py` | 6.9 KB | DistilBERT head training | Phase 3B |
| `requirements.txt` | 580 B | Dependencies | Environment |
| `experiments/fuzzy_asrs/fuzzy_labeler.py` | 24 KB | Phase 2 fuzzy labeler | Phase 2, 3, 3B |
| `experiments/fuzzy_asrs/run_fuzzy_sample.py` | 2 KB | Phase 2 demo | Phase 2 |
| `experiments/fuzzy_asrs/run_phase1_baseline.py` | 8.3 KB | Phase 1 runner | Phase 1 |
| `experiments/fuzzy_asrs/run_phase3_full_labeling.py` | 6.8 KB | Phase 3 full labeling | Phase 3 |
| `experiments/fuzzy_asrs/run_phase3_comparison.py` | 5.5 KB | Phase 3 comparison | Phase 3 |
| `experiments/fuzzy_asrs/run_phase3_sensitivity.py` | 5.5 KB | Phase 3 sensitivity | Phase 3 |
| `experiments/fuzzy_asrs/crisp_baseline_rows.csv` | 63.5 MB | Phase 1 baseline data | Phase 1, 2, 3, 3B |
| `experiments/fuzzy_asrs/crisp_label_distribution.csv` | 196 B | Phase 1 distribution | Phase 1, 3 |
| `experiments/fuzzy_asrs/results/fuzzy_full_results.csv` | 68 MB | Phase 3 fuzzy labels | Phase 3, 3B |
| `experiments/fuzzy_asrs/results/comparison_summary.json` | 438 B | Phase 3 summary | Phase 3 |
| `experiments/fuzzy_asrs/results/crisp_vs_fuzzy_transition.csv` | 302 B | Phase 3 transition | Phase 3 |
| `experiments/fuzzy_asrs/results/detailed_transitions.csv` | 2.6 MB | Phase 3 transitions | Phase 3 |
| `experiments/fuzzy_asrs/results/sensitivity/tau_sensitivity.json` | 1.3 KB | Phase 3 tau sensitivity | Phase 3 |
| `experiments/fuzzy_asrs/results/sensitivity/delta_sensitivity.json` | 1.3 KB | Phase 3 delta sensitivity | Phase 3 |
| `experiments/fuzzy_asrs/results/phase3_full_labeling.log` | 2.4 KB | Phase 3 log | Phase 3 |
| `data/processed/asrs_master.csv` | 142.6 MB | Master ASRS dataset | Data loading |
| `data/processed/ntsb_master.csv` | 97.5 MB | Master NTSB dataset | Reference |
| `src/build_asrs_master.py` | 5.7 KB | ASRS builder | Data pipeline |
| `src/build_ntsb_master.py` | 9.8 KB | NTSB builder | Data pipeline |
| `src/validate_and_report.py` | 10.5 KB | Validation | Data pipeline |
| `asrs_full.csv` | 72.1 MB | Full ASRS dataset | Phase 1, 3 |
| `docs/rapid_audit.md` | 15.1 KB | Audit docs | Reference |
| `ASRS-HFACS_4Week_Build_Plan.pdf` | 81 MB | Project plan | Reference |
| `final pitch.pdf` | 5.3 MB | Presentation | Reference |
| `research papers/` | — | Reference papers | Reference |
| `week3_results/best_model_distilbert-base-uncased.pt` | 265.5 MB | NTSB-trained DistilBERT | Phase 3B reference |

---

### ARCHIVE — Historical Experiments (Move to `archive/`)

| Path | Size | Purpose | Reason |
|------|------|---------|--------|
| `weak_labeling_results/` | ~59 MB | Week 2 weak label experiments | Superseded by Phase 3 |
| `week2_results/` | ~35 MB | Week 2 classifier training | Superseded by Phase 3 |
| `week3_results/` (except DistilBERT checkpoint) | ~3 MB | Week 3 transformer results | Superseded |
| `week4_results/` | ~1.2 MB | Week 4 explainability/HP tuning | Historical |
| `experiments/baseline_audit/` | ~2 MB | Baseline audit logs | Historical |
| `hfacs_baseline_pipeline.py` | 17 KB | Original pipeline | Superseded |
| `hfacs_baseline_pipeline.ipynb` | 9.8 KB | Original notebook | Superseded |
| `week2_classifier_training.py` | 18 KB | Week 2 training | Superseded |
| `week2_dataset_preparation.py` | 14 KB | Week 2 prep | Superseded |
| `week2_evaluation.py` | 14.5 KB | Week 2 eval | Superseded |
| `train_all_models.py` | 9.9 KB | Week 2 all models | Superseded |
| `weak_labeler_cluster.py` | 13.7 KB | Cluster labeler | Superseded |
| `weak_labeling_framework.py` | 16.5 KB | Framework | Superseded |
| `model_comparison.py` | 12.6 KB | Week 2 comparison | Superseded |
| `llm_cot_baseline.py` | 18.4 KB | LLM CoT baseline | Historical |
| `explainability.py` | 8.7 KB | Explainability | Historical |

---

### DELETE CANDIDATE — Temporary/Redundant

| Path | Size | Reason |
|------|------|--------|
| `__pycache__/` | ~2 MB | Python cache, regenerable |
| `experiments/fuzzy_asrs/__pycache__/` | ~1 MB | Python cache, regenerable |
| `test_cp1252.csv` | 22.6 MB | Encoding test, not used |
| `test_latin-1.csv` | 22.6 MB | Encoding test, not used |
| `ntsb_data.csv` | 786 B | Small test file, redundant |
| `notebook_titles.csv` | 540 B | Notebook metadata, not used |

---

### REVIEW NEEDED — Personal/Untracked Files

| Path | Size | Notes |
|------|------|-------|
| `build_plan.txt` | 12.7 KB | Appears personal |
| `cluster_config.json` | 11.4 KB | Appears personal |
| `generate_all_labels.py` | 575 B | Appears utility script |
| `generate_asrs_columns_report.py` | 4.8 KB | Appears utility script |
| `detect_encoding.py` | 1.7 KB | Appears utility script |
| `flight_phase_analysis.py` | 3.5 KB | Appears utility script |
| `hyperparameter_tuning.py` | 56 B | Appears placeholder |
| `regenerate_weak_labeling.py` | 8.4 KB | Appears utility script |
| `standardize_ntsb_fixed.py` | 3.9 KB | Appears utility script |
| `test_ntsb.py` | 11.2 KB | Test script |
| `write_ht.py` | 24 KB | Unknown purpose |

---

## C. Space Recovery Estimate

| Action | Space Recovered |
|--------|----------------|
| Delete `__pycache__` directories | ~3 MB |
| Delete test CSV files | ~45 MB |
| Archive historical experiments | ~180 MB (moved, not deleted) |
| **Total deletable** | **~48 MB** |
| **Total archivable** | **~180 MB** |

---

## D. Dependency Check Results

1. **Phase 3B DistilBERT training** depends on:
   - `transformer_model.py` (KEEP)
   - `train_distilbert_head.py` (KEEP)
   - `week3_results/best_model_distilbert-base-uncased.pt` (KEEP as reference)
   - `weak_labeling_results/dataset_majority_*.csv` (ARCHIVE but needed for reference)

2. **Phase 3 fuzzy experiments** depend on:
   - `keyword_weak_labeler.py` (KEEP)
   - `experiments/fuzzy_asrs/fuzzy_labeler.py` (KEEP)
   - `experiments/fuzzy_asrs/crisp_baseline_rows.csv` (KEEP)
   - `data/processed/asrs_master.csv` (KEEP)

3. **No current script imports** from week2_results, week4_results, or most historical scripts.

4. **asrs_full.csv** is used by `run_phase1_baseline.py` for fallback but primary source is Arrow cache.

---

## E. Proposed Actions

### Immediate Safe Cleanup (Auto-executable)
```
DELETE:
  __pycache__/
  experiments/fuzzy_asrs/__pycache__/
  test_cp1252.csv
  test_latin-1.csv
  ntsb_data.csv
  notebook_titles.csv
```

### Archive (Requires directory creation)
```
MOVE TO archive/:
  weak_labeling_results/
  week2_results/
  week3_results/ (except best_model_distilbert-base-uncased.pt)
  week4_results/
  experiments/baseline_audit/
  hfacs_baseline_pipeline.py
  hfacs_baseline_pipeline.ipynb
  week2_classifier_training.py
  week2_dataset_preparation.py
  week2_evaluation.py
  train_all_models.py
  weak_labeler_cluster.py
  weak_labeling_framework.py
  model_comparison.py
  llm_cot_baseline.py
  explainability.py
```

### Review Required (Manual decision)
```
DECISION NEEDED:
  build_plan.txt
  cluster_config.json
  generate_all_labels.py
  generate_asrs_columns_report.py
  detect_encoding.py
  flight_phase_analysis.py
  hyperparameter_tuning.py
  regenerate_weak_labeling.py
  standardize_ntsb_fixed.py
  test_ntsb.py
  write_ht.py
```

---

## F. Verification After Cleanup

Required tests to run after cleanup:

1. **Phase 2 unit tests**: `python experiments/fuzzy_asrs/fuzzy_labeler.py`
2. **Phase 3 validation**: Verify fuzzy_full_results.csv loads and has correct columns
3. **DistilBERT import test**: `python -c "from transformer_model import HFACSClassifier"`
4. **Phase 1 baseline**: `python experiments/fuzzy_asrs/run_phase1_baseline.py` (smoke test, limited rows)

---

## G. Final Directory Structure (Proposed)

```
D:\SC project\
├── .git/
├── .gitignore
├── .venv/
├── PHASE_2_COMPLETION_REPORT.md
├── PHASE_3B_DISTILBERT_ASSET_AUDIT.md
├── README_WEAK_LABELING.md
├── config.py
├── data_loading.py
├── keyword_weak_labeler.py
├── evaluation.py
├── transformer_model.py
├── train_distilbert_head.py
├── requirements.txt
├── asrs_full.csv
├── ASRS-HFACS_4Week_Build_Plan.pdf
├── final pitch.pdf
├── docs/
│   └── rapid_audit.md
├── data/
│   └── processed/
│       ├── asrs_master.csv
│       └── ntsb_master.csv
├── src/
│   ├── build_asrs_master.py
│   ├── build_ntsb_master.py
│   └── validate_and_report.py
├── experiments/
│   └── fuzzy_asrs/
│       ├── fuzzy_labeler.py
│       ├── run_fuzzy_sample.py
│       ├── run_phase1_baseline.py
│       ├── run_phase3_full_labeling.py
│       ├── run_phase3_comparison.py
│       ├── run_phase3_sensitivity.py
│       ├── crisp_baseline_rows.csv
│       ├── crisp_label_distribution.csv
│       └── results/
│           ├── fuzzy_full_results.csv
│           ├── comparison_summary.json
│           ├── crisp_vs_fuzzy_transition.csv
│           ├── detailed_transitions.csv
│           ├── phase3_full_labeling.log
│           └── sensitivity/
│               ├── tau_sensitivity.json
│               └── delta_sensitivity.json
├── week3_results/
│   └── best_model_distilbert-base-uncased.pt
├── research papers/
└── archive/  (NEW)
    ├── weak_labeling_results/
    ├── week2_results/
    ├── week3_results/
    ├── week4_results/
    ├── experiments/baseline_audit/
    ├── hfacs_baseline_pipeline.py
    ├── hfacs_baseline_pipeline.ipynb
    ├── week2_classifier_training.py
    ├── week2_dataset_preparation.py
    ├── week2_evaluation.py
    ├── train_all_models.py
    ├── weak_labeler_cluster.py
    ├── weak_labeling_framework.py
    ├── model_comparison.py
    ├── llm_cot_baseline.py
    └── explainability.py
```

---

## H. Approval Required

**Please approve the DELETE CANDIDATE list (6 files, ~48 MB) before proceeding with permanent deletion.**

The ARCHIVE move is safe (reversible) and preserves all historical work.

The REVIEW NEEDED files should be examined for personal/utility value before deciding.