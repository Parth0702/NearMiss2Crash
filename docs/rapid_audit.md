# Rapid Audit Report: ASRS-HFACS / NearMiss2Crash

**Date:** October 21, 2026  
**Auditor:** Antigravity (Pair Programming Agent)  
**Corpus / Project:** `d:\SC project` (`Parth0702/NearMiss2Crash`)  
**Git Branch:** `cleanup`  
**Git Status:** Untracked audit directories (`docs/`, `experiments/`), working tree clean (`HEAD` at commit `1ef235b`)  
**Safety Anchor Tag:** `v0-before-cleanup`

---

## 1. Executive Summary & Exit Gate Verification

This rapid audit documents repository working-tree status, canonical weak labeling implementation, text narrative integrity, Python environment dependencies, baseline reproduction, and the **Phase 1 single-reporter text decision**.

### Exit Gate Answers (Phase 1 Verified)

| Question | Verified Empirical Answer |
| :--- | :--- |
| **Which actual file and column are being used?** | File: [`experiments/fuzzy_asrs/crisp_baseline_rows.csv`](file:///d:/SC%20project/experiments/fuzzy_asrs/crisp_baseline_rows.csv) (derived from local cached HuggingFace Arrow dataset `asrs-aviation-reports-train.arrow`). Active narrative column: **`Report 1_Narrative`** (stored in column `text`, tagged with `input_source = "Report 1_Narrative"`). |
| **Did we use Report 1 alone or the combined text fallback?** | **Report 1 alone (Preferred route).** The original source dataset was locally available in the HuggingFace cache (`asrs-aviation-reports-train.arrow`), enabling instant local extraction without network calls or rebuilding. The combined-text fallback was **not** used. |
| **Were crisp labels regenerated on precisely that input?** | **Yes.** The canonical [`KeywordWeakLabeler`](file:///d:/SC%20project/keyword_weak_labeler.py) was run directly on `Report 1_Narrative` across all 38,655 rows, producing the updated crisp baseline in [`experiments/fuzzy_asrs/crisp_baseline_rows.csv`](file:///d:/SC%20project/experiments/fuzzy_asrs/crisp_baseline_rows.csv). |
| **What are the newly measured class counts?** | Total reports: **38,655**.<br>• **Decision Error:** 16,906 (43.74% of all, 80.28% of labeled)<br>• **Unlabeled:** 17,597 (45.52% of all)<br>• **Skill-Based Error:** 2,826 (7.31% of all, 13.42% of labeled)<br>• **Perceptual Error:** 659 (1.70% of all, 3.13% of labeled)<br>• **Violation:** 667 (1.73% of all, 3.17% of labeled)<br>Total labeled: **21,058** (54.48%); Total unlabeled: **17,597** (45.52%). |
| **What is the reproducible command?** | `& '.\.venv\Scripts\python.exe' experiments/fuzzy_asrs/run_phase1_baseline.py`<br>Logged to [`experiments/fuzzy_asrs/phase1_run.log`](file:///d:/SC%20project/experiments/fuzzy_asrs/phase1_run.log). |

---

## 2. Repository & Environment Status

### 2.1 Git Status & Recent History
- **Repository Path:** `D:\SC project`
- **Active Branch:** `cleanup`
- **HEAD Commit:** `1ef235b`
- **Recent Commits:**
  1. `1ef235b` — `chore(git): ignore large generated CSVs and track data/processed directory`
  2. `af17ccb` — `docs(dataset): add audit and validation report with blinded samples`
  3. `b7c4531` — `feat(dataset): add builder and validation scripts for ASRS and NTSB master datasets`
  4. `1f4f4d2` — `feat(config): add central config and generate ASRS column audit report`
  5. `1d32395` — `Regenerate weak labeling results from real ASRS dataset (asrs_full.csv)`

### 2.2 Python Environment & Package Versions
- **Python Executable:** `D:\SC project\.venv\Scripts\python.exe`
- **Python Version:** `3.14.5`
- **Package Status:**
  - `pandas`: **3.0.6**
  - `numpy`: **2.5.4**
  - `scikit-learn`: **1.9.1**
  - `torch`: **2.14.1**
  - `transformers`: **5.19.0**
  - `scikit-fuzzy`: **NOT INSTALLED** (`WARNING: Package(s) not found: scikit-fuzzy`)

> [!NOTE]
> The existing `.venv` supports all standard TF-IDF, scikit-learn, and transformer workflows. Any subsequent soft-labeling or membership-function modules that require `scikit-fuzzy` will need installation.

---

## 3. Canonical Keyword Weak Labeler Implementation

**Implementation File:** [`keyword_weak_labeler.py`](file:///d:/SC%20project/keyword_weak_labeler.py)  
**Class:** [`KeywordWeakLabeler`](file:///d:/SC%20project/keyword_weak_labeler.py#L54-L192)

### 3.1 Input Interface
- Expected input: `pd.DataFrame` containing narrative text in column `text_column` (default `'text'`).
- Optional evaluation input: `label_column` (default `'label'`), used only within `.fit()` when auditing keyword consistency ratios.

### 3.2 Target Classes & Fallback States
- **Core 4 Unsafe-Act Classes:**
  1. `"Skill-Based Error"`
  2. `"Decision Error"`
  3. `"Violation"`
  4. `"Perceptual Error"`
- **Fallback State:** `"Unlabeled"` (assigned whenever no seed keywords match).
- **Other/Unknown State:** No internal `"Other"` or `"Unknown"` state is emitted by `keyword_weak_labeler.py`.

### 3.3 Seed Keyword Dictionaries (39 Keywords)

```python
SEED_KEYWORDS = {
    "Skill-Based Error": [
        "checklist", "forgot", "omitted", "lapsed", "routine", "habit",
        "inadvertent", "slip", "memory", "missed call", "misread"
    ],
    "Decision Error": [
        "decided", "elected", "chose", "continued", "judgment", "risk",
        "weather", "visual approach", "divert", "below minimums"
    ],
    "Violation": [
        "knowingly", "intentionally", "skipped", "unauthorized",
        "rule", "sop", "regulation", "sterile cockpit", "exceeded"
    ],
    "Perceptual Error": [
        "disorientation", "spatial", "illusion", "misjudged distance",
        "visibility", "flare", "glare", "featureless terrain", "depth perception"
    ]
}
```

### 3.4 Matching Engine & Tie-Breaking Behavior
1. **Regex Matching:** Case-insensitive word boundary matching: `r'\b' + re.escape(keyword.lower()) + r'\b'`.
2. **Collection:** All matching keywords append their class name to `matched_classes` in dictionary iteration order.
3. **Tie-breaking / Crisp Assignment:**
   ```python
   if matched_classes:
       predicted_label = pd.Series(matched_classes).mode()[0]
   else:
       predicted_label = "Unlabeled"
   ```
   - If one class has more matches, `mode()[0]` selects that majority class.
   - **Tie Resolution:** When two or more classes have equal frequencies (e.g., 1 match for Skill-Based Error and 1 match for Decision Error), `pd.Series.mode()` outputs the tied values in the exact order they appeared in `matched_classes`.
   - Because keywords are evaluated sequentially in dictionary order (`Skill-Based Error` $\rightarrow$ `Decision Error` $\rightarrow$ `Violation` $\rightarrow$ `Perceptual Error`), **ties are systematically resolved in favor of the earlier dictionary class**.

### 3.5 Evidence Availability
- Output columns: `weak_label` (`str`), `matched_keywords` (`str`, comma-separated).
- **Evidence Counts:** Does **not** produce per-class counts (e.g., `{'Decision Error': 2, 'Skill-Based Error': 1}`) or probability vectors. Only the single winning label string and joined string list are stored.

---

## 4. Text-Input Decision & Narrative Leakage Analysis

### 4.1 Decision: Preferred Route (Report 1 Alone)
- **Decision:** Extract and use **`Report 1_Narrative`** exclusively as the primary text column for both crisp baseline regeneration and forthcoming fuzzy methods.
- **Availability:** Readily accessible in the local Arrow cache:
  `C:\Users\parth\.cache\huggingface\datasets\elihoole___asrs-aviation-reports\default\0.0.0\f1e681e92cddae20d01fc498d685f1cf6a052d34\asrs-aviation-reports-train.arrow`
- **Execution Time:** Extracted and aligned with 100% unique `ASRS_<ACN>` IDs in under 2 seconds.

### 4.2 Rationale & Leakage Analysis
1. **Secondary Narrative Mixing (`Report 2_Narrative`):**
   In the legacy combined setup (`asrs_full.csv` and `data/processed/asrs_master.csv`), `Report 2_Narrative` was appended in 9,111 rows (23.57% of reports). `Report 2` is filed by secondary observers (co-pilots, ATC controllers, cabin crew, or ground dispatchers). Appending secondary reports introduces external observations and multi-party perspectives into a field designated `pilot_first_person`.
2. **Analyst Synopsis (`Report 1.2_Synopsis`):**
   `Report 1.2_Synopsis` was present in the fallback chain of `build_asrs_master.py`. Because `Report 1_Narrative` is 100% populated (38,655 rows), synopsis was never triggered. Synopsis constitutes human-analyst post-hoc summary text and would represent direct label leakage if used.
3. **Analyst Codes:**
   None of the 103 analyst-coded categorical columns (`Assessments_Contributing Factors`, `Events_Anomaly`, `Person 1.7_Human Factors`, etc.) were merged into text; they remain isolated in `raw_codes` JSON strings.
4. **Consistency Principle:**
   Running fuzzy labeling on `Report 1_Narrative` alone while scoring against combined-text crisp labels would invalidate cross-method comparisons. Regenerating crisp labels on pure `Report 1_Narrative` ensures an apples-to-apples experimental baseline.

---

## 5. Phase 1 Empirical Results: Report 1 Alone vs. Combined Text

### 5.1 Text Integrity Diagnostics (`Report 1_Narrative`)
- **Total Rows:** 38,655
- **Unique Report IDs:** 38,655 (0 duplicate IDs)
- **Null Texts:** 0
- **Empty Texts (`""`):** 0
- **Duplicate Narratives:** 5 (distinct incident reports sharing standard boilerplate notices)
- **Mean Character Length:** 1,586.7 characters (Median: 1,273, Min: 11, Max: 11,991)
- **Mean Word Count:** 280.2 words (Median: 225, Min: 2, Max: 2,345)

### 5.2 Crisp Label Distribution (Report 1 Alone vs. Combined)

| HFACS Class / Status | Report 1 Only Count | % of All Reports | % of Labeled (R1) | Combined (1+2) Count | % of All (Comb) | Difference (R1 - Comb) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Decision Error** | 16,906 | 43.74% | 80.28% | 17,875 | 46.24% | **-969 (-2.51%)** |
| **Unlabeled** | 17,597 | 45.52% | — | 16,609 | 42.97% | **+988 (+2.56%)** |
| **Skill-Based Error** | 2,826 | 7.31% | 13.42% | 2,833 | 7.33% | **-7 (-0.02%)** |
| **Violation** | 667 | 1.73% | 3.17% | 680 | 1.76% | **-13 (-0.03%)** |
| **Perceptual Error** | 659 | 1.70% | 3.13% | 658 | 1.70% | **+1 (+0.00%)** |
| **Total Labeled** | **21,058** | **54.48%** | **100.00%** | **22,046** | **57.03%** | **-988 (-2.55%)** |
| **Total Reports** | **38,655** | **100.00%** | — | **38,655** | **100.00%** | **0 (0.00%)** |

### 5.3 Secondary Report Impact Analysis
- **Exact Label Agreement:** 37,374 / 38,655 (**96.69%**)
- **Labels Altered by Secondary Text:** 1,281 (**3.31%**)
- **Mechanism:** In 988 cases, an otherwise `"Unlabeled"` primary report acquired a label (predominantly `Decision Error`, +969) purely because the co-pilot or air traffic controller's secondary narrative contained keywords such as `"weather"`, `"divert"`, or `"decided"`. Using Report 1 alone eliminates this secondary narrative distortion.

---

## 6. Baseline Reproduction & Cross-Domain Benchmark Results

The unchanged baseline evaluation script [`evaluation.py`](file:///d:/SC%20project/evaluation.py) was executed to establish reference performance.

### 6.1 Benchmark Protocol Summary
- **Training Set:** 17,636 labeled ASRS reports (80% stratified split).
- **In-Domain Test Set:** 4,410 held-out labeled ASRS reports (20% stratified split).
- **Cross-Domain Test Set:** [`ntsb_processed_test.csv`](file:///d:/SC%20project/ntsb_processed_test.csv) (2,000 Part 91 accident reports).
- **Features:** TF-IDF fitted strictly on ASRS training set (`max_features=10,000`, `ngram_range=(1,2)`).
- **Bootstrapping:** 1,000 bootstrap iterations for 95% Confidence Intervals.

### 6.2 Baseline Performance Metrics

#### Protocol A: Labeled-Only Benchmark (N_asrs=4,410, N_ntsb=1,413)

| Model | Held-Out ASRS Macro-F1 [95% CI] | NTSB Consistent Macro-F1 [95% CI] | Absolute Drop | Relative Drop (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.4104 [0.3849, 0.4391] | 0.4657 [0.4399, 0.4915] | **+0.0553** | **+13.47%** |
| **Linear SVM** | 0.6501 [0.6162, 0.6811] | 0.7396 [0.7104, 0.7678] | **+0.0895** | **+13.77%** |
| **Random Forest** | 0.2284 [0.2246, 0.2328] | 0.1781 [0.1729, 0.1836] | **-0.0503** | **-22.01%** |
| **Naive Bayes** | 0.2518 [0.2428, 0.2605] | 0.1893 [0.1794, 0.2019] | **-0.0625** | **-24.82%** |

#### Protocol B: All 2,000 NTSB Rows (Unlabeled Penalized as Wrong)

| Model | 4-Class Macro-F1 (FP Penalized) | 5-Class Macro-F1 (Unlabeled F1=0) [95% CI] |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.4164 | 0.3327 [0.3122, 0.3517] |
| **Linear SVM** | 0.6235 | 0.4987 [0.4751, 0.5197] |
| **Random Forest** | 0.1405 | 0.1125 [0.1080, 0.1167] |
| **Naive Bayes** | 0.1516 | 0.1211 [0.1134, 0.1300] |

#### Sensitivity Check: Disparate Preprocessor Labels vs. Consistent Weak Labeler

| Model | Consistent NTSB Macro-F1 | Preprocessor NTSB Macro-F1 [95% CI] | Procedure Gap |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.4657 | 0.2553 [0.2375, 0.2740] | **+0.2103** |
| **Linear SVM** | 0.7396 | 0.4541 [0.4322, 0.4768] | **+0.2855** |
| **Random Forest** | 0.1781 | 0.1000 [0.0938, 0.1063] | **+0.0781** |
| **Naive Bayes** | 0.1893 | 0.1050 [0.0979, 0.1125] | **+0.0843** |

---

## 7. Generated Phase 1 Artifacts

All outputs generated during Phase 1 have been organized under dedicated experiment directories:

1. [`experiments/fuzzy_asrs/run_phase1_baseline.py`](file:///d:/SC%20project/experiments/fuzzy_asrs/run_phase1_baseline.py): Script generating the single-reporter crisp baseline.
2. [`experiments/fuzzy_asrs/crisp_baseline_rows.csv`](file:///d:/SC%20project/experiments/fuzzy_asrs/crisp_baseline_rows.csv): 38,655 rows with columns `[report_id, input_source, text, crisp_label, matched_keywords]`.
3. [`experiments/fuzzy_asrs/crisp_label_distribution.csv`](file:///d:/SC%20project/experiments/fuzzy_asrs/crisp_label_distribution.csv): Summary table of class counts and percentages on Report 1 alone.
4. [`experiments/fuzzy_asrs/phase1_run.log`](file:///d:/SC%20project/experiments/fuzzy_asrs/phase1_run.log): Complete console execution log of the Phase 1 run.
5. [`experiments/baseline_audit/baseline_run.log`](file:///d:/SC%20project/experiments/baseline_audit/baseline_run.log): Execution log of `evaluation.py`.
6. [`experiments/baseline_audit/baseline_metrics.json`](file:///d:/SC%20project/experiments/baseline_audit/baseline_metrics.json) & [`baseline_results.csv`](file:///d:/SC%20project/experiments/baseline_audit/baseline_results.csv): Structured metrics for model comparison.

---

## 8. Known Methodological Limitations & Risks

1. **Rule-Derived Ground Truth:** All weak labels are generated from deterministic keyword heuristics (Madeira et al., 2021). Scores measure model alignment with lexical rules rather than clinical human psychological consensus.
2. **Lexical Inductive Bias:** TF-IDF feature weights overlap directly with the 39 seed keywords, inflating linear model performance when test sets share identical keyword annotations.
3. **Class Imbalance:** Decision Error comprises 80.28% of labeled reports, while Violation (3.17%) and Perceptual Error (3.13%) remain scarce.
4. **Missing Fuzzy Tooling:** `scikit-fuzzy` is not currently installed in the project virtual environment.
