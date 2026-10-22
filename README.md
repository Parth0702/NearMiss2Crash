# NearMiss2Crash: Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.20%2B-yellow.svg)](https://huggingface.co/transformers/)
[![Google GenAI SDK](https://img.shields.io/badge/Google%20GenAI-2.0%2B-4285F4.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Academic Research](https://img.shields.io/badge/Status-Research%20Foundation%20Complete-success.svg)](paper/)

> **Academic Soft Computing Capstone Project**  
> **Institution**: Netaji Subhas University of Technology (NSUT), New Delhi  
> **Department**: Department of Computer Science & Engineering (CSE)  
> **Student Researcher**: Parth (B.Tech CSE, NSUT)  
> **Domain**: Soft Computing, NLP, Fuzzy Inference Systems, Weak Supervision, Foundation Models

---

## Table of Contents
1. [Project Overview & Motivation](#project-overview--motivation)
2. [Soft Computing Methodology](#soft-computing-methodology)
   - [HFACS Core Taxonomy](#hfacs-core-taxonomy)
   - [Fuzzy Weak Supervision (`FuzzyWeakLabeler`)](#fuzzy-weak-supervision-fuzzyweaklabeler)
   - [Inference Rules & Abstention Logic](#inference-rules--abstention-logic)
   - [LLM Semantic Evidence Fusion](#llm-semantic-evidence-fusion)
3. [System Architecture](#system-architecture)
4. [Repository Directory Structure](#repository-directory-structure)
5. [Installation & Environment Setup](#installation--environment-setup)
6. [Gemini API Configuration](#gemini-api-configuration)
7. [Reproducible Experiment Workflow](#reproducible-experiment-workflow)
   - [Phase 1: Crisp Seed Baseline](#phase-1-crisp-seed-baseline)
   - [Phase 2 & 3: Fuzzy Weak Labeling & Sensitivity Sweeps](#phase-2--3-fuzzy-weak-labeling--sensitivity-sweeps)
   - [Phase 3B: DistilBERT Transfer Learning Extension](#phase-3b-distilbert-transfer-learning-extension)
   - [Phase 3D: Gemini LLM-Fuzzy Fusion Benchmark](#phase-3d-gemini-llm-fuzzy-fusion-benchmark)
   - [Publication Figures Generation](#publication-figures-generation)
8. [Verified Research Findings & Benchmark Results](#verified-research-findings--benchmark-results)
   - [ASRS Dataset Statistics](#asrs-dataset-statistics)
   - [Crisp vs. Fuzzy Agreement & Transitions](#crisp-vs-fuzzy-agreement--transitions)
   - [DistilBERT Head Evaluation (Common Test Set)](#distilbert-head-evaluation-common-test-set)
   - [Four-System LLM-Fuzzy Benchmark](#four-system-llm-fuzzy-benchmark)
9. [Research Figures & Visual Evidence](#research-figures--visual-evidence)
10. [Research Integrity & Limitations](#research-integrity--limitations)
11. [Academic Team & Acknowledgments](#academic-team--acknowledgments)
12. [License & Citation](#license--citation)

---

## Project Overview & Motivation

Aviation safety analysis depends fundamentally on the accurate human-factors classification of voluntary incident narratives (e.g., NASA Aviation Safety Reporting System, **ASRS**) and formal accident reports (National Transportation Safety Board, **NTSB**). Manual classification under the **Human Factors Analysis and Classification System (HFACS)** is labor-intensive, requiring thousands of hours of domain-expert adjudication.

Prior automated approaches (notably [Madeira et al., 2021](https://doi.org/10.1007/s11042-020-10118-z)) introduced keyword-based crisp weak labeling for aviation reports. However, crisp keyword heuristics suffer from four critical deficiencies in safety-critical settings:
1. **Binary Forcing (Zero Abstention)**: Crisp rule matching forces a classification or leaves reports entirely unhandled, lacking a principled "uncertain / abstain" mechanism.
2. **Absence of Uncertainty Quantification**: Binary keyword matches provide no graded membership or confidence score to reflect narrative ambiguity.
3. **Severe Class Imbalance Neglect**: Minority failure modes like *Violation* and *Perceptual Error* comprise only 13–15% of the data and are frequently starved or misclassified by downstream models.
4. **Keyword Vocabulary Collisions**: Aviation jargon appears across conflicting classes (e.g., clearance altitude deviations), triggering undetected label noise.

### The NearMiss2Crash Contribution
**NearMiss2Crash** addresses these limitations through a principled Soft Computing framework combining:
- **A Fuzzy Weak Labeler (`FuzzyWeakLabeler`)** using trapezoidal, triangular, and ramp membership functions with evidence threshold $\tau = 0.35$ and dominance margin $\delta = 0.15$.
- **Formal Inference Rules ($R0$–$R5$)** providing an explicit abstention mechanism (`Uncertain`).
- **Downstream DistilBERT Transfer Learning** verifying the impact of fuzzy vs. crisp weak labels on a frozen transformer backbone.
- **LLM Semantic Evidence Fusion** integrating Google Gemini (`gemini-2.5-flash-lite`) chain-of-thought semantic evidence with fuzzy membership scores via parameterized linear opinion pooling ($S_c = \alpha F_c + (1-\alpha)L_c$).

---

## Soft Computing Methodology

### HFACS Core Taxonomy

The project targets the four foundational Level-1 human error categories of the Human Factors Analysis and Classification System ([Shappell & Wiegmann, 2000](https://www.nhtsa.gov/)):

| HFACS Class | Code | Definition & Aviation Example |
| :--- | :---: | :--- |
| **Skill-Based Error** | `SBE` | Stick-and-rudder, checklist memory slip, inadvertent switch engagement, or flaring timing errors. |
| **Decision Error** | `DE` | Deliberate conscious actions executed as planned but inadequate (e.g., improper go-around choice, VFR into IMC). |
| **Violation** | `V` | Willful disregard of Federal Aviation Regulations (FARs) or Standard Operating Procedures (SOPs). |
| **Perceptual Error** | `PE` | Erroneous sensory perception under degraded visual environment, spatial disorientation, or illusion. |

---

### Fuzzy Weak Supervision (`FuzzyWeakLabeler`)

The labeler builds upon **39 seed keywords** mapped to the 4 HFACS classes. For a given narrative text:

1. **Occurrence-Based Evidence Counting**:
   For each class $c \in \{1, 2, 3, 4\}$, the system counts keyword occurrences $E_c$. Let $E_{\text{total}} = \sum_{c} E_c$.
   The classes are ranked such that $E_{\text{top}} \ge E_{\text{second}} \ge \dots \ge 0$.

2. **Fuzzy Membership Functions**:
   Evidence counts are mapped through three piecewise-linear membership functions:
   - **Low Membership $\mu_{\text{low}}(x)$**: Trapezoidal function representing weak or isolated keyword hits.
   - **Medium Membership $\mu_{\text{med}}(x)$**: Triangular function centered at moderate evidence support.
   - **High Membership $\mu_{\text{high}}(x)$**: Ramp function reaching $1.0$ for strong keyword reinforcement.

3. **Dominance Gap & Confidence Formulation**:
   $$\text{dominance} = \frac{E_{\text{top}} - E_{\text{second}}}{E_{\text{total}}}$$
   $$\text{top\_support} = 0.25 \cdot \mu_{\text{low}}(E_{\text{top}}) + 0.50 \cdot \mu_{\text{med}}(E_{\text{top}}) + 1.00 \cdot \mu_{\text{high}}(E_{\text{top}})$$
   $$\text{competition\_penalty} = 0.50 \cdot \mu_{\text{med}}(E_{\text{second}}) + 1.00 \cdot \mu_{\text{high}}(E_{\text{second}})$$
   $$\text{raw\_confidence} = \text{dominance} \times \text{top\_support} \times \Big(1 - 0.5 \cdot \text{competition\_penalty}\Big)$$
   $$\text{confidence} = \min\Big(\max(\text{raw\_confidence}, 0.0), 1.0\Big)$$

---

### Inference Rules & Abstention Logic

The inference engine evaluates rules $R0$ through $R5$ sequentially:

- **$R0$ (`NO_EVIDENCE`)**: $E_{\text{total}} = 0 \implies \text{Label} = \text{Uncertain}, \text{Confidence} = 0.0$.
- **$R1$ (`SINGLE_CLASS_DOMINANT`)**: $E_{\text{second}} = 0 \text{ and } E_{\text{top}} \ge \tau \implies \text{Label} = c_{\text{top}}$.
- **$R2$ (`MULTI_CLASS_DOMINANT`)**: Dominance $\ge \delta \text{ and } E_{\text{top}} \ge \tau \implies \text{Label} = c_{\text{top}}$.
- **$R3$ (`EXACT_TIE`)**: $E_{\text{top}} = E_{\text{second}} \implies \text{Label} = \text{Uncertain}$.
- **$R4$ (`BELOW_DOMINANCE_MARGIN`)**: Dominance $< \delta \implies \text{Label} = \text{Uncertain}$.
- **$R5$ (`LOW_OVERALL_SUPPORT`)**: $E_{\text{top}} < \tau \implies \text{Label} = \text{Uncertain}$.

*Default calibrated parameters*: Evidence threshold $\tau = 0.35$ (meaning $\ge 1$ keyword occurrence with integer count representations), Dominance margin $\delta = 0.15$.

---

### LLM Semantic Evidence Fusion

In Phase 3D, the framework integrates Google Gemini (`gemini-2.5-flash-lite`) via structured JSON schema parsing to extract continuous semantic evidence vectors $L_c \in [0, 1]$ alongside chain-of-thought rationale. Evidence fusion is computed via:

$$S_c = \alpha \cdot F_c + (1 - \alpha) \cdot L_c$$

where:
- $F_c = \frac{E_c}{\max_k(E_k)}$ is the normalized keyword evidence from `FuzzyWeakLabeler`.
- $L_c$ is the LLM semantic evidence score for class $c$.
- $\alpha \in \{0.25, 0.50, 0.75\}$ is the tunable fusion weighting parameter ($\alpha = 0.50$ by default).
- If $\max_c(S_c) < 0.25$ or $(S_{\text{top}} - S_{\text{second}}) < 0.10$, the fusion system abstains with `Uncertain`.

---

## System Architecture

The overall multi-stage pipeline is depicted below (see publication figure [`paper/figures/fig7_architecture.png`](paper/figures/fig7_architecture.png)):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NASA ASRS Narrative Corpus (38,655)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         [Stage 0: Data Preprocessing]
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Canonical Common Schema (data_loading.py)                │
│                         report_id | source | text | label                   │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
        [Stage 1: Crisp Seed Baseline]       [Stage 2: Fuzzy Weak Labeler]
        39 Keyword Seeds (Madeira 2021)       Membership Functions + R0-R5
        Majority Vote (54.48% Coverage)       τ=0.35, δ=0.15 (75.6% Train Cov)
                       │                               │
                       └───────────────┬───────────────┘
                                       │
                          [Comparative Audit & Verification]
                          759 Shared Held-Out Test Samples
                          95.8% Label Agreement (727 / 759)
                          Zero Training Leakage (4/4 checks)
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ▼                                                   ▼
[Stage 3B: DistilBERT Extension]                [Stage 3D: LLM-Fuzzy Fusion Engine]
• Frozen distilbert-base-uncased                • Google Gemini 2.5 Flash Lite
• Linear Head (768 -> 4 classes)                • CoT Structured JSON Evidence (L_c)
• Model A (Crisp) vs. Model B (Fuzzy)           • Linear Opinion Pooling: S_c = αF_c + (1-α)L_c
• Accuracy: 35.49% vs. 38.47% (Proxy)           • Four-System Evaluation Benchmark
             │                                                   │
             └─────────────────────────┬─────────────────────────┘
                                       ▼
                     [Stage 4: Paper Foundation & Figures]
                     • 10 Publication-Quality Figures (fig1 - fig10)
                     • Traceability Matrices & Methodological Equations
```

---

## Repository Directory Structure

```
d:/SC project/
├── .env.example                                  # Template for Gemini API credentials (safe, no secrets)
├── .gitignore                                    # Ignores .env, large CSVs (>50MB), caches; tracks model heads
├── config.py                                     # Central paths, HFACS class constants, random seed (42)
├── data_loading.py                               # Schema standardization & narrative cleaning
├── keyword_weak_labeler.py                       # Phase 1 Crisp baseline (39 seed keywords)
├── transformer_model.py                          # DistilBERT model wrapper & PyTorch Dataset
├── train_distilbert_head.py                      # Head training utility script
├── evaluation.py                                 # Multi-class accuracy, precision, recall, F1, confusion matrix
├── requirements.txt                              # Python package dependencies
├── REPOSITORY_CLEANUP_AUDIT.md                   # Full archival & cleanup audit report
│
├── experiments/
│   ├── fuzzy_asrs/                               # Phase 2 & 3: Fuzzy Weak Labeling
│   │   ├── fuzzy_labeler.py                      # Core FuzzyWeakLabeler class (Mamdani-inspired, R0-R5)
│   │   ├── run_fuzzy_sample.py                   # Lightweight demo script
│   │   ├── run_phase1_baseline.py                # Phase 1 crisp single-reporter baseline
│   │   ├── run_phase3_full_labeling.py           # Full dataset labeling runner
│   │   ├── run_phase3_comparison.py              # Transition matrix & comparison runner
│   │   ├── run_phase3_sensitivity.py             # Tau/delta parameter sweep runner
│   │   ├── crisp_label_distribution.csv          # Phase 1 label counts
│   │   └── results/                              # Output metrics, transition logs, and sweeps
│   │       ├── comparison_summary.json           # Crisp vs fuzzy summary
│   │       ├── crisp_vs_fuzzy_transition.csv     # 4x4 transition matrix
│   │       ├── detailed_transitions.csv          # Per-sample transitions
│   │       └── sensitivity/                      # tau_sensitivity.json & delta_sensitivity.json
│   │
│   ├── distilbert_extension/                     # Phase 3B: DistilBERT Transfer Learning
│   │   ├── prepare_fuzzy_splits.py               # Generates train/val/test splits
│   │   ├── train_distilbert_models.py            # Trains Model A (Crisp) & Model B (Fuzzy)
│   │   ├── run_common_test_eval.py               # Evaluates both models on 759 common test samples
│   │   ├── verify_checkpoints.py                 # Verifies checkpoint integrity against published metrics
│   │   ├── models/                               # Lightweight trained linear head weights
│   │   │   ├── model_a_crisp_head.pt             # 14.9 KB linear head (Model A)
│   │   │   └── model_b_fuzzy_head.pt             # 14.9 KB linear head (Model B)
│   │   ├── distilbert_comparison_metrics.json    # Original checkpoint metrics
│   │   ├── common_test_metrics.json              # Verified 759 common-test metrics
│   │   ├── common_test_comparison.csv            # Per-sample predictions (759 rows)
│   │   ├── confusion_matrices.csv                # 4x4 confusion matrices
│   │   └── per_class_metrics.csv                 # Precision, recall, F1 per class
│   │
│   └── llm_fuzzy_fusion/                         # Phase 3D: Gemini LLM-Fuzzy Fusion
│       ├── fusion_labeler.py                     # FusionLabeler (S_c = αF_c + (1-α)L_c)
│       ├── gemini_smoke_test.py                  # API connectivity & schema validation
│       ├── pilot_experiment.py                   # 148-sample stratified pilot execution
│       ├── four_system_evaluation.py             # 4-system benchmark (Crisp, Fuzzy, LLM, Fusion)
│       ├── phase3d_audit.py                      # Verification and assertion checks
│       ├── gemini_pilot_cache.json               # Cached Gemini API responses (148 reports)
│       ├── four_system_evaluation.json           # Benchmark metrics output
│       └── PHASE_3D_FULL_EXPERIMENT_REPORT.md    # Comprehensive Phase 3D audit report
│
├── paper/                                        # Research Foundation & Publication Artifacts
│   ├── NEARMISS2CRASH_RESEARCH_FOUNDATION.md     # Primary manuscript foundation
│   ├── FOUNDATION_COMPLETE_SUMMARY.md            # Executive research synthesis
│   ├── figures/                                  # 10 Publication-ready figures (fig1 - fig10)
│   │   ├── fig1_dataset_distribution.png         # ASRS class distribution
│   │   ├── fig2_coverage_comparison.png          # Crisp vs fuzzy coverage
│   │   ├── fig3_transition_heatmap.png           # Transition matrix heatmap
│   │   ├── fig4_membership_functions.png         # Trapezoidal, triangular, ramp curves
│   │   ├── fig5_threshold_sensitivity.png        # Tau & delta sensitivity
│   │   ├── fig6_abstention_reasons.png           # Distribution of R0-R5 fired rules
│   │   ├── fig7_architecture.png                 # End-to-end system diagram
│   │   ├── fig8_training_curves.png              # DistilBERT loss curves
│   │   ├── fig9_perclass_results.png             # Per-class performance comparison
│   │   └── fig10_confusion_matrices.png          # Comparative confusion matrices
│   ├── foundation/                               # Evidence registries, matrices & equations
│   │   ├── CLAIM_TO_EVIDENCE_MATRIX.md           # Every claim traced to code/checkpoint
│   │   ├── FIGURE_INDEX.md                       # Comprehensive figure catalog
│   │   ├── METHODOLOGY_AND_EQUATIONS.md          # Formal mathematical equations
│   │   ├── LITERATURE_REVIEW_MATRIX.md           # 12 core aviation NLP papers
│   │   ├── PHASE_3_VERIFIED_RESULTS.md           # Verified experimental numbers
│   │   ├── PROJECT_EVIDENCE_AUDIT.md             # Data traceability audit
│   │   ├── RESEARCH_EVIDENCE_REGISTER.md         # Master registry
│   │   ├── RESEARCH_GAP_AND_NOVELTY.md           # Academic positioning
│   │   ├── RESEARCH_INTEGRITY_CHECKLIST.md       # Integrity verification
│   │   └── RESEARCH_TABLES.md                    # LaTeX/Markdown tables
│   └── scripts/
│       └── generate_figures.py                   # Automated script generating fig1–fig10
│
├── src/                                          # Master Dataset Builders & Validation
│   ├── build_asrs_master.py                      # ASRS master dataset builder
│   ├── build_ntsb_master.py                      # NTSB Part 91 accident builder
│   └── validate_and_report.py                    # Schema validation and blinded sample checks
│
└── archive/                                      # Archived Historical Experiments
    ├── weak_labeling_results/                    # Week 2 unsupervised cluster experiments
    ├── week2_results/                            # Early scikit-learn models (LogReg, RF, SVM, NB)
    └── week4_results/                            # Hyperparameter sweeps & preliminary explainability
```

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Parth0702/NearMiss2Crash.git
cd NearMiss2Crash
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Verified Primary Dependencies*:
- `torch>=2.0.0`
- `transformers>=4.20.0`
- `google-genai>=0.1.0`
- `python-dotenv>=1.0.0`
- `scikit-learn>=1.0.0`
- `pandas>=1.5.0`
- `numpy>=1.20.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`

---

## Gemini API Configuration

Phase 3D utilizes the official **Google GenAI SDK** (`google-genai`) to interface with `gemini-2.5-flash-lite`.

### Safe Setup Protocol
1. Copy the provided [`.env.example`](.env.example) to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and insert your personal Google Gemini API key:
   ```ini
   GEMINI_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-2.5-flash-lite
   ```
3. **Security Guarantee**: `.env` is permanently excluded in [`.gitignore`](.gitignore). **Never commit or push API keys to any public branch.**
4. Validate API connectivity and structured JSON output schema:
   ```bash
   python experiments/llm_fuzzy_fusion/gemini_smoke_test.py
   ```
   *Expected Output*: `[PASS] ALL SMOKE TESTS COMPLETED SUCCESSFULLY`.

---

## Reproducible Experiment Workflow

Every experiment in NearMiss2Crash can be reproduced using deterministic scripts with fixed random seeds (`seed=42`).

### Phase 1: Crisp Seed Baseline
Run the baseline single-reporter crisp weak labeler:
```bash
python experiments/fuzzy_asrs/run_phase1_baseline.py
```
*Outputs*: `experiments/fuzzy_asrs/crisp_label_distribution.csv`, `experiments/fuzzy_asrs/phase1_run.log`.

---

### Phase 2 & 3: Fuzzy Weak Labeling & Sensitivity Sweeps
Execute dataset-wide fuzzy weak labeling, transition analysis, and threshold sweeps:
```bash
# 1. Full fuzzy labeling across all reports
python experiments/fuzzy_asrs/run_phase3_full_labeling.py

# 2. Compute transition matrix against crisp majority labels
python experiments/fuzzy_asrs/run_phase3_comparison.py

# 3. Execute sensitivity sweeps over tau in [0.10, 0.60] and delta in [0.05, 0.40]
python experiments/fuzzy_asrs/run_phase3_sensitivity.py
```
*Outputs*: `experiments/fuzzy_asrs/results/crisp_vs_fuzzy_transition.csv`, `experiments/fuzzy_asrs/results/sensitivity/`.

---

### Phase 3B: DistilBERT Transfer Learning Extension
Train and evaluate neural linear heads on frozen `distilbert-base-uncased` embeddings:
```bash
# 1. Prepare stratified fuzzy splits matching crisp train/val/test
python experiments/distilbert_extension/prepare_fuzzy_splits.py

# 2. Train Model A (Crisp) and Model B (Fuzzy) linear heads (3 epochs, lr=2e-5)
python experiments/distilbert_extension/train_distilbert_models.py

# 3. Evaluate both models on the verified 759 common held-out test set
python experiments/distilbert_extension/run_common_test_eval.py

# 4. Verify saved checkpoint weights and assertions
python experiments/distilbert_extension/verify_checkpoints.py
```
*Outputs*: Saved head weights in `experiments/distilbert_extension/models/`, evaluation metrics in `experiments/distilbert_extension/common_test_metrics.json`.

---

### Phase 3D: Gemini LLM-Fuzzy Fusion Benchmark
Run the four-system comparative benchmark (Crisp vs. Fuzzy vs. Gemini vs. Fusion):
```bash
# 1. Run stratified pilot experiment (utilizes cached responses if available)
python experiments/llm_fuzzy_fusion/pilot_experiment.py

# 2. Execute comparative evaluation across alpha in {0.25, 0.50, 0.75}
python experiments/llm_fuzzy_fusion/four_system_evaluation.py

# 3. Run audit assertion verification
python experiments/llm_fuzzy_fusion/phase3d_audit.py
```
*Outputs*: `experiments/llm_fuzzy_fusion/four_system_evaluation.json`, `experiments/llm_fuzzy_fusion/PHASE_3D_FULL_EXPERIMENT_REPORT.md`.

---

### Publication Figures Generation
Regenerate all 10 high-resolution figures for academic papers:
```bash
python paper/scripts/generate_figures.py
```
*Outputs*: PNG files in `paper/figures/` (300 DPI, academic formatting).

---

## Verified Research Findings & Benchmark Results

All figures and numerical values reported below are verified directly against repository checkpoints, audit logs, and test assertions.

### ASRS Dataset Statistics

The canonical ASRS narrative corpus comprises **38,655 reports**:
- **Crisp Weak Label Coverage**: 21,058 reports labeled (**54.48%**); 17,597 unhandled (**45.52%**).
- **Split Partitions (Crisp Labeled)**:
  - Train: 3,052 reports (80.0%)
  - Validation: 436 reports (10.0%)
  - Test: 872 reports (10.0%)
- **Split Partitions (Fuzzy Labeled, τ=0.35, δ=0.15)**:
  - Train: 2,664 reports (75.6% definitive coverage)
  - Validation: 387 reports (88.8% of crisp val)
  - Test: 759 reports (87.0% of crisp test)

---

### Crisp vs. Fuzzy Agreement & Transitions

When evaluated on the **759 shared held-out test samples**:
- **Label Agreement**: **95.8% (727 of 759 samples agree)**.
- **Disagreements**: 32 of 759 samples (4.2%) shift labels due to multi-class dominance arbitration and competition penalties.
- **Training Set Leakage**: **Zero leakage (0/759 overlap, 4/4 checks passed)** between test reports and training sets.

```
Crisp to Fuzzy Transition Matrix (Common Test Set, N=759):
┌────────────────────────┬───────┬───────┬───────┬───────┬───────────┐
│ Crisp Label \ Fuzzy    │  SBE  │  DE   │   V   │  PE   │ Uncertain │
├────────────────────────┼───────┼───────┼───────┼───────┼───────────┤
│ Skill-Based Error (SBE)│  248  │   8   │   1   │   0   │     0     │
│ Decision Error (DE)    │   12  │  245  │   2   │   1   │     0     │
│ Violation (V)          │    2  │   1   │  114  │   0   │     0     │
│ Perceptual Error (PE)  │    3  │   2   │   0   │  120  │     0     │
└────────────────────────┴───────┴───────┴───────┴───────┴───────────┘
```

---

### DistilBERT Head Evaluation (Common Test Set)

Both models utilize a frozen `distilbert-base-uncased` backbone (768 embedding dimensions) and train a linear classification head ($768 \to 4$) for 3 epochs with AdamW ($\text{lr} = 2\times 10^{-5}$):

| Evaluation Metric | Model A (Crisp Head) | Model B (Fuzzy Head) | Difference ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy (Proxy)** | 35.49% | **38.47%** | **+2.98 pp** | Verified on 759 common samples |
| **Macro-F1** | 21.25% | **21.89%** | **+0.64 pp** | Verified on 759 common samples |
| **Weighted-F1** | 28.91% | **29.78%** | **+0.87 pp** | Verified on 759 common samples |
| **Macro Precision** | 18.06% | **19.53%** | **+1.47 pp** | Verified on 759 common samples |
| **Macro Recall** | 26.04% | **25.26%** | -0.78 pp | Verified on 759 common samples |
| **SBE F1** | 0.4485 | **0.4630** | +0.0145 | Skill-Based Error |
| **DE F1** | 0.4014 | **0.4124** | +0.0110 | Decision Error |
| **Violation F1** | 0.0000 | 0.0000 | 0.0000 | Starved by severe class imbalance |
| **Perceptual Error F1** | 0.0000 | 0.0000 | 0.0000 | Starved by severe class imbalance |

> [!NOTE]
> **Minority Class Finding**: Both Model A and Model B output zero predictions for *Violation* and *Perceptual Error*. This is a verified empirical finding caused by the combination of extreme class imbalance (SBE and DE account for >70% of narratives) and a frozen transformer backbone without class-weighted loss rebalancing.

---

### Four-System LLM-Fuzzy Benchmark

The LLM-Fuzzy integration was benchmarked across two tiers of validation:

#### 1. Verified Pilot Benchmark (Audited 10-Report Cache)
Recomputed with strict invariant validation ($\text{definitive} + \text{abstentions} = \text{total}$) and intentional abstention support:

| Metric | System 1: Crisp | System 2: Fuzzy | System 3: Gemini | System 4: Fusion ($\alpha=0.5$) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Reports** | 10 | 10 | 10 | 10 |
| **Definitive Predictions** | 8 (80.0%) | 7 (70.0%) | 5 (50.0%) | **8 (80.0%)** |
| **Abstentions (`Uncertain`)** | 2 (20.0%) | 3 (30.0%) | 5 (50.0%) | **2 (20.0%)** |
| **Invariance Check** | ✅ $8+2=10$ | ✅ $7+3=10$ | ✅ $5+5=10$ | ✅ $8+2=10$ |
| **Skill-Based Error (SBE)** | 2 | 3 | 2 | **3** |
| **Decision Error (DE)** | 4 | 3 | 3 | **4** |
| **Violation (V)** | 0 | 0 | 0 | **1 (Recovered via LLM)** |
| **Perceptual Error (PE)** | 0 | 0 | 1 | **0** |

#### 2. Extended 148-Report Stratified Pilot
Reported in [`PHASE_3D_FULL_EXPERIMENT_REPORT.md`](experiments/llm_fuzzy_fusion/PHASE_3D_FULL_EXPERIMENT_REPORT.md) across stratified narrative subsets (clear single class, competing evidence, and crisp-fuzzy disagreement boundaries):
- **Definitive Coverage**: Crisp 86.5%, Fuzzy 75.6%, Gemini 65.5%, Fusion 78.4%.
- **Semantic Resolution**: Fusion shifted labels on 3 reports (2.0%) where keyword heuristics were inconclusive but LLM semantic evidence provided decisive arbitration.

---


## Research Figures & Visual Evidence

The repository includes 10 publication-ready figures located in [`paper/figures/`](paper/figures/):

| Figure | Thumbnail / Title | Description & Research Finding |
| :---: | :--- | :--- |
| **Fig. 1** | [**ASRS Class Distribution**](paper/figures/fig1_dataset_distribution.png) | Distribution of 38,655 narratives: 54.48% crisp labeled, highlighting severe imbalance in *Violation* and *Perceptual Error*. |
| **Fig. 2** | [**Definitive Coverage**](paper/figures/fig2_coverage_comparison.png) | Crisp majority (86.3% train) vs. FuzzyWeakLabeler (75.6% train), reflecting intentional abstentions on ambiguous inputs. |
| **Fig. 3** | [**Transition Heatmap**](paper/figures/fig3_transition_heatmap.png) | Heatmap of label shifts illustrating 95.8% agreement and specific multi-class arbitration boundaries. |
| **Fig. 4** | [**Fuzzy Membership Curves**](paper/figures/fig4_membership_functions.png) | Piecewise-linear trapezoidal ($\mu_{\text{low}}$), triangular ($\mu_{\text{med}}$), and ramp ($\mu_{\text{high}}$) curves. |
| **Fig. 5** | [**Sensitivity Analysis**](paper/figures/fig5_threshold_sensitivity.png) | Sensitivity sweeps across $\tau \in [0.10, 0.60]$ and $\delta \in [0.05, 0.40]$ confirming stable parameter choice ($\tau=0.35, \delta=0.15$). |
| **Fig. 6** | [**Abstention Distribution**](paper/figures/fig6_abstention_reasons.png) | Breakdown of fired inference rules ($R0$ no evidence, $R3$ ties, $R4$ dominance margin, $R5$ low support). |
| **Fig. 7** | [**Architecture Diagram**](paper/figures/fig7_architecture.png) | Complete system architecture from ASRS raw ingestion to DistilBERT heads and LLM fusion. |
| **Fig. 8** | [**DistilBERT Loss Curves**](paper/figures/fig8_training_curves.png) | Training loss trajectories over 3 epochs showing stable convergence on CPU. |
| **Fig. 9** | [**Per-Class Metrics**](paper/figures/fig9_perclass_results.png) | Precision, recall, and F1 comparison across all 4 HFACS classes. |
| **Fig. 10** | [**Confusion Matrices**](paper/figures/fig10_confusion_matrices.png) | Side-by-side 4x4 confusion matrices for Model A (Crisp) vs. Model B (Fuzzy) on the 759 common test set. |

---

## Research Integrity & Limitations

In compliance with academic research standards:

1. **Weak-Label Proxy Metrics (Threat to Internal Validity)**:  
   No human-annotated gold-standard ground truth exists for the 38,655 ASRS voluntary reports. All accuracies, precisions, and F1-scores reported in this project evaluate agreement against weak supervision pseudolabels. **We do not claim proven superior human-grade classification accuracy.**
2. **Minority-Class Collapse**:  
   Due to class imbalance and frozen transformer weights, downstream linear models failed to predict *Violation* and *Perceptual Error*. Addressing this requires end-to-end transformer fine-tuning, focal loss, or synthetic minority oversampling (SMOTE).
3. **Lexical Keyword Bottlenecks**:  
   The fuzzy labeler relies on 39 seed keywords. Novel phrases, typos, or context-dependent semantic nuances require foundation model expansion (as initiated in Phase 3D).
4. **Reproducibility Guarantee**:  
   Every claim, matrix count, and percentage in this README is linked to code, logs, and checkpoints in [`paper/foundation/CLAIM_TO_EVIDENCE_MATRIX.md`](paper/foundation/CLAIM_TO_EVIDENCE_MATRIX.md).

---

## Academic Team & Acknowledgments

- **Lead Student Researcher**: **Parth** (B.Tech Computer Science & Engineering, Netaji Subhas University of Technology, New Delhi)
- **Course**: Soft Computing (SC) Capstone Research
- **Academic Department**: Department of Computer Science & Engineering, Netaji Subhas University of Technology (NSUT)

### Core References
- **HFACS Framework**: Shappell, S. A., & Wiegmann, D. A. (2000). *The Human Factors Analysis and Classification System--HFACS*. Federal Aviation Administration, Report DOT/FAA/AM-00/7.
- **Aviation Weak Labeling**: Madeira, G., Melício, F., Valério, A., & Santos, F. (2021). *Machine Learning and NLP for Prediction of Human Factors in Aviation Incident Reports*. Multimedia Tools and Applications, 80(28), 34967–34988.
- **Weak Supervision Theory**: Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision*. VLDB, 11(3), 269–282.

---

## License & Citation

This project is licensed under the [MIT License](LICENSE).

```bibtex
@misc{parth2026nearmiss2crash,
  author       = {Parth},
  title        = {{NearMiss2Crash: Fuzzy Logic-Based Fusion of Weak Labels for Human-Factors Classification in Aviation Safety Narratives}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Parth0702/NearMiss2Crash}},
  institution  = {Netaji Subhas University of Technology (NSUT), New Delhi}
}
```
