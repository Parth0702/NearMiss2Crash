# Phase 3B DistilBERT Asset Audit

## A. Executive Summary
This audit identifies existing DistilBERT assets in the NearMiss2Crash project, assesses their training status, and determines what is needed for the crisp-vs-fuzzy DistilBERT comparison.

**Status**: No Phase 3B-specific assets exist. The project has a trained DistilBERT checkpoint from Week 3 cross-domain experiments, but it was trained on NTSB data with different weak labels, not on ASRS crisp or fuzzy labels.

## B. Existing DistilBERT Assets

| Asset | Path | Status | Notes |
|-------|------|--------|-------|
| Trained checkpoint | `week3_results/best_model_distilbert-base-uncased.pt` | **TRAINED** | Trained on NTSB Part 91 data (2,000 samples) using a 3-class or 4-class label mapping from preprocessor/weak labels |
| Model definition | `transformer_model.py` | **AVAILABLE** | `HFACSClassifier` class for multi-label (sigmoid) or single-label (softmax) classification |
| Training script (NTSB) | `train_distilbert_head.py` | **AVAILABLE** | Trains classification head on frozen DistilBERT embeddings; uses NTSB data |
| Training script (ASRS) | `transformer_model.py` main() | **AVAILABLE** | Full fine-tuning pipeline for ASRS weak labels (majority/keyword/cluster) |
| ASRS weak label datasets | `weak_labeling_results/dataset_majority_*.csv` | **AVAILABLE** | Pre-split train/val/test for crisp weak labels (keyword, cluster, majority) |
| Fuzzy label datasets | None | **MISSING** | No pre-split datasets with fuzzy labels exist |

## C. Model Architecture Verification

| Property | Value | Verified |
|----------|-------|----------|
| Base model | `distilbert-base-uncased` | ✓ |
| Hidden size | 768 | ✓ |
| Output classes | 4 (HFACS core) | ✓ |
| Classifier head | Single `nn.Linear(768, 4)` | ✓ |
| Loss function | `BCEWithLogitsLoss` (multi-label) or `CrossEntropyLoss` (single-label) | ✓ |
| Max sequence length | 128 | ✓ |
| Tokenizer | `AutoTokenizer.from_pretrained("distilbert-base-uncased")` | ✓ |

## D. Existing Trained Model Assessment

### `week3_results/best_model_distilbert-base-uncased.pt`
- **Training data**: NTSB Part 91 (2,000 samples from `ntsb_processed_test.csv`)
- **Training method**: Frozen DistilBERT embeddings + trainable linear head (50 epochs)
- **Label source**: `ntsb_processed_test.csv` column `label` mapped to 4 HFACS classes
- **Training provenance**: Verified via `train_distilbert_head.py` execution
- **Compatibility with Phase 3B**: **INCOMPATIBLE**
  - Trained on different domain (NTSB vs ASRS)
  - Different label generation method (NTSB preprocessor vs ASRS crisp/fuzzy)
  - Not trained on ASRS crisp or fuzzy labels
  - Cannot be used for crisp-vs-fuzzy comparison without retraining

### `transformer_model.py` training outputs
- **Training data**: ASRS weak labels from `weak_labeling_results/dataset_majority_*.csv`
- **Training method**: Full fine-tuning (3 epochs, AdamW, lr=2e-5)
- **Label source**: Majority/keyword/cluster weak labels
- **Status**: Checkpoint exists at `week3_results/best_model_distilbert-base-uncased.pt` but it's the NTSB one; ASRS training outputs may not have been saved separately

## E. Dataset Requirements for Phase 3B

| Dataset | Status | Action Needed |
|---------|--------|---------------|
| ASRS crisp labels (train/val/test) | Available in `weak_labeling_results/dataset_majority_*.csv` | ✓ Use existing splits |
| ASRS fuzzy labels (train/val/test) | **MISSING** | Create by applying `FuzzyWeakLabeler` to ASRS narratives, then split with same seed |
| Fixed test set IDs | Available from crisp splits | ✓ Reuse same test IDs for fair comparison |

## F. Dependencies

| Package | Version | Available |
|---------|---------|-----------|
| torch | 2.14.1 | ✓ |
| transformers | 5.19.0 | ✓ |
| scikit-learn | 1.9.1 | ✓ |
| pandas | 3.0.6 | ✓ |
| numpy | 2.5.4 | ✓ |

## G. Hardware Resources

| Resource | Status |
|----------|--------|
| CUDA | Available if `torch.cuda.is_available()` |
| CPU | Fallback available |
| RAM | Sufficient for DistilBERT (batch_size=16) |
| Disk | Checkpoint ~265 MB |

## H. Conclusions for Phase 3B

1. **No existing crisp-trained ASRS DistilBERT model** - the `week3_results` checkpoint is NTSB-trained
2. **No existing fuzzy-trained ASRS DistilBERT model** - never created
3. **Both models need to be trained** from scratch using the Phase 3B pipeline
4. **Training infrastructure exists** - `transformer_model.py` provides the complete training loop
5. **ASRS data splits exist for crisp labels** - `weak_labeling_results/dataset_majority_*.csv`
6. **Fuzzy label splits need to be created** - apply `FuzzyWeakLabeler` to ASRS data, then split with seed 42
7. **No Phase 3B reports exist yet** - will be created during this phase

## I. Recommended Next Steps

1. Create ASRS fuzzy label dataset by running `FuzzyWeakLabeler` on all ASRS narratives
2. Create train/val/test splits for fuzzy labels using same IDs as crisp splits
3. Train Model A (crisp) using `transformer_model.py` pipeline with crisp data
4. Train Model B (fuzzy) using same pipeline with fuzzy data
5. Evaluate both on common held-out test set
6. Generate comparison report

---

**Audit Date**: October 22, 2026  
**Auditor**: Automated verification