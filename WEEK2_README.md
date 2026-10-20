# Week 2: Supervised Learning Pipeline

Week 2 implements the training and evaluation of supervised classifiers using weakly labeled HFACS dataset.

## Overview

Previous Week (Week 1) built:
- Weak labeling framework (keyword-based + cluster-based)
- Generated weakly labeled datasets

Week 2 (Current) builds:
- ✅ Dataset preparation module (`week2_dataset_preparation.py`)
- ✅ Classifier training module (`week2_classifier_training.py`)
- ✅ Model evaluation module (`week2_evaluation.py`)
- 🔄 Comprehensive experiment notebook (`week2_experiments.ipynb` - next)

## Project Structure

```
SC project/
├── weak_labeling_framework.py          # Week 1: Unified framework
├── week2_dataset_preparation.py        # Week 2: Dataset preparation
├── week2_classifier_training.py        # Week 2: Classifier training
├── week2_evaluation.py                 # Week 2: Model evaluation
├── week2_experiments.ipynb             # Week 2: Interactive experiments
├── datasets/                           # Prepared datasets (output)
├── week2_results/                      # Trained models & results (output)
└── WEAK_LABELING_COMPARISON.md         # Detailed comparison doc
```

## Quick Start

### Prerequisites

Ensure you've run Week 1 weak labeling first:
```bash
python weak_labeling_framework.py
```

This generates `weak_labeling_results/combined_labels_*.csv`.

### Step 1: Prepare Labeled Datasets

```bash
python week2_dataset_preparation.py
```

**Output:**
```
weak_labeling_results/
├── dataset_majority_train.csv              # 720 samples
├── dataset_majority_validation.csv         # 80 samples
├── dataset_majority_test.csv               # 200 samples
├── dataset_keyword_first_train.csv         # ...
├── dataset_keyword_first_validation.csv
├── dataset_keyword_first_test.csv
├── dataset_cluster_first_train.csv
├── dataset_cluster_first_validation.csv
├── dataset_cluster_first_test.csv
├── dataset_stats_*.json                    # Class distributions
└── weights_*.npy                           # Balancing weights
```

### Step 2: Train Classifiers

```bash
python week2_classifier_training.py
```

**Trains 4 classifiers:**
1. Logistic Regression
2. Random Forest
3. SVM (Support Vector Machine)
4. Naive Bayes

**Output:**
```
week2_results/
├── classifier_logistic_regression_report.txt
├── classifier_random_forest_report.txt
├── classifier_svm_report.txt
├── classifier_naive_bayes_report.txt
└── model_comparison.csv                    # Performance comparison
```

### Step 3: Evaluate Models

```bash
python week2_evaluation.py
```

**Performs:**
- Cross-validation across strategies
- Test set evaluation
- Model comparison
- Error analysis
- Visualization generation

**Output:**
```
week2_results/
├── strategy_comparison.csv                  # Strategy vs model performance
├── model_comparison_visualization.png      # Performance charts
└── results_summary.json                     # Overall summary
```

### Interactive Experiments

Use Jupyter Notebook for interactive exploration:
```bash
jupyter notebook week2_experiments.ipynb
```

## Key Features

### 1. Dataset Preparation

Handles:
- Loading weakly labeled data from multiple strategies
- Calculating class-balancing weights (inverse frequency)
- Stratified train/validation/test split (72%/8%/20%)
- Handling class imbalance
- Quality checks and validation

**Class Distribution Handling:**
```
Strategy "majority":
  Skill-Based Error:    35.2%
  Decision Error:       28.5%
  Violation:            18.3%
  Perceptual Error:     18.0%
```

### 2. Classifier Training

Supports 4 models:

**Logistic Regression:**
- Linear classifier, fast, interpretable
- Handles imbalanced data via class weights
- Suitable as baseline

**Random Forest:**
- Ensemble of decision trees
- Handles non-linear relationships
- Better accuracy than Logistic Regression
- Slower, harder to interpret

**SVM (Linear Kernel):**
- Good for high-dimensional text data
- Effective with proper regularization
- Can be computationally expensive

**Naive Bayes:**
- Fast probabilistic classifier
- Works well with sparse text data
- Baseline for comparison

### 3. Model Evaluation

Metrics tracked:
- **Accuracy**: Overall correctness
- **Precision (macro)**: True positives across classes
- **Recall (macro)**: Coverage of all classes
- **F1 Score (macro)**: Harmonic mean of precision/recall
- **Hamming Loss**: Proportion of mislabeled samples

Per-class metrics for each HFACS class:
- Skill-Based Error
- Decision Error
- Violation
- Perceptual Error

### 4. Label Combination Strategies

Three strategies for combining keyword and cluster labels:

**1. Majority Vote**
```
If keyword_label == cluster_label:
    final_label = keyword_label
else:
    final_label = "Unlabeled"
```
- Disagreement → unlabeled (conservative)

**2. Keyword First**
```
If keyword_label == cluster_label:
    final_label = keyword_label  # Expert wins
else:
    final_label = cluster_label  # Fallback to unsupervised
```
- Expert knowledge prioritized when certain

**3. Cluster First**
```
If keyword_label == cluster_label:
    final_label = cluster_label  # Data-driven
else:
    final_label = keyword_label  # Expert fallback
```
- Data-driven when uncertain

## Usage Examples

### Example 1: Train on Majority Strategy

```python
from week2_classifier_training import Week2ClassifierTrainer

trainer = Week2ClassifierTrainer(vectorizer_type='tfidf', max_features=10000)

# Load data
train_df = pd.read_csv("weak_labeling_results/dataset_majority_train.csv")

# Train all models
results = trainer.train_all_classifiers(train_df)

# Load trained model
model = results['random_forest']['pipeline']
```

### Example 2: Custom Evaluation

```python
from week2_evaluation import WeekendModelEvaluator

evaluator = WeekendModelEvaluator(output_dir="custom_results")

# Compare strategies
results = evaluator.compare_across_strategies(
    ["majority", "keyword_first", "cluster_first"]
)

# Evaluate a specific model
test_df = pd.read_csv("weak_labeling_results/dataset_majority_test.csv")
metrics = evaluator.evaluate_on_test_set(
    "random_forest",
    test_df,
    model_pipeline
)
```

### Example 3: Custom Configuration

```python
from week2_classifier_training import Week2ClassifierTrainer

trainer = Week2ClassifierTrainer(
    vectorizer_type='tfidf',
    max_features=5000,              # Fewer features
    ngram_range=(1, 3),            # 1-gram, 2-gram, 3-gram
    use_gpu=False
)

# Train with custom settings
results = trainer.train_all_classifiers(train_df)
```

## Expected Results

Based on similar aviation safety ML research (~60-75% F1 score):

**Typical Performance Range:**
```
Model           Accuracy  F1 (macro)  Recall (macro)  Precision (macro)
─────────────────────────────────────────────────────────────────────────────
Logistic Reg    0.68      0.66        0.64            0.67
Random Forest   0.73      0.71        0.70            0.72
SVM             0.71      0.69        0.68            0.70
Naive Bayes     0.62      0.60        0.58            0.62
```

**Interpretation:**
- Random Forest typically performs best
- Logistic Regression provides good baseline
- Precision/Recall trade-offs vary by class
- Some classes easier to predict than others

## Troubleshooting

### Issue: "Data file not found"

**Solution:** Run `weak_labeling_framework.py` first to generate weakly labeled datasets.

### Issue: Memory error during training

**Solution:**
```python
# Reduce model complexity
trainer = Week2ClassifierTrainer(
    max_features=5000,  # Instead of 10000
    ngram_range=(1, 2)  # Slower, fewer features
)

# Or reduce tree count in Random Forest
# In week2_classifier_training.py:
self.classifier = RandomForestClassifier(
    n_estimators=100,  # Instead of 200
    ...
)
```

### Issue: Class imbalance

**Solutions:**
1. Use class weights (auto-enabled)
2. Balance train/validation/test sets
3. Increase regularization
4. Consider oversampling minority classes

### Issue: Overfitting

**Solutions:**
1. Add more regularization
2. Reduce model complexity
3. Increase training data
4. Use cross-validation
5. Fewer n-gram features

## Next Steps

### Week 3 (Planned):
- [ ] Advanced model configurations (BERT, LSTM, Fine-tuning)
- [ ] Hyperparameter optimization
- [ ] A/B testing strategies
- [ ] Industry benchmarking

### Experiment Ideas:
1. **Ensemble methods**: Combine top 2-3 models
2. **Feature engineering**: Add handcrafted features
3. **Domain adaptation**: Adjust for different data sources (NTSB vs ASRS)
4. **Active learning**: Select most uncertain samples for manual labeling
5. **Transfer learning**: Pre-train on general aviation datasets

## References

### Methodology Papers
1. **Madeira et al. (2021)** - Keyword-based weak labeling
2. **Macedo et al. (2026)** - Cluster-based weak labeling
3. **Liu et al. (2025)** - HFACS-guided chain-of-thoughts

### Implementation Notes
- Vectorization: TF-IDF with 10,000 max features
- N-grams: 1-2 grams (captures phrases)
- Class weights: Inverse frequency balancing
- CV: 5-fold stratified cross-validation

## License

Week 2 components build on weak labeling framework (Week 1). Both implemented within this project scope.

---

**Status:** ✅ WEEK 2 CORE MODULES COMPLETE - Ready for iterative improvement