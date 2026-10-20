# HFACS Weak Labeling Framework

A systematic framework for weak labeling aviation incident reports using two complementary approaches:
1. **Keyword-Based Weak Labeling** (expert-driven, interpretable)
2. **Cluster-Based Weak Labeling** (unsupervised, semantic understanding)

---

## Quick Start

### Run the Full Pipeline

```bash
python weak_labeling_framework.py
```

This will:
1. Load `asrs_data.csv`
2. Run keyword-based labeling
3. Run cluster-based labeling (BERT+UMAP+HDBSCAN)
4. Compare results (agreement statistics)
5. Combine labels using majority vote
6. Save all outputs to `weak_labeling_results/`

### Input Data

Ensure you have `asrs_data.csv` with columns:
```csv
report_id, acn, narrative, ...
```

The framework automatically standardizes schemas using `data_loading.py`.

---

## Project Structure

```
SC project/
├── ASRS-HFACS_4Week_Build_Plan.pdf     # Build plan reference
├── weak_labeling_framework.py           # Unified framework (NEW)
├── weak_labeler_cluster.py             # Cluster-based weak labeling
├── keyword_weak_labeler.py             # Keyword-based weak labeling
├── data_loading.py                     # Data loading & schema standardization
├── cluster_config.json                 # Cluster keyword mappings (generated)
├── WEAK_LABELING_COMPARISON.md         # Detailed comparison (NEW)
├── README_WEAK_LABELING.md             # This file
├── requirements.txt                    # Dependencies
├── asrs_data.csv                       # Input data
└── experiments/                        # (For your experiments)
```

---

## Outputs

After running `weak_labeling_framework.py`, you'll find:

```
weak_labeling_results/
├── keyword_audit_log.csv              # Keyword validation results
├── cluster_keywords.json              # Top words for each cluster
├── label_comparison.csv               # Keyword vs Cluster agreement
├── agreement_stats.json               # Summary statistics
├── combined_labels_majority.csv       # Final combined labels
└── combined_labels_keyword_first.csv  # Alternative combination methods
```

---

## Usage Examples

### 1. Run Independently

```python
from weak_labeling_framework import UnifiedWeakLabelingFramework

framework = UnifiedWeakLabelingFramework()

# Run keyword-based
keyword_df = framework.keywords_labeler.transform(df)

# Run cluster-based
cluster_df = framework.cluster_labeler.transform(df)
```

### 2. Compare Results

```python
comparison_df, stats = framework.compare_results(keyword_df, cluster_df)

print(f"Agreement rate: {stats['agreement_stats']['agreement_rate']:.2f}%")
```

### 3. Combine Labels

```python
# Method 1: Majority vote (experts and data agree)
combined_df = framework.combine_labels(
    keyword_df, cluster_df, method='majority'
)

# Method 2: Keyword first (experts override when they agree)
combined_df = framework.combine_labels(
    keyword_df, cluster_df, method='keyword_first'
)

# Method 3: Cluster first (data-driven when experts agree)
combined_df = framework.combine_labels(
    keyword_df, cluster_df, method='cluster_first'
)
```

### 4. Custom Configuration

```python
custom_config = {
    "keyword_labeler": {
        "sample_size": 20,
        "threshold_ratio": 0.85,  # Stricter threshold
        "random_state": 42
    },
    "cluster_labeler": {
        "bert_model": "sentence-transformers/all-MiniLM-L6-v2",
        "min_cluster_size": 20,  # Larger clusters
        "min_samples": 10,
        "n_neighbors": 20,
        "random_state": 42
    },
    "comparison": {
        "output_dir": "my_experiments",
        "include_outliers": False,
        "min_samples_for_analysis": 5
    }
}

framework = UnifiedWeakLabelingFramework(config=custom_config)
results = framework.run_full_pipeline(data_path="data.csv")
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- transformers (HuggingFace)
- torch
- umap-learn
- hdbscan

---

## Understanding the Two Approaches

### Keyword-Based (Expert-Driven)

**What it does:**
- Uses expert-defined keywords mapped to HFACS classes
- Validates keywords with consistency check (12/15 = 80% agreement)
- Labels all documents containing validated keywords

**When to use:**
- ✅ You have labeled training data for validation
- ✅ You need interpretability and audit trails
- ✅ You want fast, lightweight labeling

**Reference:** Madeira et al. (2021) - "Machine Learning and NLP for Prediction of Human Factors in Aviation Incident Reports"

### Cluster-Based (Data-Driven)

**What it does:**
- Converts documents to BERT embeddings
- Reduces dimensionality with UMAP
- Groups semantically similar documents with HDBSCAN
- Names clusters with TF-IDF top words

**When to use:**
- ✅ You have no labeled data
- ✅ You want unsupervised discovery
- ✅ You have large datasets (1000+ documents)

**Reference:** Macedo et al. (2026) - "Automated identification of pilot failures in aviation accidents using a BERT-based classifier and topic modeling"

---

## Key Differences

| Aspect | Keyword-Based | Cluster-Based |
|--------|---------------|---------------|
| **Method** | Rule-based keywords | BERT embeddings + clustering |
| **Speed** | Seconds | Minutes to hours |
| **Interpretability** | High (audit keywords) | Low (cluster semantics) |
| **Ground Truth** | Required for validation | Not required |
| **Coverage** | Limited to keyword matches | All documents (minus outliers) |
| **Best For** | Transparency & expertise | Unsupervised discovery |

**See `WEAK_LABELING_COMPARISON.md` for detailed analysis.**

---

## Experimental Workflow

### Recommended Phase-by-Phase Approach

#### Phase 1: Initial Exploration (Keyword-First)
```bash
python keyword_weak_labeler.py
```
- Quick, interpretable results
- Understand data and identify potential issues
- Generate `cluster_config.json` for manual assignment

#### Phase 2: Hybrid Validation
```bash
python weak_labeler_cluster.py
python weak_labeling_framework.py
```
- Compare agreement rates between approaches
- Identify strengths/weaknesses of each method
- Analyze per-class performance

#### Phase 3: Manual Assignment
1. Review clusters in `cluster_config.json`
2. Manually assign HFACS classes to clusters
3. Update `cluster_config.json` with expert labels
4. Use as weak labels for supervised learning

#### Phase 4: Hybrid Approach
- Combine both approaches using majority vote
- Use keyword results as confidence weights
- Iterate and improve based on cluster insights

---

## Troubleshooting

### Memory Issues

If BERT embeddings are too memory-intensive:

```python
cluster_labeler = ClusterBasedWeakLabeler(
    bert_model_name="sentence-transformers/all-MiniLM-L6-v2",
    min_cluster_size=2,  # Smaller sample size
    batch_size=16  # Process in smaller batches
)
```

### Slow Performance

- Use lighter BERT model: `all-MiniLM-L6-v2` is already lightweight
- Reduce sample size for testing: `min_cluster_size=10`, `min_samples=5`
- Process in batches manually (customize `_get_embeddings()`)

### Low Agreement Rates

- Adjust keyword threshold: `threshold_ratio=0.70` (more permissive)
- Increase sample size: `sample_size=20`
- Review seed keywords (add synonyms, adjust examples)
- Check if ground truth labels are accurate

---

## Citation

If you use this framework, please cite the reference papers:

```bibtex
@article{madeira2021,
  author = {Madeira, G. and Melício, F. and Valério, A. and Santos, F.},
  title = {Machine Learning and NLP for Prediction of Human Factors in Aviation Incident Reports},
  year = {2021}
}

@article{macedo2026,
  author = {Macedo, R. and Ramos, P. and Maior, D. and Moura, J. and Lins, R.},
  title = {Automated identification of pilot failures in aviation accidents using a BERT-based classifier and topic modeling},
  year = {2026}
}
```

---

## License

This project implements weak labeling methods from the cited papers. Please refer to their licenses for usage restrictions.

---

## Contact

For questions or issues with the framework:
1. Review `WEAK_LABELING_COMPARISON.md` for detailed methodology
2. Check the inline comments in each module
3. Run the test scripts to verify installation

---

## Next Steps

- ✅ Run `weak_labeling_framework.py` on your data
- ✅ Review the output in `weak_labeling_results/`
- ✅ Read `WEAK_LABELING_COMPARISON.md` to understand differences
- ✅ Experiment with different combinations of approaches
- ✅ Choose the best method for your use case