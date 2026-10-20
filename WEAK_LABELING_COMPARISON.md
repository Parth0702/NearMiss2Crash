# Weak Labeling Approaches Comparison

## Overview

This document explains the two weak labeling approaches implemented in this project and how they differ.

---

## 1. Keyword-Based Weak Labeling

### Reference Paper
**Madeira, Melício, Valério & Santos (2021)**
> "Machine Learning and NLP for Prediction of Human Factors in Aviation Incident Reports"

### Core Concept
Uses expert-defined seed keywords mapped to HFACS classes, with a **consistency criterion** for validation.

### Key Features

#### 1.1 Seed Keyword Dictionary
Expert-curated keywords for each HFACS class:

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

#### 1.2 Consistency Criterion (Verbatim from Madeira et al.)
> "For a candidate keyword, sample 15 random documents containing it. If >= 12 of 15 (80%) agree on one HFACS-ML class, label all documents containing that keyword with that class. Otherwise discard the keyword entirely -- never force a label."

**Key Points:**
- **Sample 15 documents** containing the keyword
- **80% agreement threshold** (12/15)
- If agreement met: **label ALL documents** with that keyword
- If agreement failed: **discard the keyword** (never force a label)

#### 1.3 Validation Process
1. For each keyword in each HFACS class:
   - Find all documents containing the keyword
   - Randomly sample 15 documents
   - Check if 80% share the same HFACS ground truth
   - If yes: add to validated keywords
   - If no: discard

#### 1.4 Label Assignment
- Apply **validated keywords** to all documents
- If multiple keywords match a document: **majority vote**
- If no keywords match: label = "Unlabeled"

### Advantages
✅ **Interpretable**: Clear rule-based logic
✅ **Expert-driven**: Leverages domain knowledge
✅ **Transparent**: Easy to audit keyword validity
✅ **Fast**: No embeddings, no clustering
✅ **Consistent**: Deterministic (fixed random seed)

### Limitations
❌ **Requires ground truth**: Needs labeled data for validation
❌ **Limited coverage**: Only documents containing keywords get labeled
❌ **Vocabulary-dependent**: Misses synonyms and context-dependent usage
❌ **Expert bias**: Only reflects the expert's vocabulary
❌ **Static**: Keywords don't adapt to new language patterns

### When to Use
- ✅ You have **labeled training data** for validation
- ✅ You want **interpretability** and transparency
- ✅ You need **fast, lightweight** labeling
- ✅ You want to **audit** and explain label decisions
- ✅ Domain experts can define good keywords

---

## 2. Cluster-Based Weak Labeling

### Reference Paper
**Macedo, Ramos, Maior, Moura & Lins (2026)**
> "Automated identification of pilot failures in aviation accidents using a BERT-based classifier and topic modeling"

### Core Concept
Uses **BERT embeddings** + **UMAP dimensionality reduction** + **HDBSCAN clustering** to group semantically similar documents.

### Key Features

#### 2.1 BERT Embeddings
Convert each document to a dense vector using a pre-trained transformer model:
```python
model = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, fast
```

**Process:**
1. Tokenize document
2. Pass through BERT encoder
3. **Mean-pool** over token embeddings (mask-aware)
4. Result: 384-dimensional vector per document

#### 2.2 UMAP Dimensionality Reduction
Reduce 384D embeddings to 2D for clustering:
```python
UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
```

**Why UMAP?**
- Preserves local and global structure
- Faster than t-SNE for large datasets
- Good for subsequent clustering

#### 2.3 HDBSCAN Clustering
Group documents based on semantic similarity:
```python
HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
```

**Key Characteristics:**
- **No predefined cluster count**: Discovers clusters automatically
- **Outlier detection**: Documents marked as -1 (noise)
- **HDBSCAN over K-Means**: Better for varying cluster sizes and densities

#### 2.4 Cluster Naming
Use **TF-IDF** to identify top keywords for each cluster:
```python
TF-IDF(max_features=50, stop_words='english')
```

**Output:**
```
Cluster 0: altitude, radio, fatigued, catch, normally
Cluster 1: skipped, pressurization, running, schedule, knowingly
Cluster 2: approach, decided, continue, judging, minimums
```

### Advantages
✅ **Unsupervised**: No ground truth required
✅ **Semantic understanding**: Captures meaning beyond keywords
✅ **Scalable**: Handles large datasets
✅ **Flexible**: Discovers new patterns
✅ **Context-aware**: Considers document context

### Limitations
❌ **Black-box**: Hard to interpret why documents grouped together
❌ **Computationally expensive**: BERT embeddings + clustering
❌ **Requires GPU**: Slower on CPU
❌ **Cluster naming**: Keywords are statistical, not semantic
❌ **Outliers**: Documents marked as noise (-1)

### When to Use
- ✅ **No labeled data** available
- ✅ You want to **discover** natural groupings
- ✅ You have **large datasets** (1000+ documents)
- ✅ You need **semantic understanding** beyond keywords
- ✅ You're willing to **manually assign** HFACS classes to clusters

---

## 3. Key Differences

| Aspect | Keyword-Based | Cluster-Based |
|--------|---------------|---------------|
| **Core Method** | Rule-based keyword matching | BERT embeddings + clustering |
| **Expert Input** | Required (seed keywords) | Minimal (cluster config) |
| **Ground Truth** | Required for validation | Not required |
| **Interpretability** | High (audit keyword validity) | Low (cluster semantics unclear) |
| **Speed** | Very fast (seconds) | Slow (minutes to hours) |
| **Coverage** | Limited to keyword matches | All documents (except outliers) |
| **Semantic Understanding** | Low (exact matches only) | High (context-aware) |
| **Scalability** | Excellent | Good (but resource-intensive) |
| **Adaptability** | Static (fixed keywords) | Dynamic (learns from data) |
| **Output** | Validated keywords → labels | Cluster IDs → manual assignment |
| **Best For** | Interpretability, audits, expert guidance | Unsupervised discovery, large datasets |
| **Implementation** | Simple rule-based logic | Complex ML pipeline |

---

## 4. Comparison Workflow

### 4.1 Independent Execution
```python
# Run both labelers independently
keyword_df = keyword_labeler.transform(df)
cluster_df = cluster_labeler.transform(df)

# Compare agreement
comparison = compare_results(keyword_df, cluster_df)
```

### 4.2 Agreement Analysis
```
Total documents: 1000
Agreement rate: 68.3%
  - Agreements: 683
  - Disagreements: 317
  - Keyword-only: 150
  - Cluster-only: 167
  - Unlabeled: 0
```

### 4.3 Per-Class Agreement
```
Skill-Based Error:
  - Keyword: 450, Cluster: 480
  - Agreements: 310 (68.9%)

Decision Error:
  - Keyword: 300, Cluster: 280
  - Agreements: 180 (60.0%)

Violation:
  - Keyword: 150, Cluster: 180
  - Agreements: 95 (63.3%)

Perceptual Error:
  - Keyword: 100, Cluster: 120
  - Agreements: 98 (98.0%)
```

### 4.4 Combining Labels

#### Method 1: Majority Vote
```python
if keyword_label == cluster_label:
    final_label = keyword_label
else:
    final_label = "Unlabeled"
```

#### Method 2: Keyword First
```python
if keyword_label == cluster_label:
    final_label = keyword_label  # Expert-driven wins
else:
    final_label = cluster_label  # Fallback to unsupervised
```

#### Method 3: Cluster First
```python
if keyword_label == cluster_label:
    final_label = cluster_label  # Data-driven wins
else:
    final_label = keyword_label  # Fallback to expert
```

---

## 5. When to Use Each Approach

### Use Keyword-Based If:
1. You have labeled validation data
2. You need to explain label decisions to stakeholders
3. You want to audit and improve keyword coverage
4. You need fast, lightweight labeling
5. Domain experts can provide good seed keywords

### Use Cluster-Based If:
1. You have no labeled data
2. You want to discover natural groupings
3. You have large datasets (1000+ documents)
4. You need semantic understanding beyond keywords
5. You're willing to manually assign HFACS classes to clusters

### Use Both If:
1. You have limited labeled data for validation
2. You want to leverage expert knowledge and data-driven discovery
3. You need to compare approaches and select best fit
4. You want to combine strengths (interpretability + semantic understanding)

---

## 6. Implementation Notes

### Keyword-Based
- **File**: `keyword_weak_labeler.py`
- **Class**: `KeywordWeakLabeler`
- **Key Method**: `evaluate_keyword()` (consistency check)
- **Output**: `keyword_audit_log.csv`, `cluster_config.json`

### Cluster-Based
- **File**: `weak_labeler_cluster.py`
- **Class**: `ClusterBasedWeakLabeler`
- **Key Methods**:
  - `_get_embeddings()` (BERT embeddings)
  - `_reduce_dimensionality()` (UMAP)
  - `_cluster_documents()` (HDBSCAN)
- **Output**: `cluster_keywords.json`

### Unified Framework
- **File**: `weak_labeling_framework.py`
- **Class**: `UnifiedWeakLabelingFramework`
- **Methods**:
  - `run_keyword_labeling()`
  - `run_cluster_labeling()`
  - `compare_results()`
  - `combine_labels()`
  - `run_full_pipeline()`
- **Output**: Comparison stats, combined labels

---

## 7. Recommendations

### Phase 1: Initial Exploration
- Run **keyword-based** labeling first (fast, interpretable)
- Use results to understand data and identify potential issues

### Phase 2: Validation
- Run **cluster-based** labeling on same data
- Compare agreement rates and per-class performance
- Identify clusters that keyword-based misses

### Phase 3: Manual Assignment
- Manually assign HFACS classes to clusters
- Update `cluster_config.json` with expert labels
- Use as weak labels for supervised learning

### Phase 4: Hybrid Approach
- Combine both approaches using majority vote
- Use keyword-based as **confidence weight** for cluster labels
- Iterate and improve seed keywords based on cluster insights

---

## 8. References

### Keyword-Based Paper
```
Madeira, G., Melício, F., Valério, A., & Santos, F.
(2021). Machine Learning and NLP for Prediction of Human Factors in Aviation Incident Reports.
```

### Cluster-Based Paper
```
Macedo, R., Ramos, P., Maior, D., Moura, J., & Lins, R.
(2026). Automated identification of pilot failures in aviation accidents using a BERT-based classifier and topic modeling.
```

### Additional Resources
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
