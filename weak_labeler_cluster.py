"""
===============================================================================
Cluster-Based Weak Labeler Module (Week 1, Day 6-7)
===============================================================================
Reference Paper:
  Macedo, Ramos, Maior, Moura & Lins (2026) -- "Automated identification of pilot
  failures in aviation accidents using a BERT-based classifier and topic modeling"
  (Section 3.2, Fig. 4-5, Table 2)

Key Pipeline (from Macedo et al.):
  1. BERT embeddings, mean-pooled per document
  2. UMAP dimensionality reduction (n_neighbors=15, min_dist=0.1)
  3. HDBSCAN clustering (no predefined cluster count)

Features:
  - BERT embeddings via HuggingFace transformers
  - UMAP dimensionality reduction
  - HDBSCAN clustering with hyperparameter grid search
  - Cluster naming using TF-IDF top words
  - JSON output of cluster → keyword mappings for manual HFACS assignment
===============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers not installed. Install with: pip install transformers torch")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[WARNING] umap-learn not installed. Install with: pip install umap-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
    HDBSCAN_CLASS = hdbscan.HDBSCAN
except ImportError:
    try:
        from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN
        HDBSCAN_AVAILABLE = True
        HDBSCAN_CLASS = sklearn_HDBSCAN
    except ImportError:
        HDBSCAN_AVAILABLE = False
        HDBSCAN_CLASS = None
        print("[WARNING] hdbscan not installed. Install with: pip install hdbscan")


DEFAULT_BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLUSTER_CONFIG = {
    "min_cluster_size": 15,
    "min_samples": 5,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "random_state": 42
}


class ClusterBasedWeakLabeler:
    """
    Implements Macedo et al. (2026) BERT+UMAP+HDBSCAN pipeline for weak labeling.
    """

    def __init__(
        self,
        bert_model_name: str = DEFAULT_BERT_MODEL,
        cluster_config: Optional[Dict] = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        random_state: int = 42
    ):
        self.bert_model_name = bert_model_name
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.random_state = random_state
        self.cluster_config = cluster_config or DEFAULT_CLUSTER_CONFIG.copy()

        self.embedder = None
        self.reducer = None
        self.clusterer = None
        self.labels = None
        self.cluster_keywords: Dict[int, List[str]] = {}

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings using mean-pooling."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install: pip install transformers torch")

        if self.embedder is None:
            print(f"[INFO] Loading BERT model: {self.bert_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.embedder = AutoModel.from_pretrained(self.bert_model_name)
            self.embedder.eval()

        import torch
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
                outputs = self.embedder(**inputs)
                # Mean-pooling over last hidden state (mask-aware)
                mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.shape).float()
                summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                batch_emb = (summed / counts).cpu().numpy()
                embeddings.append(batch_emb)

        return np.vstack(embeddings)

    def _reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction."""
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn required. Install: pip install umap-learn")

        if self.reducer is None:
            print(f"[INFO] Applying UMAP (n_neighbors={self.n_neighbors}, min_dist={self.min_dist})")
            self.reducer = umap.UMAP(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state,
                n_components=2
            )

        return self.reducer.fit_transform(embeddings)

    def _cluster_documents(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply HDBSCAN clustering."""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan required. Install: pip install hdbscan")

        if self.clusterer is None:
            print(f"[INFO] Applying HDBSCAN (min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples})")
            self.clusterer = HDBSCAN_CLASS(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )

        self.labels = self.clusterer.fit_predict(embeddings)
        return self.labels

    def _name_clusters(self, texts: List[str], labels: np.ndarray) -> Dict[int, List[str]]:
        """
        Name each cluster using class-based TF-IDF top words.
        Returns dict: cluster_id -> list of top 5 words.
        """
        cluster_texts = {c: [] for c in np.unique(labels) if c != -1}  # -1 = outlier
        for text, label in zip(texts, labels):
            if label != -1 and label in cluster_texts:
                cluster_texts[label].append(text)

        cluster_keywords = {}
        for cluster_id, cluster_docs in cluster_texts.items():
            if len(cluster_docs) > 0:
                # TF-IDF to get top words
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(cluster_docs)
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.max(axis=0).toarray()[0]

                # Get top 5 words
                top_indices = np.argsort(tfidf_scores)[-5:][::-1]
                top_words = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                cluster_keywords[cluster_id] = top_words

        # Convert numpy int64 keys to standard Python int for JSON serialization
        self.cluster_keywords = {int(k): v for k, v in cluster_keywords.items()}
        return self.cluster_keywords

    def fit(self, texts: List[str]) -> np.ndarray:
        """
        Run full BERT+UMAP+HDBSCAN pipeline.
        Returns cluster labels (numpy array).
        """
        print("[INFO] Step 1/3: Computing BERT embeddings...")
        embeddings = self._get_embeddings(texts)

        print("[INFO] Step 2/3: UMAP dimensionality reduction...")
        reduced_embeddings = self._reduce_dimensionality(embeddings)

        print("[INFO] Step 3/3: HDBSCAN clustering...")
        labels = self._cluster_documents(reduced_embeddings)

        outlier_rate = (labels == -1).sum() / len(labels) * 100
        print(f"[INFO] Outlier rate: {outlier_rate:.2f}%")

        print("[INFO] Naming clusters with TF-IDF top words...")
        self._name_clusters(texts, labels)

        return labels

    def transform(self, texts: List[str]) -> pd.DataFrame:
        """
        Apply fitted model to new documents and return weak labels.
        """
        labels = self.fit(texts)

        # Load cluster-to-class mapping from cluster_config.json if it exists
        mapping = {}
        config_path = "cluster_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                for cid, cinfo in config_data.get("clusters", {}).items():
                    if "hfacs_class" in cinfo and cinfo["hfacs_class"] != "Unlabeled":
                        mapping[int(cid)] = cinfo["hfacs_class"]
            except Exception as e:
                print(f"[WARNING] Could not load cluster config mapping: {e}")

        # If mapping is empty, try to load from weak_labeling_results/cluster_config.json
        if not mapping:
            config_path = os.path.join("weak_labeling_results", "cluster_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    for cid, cinfo in config_data.get("clusters", {}).items():
                        if "hfacs_class" in cinfo and cinfo["hfacs_class"] != "Unlabeled":
                            mapping[int(cid)] = cinfo["hfacs_class"]
                except Exception as e:
                    print(f"[WARNING] Could not load cluster config mapping: {e}")

        # Map cluster IDs to HFACS classes (or "Unlabeled" if not mapped or outlier)
        weak_labels = []
        for label in labels:
            if label in mapping:
                weak_labels.append(mapping[label])
            else:
                weak_labels.append("Unlabeled")

        df = pd.DataFrame({
            'text': texts,
            'cluster_id': labels,
            'weak_label': weak_labels
        })

        # Add cluster keywords
        for cluster_id, keywords in self.cluster_keywords.items():
            df.loc[df['cluster_id'] == cluster_id, 'cluster_keywords'] = ", ".join(keywords)

        return df

    def save_cluster_config(self, output_path: str = "cluster_config.json"):
        """
        Save cluster keyword mappings to JSON for manual HFACS assignment.
        """
        # Load existing config to preserve manual assignments
        existing_clusters = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    old_config = json.load(f)
                existing_clusters = old_config.get("clusters", {})
            except Exception as e:
                print(f"[WARNING] Could not read existing cluster config for preservation: {e}")

        config = {
            "metadata": {
                "n_neighbors": self.n_neighbors,
                "min_dist": self.min_dist,
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "random_state": self.random_state
            },
            "clusters": {}
        }

        for cluster_id, keywords in self.cluster_keywords.items():
            cid_str = str(cluster_id)
            # Default values
            hfacs_class = "Unlabeled"
            desc = "Manually assign to one HFACS class (Skill-Based Error / Decision Error / Violation / Perceptual Error)"

            # Preserve if existed
            if cid_str in existing_clusters:
                hfacs_class = existing_clusters[cid_str].get("hfacs_class", "Unlabeled")
                desc = existing_clusters[cid_str].get("description", desc)

            config["clusters"][cid_str] = {
                "keywords": keywords,
                "hfacs_class": hfacs_class,
                "description": desc
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"[INFO] Cluster config saved to: {output_path}")


if __name__ == "__main__":
    print("===============================================================================")
    print(" TESTING CLUSTER-BASED WEAK LABELER (Macedo et al. 2026)")
    print("===============================================================================")

    from data_loading import standardize_schema

    asrs_file = "asrs_full.csv" if os.path.exists("asrs_full.csv") else "asrs_data.csv"
    if os.path.exists(asrs_file):
        raw_df = pd.read_csv(asrs_file)
        df_asrs = standardize_schema(raw_df, source_name="ASRS")

        # Use smaller sample for quick test
        sample_df = df_asrs.sample(n=min(100, len(df_asrs)), random_state=42)

        labeler = ClusterBasedWeakLabeler(
            min_cluster_size=2,  # Small cluster size for small test sample
            min_samples=1,
            n_neighbors=3,
            min_dist=0.1
        )

        labeled_df = labeler.transform(sample_df['text'].tolist())

        print("\n--- Cluster Summary ---")
        unique_clusters = labeled_df['cluster_id'].unique()
        print(f"Number of clusters: {len([c for c in unique_clusters if c != -1])}")
        print(f"Number of outliers: {(labeled_df['cluster_id'] == -1).sum()}")

        print("\n--- Cluster Keywords ---")
        for cluster_id, keywords in labeler.cluster_keywords.items():
            print(f"Cluster {cluster_id}: {', '.join(keywords)}")

        print("\n--- Sample Weak Labels ---")
        print(labeled_df[['text', 'cluster_id', 'cluster_keywords']].head(10))

        labeler.save_cluster_config("cluster_config.json")
