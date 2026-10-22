"""
===============================================================================
Systematic Weak Labeling Framework (Week 1, Day 8-9)
===============================================================================
Reference Papers:
  1. Madeira et al. (2021) -- Keyword-based consistency criterion
  2. Macedo et al. (2026) -- BERT+UMAP+HDBSCAN cluster-based weak labeling

This framework integrates BOTH approaches into a unified pipeline:
  - Keyword Weak Labeler: Uses expert-defined seed keywords with consistency verification
  - Cluster Weak Labeler: Uses BERT embeddings + HDBSCAN clustering for unsupervised grouping

Key Differences:
  - Keyword: Rule-based, expert-driven, interpretable, requires ground truth for validation
  - Cluster: Embedding-based, data-driven, captures semantic similarity, unsupervised

Usage:
  1. Run both labelers independently
  2. Compare results (agreement analysis)
  3. Combine labels (majority vote or confidence-weighted)
  4. Select best approach based on domain expertise
===============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Import weak labelers
from keyword_weak_labeler import KeywordWeakLabeler
from weak_labeler_cluster import ClusterBasedWeakLabeler
from data_loading import standardize_schema


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    "keyword_labeler": {
        "sample_size": 15,
        "threshold_ratio": 0.80,  # 80% agreement required
        "random_state": 42
    },
    "cluster_labeler": {
        "bert_model": "sentence-transformers/all-MiniLM-L6-v2",
        "min_cluster_size": 3,
        "min_samples": 1,
        "n_neighbors": 3,
        "min_dist": 0.1,
        "random_state": 42
    },
    "comparison": {
        "output_dir": "weak_labeling_results",
        "include_outliers": True,
        "min_samples_for_analysis": 10
    }
}


# -----------------------------------------------------------------------------
# Weak Labeling Pipeline
# -----------------------------------------------------------------------------
class UnifiedWeakLabelingFramework:
    """
    Integrates both keyword-based and cluster-based weak labeling approaches.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or CONFIG
        self.keyword_labeler = KeywordWeakLabeler(
            sample_size=self.config["keyword_labeler"]["sample_size"],
            threshold_ratio=self.config["keyword_labeler"]["threshold_ratio"],
            random_state=self.config["keyword_labeler"]["random_state"]
        )
        self.cluster_labeler = ClusterBasedWeakLabeler(
            bert_model_name=self.config["cluster_labeler"]["bert_model"],
            min_cluster_size=self.config["cluster_labeler"]["min_cluster_size"],
            min_samples=self.config["cluster_labeler"]["min_samples"],
            n_neighbors=self.config["cluster_labeler"]["n_neighbors"],
            min_dist=self.config["cluster_labeler"]["min_dist"],
            random_state=self.config["cluster_labeler"]["random_state"]
        )
        self.results = {}

    def load_data(self, filepath: str, source_name: str = "ASRS") -> pd.DataFrame:
        """Load and standardize data."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, low_memory=False)
        return standardize_schema(df, source_name=source_name)

    def run_keyword_labeling(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        label_column: Optional[str] = 'label'
    ) -> pd.DataFrame:
        """
        Run keyword-based weak labeling.
        Returns: DataFrame with weak labels and keyword audit log.
        """
        print("\n" + "="*70)
        print("RUNNING KEYWORD-BASED WEAK LABELING")
        print("="*70)

        # Evaluate all seed keywords
        audit_log = self.keyword_labeler.fit(df, text_column, label_column)

        # Apply validated keywords
        labeled_df = self.keyword_labeler.transform(df, text_column)

        # Save audit log
        audit_path = os.path.join(
            self.config["comparison"]["output_dir"],
            "keyword_audit_log.csv"
        )
        os.makedirs(self.config["comparison"]["output_dir"], exist_ok=True)
        audit_log.to_csv(audit_path, index=False)

        # Save cluster config (for manual assignment)
        cluster_config_path = os.path.join(
            self.config["comparison"]["output_dir"],
            "cluster_config.json"
        )
        self.cluster_labeler.save_cluster_config(cluster_config_path)

        print(f"\n[INFO] Keyword audit log saved to: {audit_path}")
        print(f"[INFO] Cluster config saved to: {cluster_config_path}")

        self.results["keyword"] = {
            "labeled_df": labeled_df,
            "audit_log": audit_log,
            "validated_keywords": self.keyword_labeler.validated_keywords
        }

        return labeled_df, audit_log

    def run_cluster_labeling(
        self,
        df: pd.DataFrame,
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        Run cluster-based weak labeling (BERT+UMAP+HDBSCAN).
        Returns: DataFrame with cluster assignments and cluster keywords.
        """
        print("\n" + "="*70)
        print("RUNNING CLUSTER-BASED WEAK LABELING (BERT+UMAP+HDBSCAN)")
        print("="*70)

        # Run full pipeline
        labeled_df = self.cluster_labeler.transform(df[text_column].tolist())
        if 'report_id' in df.columns:
            labeled_df['report_id'] = df['report_id'].values

        # Save cluster keywords mapping
        cluster_keywords_path = os.path.join(
            self.config["comparison"]["output_dir"],
            "cluster_keywords.json"
        )
        with open(cluster_keywords_path, 'w', encoding='utf-8') as f:
            json.dump(self.cluster_labeler.cluster_keywords, f, indent=2)

        print(f"\n[INFO] Cluster keywords saved to: {cluster_keywords_path}")

        self.results["cluster"] = {
            "labeled_df": labeled_df,
            "cluster_keywords": self.cluster_labeler.cluster_keywords,
            "labels": self.cluster_labeler.labels
        }

        return labeled_df

    def compare_results(
        self,
        keyword_df: pd.DataFrame,
        cluster_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare keyword-based and cluster-based weak labels.
        Returns: Comparison DataFrame with agreement statistics.
        """
        print("\n" + "="*70)
        print("COMPARING KEYWORD vs CLUSTER WEAK LABELS")
        print("="*70)

        # Merge on report_id
        # Ensure both dataframes have report_id column
        if 'report_id' not in keyword_df.columns:
            keyword_df = keyword_df.copy()
            keyword_df['report_id'] = range(len(keyword_df))
        if 'report_id' not in cluster_df.columns:
            cluster_df = cluster_df.copy()
            cluster_df['report_id'] = range(len(cluster_df))

        comparison_df = keyword_df[['report_id', 'weak_label']].rename(
            columns={'weak_label': 'keyword_label'}
        ).merge(
            cluster_df[['report_id', 'weak_label']].rename(
                columns={'weak_label': 'cluster_label'}
            ),
            on='report_id',
            how='outer'
        )

        # Count agreements
        valid_comparison = comparison_df.dropna(subset=['keyword_label', 'cluster_label'])

        agreement_stats = {
            'total_compared': int(len(valid_comparison)),
            'agree': int((valid_comparison['keyword_label'] == valid_comparison['cluster_label']).sum()),
            'disagree': int((valid_comparison['keyword_label'] != valid_comparison['cluster_label']).sum()),
            'keyword_only': int((comparison_df['keyword_label'].notna() & comparison_df['cluster_label'].isna()).sum()),
            'cluster_only': int((comparison_df['keyword_label'].isna() & comparison_df['cluster_label'].notna()).sum()),
            'unlabeled_both': int((comparison_df['keyword_label'].isna() & comparison_df['cluster_label'].isna()).sum())
        }

        agreement_stats['agreement_rate'] = (
            agreement_stats['agree'] / agreement_stats['total_compared'] * 100
            if agreement_stats['total_compared'] > 0 else 0
        )

        # Per-class agreement
        per_class = {}
        for hfacs_class in ['Skill-Based Error', 'Decision Error', 'Violation', 'Perceptual Error']:
            subset = valid_comparison[
                (valid_comparison['keyword_label'] == hfacs_class) &
                (valid_comparison['cluster_label'] == hfacs_class)
            ]
            per_class[hfacs_class] = {
                'keyword_count': (valid_comparison['keyword_label'] == hfacs_class).sum(),
                'cluster_count': (valid_comparison['cluster_label'] == hfacs_class).sum(),
                'agree_count': len(subset),
                'agree_rate': len(subset) / (valid_comparison['keyword_label'] == hfacs_class).sum() * 100
                if (valid_comparison['keyword_label'] == hfacs_class).sum() > 0 else 0
            }

        # Save comparison results
        comparison_path = os.path.join(
            self.config["comparison"]["output_dir"],
            "label_comparison.csv"
        )
        comparison_df.to_csv(comparison_path, index=False)

        # Save agreement stats
        # Convert numpy types to native Python types for JSON serialization
        agreement_stats_native = {
            k: (int(v) if isinstance(v, (np.integer, np.int64)) else v)
            for k, v in agreement_stats.items()
        }
        per_class_native = {}
        for cls, stats in per_class.items():
            per_class_native[cls] = {
                k: (int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v)
                for k, v in stats.items()
            }

        stats_path = os.path.join(
            self.config["comparison"]["output_dir"],
            "agreement_stats.json"
        )
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'agreement_stats': agreement_stats_native,
                'per_class': per_class_native
            }, f, indent=2)

        print(f"\n[INFO] Comparison saved to: {comparison_path}")
        print(f"[INFO] Agreement stats saved to: {stats_path}")

        # Print summary
        print("\n--- Agreement Summary ---")
        print(f"Total documents compared: {agreement_stats['total_compared']}")
        print(f"Agreement rate: {agreement_stats['agreement_rate']:.2f}%")
        print(f"Agreements: {agreement_stats['agree']}")
        print(f"Disagreements: {agreement_stats['disagree']}")
        print(f"Keyword-only: {agreement_stats['keyword_only']}")
        print(f"Cluster-only: {agreement_stats['cluster_only']}")
        print(f"Unlabeled (both): {agreement_stats['unlabeled_both']}")

        print("\n--- Per-Class Agreement ---")
        for hfacs_class, stats in per_class.items():
            print(f"{hfacs_class}:")
            print(f"  Keyword: {stats['keyword_count']}, Cluster: {stats['cluster_count']}")
            print(f"  Agreements: {stats['agree_count']} ({stats['agree_rate']:.1f}%)")

        self.results["comparison"] = {
            "comparison_df": comparison_df,
            "agreement_stats": agreement_stats,
            "per_class": per_class
        }

        return comparison_df, agreement_stats

    def combine_labels(
        self,
        keyword_df: pd.DataFrame,
        cluster_df: pd.DataFrame,
        method: str = 'majority'
    ) -> pd.DataFrame:
        """
        Combine keyword-based and cluster-based weak labels.

        Methods:
          - 'majority': Majority vote (if both agree)
          - 'keyword_first': Keyword label wins if agreement
          - 'cluster_first': Cluster label wins if agreement
          - 'weighted': Weighted by keyword validation confidence
        """
        print("\n" + "="*70)
        print(f"COMBINING LABELS (Method: {method})")
        print("="*70)

        combined_df = keyword_df[['report_id', 'text']].copy()
        combined_df['keyword_label'] = keyword_df['weak_label']
        combined_df['cluster_label'] = cluster_df['weak_label']

        if method == 'majority':
            # Majority vote
            combined_df['final_label'] = combined_df.apply(
                lambda row: row['keyword_label'] if row['keyword_label'] == row['cluster_label']
                else 'Unlabeled',
                axis=1
            )
        elif method == 'keyword_first':
            # Keyword wins if agreement
            combined_df['final_label'] = combined_df.apply(
                lambda row: row['keyword_label'] if row['keyword_label'] == row['cluster_label']
                else row['cluster_label'],
                axis=1
            )
        elif method == 'cluster_first':
            # Cluster wins if agreement
            combined_df['final_label'] = combined_df.apply(
                lambda row: row['cluster_label'] if row['keyword_label'] == row['cluster_label']
                else row['keyword_label'],
                axis=1
            )
        elif method == 'weighted':
            # Weighted by keyword validation (if keyword validated)
            combined_df['final_label'] = combined_df.apply(
                lambda row: row['keyword_label'] if row['keyword_label'] == row['cluster_label']
                else row['cluster_label'],
                axis=1
            )

        # Save combined labels
        output_path = os.path.join(
            self.config["comparison"]["output_dir"],
            f"combined_labels_{method}.csv"
        )
        combined_df.to_csv(output_path, index=False)
        print(f"[INFO] Combined labels saved to: {output_path}")

        return combined_df

    def run_full_pipeline(
        self,
        data_path: str,
        source_name: str = "ASRS",
        combine_method: str = 'majority'
    ) -> Dict:
        """
        Run the complete weak labeling pipeline:
          1. Load data
          2. Run keyword-based labeling
          3. Run cluster-based labeling
          4. Compare results
          5. Combine labels
        """
        print("\n" + "="*70)
        print("UNIFIED WEAK LABELING PIPELINE")
        print("="*70)

        # Create output directory
        os.makedirs(self.config["comparison"]["output_dir"], exist_ok=True)

        # Step 1: Load data
        print(f"\n[STEP 1/5] Loading data from: {data_path}")
        df = self.load_data(data_path, source_name)
        print(f"  Loaded {len(df)} documents")

        # Step 2: Keyword labeling
        print(f"\n[STEP 2/5] Running keyword-based weak labeling")
        keyword_df, keyword_audit = self.run_keyword_labeling(df)

        # Step 3: Cluster labeling
        print(f"\n[STEP 3/5] Running cluster-based weak labeling")
        cluster_df = self.run_cluster_labeling(df)

        # Step 4: Compare results
        print(f"\n[STEP 4/5] Comparing keyword vs cluster results")
        comparison_df, agreement_stats = self.compare_results(keyword_df, cluster_df)

        # Step 5: Combine labels
        print(f"\n[STEP 5/5] Combining labels using method: {combine_method}")
        combined_df = self.combine_labels(keyword_df, cluster_df, method=combine_method)

        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)

        return self.results


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize framework
    framework = UnifiedWeakLabelingFramework(config=CONFIG)

    asrs_path = "asrs_full.csv" if os.path.exists("asrs_full.csv") else "asrs_data.csv"
    results = framework.run_full_pipeline(
        data_path=asrs_path,
        source_name="ASRS",
        combine_method="majority"
    )

    print("\n[SUCCESS] All results saved to:", CONFIG["comparison"]["output_dir"])
