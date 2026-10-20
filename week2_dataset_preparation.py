"""
===============================================================================
Week 2: Dataset Preparation for Supervised Training
===============================================================================
Based on weak labeling framework, prepare labeled dataset for classifier training.

This module:
  1. Loads weakly labeled data from both approaches
  2. Combines labels (majority vote or confidence-weighted)
  3. Splits into train/validation/test sets
  4. Exports clean, ready-to-train dataset
===============================================================================
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class WeekendDatasetPreparation:
    """
    Prepares weakly labeled dataset for supervised learning.
    """

    def __init__(
        self,
        weak_labeling_dir: str = "weak_labeling_results",
        strategies: List[str] = ["majority", "keyword_first", "cluster_first"],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize dataset preparation.

        Args:
            weak_labeling_dir: Directory containing weak labeling results
            strategies: List of label combination strategies to use
            test_size: Proportion for test set (default 0.2)
            val_size: Proportion for validation set (default 0.1)
            random_state: Random seed for reproducibility
        """
        self.weak_labeling_dir = weak_labeling_dir
        self.strategies = strategies
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.label_sets = {}  # strategy -> labeled DataFrame
        self.raw_data = None
        self.sparse_matrix = None

    def load_weak_labels(self, strategy: str) -> pd.DataFrame:
        """
        Load weakly labeled data for a given strategy.

        Args:
            strategy: Label combination method ('majority', 'keyword_first', 'cluster_first')

        Returns:
            DataFrame with report_id, text, label, and metadata
        """
        label_path = os.path.join(self.weak_labeling_dir, f"combined_labels_{strategy}.csv")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Labeled data not found: {label_path}")

        df = pd.read_csv(label_path)

        # Ensure required columns exist
        # The weak labeling framework uses 'final_label', so map it to 'label'
        if 'final_label' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'final_label': 'label'})
        elif 'label' not in df.columns:
            raise ValueError(f"Missing required columns in {label_path}")

        self.label_sets[strategy] = df

        print(f"[INFO] Loaded {len(df)} labeled documents for strategy: {strategy}")
        return df

    def load_raw_data(self, data_path: Optional[str] = None, source: str = "ASRS") -> pd.DataFrame:
        if data_path is None:
            data_path = "asrs_full.csv" if os.path.exists("asrs_full.csv") else "asrs_data.csv"
        """
        Load raw data for documentation.

        Args:
            data_path: Path to raw data file
            source: Source identifier

        Returns:
            DataFrame with raw data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.raw_data = pd.read_csv(data_path, low_memory=False)

        # Standardize if possible
        try:
            from data_loading import standardize_schema
            self.raw_data = standardize_schema(self.raw_data, source_name=source)
        except:
            print(f"[WARNING] Could not standardize data schema")

        print(f"[INFO] Loaded {len(self.raw_data)} raw documents from {source}")
        return self.raw_data

    def get_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get class distribution in labeled dataset.

        Args:
            df: Labeled DataFrame

        Returns:
            DataFrame with class counts and percentages
        """
        hfacs_classes = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]

        dist = []
        for hfacs_class in hfacs_classes:
            count = (df['label'] == hfacs_class).sum()
            percent = count / len(df) * 100 if len(df) > 0 else 0
            dist.append({
                'HFACS Class': hfacs_class,
                'Count': count,
                'Percentage': f"{percent:.2f}%"
            })

        dist_df = pd.DataFrame(dist)
        dist_df = dist_df.sort_values('Count', ascending=False)

        return dist_df

    def calculate_balancing_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate class balancing weights using inverse frequency weighting.

        Args:
            df: Labeled DataFrame

        Returns:
            Array of weights for each class (one per row)
        """
        hfacs_classes = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]

        # Get class frequencies
        class_counts = df['label'].value_counts()
        total = len(df)

        # Calculate inverse frequency weights
        weights = []
        for label in df['label']:
            if label in class_counts:
                freq = class_counts[label]
                weight = total / (len(hfacs_classes) * freq)
                weights.append(weight)

        return np.array(weights)

    def split_dataset(
        self,
        df: pd.DataFrame,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets.

        Args:
            df: Input DataFrame
            stratify: Whether to stratify on labels

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Determine if we can stratify by checking per-class sample counts
        label_counts = df['label'].value_counts()
        min_class_count = label_counts.min()
        n_classes = len(label_counts)

        # If dataset is too small (< 20 samples), stratified split is numerically impossible with test_size=1
        if len(df) < 20:
            stratify = False

        if stratify:
            # First split: 20% test, 80% remaining
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df['label']
            )

            # Second split: 12.5% val (10% of original), 67.5% train (80% of original)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
                stratify=train_val_df['label']
            )
        else:
            # Non-stratified split
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state
            )

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state
            )

        print(f"\n[INFO] Dataset split:")
        print(f"  Train: {len(train_df)} samples ({(1-self.test_size-self.val_size)*100:.1f}%)")
        print(f"  Validation: {len(val_df)} samples ({self.val_size*100:.1f}%)")
        print(f"  Test: {len(test_df)} samples ({self.test_size*100:.1f}%)")

        return train_df, val_df, test_df

    def analyze_class_imbalance(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Analyze and report class imbalance across splits.

        Args:
            train_df: Training set
            val_df: Validation set
            test_df: Test set
        """
        print("\n[INFO] Class Distribution by Split:")

        hfacs_classes = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]
        table_data = []

        for hfacs_class in hfacs_classes:
            train_count = (train_df['label'] == hfacs_class).sum()
            val_count = (val_df['label'] == hfacs_class).sum()
            test_count = (test_df['label'] == hfacs_class).sum()
            train_pct = train_count / len(train_df) * 100 if len(train_df) > 0 else 0
            val_pct = val_count / len(val_df) * 100 if len(val_df) > 0 else 0
            test_pct = test_count / len(test_df) * 100 if len(test_df) > 0 else 0

            table_data.append({
                'HFACS Class': hfacs_class,
                'Train': f"{train_count} ({train_pct:.1f}%)",
                'Validation': f"{val_count} ({val_pct:.1f}%)",
                'Test': f"{test_count} ({test_pct:.1f}%)"
            })

        imbalance_report = pd.DataFrame(table_data)
        print(f"\n{imbalance_report.to_string(index=False)}")

        return imbalance_report

    def create_output_directory(self) -> Path:
        """Create output directory for prepared datasets."""
        output_dir = Path("datasets")
        output_dir.mkdir(exist_ok=True)

        # Create subdirectories for each strategy
        for strategy in self.strategies:
            strategy_dir = output_dir / strategy
            strategy_dir.mkdir(exist_ok=True)

        return output_dir

    def save_dataset(self, df: pd.DataFrame, strategy: str, split_name: str):
        """
        Save dataset split to CSV file.

        Args:
            df: DataFrame to save
            strategy: Label combination strategy
            split_name: Name of split (train/val/test)
        """
        output_path = os.path.join(
            self.weak_labeling_dir,
            f"dataset_{strategy}_{split_name}.csv"
        )
        df.to_csv(output_path, index=False)

    def prepare_all_strategies(self, raw_data_path: Optional[str] = None):
        """
        Prepare labeled datasets for all strategies.

        Args:
            raw_data_path: Path to raw data (optional, for documentation)
        """
        print("\n" + "="*70)
        print("WEEK 2: DATASET PREPARATION FOR SUPERVISED TRAINING")
        print("="*70)

        # Load unlabeled raw data
        if raw_data_path and os.path.exists(raw_data_path):
            self.load_raw_data(raw_data_path)

        # Prepare datasets for each strategy
        for strategy in self.strategies:
            print(f"\n{'='*70}")
            print(f"PREPARING DATASET FOR STRATEGY: {strategy.upper()}")
            print(f"{'='*70}")

            # Check if labeled data exists
            label_path = os.path.join(self.weak_labeling_dir, f"combined_labels_{strategy}.csv")
            if not os.path.exists(label_path):
                print(f"[WARNING] Labeled data not found: {label_path}")
                print(f"[WARNING] Skipping strategy: {strategy}")
                continue

            # Load labeled data
            df = self.load_weak_labels(strategy)

            # Show distribution
            print(f"\n[INFO] Label distribution:")
            dist_df = self.get_distribution(df)
            print(f"\n{dist_df.to_string(index=False)}")

            # Calculate balancing weights
            weights = self.calculate_balancing_weights(df)
            print(f"[INFO] Class balancing weights calculated")

            # Split dataset
            train_df, val_df, test_df = self.split_dataset(df, stratify=True)

            # Analyze imbalance
            imbalance_report = self.analyze_class_imbalance(train_df, val_df, test_df)

            # Save dataset splits
            self.save_dataset(train_df, strategy, "train")
            self.save_dataset(val_df, strategy, "validation")
            self.save_dataset(test_df, strategy, "test")

            # Save statistics
            stats = {
                'strategy': strategy,
                'total_samples': len(df),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'imbalance_report': imbalance_report.to_dict(orient='records')
            }

            stats_path = os.path.join(self.weak_labeling_dir, f"dataset_stats_{strategy}.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, default=str)

            print(f"\n[SUCCESS] Dataset saved for strategy: {strategy}")
            print(f"  - Train: weak_labeling_results/dataset_{strategy}_train.csv ({len(train_df)})")
            print(f"  - Validation: weak_labeling_results/dataset_{strategy}_validation.csv ({len(val_df)})")
            print(f"  - Test: weak_labeling_results/dataset_{strategy}_test.csv ({len(test_df)})")
            print(f"  - Statistics: weak_labeling_results/dataset_stats_{strategy}.json")

            # Save balancing weights for this strategy
            weights_path = os.path.join(self.weak_labeling_dir, f"weights_{strategy}.npy")
            np.save(weights_path, weights)
            print(f"  - Weights: weak_labeling_results/weights_{strategy}.npy")

        print("\n" + "="*70)
        print("DATASET PREPARATION COMPLETED")
        print("="*70)

        return self.label_sets


if __name__ == "__main__":
    # Prepare datasets for all label combination strategies
    preparation = WeekendDatasetPreparation(
        weak_labeling_dir="weak_labeling_results",
        strategies=["majority", "keyword_first", "cluster_first"],
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Run preparation
    asrs_path = "asrs_full.csv" if os.path.exists("asrs_full.csv") else "asrs_data.csv"
    label_sets = preparation.prepare_all_strategies(raw_data_path=asrs_path)

    print("\n[SUCCESS] All datasets prepared for Week 2 training!")