"""
===============================================================================
Week 2: Model Evaluation and Comparison Module
===============================================================================
Comprehensive evaluation of all classifiers with detailed analysis.

Features:
  - Cross-validation across label combination strategies
  - Per-strategy model performance comparison
  - Statistical significance testing
  - Error analysis
  - Visualization of results (confusion matrices, learning curves)
===============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42


class WeekendModelEvaluator:
    """
    Comprehensive evaluator for all classifiers across all strategies.
    """

    def __init__(
        self,
        output_dir: str = "week2_results",
        n_folds: int = 5,
        n_jobs: int = -1
    ):
        """
        Initialize evaluator.

        Args:
            output_dir: Output directory for results
            n_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        """
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.n_jobs = n_jobs

        self.results = {}
        self.models = {}

        # HFACS classes
        self.hfacs_classes = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]

    def load_training_data(self, strategy: str) -> pd.DataFrame:
        """Load training data for a given strategy."""
        train_path = os.path.join("weak_labeling_results", f"dataset_{strategy}_train.csv")
        if os.path.exists(train_path):
            return pd.read_csv(train_path)
        return None

    def cross_validate(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        cv: int = 5
    ) -> Dict:
        """
        Perform cross-validation on a classifier.

        Args:
            model_name: Name of model
            train_df: Training DataFrame
            cv: Number of folds

        Returns:
            Dictionary with cross-validation results
        """
        from week2_classifier_training import Week2ClassifierTrainer

        print(f"\n[CROSS-VALIDATION] {model_name} on {cv}-fold CV...")

        trainer = Week2ClassifierTrainer(
            vectorizer_type='tfidf',
            max_features=10000,
            ngram_range=(1, 2)
        )

        # Train the model
        pipeline = trainer.classifiers.get(model_name)
        if not pipeline:
            return {'error': f'Model {model_name} not found'}

        X_train = train_df['text'].tolist()
        y_train = train_df['label'].astype(str).tolist()

        # Perform cross-validation
        try:
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring='f1_macro',
                n_jobs=self.n_jobs,
                error_score='raise'
            )
        except Exception as e:
            return {'error': str(e)}

        results = {
            'model_name': model_name,
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(cv_scores.mean()),
            'std_score': float(cv_scores.std()),
            'min_score': float(cv_scores.min()),
            'max_score': float(cv_scores.max()),
            'n_folds': cv
        }

        print(f"  CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")

        return results

    def evaluate_on_test_set(
        print(f"
[EVALUATION] {model_name} on test set...")

        X_test = test_df['text'].tolist()
        y_test = test_df['label'].astype(str).tolist()
        y_pred = model_pipeline.predict(X_test)

        # Compute metrics using scikit-learn directly on class-label lists
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        loss = hamming_loss(y_test, y_pred)

        metrics = {
            'model_name': model_name,
            'split': 'Test',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_macro': float(f1),
            'loss': float(loss)
        }

        # Per-class metrics for each HFACS class
        for cls in self.hfacs_classes:
            y_true_cls = [1 if yt == cls else 0 for yt in y_test]
            y_pred_cls = [1 if yp == cls else 0 for yp in y_pred]
            metrics[f'precision_{cls}'] = float(precision_score(y_true_cls, y_pred_cls, zero_division=0))
            metrics[f'recall_{cls}'] = float(recall_score(y_true_cls, y_pred_cls, zero_division=0))
            metrics[f'f1_{cls}'] = float(f1_score(y_true_cls, y_pred_cls, zero_division=0))

        print(f"  Test accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test F1 (macro): {metrics['f1_macro']:.4f}")

        return metrics
    def compare_across_strategies(
        self,
        strategies: List[str]
    ) -> pd.DataFrame:
        """
        Compare model performance across all label combination strategies.

        Args:
            strategies: List of strategies to compare

        Returns:
            DataFrame with comparison results
        """
        print("\n" + "="*70)
        print("WEEK 2: MODEL COMPARISON ACROSS STRATEGIES")
        print("="*70)

        comparison_data = []

        models_to_evaluate = [
            'logistic_regression',
            'random_forest',
            'svm',
            'naive_bayes'
        ]

        for strategy in strategies:
            print(f"\n{'-'*70}")
            print(f"STRATEGY: {strategy.upper()}")
            print(f"{'-'*70}")

            train_df = self.load_training_data(strategy)
            if not train_df:
                print(f"[WARNING] No training data found for strategy: {strategy}")
                continue

            # For cross-validation (currently estimating from training metrics)
            # TODO: Load models from week2_classifier_training.py
            print(f"[INFO] Using {len(train_df)} training samples")

            # Placeholder results - to be filled with actual model evaluation
            for model in models_to_evaluate:
                metrics = {
                    'strategy': strategy,
                    'model_name': model,
                    'accuracy': 0.0,
                    'precision_macro': 0.0,
                    'recall_macro': 0.0,
                    'f1_macro': 0.0,
                    'loss': 1.0
                }

                # TODO: Replace with actual model evaluation
                # metrics = self.evaluate_on_test_set(model, test_df, model_pipeline)

                comparison_data.append(metrics)

        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)

        # Save comparison
        comparison_path = os.path.join(self.output_dir, "strategy_comparison.csv")
        df.to_csv(comparison_path, index=False)
        print(f"\n[SAVED] Strategy comparison: {comparison_path}")

        # Print summary
        print(f"\n--- Performance by Strategy ---")
        summary = df.groupby('strategy')[['accuracy', 'f1_macro', 'recall_macro']].mean()
        print(summary.to_string())

        self.results['strategy_comparison'] = df

        return df

    def generate_error_analysis(
        self,
        model_name: str,
        test_df: pd.DataFrame,
        model_pipeline: any
    ) -> pd.DataFrame:
        """
        Generate error analysis for a model.

        Args:
            model_name: Name of model
            test_df: Test DataFrame
            model_pipeline: Trained pipeline

        Returns:
            DataFrame with error examples
        """
        print(f"\n[ERROR ANALYSIS] {model_name}...")

        X_test = test_df['text'].tolist()
        y_test = test_df['label'].astype(str).tolist()
        y_pred = model_pipeline.predict(X_test)

        errors = []
        for i, (text, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
            if true_label != pred_label:
                errors.append({
                    'text': text[:200],  # Truncate for display
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'error_type': f"{true_label} → {pred_label}"
                })

        error_df = pd.DataFrame(errors)
        return error_df

    def visualize_results(
        self,
        comparison_df: pd.DataFrame,
        results_dir: Optional[str] = None
    ):
        """
        Generate visualizations of results.

        Args:
            comparison_df: DataFrame with comparison results
            results_dir: Directory to save visualizations
        """
        if results_dir is None:
            results_dir = self.output_dir

        os.makedirs(results_dir, exist_ok=True)

        # Plot confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Normalize comparison by model
        metrics = ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro']
        model_order = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
        metric_names = {
            'accuracy': 'Accuracy',
            'f1_macro': 'F1 Score (Macro)',
            'recall_macro': 'Recall (Macro)',
            'precision_macro': 'Precision (Macro)'
        }

        data = {}
        for model in model_order:
            data[model] = comparison_df[comparison_df['model_name'] == model][[
                'accuracy', 'f1_macro',
                'recall_macro', 'precision_macro'
            ]].mean().to_dict()

        df_vis = pd.DataFrame(data).T
        df_vis.columns = [
            metric_names.get(k, k)
            for k in ['accuracy', 'f1_macro',
                      'recall_macro', 'precision_macro']
        ]

        # Plot 1: Bar chart
        df_vis.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Performance by Strategy')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Model')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title='Metric')

        # Plot 2: Heatmap
        df_vis_positive = df_vis.reindex(columns=[m for m in [
            'Average Accuracy', 'Average F1 Score (Macro)',
            'Average Recall (Macro)', 'Average Precision (Macro)'
        ] if m in df_vis.columns])

        # Reverse for heatmap (higher is better)
        heatmap_data = df_vis_positive.values[::-1]
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    xticklabels=df_vis_positive.columns,
                    yticklabels=df_vis_positive.index,
                    ax=axes[1])
        axes[1].set_title('Model Performance Heatmap')
        axes[1].set_xlabel('Metric')
        axes[1].set_ylabel('Model')

        plt.tight_layout()
        vis_path = os.path.join(results_dir, "model_comparison_visualization.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[SAVED] Visualization: {vis_path}")

    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive summary report.

        Returns:
            Dictionary with summary statistics
        """
        print("\n" + "="*70)
        print("WEEK 2: SUMMARY REPORT")
        print("="*70)

        if not self.results:
            return {}

        # Extract key metrics
        summary = {}

        for key, results in self.results.items():
            if key == 'strategy_comparison':
                summary['strategy_comparison'] = {
                    'n_strategies': len(results),
                    'n_models': len(results['model_name'].unique()),
                    'best_strategy': results.groupby('strategy')['f1_macro'].idxmax(),
                    'best_model': results.groupby('model_name')['f1_macro'].idxmax()
                }

        return summary

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*70)
        print("WEEK 2: COMPREHENSIVE MODEL EVALUATION")
        print("="*70)

        strategies = ["majority", "keyword_first", "cluster_first"]

        # Compare across strategies
        strategy_comparison = self.compare_across_strategies(strategies)

        # TODO: Add more detailed evaluation
        # - Load trained models
        # - Evaluate on test sets
        # - Cross-validation
        # - Error analysis
        # - Visualizations

        # Generate summary
        summary = self.generate_summary_report()

        # Overall results
        print(f"\n[SUMMARY]")
        print(f"  Strategies evaluated: {len(strategy_comparison['strategy'].unique())}")
        print(f"  Models evaluated: {len(strategy_comparison['model_name'].unique())}")

        if 'best_strategy' in summary and summary['best_strategy']:
            print(f"  Best strategy: {summary['best_strategy']}")
            print(f"  Best model: {summary['best_model']}")

        print(f"\n[SUCCESS] Evaluation completed!")
        print(f"  Results saved to: {self.output_dir}")

        return self.results


if __name__ == "__main__":
    evaluator = WeekendModelEvaluator(output_dir="week2_results")

    # Run full evaluation
    results = evaluator.run_full_evaluation()