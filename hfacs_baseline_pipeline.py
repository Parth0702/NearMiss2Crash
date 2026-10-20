"""
===============================================================================
Aviation Human Factors (HFACS) Baseline Text Classification Pipeline
===============================================================================
Research Context:
This script implements an end-to-end baseline multi-class text classification 
pipeline for categorizing aviation incident/accident narratives into 4 HFACS 
Unsafe Act categories:
  1. Skill-Based Error (e.g., memory lapses, routine checklist oversights)
  2. Decision Error    (e.g., inappropriate tactical choices, miscalculated risk)
  3. Violation         (e.g., willful disregard of rules/SOPs)
  4. Perceptual Error  (e.g., visual illusions, spatial disorientation)

Domain Transfer Setup:
- Primary Training Domain: ASRS (Aviation Safety Reporting System) - Self-reported near-misses
- Transfer Test Domain:    NTSB (National Transportation Safety Board) - Official accident reports

Author: B.Tech Research Project Baseline Pipeline
===============================================================================
"""

import re
import string
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless file saving
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules for NLP and classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLTK modules for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Automatically download required NLTK resources if not present
def setup_nltk():
    """Download required NLTK data resources quietly."""
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

setup_nltk()

# Define the canonical HFACS Unsafe Act categories
HFACS_CLASSES = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]


# =============================================================================
# Step 1: Data Loading Module
# =============================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads incident report dataset from CSV file and validates expected columns.
    
    Expected CSV Columns:
      - report_id : Unique identifier for the report
      - source    : Data source (e.g., 'ASRS', 'NTSB')
      - text      : Raw narrative text of the aviation incident
      - label     : HFACS unsafe act classification category
      
    Parameters:
      filepath (str): Path to CSV file.
      
    Returns:
      pd.DataFrame: Cleaned pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at path: {filepath}")
        
    df = pd.read_csv(filepath)
    required_cols = {'report_id', 'source', 'text', 'label'}
    
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV file '{filepath}' is missing required columns: {missing}")
        
    print(f"[INFO] Successfully loaded '{filepath}': {len(df)} records found.")
    print(f"[INFO] Class Distribution:\n{df['label'].value_counts()}\n")
    return df


# =============================================================================
# Step 2: Text Preprocessing Module
# =============================================================================
def preprocess_text(text: str) -> str:
    """
    Standard NLP preprocessing pipeline:
      1. Lowercasing: Normalizes character casing to reduce vocabulary size.
      2. Punctuation Removal: Strips non-alphanumeric punctuation marks.
      3. Tokenization: Splits narrative into discrete word tokens.
      4. Stopword Removal: Filters out high-frequency non-informative English words.
      
    Parameters:
      text (str): Raw input text narrative.
      
    Returns:
      str: Space-separated preprocessed text string ready for vectorization.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase text
    text_clean = text.lower()
    
    # 2. Remove punctuation
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenize text using NLTK word_tokenize (with regex fallback)
    try:
        tokens = word_tokenize(text_clean)
    except Exception:
        tokens = text_clean.split()
        
    # 4. Filter out standard English stopwords & non-alphabetic tokens
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'was', 'were', 'it', 'is', 'be'}
        
    filtered_tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
    
    # Rejoin tokens back into clean text string
    return " ".join(filtered_tokens)


def preprocess_dataset(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Applies text preprocessing across a pandas DataFrame column.
    
    Parameters:
      df (pd.DataFrame): Input dataset dataframe.
      text_column (str): Column name containing raw text.
      
    Returns:
      pd.DataFrame: DataFrame containing a new column 'clean_text'.
    """
    df = df.copy()
    df['clean_text'] = df[text_column].apply(preprocess_text)
    return df


# =============================================================================
# Step 3 & 4: Model Training & Vectorization Module
# =============================================================================
def train_baseline(X_train: pd.Series, y_train: pd.Series):
    """
    Constructs TF-IDF Feature Representation and Trains Baseline Models:
      1. TF-IDF (Term Frequency - Inverse Document Frequency): Transforms text 
         into weighted numerical n-gram features (unigrams + bigrams).
      2. Logistic Regression: Linear probabilistic classifier with L2 regularization.
      3. Linear Support Vector Machine (LinearSVC): Max-margin linear classifier.
      
    Parameters:
      X_train (pd.Series): Preprocessed text narratives for training.
      y_train (pd.Series): Ground-truth HFACS category labels.
      
    Returns:
      vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
      models (dict): Dictionary of fitted scikit-learn models.
    """
    print("[INFO] Building TF-IDF Feature Representation (unigrams + bigrams)...")
    # Sublinear tf scaling (1 + log(tf)) helps damp the effect of repetitive words
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    print(f"[INFO] Vocabulary size extracted: {len(vectorizer.get_feature_names_out())} features.")
    
    models = {}
    
    # Model 1: Logistic Regression
    print("[INFO] Training Model 1: Logistic Regression...")
    log_reg = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
    log_reg.fit(X_train_tfidf, y_train)
    models['Logistic Regression'] = log_reg
    
    # Model 2: Linear Support Vector Machine (LinearSVC)
    print("[INFO] Training Model 2: Linear Support Vector Machine (LinearSVC)...")
    svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight='balanced', dual='auto')
    svm_model.fit(X_train_tfidf, y_train)
    models['Linear SVM (LinearSVC)'] = svm_model
    
    return vectorizer, models


# =============================================================================
# Step 5: Model Evaluation & Visualization Module
# =============================================================================
def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str, save_filename: str = None):
    """
    Generates and displays a formatted Seaborn heatmap for the confusion matrix.
    
    Parameters:
      cm (np.ndarray): Calculated confusion matrix array.
      labels (list): List of class label names in order.
      title (str): Plot title.
      save_filename (str, optional): File path to save plot PNG image.
    """
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar=True, square=True)
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Predicted HFACS Category', fontsize=10)
    plt.ylabel('Actual (Ground Truth) Category', fontsize=10)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300)
        print(f"[INFO] Saved confusion matrix plot to: {save_filename}", flush=True)
    plt.close()


def evaluate_model(model, vectorizer, X_test: pd.Series, y_test: pd.Series, 
                   model_name: str, dataset_name: str = "In-Domain Test Set", 
                   labels: list = None, save_fig: bool = False):
    """
    Evaluates a trained classifier on a test dataset.
    
    Computes:
      - Precision, Recall, Macro & Weighted F1-scores per class
      - Overall Accuracy
      - Confusion Matrix
      
    Parameters:
      model: Trained scikit-learn classifier model.
      vectorizer: Fitted TF-IDF vectorizer.
      X_test (pd.Series): Preprocessed text narratives.
      y_test (pd.Series): Ground-truth labels.
      model_name (str): Identifier name for the model.
      dataset_name (str): Description of the evaluation dataset.
      labels (list): Class names.
      save_fig (bool): Whether to save confusion matrix image.
    """
    print(f"\n===============================================================================")
    print(f" EVALUATION: {model_name} | {dataset_name}")
    print(f"===============================================================================")
    
    # Transform test features using the existing fitted vectorizer
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    if labels is None:
        labels = sorted(list(set(y_test).union(set(y_pred))))
        
    # Print Classification Report
    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report (Precision, Recall, F1-Score per HFACS Class):")
    print(report)
    
    # Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Plot heatmap
    safe_name = re.sub(r'\W+', '_', f"{model_name}_{dataset_name}").lower()
    fig_filename = f"cm_{safe_name}.png" if save_fig else None
    plot_confusion_matrix(cm, labels, f"Confusion Matrix: {model_name}\n({dataset_name})", fig_filename)
    
    return y_pred, report, cm


# =============================================================================
# Step 6: Cross-Domain Transferability Evaluation Module
# =============================================================================
def transfer_test(models: dict, vectorizer, ntsb_filepath: str, labels: list = HFACS_CLASSES):
    """
    Cross-Domain Transfer Test:
    Evaluates the trained model (fitted ONLY on ASRS near-miss data) on the separate 
    NTSB accident report dataset without any re-training or domain fine-tuning.
    
    Academic Rationale for Transfer Test:
    -----------------------------------
    ASRS reports are voluntary, self-reported narratives written by pilots/controllers 
    shortly after near-miss events (informal, first-person perspective).
    NTSB reports are official, third-person formal investigations written by accident 
    investigators (formal tone, past tense, factual focus).
    Evaluating cross-domain performance assesses the generalizability and robustness 
    of TF-IDF + linear classifier baselines across linguistic domain shifts.
    
    Parameters:
      models (dict): Trained dictionary of baseline models.
      vectorizer: Fitted TF-IDF vectorizer.
      ntsb_filepath (str): Path to NTSB CSV dataset.
      labels (list): List of target HFACS labels.
    """
    print("\n" + "#"*79)
    print(" STARTING CROSS-DOMAIN TRANSFER TEST (ASRS Trained -> NTSB Tested)")
    print("#"*79)
    
    df_ntsb = load_data(ntsb_filepath)
    df_ntsb = preprocess_dataset(df_ntsb, text_column='text')
    
    X_ntsb = df_ntsb['clean_text']
    y_ntsb = df_ntsb['label']
    
    for model_name, model in models.items():
        evaluate_model(
            model=model,
            vectorizer=vectorizer,
            X_test=X_ntsb,
            y_test=y_ntsb,
            model_name=model_name,
            dataset_name="Cross-Domain Transfer Test (NTSB Data)",
            labels=labels,
            save_fig=True
        )


# =============================================================================
# Step 7: Main Execution Pipeline
# =============================================================================
def run_pipeline(asrs_filepath: str, ntsb_filepath: str):
    """
    Runs the complete modular baseline pipeline end-to-end.
    
    Parameters:
      asrs_filepath (str): Path to ASRS CSV file.
      ntsb_filepath (str): Path to NTSB CSV file.
    """
    print("\n===============================================================================")
    print(" STEP 1 & 2: LOADING & PREPROCESSING ASRS TRAINING DATA")
    print("===============================================================================")
    
    # Load ASRS Dataset
    df_asrs = load_data(asrs_filepath)
    
    # Preprocess text
    print("[INFO] Preprocessing text narratives (lowercasing, punctuation, stopwords, tokenization)...")
    df_asrs = preprocess_dataset(df_asrs, text_column='text')
    
    X = df_asrs['clean_text']
    y = df_asrs['label']
    
    # Step 4: Train/Test Split (80/20 Stratified)
    print("\n===============================================================================")
    print(" STEP 4: TRAIN / TEST SPLIT (80% Train, 20% In-Domain Test)")
    print("===============================================================================")
    
    # Handle sample size check for stratified split safety
    min_class_count = y.value_counts().min()
    use_stratify = y if min_class_count >= 2 else None
    if use_stratify is None:
        print("[WARNING] Extremely small dataset sample detected. Disabling stratification for split.")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=use_stratify
    )
    
    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] In-domain test set size: {len(X_test)} samples")
    
    # Step 3: Train Baseline Models
    print("\n===============================================================================")
    print(" STEP 3: TRAINING BASELINE MODELS (TF-IDF + LOGISTIC REGRESSION / SVM)")
    print("===============================================================================")
    vectorizer, models = train_baseline(X_train, y_train)
    
    # Step 5: Evaluate In-Domain Test Set
    print("\n===============================================================================")
    print(" STEP 5: IN-DOMAIN EVALUATION (ASRS Test Set)")
    print("===============================================================================")
    
    all_classes = sorted(list(y.unique()))
    
    for model_name, model in models.items():
        evaluate_model(
            model=model,
            vectorizer=vectorizer,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            dataset_name="In-Domain Evaluation (ASRS Test Set)",
            labels=all_classes,
            save_fig=True
        )
        
    # Step 6: Cross-Domain Transfer Test on NTSB Dataset
    transfer_test(models, vectorizer, ntsb_filepath, labels=all_classes)
    
    print("\n===============================================================================")
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("===============================================================================")


if __name__ == "__main__":
    # Default CSV filenames (assumed to be in working directory)
    asrs_csv = "asrs_full.csv" if os.path.exists("asrs_full.csv") else "asrs_data.csv"
    ntsb_csv = "ntsb_data.csv"
    
    # Check if arguments provided via command line
    if len(sys.argv) >= 3:
        asrs_csv = sys.argv[1]
        ntsb_csv = sys.argv[2]
        
    print(f"[START] Running HFACS Aviation Human-Factors Baseline Pipeline...")
    print(f"[START] ASRS dataset path: {asrs_csv}")
    print(f"[START] NTSB dataset path: {ntsb_csv}")
    
    run_pipeline(asrs_filepath=asrs_csv, ntsb_filepath=ntsb_csv)
