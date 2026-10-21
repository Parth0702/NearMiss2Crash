import sys
import os
import time
import pandas as pd
sys.path.insert(0, os.path.abspath("."))

from evaluation import run_evaluation_pipeline, HFACS_CLASSES, bootstrap_f1, get_unified_keyword_labeler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

print("[INFO] Running quick sample sanity test of baseline pipeline...")
t0 = time.time()

# Load 500 rows from asrs_full.csv
df = pd.read_csv("asrs_full.csv")
df_labeled = df[df["label"] != "Unlabeled"].sample(n=500, random_state=42)

train_df, test_df = train_test_split(df_labeled, test_size=0.2, random_state=42, stratify=df_labeled["label"])
print(f"Sample train: {len(train_df)}, test: {len(test_df)}")

vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
X_train = vec.fit_transform(train_df["text"].fillna(""))
y_train = train_df["label"].values

X_test = vec.transform(test_df["text"].fillna(""))
y_test = test_df["label"].values

clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
b_mean, b_ci = bootstrap_f1(y_test, preds, n=100)

print(f"Sample test Macro-F1: {macro_f1:.4f}, Bootstrap mean: {b_mean:.4f}, CI: {b_ci}")
print(f"Completed in {time.time() - t0:.2f} seconds.")
