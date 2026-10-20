"""Helper: writes hyperparameter_tuning.py with precision-focused F0.5 tuning."""
import pathlib

CODE = r'''#!/usr/bin/env python3
"""
Precision-Focused Hyperparameter Tuning  --  F0.5 Objective
Objectives: Maximise Macro-Precision, minimise false positives
Techniques:
  1. F0.5 CV objective (precision weighted 2x over recall)
  2. Precision-first search spaces (L1/ElasticNet LR, tight SVM C,
     RF min_impurity_decrease, high NB alpha)
  3. CalibratedClassifierCV (isotonic regression)
  4. Per-class decision threshold tau_c with abstention fallback
"""
import sys, os, json, random, warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
warnings.filterwarnings("ignore")

if sys.stdout.encoding != "utf-8":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
    recall_score, fbeta_score, classification_report)
import joblib

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

HFACS_CLASSES = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]
OUT_DIR = Path("week4_results/hyperparameter_tuning")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def macro_fbeta(y_true, y_pred, beta=0.5):
    return fbeta_score(y_true, y_pred, beta=beta, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset():
    data_path = "ntsb_processed_test.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df)} samples")
    print("[INFO] Class Distribution:")
    for cls_name, cnt in df["label"].value_counts().items():
        print(f"       - {cls_name:22s}: {cnt}")
    X = df["clean_text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f"[INFO] Train/CV pool: {len(X_train)} | Holdout Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Precision-first search spaces
# ---------------------------------------------------------------------------
SHARED_VEC = {
    "vec__ngram_range":  [(1,1),(1,2)],
    "vec__max_features": [800,1500,2500,4000],
    "vec__sublinear_tf": [True],
    "vec__min_df":       [3,5,8,12],
    "vec__max_df":       [0.75,0.80,0.85,0.90],
}

MODEL_SPACES = {
    "logistic_regression": {
        "clf__C_log10":      (-2.5, 0.0),
        "clf__penalty":      ["l1","l2","elasticnet"],
        "clf__class_weight": ["balanced", None],
        "clf__l1_ratio":     [0.0,0.3,0.5,0.7,1.0],
    },
    "svm": {
        "clf__C_log10":      (-2.5,-0.3),
        "clf__loss":         ["squared_hinge"],
        "clf__class_weight": ["balanced", None],
    },
    "random_forest": {
        "clf__n_estimators":          [100,200],
        "clf__max_depth":             [4,6,8,12],
        "clf__min_samples_split":     [10,20,40],
        "clf__min_samples_leaf":      [4,8,16],
        "clf__max_features":          ["sqrt","log2"],
        "clf__min_impurity_decrease": [0.0,0.001,0.005],
    },
    "naive_bayes": {
        "clf__alpha_log10": (0.0, 1.3),
        "clf__fit_prior":   [True, False],
    },
}


def sample_params(space):
    p = {}
    for k, v in space.items():
        if isinstance(v, list):    p[k] = random.choice(v)
        elif isinstance(v, tuple): p[k] = random.uniform(v[0], v[1])
    return p


def build_pipeline(model_name, params):
    vec = TfidfVectorizer(
        ngram_range  = params.get("vec__ngram_range", (1,2)),
        max_features = params.get("vec__max_features", 2500),
        sublinear_tf = params.get("vec__sublinear_tf", True),
        min_df       = params.get("vec__min_df", 5),
        max_df       = params.get("vec__max_df", 0.85),
        stop_words   = "english")

    if model_name == "logistic_regression":
        penalty  = params.get("clf__penalty", "l2")
        c_val    = 10.0 ** params.get("clf__C_log10", -1.0)
        solver   = "saga" if penalty in ("l1","elasticnet") else "lbfgs"
        l1_ratio = params.get("clf__l1_ratio", 0.5) if penalty == "elasticnet" else None
        clf = LogisticRegression(C=c_val, penalty=penalty, solver=solver,
            l1_ratio=l1_ratio,
            class_weight=params.get("clf__class_weight", "balanced"),
            max_iter=2000, random_state=RANDOM_STATE)

    elif model_name == "svm":
        c_val = 10.0 ** params.get("clf__C_log10", -1.0)
        base  = LinearSVC(C=c_val, loss=params.get("clf__loss", "squared_hinge"),
            dual="auto", class_weight=params.get("clf__class_weight", "balanced"),
            max_iter=3000, random_state=RANDOM_STATE)
        clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")

    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators          = params.get("clf__n_estimators", 100),
            max_depth             = params.get("clf__max_depth", 8),
            min_samples_split     = params.get("clf__min_samples_split", 10),
            min_samples_leaf      = params.get("clf__min_samples_leaf", 4),
            max_features          = params.get("clf__max_features", "sqrt"),
            min_impurity_decrease = params.get("clf__min_impurity_decrease", 0.0),
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

    elif model_name == "naive_bayes":
        alpha = 10.0 ** params.get("clf__alpha_log10", 0.5)
        clf = MultinomialNB(alpha=alpha, fit_prior=params.get("clf__fit_prior", True))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("vec", vec), ("clf", clf)])


def build_baseline(model_name):
    defaults = {
        "logistic_regression": {
            "vec__ngram_range":(1,2),"vec__max_features":10000,"vec__sublinear_tf":True,
            "vec__min_df":1,"vec__max_df":1.0,"clf__C_log10":1.0,
            "clf__class_weight":"balanced","clf__penalty":"l2"},
        "svm": {
            "vec__ngram_range":(1,2),"vec__max_features":10000,"vec__sublinear_tf":True,
            "vec__min_df":1,"vec__max_df":1.0,"clf__C_log10":0.7,
            "clf__loss":"squared_hinge","clf__class_weight":"balanced"},
        "random_forest": {
            "vec__ngram_range":(1,2),"vec__max_features":5000,"vec__sublinear_tf":True,
            "vec__min_df":1,"vec__max_df":1.0,"clf__n_estimators":100,
            "clf__max_depth":None,"clf__min_samples_split":2,"clf__min_samples_leaf":1,
            "clf__max_features":"sqrt","clf__min_impurity_decrease":0.0},
        "naive_bayes": {
            "vec__ngram_range":(1,2),"vec__max_features":5000,"vec__sublinear_tf":True,
            "vec__min_df":1,"vec__max_df":1.0,"clf__alpha_log10":-2.0,"clf__fit_prior":True},
    }
    p = defaults[model_name]
    return build_pipeline(model_name, p), p


# ---------------------------------------------------------------------------
# 5-Fold CV with F0.5 objective + gap penalty
# ---------------------------------------------------------------------------
def evaluate_cv(pipe, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    Xa, ya = np.array(X, dtype=object), np.array(y, dtype=object)
    tr_fb, vl_fb, vl_pr, vl_rc, vl_f1, vl_ac = [], [], [], [], [], []
    for tr_i, vl_i in skf.split(Xa, ya):
        Xtr, ytr = Xa[tr_i].tolist(), ya[tr_i].tolist()
        Xv,  yv  = Xa[vl_i].tolist(), ya[vl_i].tolist()
        pipe.fit(Xtr, ytr)
        tr_fb.append(macro_fbeta(ytr, pipe.predict(Xtr)))
        pv = pipe.predict(Xv)
        vl_fb.append(macro_fbeta(yv, pv))
        vl_pr.append(precision_score(yv, pv, average="macro", zero_division=0))
        vl_rc.append(recall_score(yv, pv, average="macro", zero_division=0))
        vl_f1.append(f1_score(yv, pv, average="macro", zero_division=0))
        vl_ac.append(accuracy_score(yv, pv))
    cv_tr = float(np.mean(tr_fb)); cv_vl = float(np.mean(vl_fb))
    gap   = max(0.0, cv_tr - cv_vl)
    return {
        "cv_train_f05":    round(cv_tr, 4),
        "cv_val_f05":      round(cv_vl, 4),
        "overfitting_gap": round(gap, 4),
        "selection_score": round(cv_vl - 0.5*gap, 4),
        "cv_val_precision":round(float(np.mean(vl_pr)), 4),
        "cv_val_recall":   round(float(np.mean(vl_rc)), 4),
        "cv_val_macro_f1": round(float(np.mean(vl_f1)), 4),
        "cv_val_accuracy": round(float(np.mean(vl_ac)), 4),
    }


# ---------------------------------------------------------------------------
# Per-class threshold tuning
# ---------------------------------------------------------------------------
def tune_thresholds(proba, y_cal, cl_order, min_recall=0.55):
    grid = np.arange(0.25, 0.85, 0.025)
    ya   = np.array(y_cal)
    thresholds = {}
    for cls in HFACS_CLASSES:
        if cls not in cl_order:
            thresholds[cls] = 0.5; continue
        col   = cl_order.index(cls)
        p_c   = proba[:, col]
        yb    = (ya == cls).astype(int)
        best_tau, best_prec = 0.5, 0.0
        for tau in grid:
            pb  = (p_c >= tau).astype(int)
            tp  = int(np.sum((pb==1)&(yb==1)))
            fp  = int(np.sum((pb==1)&(yb==0)))
            fn  = int(np.sum((pb==0)&(yb==1)))
            pr  = tp/(tp+fp+1e-9)
            rc  = tp/(tp+fn+1e-9)
            if rc >= min_recall and pr > best_prec:
                best_prec = pr; best_tau = tau
        thresholds[cls] = round(float(best_tau), 3)
    return thresholds


def predict_thresh(proba, cl_order, thresholds, abstain="Ambiguous"):
    preds = []
    for row in proba:
        best_cls, best_p = abstain, -1.0
        for cls in HFACS_CLASSES:
            if cls not in cl_order: continue
            col = cl_order.index(cls)
            p   = row[col]; tau = thresholds.get(cls, 0.5)
            if p >= tau and p > best_p:
                best_p = p; best_cls = cls
        preds.append(best_cls)
    return preds


# ---------------------------------------------------------------------------
# fANOVA variance decomposition
# ---------------------------------------------------------------------------
def fanova_importance(trials_df, model_name, target="cv_val_f05"):
    mdf = trials_df[trials_df["model"]==model_name].copy()
    if len(mdf) < 5: return {}
    tv = mdf[target].var()
    if tv == 0 or np.isnan(tv): return {}
    imp = {}
    for col in [c for c in mdf.columns if c.startswith("vec__") or c.startswith("clf__")]:
        s = mdf[col].dropna()
        if s.empty: continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            try:    binned = pd.qcut(s, q=min(4,s.nunique()), duplicates="drop")
            except: binned = s.astype(str)
        else: binned = s.astype(str)
        vi  = s.index
        grp = mdf.loc[vi].groupby(binned, observed=False)[target]
        gm  = grp.mean(); gc = grp.count(); om = mdf.loc[vi,target].mean()
        ssb = float(np.sum(gc*(gm-om)**2)); sst = float(np.sum((mdf.loc[vi,target]-om)**2))
        r   = ssb/sst if sst > 0 else 0.0
        imp[col.replace("vec__","TF-IDF: ").replace("clf__","Clf: ")] = max(0.0, r)
    return imp


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------
def run_hyperparameter_optimization(num_trials_per_model=30):
    print("="*80)
    print(" WEEK 4: PRECISION-FOCUSED HYPERPARAMETER TUNING  (F0.5 OBJECTIVE)")
    print("   Primary goal: Maximize Precision, Minimize False Positives")
    print("="*80)

    X_train, X_test, y_train, y_test = load_dataset()
    # Calibration set (20% of train) for isotonic calibration + threshold search
    X_tr_main, X_cal, y_tr_main, y_cal = train_test_split(
        X_train, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train)

    all_trials = []; baseline_recs = {}; tuned_recs = {}
    full_spaces = {m: {**SHARED_VEC, **s} for m, s in MODEL_SPACES.items()}

    for model_name, full_space in full_spaces.items():
        print("\n" + "#"*78)
        print(f" OPTIMIZING: {model_name.upper()}  ({num_trials_per_model} Trials | F0.5 | 5-Fold CV)")
        print("#"*78)

        # --- Baseline (default, unregularised) ---
        bp, _ = build_baseline(model_name)
        bcv   = evaluate_cv(bp, X_train, y_train)
        bp.fit(X_train, y_train)
        bpreds = bp.predict(X_test)
        bprec  = precision_score(y_test, bpreds, average="macro", zero_division=0)
        bf05   = macro_fbeta(y_test, bpreds)
        bf1    = f1_score(y_test, bpreds, average="macro", zero_division=0)
        baseline_recs[model_name] = {
            "cv_train_f05": bcv["cv_train_f05"], "cv_val_f05": bcv["cv_val_f05"],
            "overfit_gap": bcv["overfitting_gap"], "test_precision": round(bprec,4),
            "test_f05": round(bf05,4), "test_f1": round(bf1,4)}
        print(f"[BASELINE] CV Train F0.5: {bcv['cv_train_f05']:.4f} | "
              f"CV Val F0.5: {bcv['cv_val_f05']:.4f} | "
              f"Gap: {bcv['overfitting_gap']:.4f} | "
              f"Holdout Precision: {bprec:.4f}")

        # --- Random search (F0.5 selection objective) ---
        best_score, best_rec, best_pipe = -999.0, None, None
        for t in range(num_trials_per_model):
            params = sample_params(full_space)
            pipe   = build_pipeline(model_name, params)
            cv_res = evaluate_cv(pipe, X_train, y_train)
            rec = {"model": model_name, "trial": t+1}
            rec.update(cv_res)
            for k, v in params.items():
                rec[k] = str(v) if isinstance(v, (list,tuple)) else v
            all_trials.append(rec)
            if cv_res["selection_score"] > best_score:
                best_score = cv_res["selection_score"]; best_rec = rec; best_pipe = pipe
            if (t+1) % 10 == 0 or t == 0:
                print(f"  Trial {t+1:02d}/{num_trials_per_model} | "
                      f"Val F0.5: {cv_res['cv_val_f05']:.4f} | "
                      f"Val Prec: {cv_res['cv_val_precision']:.4f} | "
                      f"Gap: {cv_res['overfitting_gap']:.4f} | "
                      f"SelScore: {cv_res['selection_score']:.4f}")

        # --- Fit best on full train, calibrate on cal set ---
        best_pipe.fit(X_train, y_train)
        try:
            cal_clf    = CalibratedClassifierCV(best_pipe["clf"], cv="prefit", method="isotonic")
            X_cal_vec  = best_pipe["vec"].transform(X_cal)
            cal_clf.fit(X_cal_vec, y_cal)
            try:    cl_order = list(cal_clf.classes_)
            except: cl_order = HFACS_CLASSES
            X_test_vec  = best_pipe["vec"].transform(X_test)
            proba_test  = cal_clf.predict_proba(X_test_vec)
            proba_cal   = cal_clf.predict_proba(X_cal_vec)
            thresholds  = tune_thresholds(proba_cal, y_cal, cl_order)
            print(f"  [THRESHOLDS] {model_name}: {thresholds}")
            preds_def   = np.array(cl_order)[np.argmax(proba_test, axis=1)]
            preds_tuned = predict_thresh(proba_test, cl_order, thresholds)
        except Exception as e:
            print(f"  [WARN] Calibration error: {e}. Using raw pipeline.")
            preds_def   = best_pipe.predict(X_test)
            preds_tuned = list(preds_def)
            thresholds  = {c: 0.5 for c in HFACS_CLASSES}
            cl_order    = HFACS_CLASSES

        # Metrics — default threshold
        hp_def = precision_score(y_test, preds_def, average="macro", zero_division=0)
        hr_def = recall_score(y_test, preds_def, average="macro", zero_division=0)
        hf_def = macro_fbeta(y_test, preds_def)
        h1_def = f1_score(y_test, preds_def, average="macro", zero_division=0)
        ha_def = accuracy_score(y_test, preds_def)

        # Metrics — tuned threshold (abstentions excluded)
        paired = [(yt, yp) for yt, yp in zip(y_test, preds_tuned) if yp != "Ambiguous"]
        n_abs  = sum(1 for yp in preds_tuned if yp == "Ambiguous")
        if paired:
            yt_f = [p[0] for p in paired]; yp_f = [p[1] for p in paired]
            hp_t = precision_score(yt_f, yp_f, average="macro", zero_division=0)
            hr_t = recall_score(yt_f, yp_f, average="macro", zero_division=0)
            hf_t = macro_fbeta(yt_f, yp_f)
            h1_t = f1_score(yt_f, yp_f, average="macro", zero_division=0)
            ha_t = accuracy_score(yt_f, yp_f)
        else:
            hp_t = hr_t = hf_t = h1_t = ha_t = 0.0; yt_f = []; yp_f = []

        joblib.dump({"pipeline": best_pipe, "thresholds": thresholds, "cl_order": cl_order},
                    OUT_DIR / f"best_{model_name}.joblib")

        tuned_recs[model_name] = {
            "cv_val_f05": best_rec["cv_val_f05"], "cv_val_precision": best_rec["cv_val_precision"],
            "overfit_gap": best_rec["overfitting_gap"],
            "test_precision_def": round(hp_def,4), "test_recall_def": round(hr_def,4),
            "test_f05_def": round(hf_def,4), "test_f1_def": round(h1_def,4), "test_acc_def": round(ha_def,4),
            "test_precision_tuned": round(hp_t,4), "test_recall_tuned": round(hr_t,4),
            "test_f05_tuned": round(hf_t,4), "test_f1_tuned": round(h1_t,4), "test_acc_tuned": round(ha_t,4),
            "abstain_count": n_abs, "thresholds": thresholds}

        pg = hp_t - bprec
        print(f"\n[OPTIMAL] {model_name}")
        print(f"  Default  tau  -> Precision: {hp_def:.4f} | F0.5: {hf_def:.4f} | Recall: {hr_def:.4f}")
        print(f"  Tuned  tau_c  -> Precision: {hp_t:.4f} | F0.5: {hf_t:.4f} | Recall: {hr_t:.4f} | Abstentions: {n_abs}/{len(y_test)}")
        print(f"  Precision gain vs baseline: {pg:+.4f} | Abstain rate: {n_abs/len(y_test)*100:.1f}%")
        if yp_f:
            print(classification_report(yt_f, yp_f, digits=4, zero_division=0))

    # --- Save all trials ---
    trials_df = pd.DataFrame(all_trials)
    trials_df.to_csv(OUT_DIR / "trials_precision.csv", index=False)
    print(f"\n[SUCCESS] Exported {len(trials_df)} trials -> {OUT_DIR}/trials_precision.csv")

    # --- fANOVA ---
    print("\n" + "="*78)
    print(" fANOVA HYPERPARAMETER IMPORTANCE (Target: CV Val F0.5)")
    print("="*78)
    imp_summary = {}
    for model_name in MODEL_SPACES.keys():
        imp_f  = fanova_importance(trials_df, model_name, "cv_val_f05")
        imp_pr = fanova_importance(trials_df, model_name, "cv_val_precision")
        imp_summary[model_name] = {"f05": imp_f, "precision": imp_pr}
        sorted_i = sorted(imp_f.items(), key=lambda x: x[1], reverse=True)
        print(f"\nModel: {model_name.upper()}")
        for rank, (param, var) in enumerate(sorted_i, 1):
            bar = "#"*int(var*30)
            print(f"  {rank:2d}. {param:32s}: {var*100:5.1f}%  [{bar:<30s}]")
        if sorted_i:
            plt.figure(figsize=(8,4.5))
            pn = [x[0] for x in sorted_i][::-1]; pv = [x[1]*100 for x in sorted_i][::-1]
            bars = plt.barh(pn, pv, color="#c0392b")
            plt.xlabel("% of Val F0.5 Variance Explained (fANOVA)", fontsize=11)
            mn = model_name.replace("_"," ").title()
            plt.title(f"Hyperparameter Importance (F0.5): {mn}", fontsize=12, fontweight="bold")
            plt.xlim(0, max(max(pv)*1.15, 10))
            for bar in bars:
                w = bar.get_width()
                plt.text(w+0.5, bar.get_y()+bar.get_height()/2, f"{w:.1f}%",
                         va="center", ha="left", fontsize=9)
            plt.tight_layout()
            plt.savefig(OUT_DIR/f"importance_{model_name}.png", dpi=300)
            plt.close()
    with open(OUT_DIR/"importance_summary.json","w",encoding="utf-8") as f:
        json.dump(imp_summary, f, indent=2)

    # --- Comparison table ---
    rows = []
    for m in MODEL_SPACES.keys():
        b = baseline_recs[m]; t = tuned_recs[m]
        rows.append({
            "Model":                         m.replace("_"," ").title(),
            "Baseline Precision":             b["test_precision"],
            "Tuned Precision (default tau)":  t["test_precision_def"],
            "Tuned Precision (class tau_c)":  t["test_precision_tuned"],
            "Precision Gain (tau_c)": round(t["test_precision_tuned"]-b["test_precision"],4),
            "Baseline F0.5":          b["test_f05"],
            "Tuned F0.5 (tau_c)":     t["test_f05_tuned"],
            "Tuned Recall (tau_c)":   t["test_recall_tuned"],
            "Abstention Rate (%)": round(t["abstain_count"]/len(y_test)*100,1),
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_DIR/"precision_comparison.csv", index=False)
    print("\n"+"="*80)
    print(" PRECISION-FOCUSED TUNING -- FINAL COMPARISON")
    print("="*80)
    print(comp.to_string(index=False))

    # Precision comparison chart
    try:
        fig, ax = plt.subplots(figsize=(11,5))
        x = np.arange(len(comp)); w = 0.26
        ax.bar(x-w, comp["Baseline Precision"],           w, label="Baseline (argmax)",  color="#95a5a6")
        ax.bar(x,   comp["Tuned Precision (default tau)"],w, label="Tuned, default tau", color="#3498db")
        ax.bar(x+w, comp["Tuned Precision (class tau_c)"],w, label="Tuned + class tau_c",color="#e74c3c")
        ax.set_ylabel("Macro Precision (higher = fewer FPs)", fontsize=11)
        ax.set_title("Precision: Baseline vs Regularised Tuning vs Class Thresholds",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(comp["Model"], fontsize=10)
        ax.set_ylim(0,1.05); ax.legend(frameon=True); ax.grid(axis="y",linestyle="--",alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUT_DIR/"precision_comparison.png", dpi=300); plt.close()
        print("[SUCCESS] precision_comparison.png saved")
    except Exception as e: print(f"[WARN] Chart error: {e}")

    # Abstention rate chart
    try:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(comp["Model"], comp["Abstention Rate (%)"], color="#e67e22", alpha=0.85)
        ax.set_ylabel("% samples abstained (human review)", fontsize=10)
        ax.set_title("Abstention Rate per Model (Below Precision Threshold)",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(comp["Abstention Rate (%)"].max()*1.3, 5))
        ax.grid(axis="y",linestyle="--",alpha=0.5)
        for i,v in enumerate(comp["Abstention Rate (%)"]):
            ax.text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR/"abstention_rate.png", dpi=300); plt.close()
        print("[SUCCESS] abstention_rate.png saved")
    except Exception as e: print(f"[WARN] Chart error: {e}")

    print("\n"+"="*80)
    print(" PRECISION-FOCUSED TUNING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_hyperparameter_optimization(num_trials_per_model=30)
'''

pathlib.Path(r"d:/SC project/hyperparameter_tuning.py").write_text(CODE, encoding="utf-8")
print(f"WRITE OK  lines: {CODE.count(chr(10))}")
