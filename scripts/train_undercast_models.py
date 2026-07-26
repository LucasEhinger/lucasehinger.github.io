#!/usr/bin/env python3
"""Train undercast-prediction models for every forecast source in one run.

Reads the per-date labeled CSVs produced by ``weather_to_csv.py`` and, for each
source in {hrrr, nam, gfs, rap, ecmwf, nbm, all}, trains Random Forest, XGBoost,
and Gradient Boosting, tunes each to its best-F1 decision threshold, and writes
the artifacts the site loads at prediction time:

    files/weather/models/preprocessor_{source}.pkl
    files/weather/models/random_forest_best_f1_{source}.pkl
    files/weather/models/gradient_boosting_best_f1_{source}.pkl
    files/weather/models/xgboost_best_f1_{source}.json
    files/weather/models/model_metadata_{source}.json

Evaluation is leakage-safe and honest:
  * Every split/fold GROUPS BY DATE, so the many forecast-hour rows of one date
    never straddle train and eval (which would leak the answer and inflate F1).
  * The decision threshold is tuned on a validation split; the reported
    precision/recall/F1 come from a separate held-out test split.
  * Grouped 5-fold CV reports threshold-free ROC-AUC / PR-AUC (mean +/- std) as
    a robustness check.
  * Class imbalance is handled with class weights (RF class_weight, XGB
    scale_pos_weight, GB balanced sample_weight) -- no SMOTE, which for tree
    models on heavily-imputed features mostly adds noise.
The deployed models + preprocessor are refit on ALL data so no rows are wasted;
the saved threshold is the validation-tuned one and the metadata records the
honest held-out (test) and CV metrics.

Per-source feature selection mirrors weather_to_json.py's prediction path (drop
fxx; for a named source keep *_{source} plus month/day; for all keep every
feature column). New CSV columns are picked up automatically.

Usage:
    python3 scripts/train_undercast_models.py            # all sources, defaults
    python3 scripts/train_undercast_models.py --sources hrrr all
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

RANDOM_STATE = 23
TARGET = "is_undercast"
SOURCES = ["hrrr", "nam", "gfs", "rap", "ecmwf", "nbm", "all"]
CV_SPLITS = 5
MODEL_NAMES = ("XGBoost", "Random Forest", "Gradient Boosting")


def load_data(csv_dir):
    """Concatenate every *_MtWashington.csv, adding date/month/day from filename."""
    paths = sorted(glob.glob(os.path.join(csv_dir, "*_MtWashington.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_MtWashington.csv files found in {csv_dir}")
    frames = []
    for p in paths:
        t = pd.read_csv(p)
        date_part = os.path.basename(p).split("_", 1)[0]
        dt = pd.to_datetime(date_part, format="%Y-%m-%d", errors="coerce")
        t["date"] = date_part  # group key: one weather event per date
        t["month"] = dt.month
        t["day"] = dt.day
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(paths)} files -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def select_features(df, source):
    """Feature matrix for a source, matching weather_to_json.py prediction time."""
    X = df.drop(columns=[TARGET, "date"], errors="ignore").drop(
        columns=["fxx"], errors="ignore"
    )
    if source != "all":
        keep = [c for c in X.columns if c.endswith(f"_{source}") or c in ("month", "day")]
        X = X[keep]
    return X


def make_preprocessor(X):
    """Fresh imputer/encoder; must be fit only on the current training rows."""
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    pre = make_column_transformer(
        (
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            cat_cols,
        ),
        (make_pipeline(SimpleImputer(strategy="median")), num_cols),
    )
    return pre, cat_cols, num_cols


def build_models(y_train):
    """Three tree ensembles, each handling class imbalance via class weights."""
    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    return {
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, scale_pos_weight=neg / pos, n_jobs=-1, verbosity=0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def fit_model(model, name, X, y):
    """GradientBoosting has no class_weight, so pass balanced sample_weight."""
    if name == "Gradient Boosting":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def best_threshold(y_true, proba):
    """Threshold in 0.05..0.95 that maximizes F1 (tuned on validation data)."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 100):
        f1 = f1_score(y_true, (proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr, best_f1


def metrics_at(y_true, proba, thr):
    """Precision/recall/F1 and error counts at a fixed threshold (test data)."""
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "fp": int(fp),
        "fn": int(fn),
    }


def grouped_cv_auc(X, y, groups):
    """Threshold-free ROC-AUC / PR-AUC per model via date-grouped stratified CV."""
    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = {name: {"roc": [], "pr": []} for name in MODEL_NAMES}
    for tr_i, te_i in cv.split(X, y, groups):
        pre, _, _ = make_preprocessor(X.iloc[tr_i])
        Xtr = pre.fit_transform(X.iloc[tr_i])
        Xte = pre.transform(X.iloc[te_i])
        ytr, yte = y.iloc[tr_i], y.iloc[te_i]
        if yte.nunique() < 2:
            continue  # AUC undefined for a single-class fold
        for name, model in build_models(ytr).items():
            fit_model(model, name, Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            scores[name]["roc"].append(roc_auc_score(yte, p))
            scores[name]["pr"].append(average_precision_score(yte, p))
    return scores


def train_source(df, source, out_dir):
    X = select_features(df, source)
    y = (df[TARGET] >= 0.5).astype(int)
    groups = df["date"]

    # --- date-grouped train / val / test (~60/20/20) ---
    dev_i, test_i = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(
            X, y, groups
        )
    )
    X_dev, y_dev, g_dev = X.iloc[dev_i], y.iloc[dev_i], groups.iloc[dev_i]
    X_test, y_test = X.iloc[test_i], y.iloc[test_i]
    tr_i, val_i = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE).split(
            X_dev, y_dev, g_dev
        )
    )
    X_tr, y_tr = X_dev.iloc[tr_i], y_dev.iloc[tr_i]
    X_val, y_val = X_dev.iloc[val_i], y_dev.iloc[val_i]

    pre, _, _ = make_preprocessor(X_tr)
    Xtr_p = pre.fit_transform(X_tr)
    Xval_p = pre.transform(X_val)
    Xtest_p = pre.transform(X_test)

    # --- fit eval models, tune threshold on val, report on held-out test ---
    metadata = {}
    for name, model in build_models(y_tr).items():
        fit_model(model, name, Xtr_p, y_tr)
        thr, val_f1 = best_threshold(y_val, model.predict_proba(Xval_p)[:, 1])
        test_m = metrics_at(y_test, model.predict_proba(Xtest_p)[:, 1], thr)
        metadata[name] = {
            "threshold_best_f1": thr,  # consumed by weather_to_json.py at predict time
            "val_f1_score": val_f1,
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_f1_score": test_m["f1_score"],
            "test_fp": test_m["fp"],
            "test_fn": test_m["fn"],
        }

    # --- grouped CV robustness metrics (threshold-free) ---
    cv = grouped_cv_auc(X, y, groups)
    for name in MODEL_NAMES:
        roc, pr = cv[name]["roc"], cv[name]["pr"]
        metadata[name].update(
            {
                "cv_roc_auc_mean": float(np.mean(roc)) if roc else None,
                "cv_roc_auc_std": float(np.std(roc)) if roc else None,
                "cv_pr_auc_mean": float(np.mean(pr)) if pr else None,
                "cv_pr_auc_std": float(np.std(pr)) if pr else None,
            }
        )

    # --- refit deployed models + preprocessor on ALL data (no rows wasted) ---
    pre_final, cat_cols, num_cols = make_preprocessor(X)
    X_all = pre_final.fit_transform(X)
    final = build_models(y)
    for name, model in final.items():
        fit_model(model, name, X_all, y)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pre_final, os.path.join(out_dir, f"preprocessor_{source}.pkl"))
    final["XGBoost"].save_model(os.path.join(out_dir, f"xgboost_best_f1_{source}.json"))
    joblib.dump(
        final["Random Forest"], os.path.join(out_dir, f"random_forest_best_f1_{source}.pkl")
    )
    joblib.dump(
        final["Gradient Boosting"],
        os.path.join(out_dir, f"gradient_boosting_best_f1_{source}.pkl"),
    )

    metadata.update(
        {
            "feature_columns": list(X.columns),
            "categorical_columns": cat_cols,
            "numerical_columns": num_cols,
            "target_column": TARGET,
            "random_state": RANDOM_STATE,
            "source": source,
            "n_rows": int(len(X)),
            "n_dates": int(groups.nunique()),
            "imbalance_strategy": "class weights (RF class_weight, XGB scale_pos_weight, "
            "GB balanced sample_weight)",
            "evaluation": "date-grouped 60/20/20 train/val/test; threshold tuned on val, "
            "metrics on test; robustness via 5-fold StratifiedGroupKFold; deployed models "
            "refit on all data",
        }
    )
    with open(os.path.join(out_dir, f"model_metadata_{source}.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    def pr_auc(name):
        v = metadata[name]["cv_pr_auc_mean"]
        return f"{v:.3f}" if v is not None else "n/a"

    print(
        f"[{source:>5}] {X.shape[1]:3d} feat / {groups.nunique()} dates | "
        f"test-F1 XGB={metadata['XGBoost']['test_f1_score']:.3f} "
        f"RF={metadata['Random Forest']['test_f1_score']:.3f} "
        f"GB={metadata['Gradient Boosting']['test_f1_score']:.3f} | "
        f"CV PR-AUC XGB={pr_auc('XGBoost')} RF={pr_auc('Random Forest')} "
        f"GB={pr_auc('Gradient Boosting')} -> saved 5 artifacts"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", default="files/weather/csv/ML/all/")
    ap.add_argument("--out-dir", default="files/weather/models/")
    ap.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    args = ap.parse_args()

    df = load_data(args.csv_dir)
    for source in args.sources:
        train_source(df, source, args.out_dir)


if __name__ == "__main__":
    main()
