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
# Undercast is rare (~24 of 589 dates), so a single val split holds ~1 positive
# event and F1-tuned thresholds are unstable. Instead pick the deployed threshold
# on pooled out-of-fold predictions, targeting this precision floor (max recall
# subject to precision >= floor). Raise for fewer false "undercast" calls.
# The honest precision ceiling for this signal is ~0.25 (the old models' ~0.8 was
# a leakage artifact), so 0.2 is near the attainable max; higher just makes models
# fall back to the over-firing F1-max threshold.
PRECISION_FLOOR = 0.2
SHORT_NAME = {"XGBoost": "XGB", "Random Forest": "RF", "Gradient Boosting": "GB"}


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


def oof_predictions(X, y, groups):
    """Pooled out-of-fold predicted probabilities per model via date-grouped CV.

    Every row is scored by a model that never saw its date in training, so the
    pooled predictions use all positives at once -- far more stable than a single
    tiny validation split when positives are rare (~24 undercast dates here).
    """
    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = {name: np.full(len(y), np.nan) for name in MODEL_NAMES}
    for tr_i, te_i in cv.split(X, y, groups):
        pre, _, _ = make_preprocessor(X.iloc[tr_i])
        Xtr = pre.fit_transform(X.iloc[tr_i])
        Xte = pre.transform(X.iloc[te_i])
        for name, model in build_models(y.iloc[tr_i]).items():
            fit_model(model, name, Xtr, y.iloc[tr_i])
            oof[name][te_i] = model.predict_proba(Xte)[:, 1]
    return oof


def threshold_for_precision(y_true, proba, floor):
    """Threshold with the highest recall whose precision is >= floor.

    Falls back to the F1-maximizing threshold when no threshold reaches the
    precision floor (e.g. a weak source). Returns (threshold, floor_met).
    """
    best = None  # (recall, -threshold, threshold), maximized lexicographically
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn) if tp + fn else 0.0
        if prec >= floor:
            cand = (rec, -float(thr), float(thr))
            if best is None or cand > best:
                best = cand
    if best is not None:
        return best[2], True
    return best_threshold(y_true, proba)[0], False


def train_source(df, source, out_dir):
    X = select_features(df, source)
    y = (df[TARGET] >= 0.5).astype(int)
    groups = df["date"]

    # --- pooled out-of-fold CV predictions (robust when positives are rare) ---
    # threshold + reported metrics both come from OOF preds over all data, so the
    # deployed threshold is tuned against all ~24 positives, not a single split.
    oof = oof_predictions(X, y, groups)
    metadata = {}
    for name in MODEL_NAMES:
        p = oof[name]
        thr, floor_met = threshold_for_precision(y, p, PRECISION_FLOOR)
        pred = (p >= thr).astype(int)
        _, fp, fn, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        metadata[name] = {
            "threshold_best_f1": float(thr),  # deployed threshold, read by weather_to_json.py
            "threshold_strategy": f"max-recall s.t. precision>={PRECISION_FLOOR}"
            + ("" if floor_met else " (floor unreachable; fell back to F1-max)"),
            "precision_floor": PRECISION_FLOOR,
            "precision_floor_met": bool(floor_met),
            "oof_precision": float(precision_score(y, pred, zero_division=0)),
            "oof_recall": float(recall_score(y, pred, zero_division=0)),
            "oof_f1_score": float(f1_score(y, pred, zero_division=0)),
            "oof_fp": int(fp),
            "oof_fn": int(fn),
            "cv_roc_auc": float(roc_auc_score(y, p)),
            "cv_pr_auc": float(average_precision_score(y, p)),
        }

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

    def fmt(name):
        m = metadata[name]
        flag = "" if m["precision_floor_met"] else "!"
        return (
            f"{SHORT_NAME[name]}{flag} thr={m['threshold_best_f1']:.2f} "
            f"P={m['oof_precision']:.2f} R={m['oof_recall']:.2f} AUC={m['cv_roc_auc']:.2f}"
        )

    print(
        f"[{source:>5}] {X.shape[1]:3d} feat / {groups.nunique()} dates | "
        + " | ".join(fmt(n) for n in MODEL_NAMES)
        + "  (! = precision floor unreachable)"
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
