#!/usr/bin/env python3
"""Train undercast-prediction models for every forecast source in one run.

Reads the per-date labeled CSVs produced by ``weather_to_csv.py`` and, for each
source in {hrrr, nam, gfs, rap, ecmwf, nbm, all}, trains Random Forest, XGBoost, and Gradient
Boosting (on SMOTE-balanced data), tunes each model to its best-F1 decision
threshold, and writes the artifacts the site loads at prediction time:

    files/weather/models/preprocessor_{source}.pkl
    files/weather/models/random_forest_best_f1_{source}.pkl
    files/weather/models/gradient_boosting_best_f1_{source}.pkl
    files/weather/models/xgboost_best_f1_{source}.json
    files/weather/models/model_metadata_{source}.json

Per-source feature selection mirrors weather_to_json.py's prediction path
exactly (drop ``fxx``; for a named source keep ``*_{source}`` plus month/day;
for ``all`` keep every feature column), so retrained models line up with the
columns the site feeds them. Any new columns present in the CSVs (e.g. the
hgt_*mb geopotential-height fields) are picked up automatically.

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
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib

RANDOM_STATE = 23
TARGET = "is_undercast"
SOURCES = ["hrrr", "nam", "gfs", "rap", "ecmwf", "nbm", "all"]


def load_data(csv_dir):
    """Concatenate every *_MtWashington.csv, adding month/day from the filename."""
    paths = sorted(glob.glob(os.path.join(csv_dir, "*_MtWashington.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_MtWashington.csv files found in {csv_dir}")
    frames = []
    for p in paths:
        t = pd.read_csv(p)
        date_part = os.path.basename(p).split("_", 1)[0]
        dt = pd.to_datetime(date_part, format="%Y-%m-%d", errors="coerce")
        t["month"] = dt.month
        t["day"] = dt.day
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(paths)} files -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def select_features(df, source):
    """Feature matrix for a source, matching weather_to_json.py prediction time."""
    X = df.drop(columns=[TARGET]).drop(columns=["fxx"], errors="ignore")
    if source != "all":
        keep = [c for c in X.columns if c.endswith(f"_{source}") or c in ("month", "day")]
        X = X[keep]
    return X


def best_f1_metrics(y_true, proba):
    """Sweep thresholds 0.05..0.95 and return metrics at the best-F1 threshold."""
    best = None
    for thresh in np.linspace(0.05, 0.95, 100):
        pred = (proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        f1 = f1_score(y_true, pred, zero_division=0)
        if best is None or f1 > best["f1_score"]:
            best = {
                "threshold_best_f1": float(thresh),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1_score": float(f1),
                "fp": int(fp),
                "fn": int(fn),
            }
    return best


def train_source(df, source, out_dir):
    X = select_features(df, source)
    y = df[TARGET]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    preprocess = make_column_transformer(
        (
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            cat_cols,
        ),
        (make_pipeline(SimpleImputer(strategy="median")), num_cols),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    y_train_bin = (y_train >= 0.5).astype(int)
    y_test_bin = (y_test >= 0.5).astype(int)

    X_train_p = preprocess.fit_transform(X_train)
    X_test_p = preprocess.transform(X_test)
    X_train_s, y_train_s = SMOTE(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(
        X_train_p, y_train_bin
    )

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_s, y_train_s)

    pos = max(int((y_train_s == 1).sum()), 1)
    neg = int((y_train_s == 0).sum())
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        scale_pos_weight=neg / pos,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train_s, y_train_s)

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb.fit(X_train_s, y_train_s)

    metadata = {}
    for name, model in (("XGBoost", xgb), ("Random Forest", rf), ("Gradient Boosting", gb)):
        proba = model.predict_proba(X_test_p)[:, 1]
        metadata[name] = best_f1_metrics(y_test_bin, proba)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(preprocess, os.path.join(out_dir, f"preprocessor_{source}.pkl"))
    xgb.save_model(os.path.join(out_dir, f"xgboost_best_f1_{source}.json"))
    joblib.dump(rf, os.path.join(out_dir, f"random_forest_best_f1_{source}.pkl"))
    joblib.dump(gb, os.path.join(out_dir, f"gradient_boosting_best_f1_{source}.pkl"))

    metadata.update(
        {
            "feature_columns": list(X.columns),
            "categorical_columns": cat_cols,
            "numerical_columns": num_cols,
            "target_column": TARGET,
            "random_state": RANDOM_STATE,
            "source": source,
        }
    )
    with open(os.path.join(out_dir, f"model_metadata_{source}.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[{source:>4}] {X.shape[1]:3d} features | "
        f"XGB F1={metadata['XGBoost']['f1_score']:.3f} "
        f"RF F1={metadata['Random Forest']['f1_score']:.3f} "
        f"GB F1={metadata['Gradient Boosting']['f1_score']:.3f} -> saved 5 artifacts"
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
