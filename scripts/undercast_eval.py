#!/usr/bin/env python3
"""Score already-trained undercast models on the holdouts, without retraining.

``train_undercast_obs.py`` fits every source and writes the fitted preprocessor
and three classifiers to disk. Anything that wants to *look* at those models --
figures, ensemble experiments, source comparisons -- needs their probabilities on
the holdout splits, and refitting to get them would be both slow and slightly
dishonest: a refit is a different model, so its numbers would not be the ones the
page quotes. This loads the exact artifacts instead.

The one thing to be careful about is row alignment. Each source keeps only the
rows where its own archive was live, so HRRR's base-rate holdout has more rows
than ECMWF's. Anything comparing two sources therefore has to intersect on
`valid_utc` + `target_lead_h` first -- see ``aligned_probabilities``.

Not a script; import it.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_undercast_obs import (  # noqa: E402
    MODEL_NAMES, MODEL_SOURCES, SHORT_NAME, SOURCES, TARGET,
    assign_splits, load_obs_data, rows_for_source, select_features,
)

MODELS_DIR = "files/weather/models/obs"
CSV_DIR = "files/weather/csv/obs"
LABELS = "files/weather/csv/MtWashington_undercast_orig.csv"
RECORD = "files/weather/obs/undercast_record.csv"
KEY = ["valid_utc", "target_lead_h"]


def load_frame(csv_dir=CSV_DIR, labels=LABELS, record=RECORD, cache=None):
    """The assembled, split-assigned frame; cached as a pickle if asked."""
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    df = assign_splits(load_obs_data(csv_dir, labels, record))
    if cache:
        df.to_pickle(cache)
    return df


def load_artifacts(source, models_dir=MODELS_DIR):
    """The fitted preprocessor, the three classifiers, and the metadata."""
    meta = json.load(open(os.path.join(models_dir, f"model_metadata_{source}.json")))
    pre = joblib.load(os.path.join(models_dir, f"preprocessor_{source}.pkl"))
    xg = xgb.XGBClassifier()
    xg.load_model(os.path.join(models_dir, f"xgboost_best_f1_{source}.json"))
    models = {
        "XGBoost": xg,
        "Random Forest": joblib.load(
            os.path.join(models_dir, f"random_forest_best_f1_{source}.pkl")),
        "Gradient Boosting": joblib.load(
            os.path.join(models_dir, f"gradient_boosting_best_f1_{source}.pkl")),
    }
    return pre, models, meta


def split_rows(df, source, split):
    """Rows of one split that this source actually covers.

    `holdout_webcam` additionally drops rows with no human label, because the
    whole point of that split is to be scored against a person looking at a
    photograph rather than against the remark screen.
    """
    sub = rows_for_source(df, source)
    sub = sub[sub["split_eff"] == split]
    if split == "holdout_webcam":
        sub = sub[sub["hand_label"].notna()]
    elif split == "train":
        sub = sub[~sub["screen_ambiguous"]]
    return sub


def truth(rows, split):
    """The label to score against: human eyes on the webcam split, else the screen."""
    if split == "holdout_webcam":
        return (rows["hand_label"] >= 0.5).astype(int).to_numpy()
    return rows[TARGET].to_numpy()


def probabilities(rows, source, pre, models):
    """{algorithm: P(undercast)} for the given rows."""
    if not len(rows):
        return {n: np.array([]) for n in models}
    X = pre.transform(select_features(rows, source))
    return {n: m.predict_proba(X)[:, 1] for n, m in models.items()}


def score_split(df, source, split, models_dir=MODELS_DIR):
    """Everything downstream needs about one (source, split) pair."""
    pre, models, meta = load_artifacts(source, models_dir)
    rows = split_rows(df, source, split)
    return {
        "rows": rows,
        "y": truth(rows, split),
        "proba": probabilities(rows, source, pre, models),
        "thresholds": {n: meta[n]["threshold"] for n in MODEL_NAMES},
        "meta": meta,
    }


def aligned_probabilities(df, sources, split, algorithm="XGBoost",
                          models_dir=MODELS_DIR):
    """Per-source probabilities on the rows EVERY listed source covers.

    Comparing sources on their own row sets compares the weather of different
    years as much as the models: HRRR's holdout reaches back to 2014, ECMWF's
    starts in 2022. Intersecting on (valid_utc, target_lead_h) makes every source
    answer the same questions.

    Returns (index_frame, {source: proba}, y).
    """
    frames, probs = {}, {}
    for s in sources:
        pre, models, _ = load_artifacts(s, models_dir)
        rows = split_rows(df, s, split)
        p = probabilities(rows, s, pre, models)[algorithm]
        f = rows[KEY].copy()
        f[s] = p
        f["_y"] = truth(rows, split)
        frames[s] = f

    common = None
    for s in sources:
        k = frames[s].set_index(KEY).index
        common = k if common is None else common.intersection(k)
    for s in sources:
        f = frames[s].set_index(KEY).loc[common]
        probs[s] = f[s].to_numpy()
        y = f["_y"].to_numpy()
    return common.to_frame(index=False), probs, y
