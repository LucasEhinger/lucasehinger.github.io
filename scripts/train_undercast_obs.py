#!/usr/bin/env python3
"""Train undercast classifiers on the per-observation forecast dataset.

Supersedes ``train_undercast_models.py``, which learned from 589 hand-labeled
DATES with forecast fields sampled at the wrong time of day. This reads the
output of ``fetch_nwp_at_obs.py``: one row per (observation, forecast lead), with
fields valid at the observation's own time, ~3,500 positives instead of 24.

Four things are done differently, each because the old way was actively wrong.

1. MISSING IS NOT THE MEDIAN.
   GRIB omits cloud ceiling/base/top where there is no cloud, so those cells
   arrive as nan -- and nan means CLEAR, which is about the most informative
   state there is for undercast. Verified on the rehearsal data: rows with a nan
   HRRR ceiling have a median low-cloud cover of 0.0%, rows with a number have
   39.9%. The old pipeline ran SimpleImputer(strategy="median") over these, which
   recoded every clear-sky row as "ceiling at 1,582 m" -- the opposite of the
   truth. Here every numeric gets a constant out-of-range fill plus a
   missingness indicator, and the cloud-geometry columns additionally get an
   explicit _no_cloud flag distinguishing "model says no cloud" (nan in the CSV)
   from "never fetched" (empty in the CSV). Those are different facts and the
   CSV preserves the difference, so the features should too.

2. EACH SOURCE TRAINS ON ITS OWN DATE WINDOW.
   The archives start at different times -- HRRR 2014, NAM 2020, GFS/RAP 2021,
   ECMWF 2022 -- so a row outside a model's window has that model's columns
   entirely blank. Training HRRR on those rows is free data; training ECMWF on
   them is training on nothing. A source keeps only rows where its own lead
   column is populated. The combined "all" source keeps only rows where every
   source is present, which is also the only situation it can be applied in.

3. THE TWO HOLDOUTS ARE NAMED, AND SEPARATED BY DATE -- NOT BY OBSERVATION.
   Negatives were subsampled 5:1, so precision measured on the training sample is
   meaningless -- it is computed against a ~17% base rate where reality is 2.7%.
   The threshold is therefore tuned on `holdout_baserate` (a full year at 3-hourly
   steps, unsampled, true base rate) and the headline numbers come from
   `holdout_webcam` (the hand-labeled days, scored against HUMAN webcam labels
   rather than the remark screen that generated the training labels).

   The sampler assigns those splits per OBSERVATION, and that is not sufficient.
   Each observation is one hour, and a day holds ~24 of them, so holding out
   03:50 while training on 02:50 of the same day holds out nothing: 535 of the
   589 webcam-holdout dates also carried training observations, and every single
   webcam holdout week overlapped a training week. A model could learn a day's
   synoptic pattern from one hour and be "tested" on the next.

   assign_splits() therefore drops any training row whose DATE falls in a holdout
   (plus a one-day buffer, since multi-day inversions correlate neighbours). It
   costs ~17% of training observations and leaves the holdouts as contiguous
   TEMPORAL blocks, which is a stronger test anyway -- a forward test rather than
   interpolation between training hours.

4. FOLDS GROUP BY WEEK, NOT BY DATE.
   A multi-day inversion makes consecutive days highly correlated, so
   date-grouped folds still leak across a Tuesday/Wednesday undercast. Grouping
   by ISO week is the conservative choice.

Metrics are also broken out BY LEAD, which is the question the project has never
actually answered: if skill is decent at ~1 h and collapses by 24 h, the models
resolve inversions but cannot forecast them; if it is poor at 1 h too, they never
resolve them at all.

Artifacts are written to a SEPARATE directory from the deployed ones, because the
live weather_to_json.py still expects the old column names. Cutting over means
updating both at once.

Usage:
    python3 scripts/train_undercast_obs.py
    python3 scripts/train_undercast_obs.py --sources hrrr all --report-only
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

RANDOM_STATE = 23
TARGET = "is_undercast"
SOURCES = ["hrrr", "nam", "gfs", "rap", "ecmwf", "nbm", "all"]
MODEL_SOURCES = ["hrrr", "nam", "gfs", "rap", "ecmwf", "nbm"]
CV_SPLITS = 5
MODEL_NAMES = ("XGBoost", "Random Forest", "Gradient Boosting")
SHORT_NAME = {"XGBoost": "XGB", "Random Forest": "RF", "Gradient Boosting": "GB"}
FILL_VALUE = -9999.0
# A lead group needs at least this many positives before its own F1-optimal
# threshold is preferred over the global one. The base-rate holdout carries ~86
# positives total, so ~29 per lead -- enough to see a trend, not enough to pin a
# threshold to two decimal places.
MIN_POS_FOR_LEAD_THRESHOLD = 25

# Columns whose emptiness is a physical statement ("no cloud") rather than a gap.
# GRIB only defines cloud geometry where cloud exists.
CLOUD_GEOMETRY = re.compile(
    r"^(cloud_ceiling|cloud_base|cloud_top|cdcb_)", re.I
)

# Never features: identifiers, audit trail, label/split bookkeeping -- and the
# lead columns, for two independent reasons.
#
# Leakage: a model's maximum forecast hour grew with its versions, so lead_hrrr
# is {1, 2, 15} for 2014-2016 rows and {1, 23, 47} for 2021+ rows. That makes it
# a near-proxy for the year, which would smuggle back the observer-drift confound
# the (year, month, hour) stratified sampling exists to remove.
#
# Deployability: target_lead_h only ever takes the values 1, 24 and 48 in
# training, but the live page forecasts every valid time out to 48 h. Feeding
# lead=14 to trees whose splits were learned on {1,24,48} is out-of-distribution
# for no benefit. Skill-versus-lead is still measured -- by GROUPING the holdouts
# on target_lead_h, which needs the column but not the feature.
DROP_ALWAYS = {TARGET, "valid_utc", "model_valid_utc", "split", "split_eff",
               "date", "week", "year", "hand_label", "target_lead_h",
               "screen_ambiguous"}


def _is_lead_col(c):
    return c.startswith("lead_")


def load_obs_data(csv_dir, labels_path=None, record_path=None):
    """Concatenate shard CSVs; add time features and the hand labels."""
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no shard CSVs in {csv_dir}")
    # keep_default_na=False so "" and "nan" stay DISTINGUISHABLE: "" means the
    # field was never fetched, "nan" means the model reported no value, which for
    # cloud geometry means clear sky. Collapsing both to NaN on read throws that
    # away irrecoverably.
    frames = [pd.read_csv(p, keep_default_na=False, dtype=str) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(paths)} shard files -> {len(raw):,} rows, {raw.shape[1]} cols")

    df = pd.DataFrame(index=raw.index)
    df["valid_utc"] = raw["valid_utc"]
    df["split"] = raw["split"]
    df[TARGET] = raw[TARGET].astype(int)
    dt = pd.to_datetime(raw["valid_utc"], format="%Y-%m-%dT%H:%M", utc=True)
    df["date"] = dt.dt.strftime("%Y-%m-%d")
    iso = dt.dt.isocalendar()
    df["week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    df["year"] = dt.dt.year

    # Cyclical time features. Raw month/day impose a false ordering (December is
    # adjacent to January, day-of-month means nothing at all), and because the
    # negatives were matched to each positive's own (year, month, hour) cell,
    # these carry no marginal signal by construction -- they are here only for
    # interactions with the atmospheric fields.
    month, hour = dt.dt.month, dt.dt.hour
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # Kept for grouping and reporting, explicitly NOT a feature (see DROP_ALWAYS).
    df["target_lead_h"] = pd.to_numeric(raw["target_lead_h"], errors="coerce")

    # Weather columns: everything except bookkeeping and the meta_ audit trail.
    skip = {"valid_utc", "model_valid_utc", "target_lead_h", "split", TARGET,
            "year", "month", "hour_utc"}
    wcols = [c for c in raw.columns if c not in skip and not c.startswith("meta_")]
    # Build every weather column up front and concat ONCE. Inserting ~160 columns
    # one at a time fragments the frame and pandas rightly complains.
    built = {}
    for c in wcols:
        col = raw[c]
        built[c] = pd.to_numeric(col.replace("", np.nan), errors="coerce")
        if CLOUD_GEOMETRY.search(c):
            # "nan" in the file = the model ran and reported no cloud. "" = the
            # field was never retrieved. Only the first is evidence of clear sky.
            built[f"{c}_no_cloud"] = (col.str.lower() == "nan").astype(int)
    df = pd.concat([df, pd.DataFrame(built, index=raw.index)], axis=1)

    if record_path and os.path.exists(record_path):
        # Re-derive the label from the undercast record instead of trusting the
        # is_undercast column the fetch job baked into each shard. The screen is
        # the thing most likely to be revised, and re-running the 12-hour GRIB
        # download just to change a label would be absurd -- the forecast fields
        # do not depend on the label at all. This also picks up the rare case
        # where a different report now represents an hour.
        rec = pd.read_csv(record_path, usecols=["valid_utc", "label"])
        lab = dict(zip(rec["valid_utc"], rec["label"]))
        mapped = df["valid_utc"].map(lab)
        missing = int(mapped.isna().sum())
        relabeled = pd.Series(
            np.where(mapped.isna(), df[TARGET], (mapped == "undercast").astype(int)),
            index=df.index,
        ).astype(int)
        changed = int((relabeled != df[TARGET]).sum())
        df[TARGET] = relabeled
        # Flag, do not drop. Ambiguous observations (scattered decks, narrow
        # near-misses) must leave TRAINING -- calling them negative puts the
        # worst label noise right on the decision boundary. But they have to stay
        # in the holdouts: the webcam holdout is scored against HUMAN labels,
        # where the screen's uncertainty is irrelevant and dropping rows would
        # discard real evaluation days; and the base-rate holdout needs the honest
        # population, since an observation the screen could not call is still one
        # the deployed model will be handed.
        df["screen_ambiguous"] = (mapped == "ambiguous").fillna(False).astype(bool)
        print(f"labels re-derived from {record_path}: {changed:,} changed, "
              f"{missing:,} not found (kept the shard value), "
              f"{int(df['screen_ambiguous'].sum()):,} flagged ambiguous")
    else:
        df["screen_ambiguous"] = False

    if labels_path and os.path.exists(labels_path):
        hand = {}
        for r in pd.read_csv(labels_path).to_dict("records"):
            try:
                d = pd.to_datetime(r["Short Date"], format="%m/%d/%y").strftime("%Y-%m-%d")
                hand[d] = float(r["Avg"])
            except (ValueError, TypeError, KeyError):
                continue
        df["hand_label"] = df["date"].map(lambda d: hand.get(d, np.nan))
        n = int(df["hand_label"].notna().sum())
        print(f"joined hand labels onto {n:,} rows "
              f"({int((df['hand_label'] >= 0.5).sum()):,} human-scored undercast)")
    else:
        df["hand_label"] = np.nan
    return df


def assign_splits(df, buffer_days=1):
    """Enforce date-level separation between train and the holdouts.

    Returns df with a `split_eff` column: the sampler's split, except training
    rows that collide with a holdout date are relabeled "dropped_overlap" and
    never trained on. See point 3 of the module docstring for why per-observation
    splitting was not enough.
    """
    hold = set()
    for sp in ("holdout_webcam", "holdout_baserate"):
        days = pd.to_datetime(df.loc[df["split"] == sp, "date"].unique())
        for off in range(-buffer_days, buffer_days + 1):
            hold.update((days + pd.Timedelta(days=off)).strftime("%Y-%m-%d"))
    collide = (df["split"] == "train") & df["date"].isin(hold)
    df = df.copy()
    df["split_eff"] = df["split"].where(~collide, "dropped_overlap")

    n_tr = int((df["split_eff"] == "train").sum())
    n_drop = int(collide.sum())
    print(f"date-level separation: dropped {n_drop:,} training rows that shared a "
          f"date with a holdout ({100*n_drop/max(n_tr+n_drop,1):.0f}% of train)")
    # Prove it worked rather than trusting it.
    tr_d = set(df.loc[df["split_eff"] == "train", "date"])
    for sp in ("holdout_webcam", "holdout_baserate"):
        hd = set(df.loc[df["split"] == sp, "date"])
        assert not (tr_d & hd), f"{len(tr_d & hd)} dates still shared with {sp}"
    print(f"  verified: 0 dates shared between train and either holdout")
    return df


def rows_for_source(df, source):
    """Rows where this source actually has data.

    Outside a model's archive window its columns are entirely blank, so those
    rows teach it nothing -- but they are perfectly good rows for whichever model
    DOES cover them. "all" requires every source present, which is both the only
    honest way to train a combined model and the only situation it can be used in.
    """
    if source == "all":
        mask = np.ones(len(df), dtype=bool)
        for m in MODEL_SOURCES:
            mask &= df[f"lead_{m}"].notna().to_numpy()
        return df[mask]
    return df[df[f"lead_{source}"].notna()]


def select_features(df, source):
    cols = [c for c in df.columns
            if c not in DROP_ALWAYS and not _is_lead_col(c)]
    time_cols = ["month_sin", "month_cos", "hour_sin", "hour_cos"]
    if source != "all":
        keep = [c for c in cols
                if c.endswith(f"_{source}")
                or c.endswith(f"_{source}_no_cloud")
                or c in time_cols]
    else:
        keep = cols
    return df[keep]


def make_preprocessor(X):
    """Constant out-of-range fill + explicit missingness indicators.

    Median imputation is wrong here (see module docstring): it rewrites "no
    cloud" as "cloud at the typical height". A constant far outside the observed
    range lets a single tree split isolate the missing rows, and add_indicator
    makes the missingness itself a first-class feature instead of something the
    model has to infer from a suspicious sentinel value.
    """
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
        (SimpleImputer(strategy="constant", fill_value=FILL_VALUE), num_cols),
        # features="all" is the point, and SimpleImputer(add_indicator=True)
        # cannot do it: that defaults to "missing-only", emitting indicators only
        # for columns that had a gap AT FIT TIME. The holdouts are different eras
        # from the training window, so a column that is complete in training and
        # missing in a holdout row (a throttled cell, a field retired or added)
        # would arrive with no indicator at all -- filled with FILL_VALUE and
        # indistinguishable from a real reading of -9999. A tree split at
        # "ceiling < 1000" then sends it down the LOW-CEILING branch and predicts
        # cloud where there is none. Emitting an indicator for every numeric makes
        # the column set identical across every split, always.
        (MissingIndicator(features="all"), num_cols),
    )
    return pre, cat_cols, num_cols


def build_models(y_train):
    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    return {
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, scale_pos_weight=neg / pos,
            n_jobs=-1, verbosity=0, n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE, n_estimators=300, max_depth=3,
            learning_rate=0.05, subsample=0.8,
        ),
    }


def fit_model(model, name, X, y):
    if name == "Gradient Boosting":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def scores(y_true, proba, thr):
    pred = (np.asarray(proba) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "n": int(len(pred)),
    }


def best_f1_threshold(y_true, proba):
    """F1-maximizing threshold, or 0.5 if no threshold separates anything.

    Without the guard a model with zero skill returns the lowest threshold tried
    (every candidate scores F1=0, and the first one wins the > comparison), which
    is the "predict undercast always" cut -- the worst possible default dressed up
    as a tuned parameter.
    """
    best = (0.5, -1.0)
    for thr in np.linspace(0.02, 0.98, 193):
        f1 = f1_score(y_true, (np.asarray(proba) >= thr).astype(int), zero_division=0)
        if f1 > best[1]:
            best = (float(thr), float(f1))
    return best[0] if best[1] > 0 else 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="files/weather/csv/obs")
    ap.add_argument("--out-dir", default="files/weather/models/obs")
    ap.add_argument("--labels", default="files/weather/csv/MtWashington_undercast_orig.csv")
    ap.add_argument("--record", default="files/weather/obs/undercast_record.csv",
                    help="authoritative labels, re-joined on valid_utc so a change "
                         "to the screen never requires re-downloading GRIB")
    ap.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    ap.add_argument("--buffer-days", type=int, default=1,
                    help="also exclude this many days either side of every "
                         "holdout date, since consecutive days correlate")
    ap.add_argument("--report-only", action="store_true",
                    help="evaluate without writing model artifacts")
    args = ap.parse_args()

    df = load_obs_data(args.csv_dir, args.labels, args.record)
    df = assign_splits(df, buffer_days=args.buffer_days)
    print()
    for source in args.sources:
        try:
            train_source(df, source, args.out_dir, report_only=args.report_only)
        except Exception as e:
            print(f"[{source:>5}] FAILED: {type(e).__name__}: {e}")


def train_source(df, source, out_dir, report_only=False):
    sub = rows_for_source(df, source)
    # Ambiguous rows are excluded from training only -- see load_obs_data.
    tr = sub[(sub["split_eff"] == "train") & ~sub["screen_ambiguous"]]
    base = sub[sub["split_eff"] == "holdout_baserate"]
    web = sub[(sub["split_eff"] == "holdout_webcam") & sub["hand_label"].notna()]
    if len(tr) < 50 or int(tr[TARGET].sum()) < 10:
        print(f"[{source:>5}] skipped: only {len(tr)} train rows / "
              f"{int(tr[TARGET].sum())} positives")
        return

    Xtr, ytr = select_features(tr, source), tr[TARGET].to_numpy()
    groups = tr["week"].to_numpy()
    print(f"[{source:>5}] {Xtr.shape[1]:3d} features | train {len(tr):,} rows "
          f"({int(ytr.sum()):,} pos, {100*ytr.mean():.1f}%) over "
          f"{tr['week'].nunique()} weeks, {tr['year'].min()}-{tr['year'].max()}")

    # --- out-of-fold predictions on the training split (week-grouped) ---------
    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = {n: np.full(len(ytr), np.nan) for n in MODEL_NAMES}
    for tr_i, te_i in cv.split(Xtr, ytr, groups):
        pre, _, _ = make_preprocessor(Xtr.iloc[tr_i])
        A = pre.fit_transform(Xtr.iloc[tr_i])
        B = pre.transform(Xtr.iloc[te_i])
        for name, model in build_models(ytr[tr_i]).items():
            fit_model(model, name, A, ytr[tr_i])
            oof[name][te_i] = model.predict_proba(B)[:, 1]

    # --- fit on all training rows, then evaluate on the untouched holdouts ----
    pre_final, cat_cols, num_cols = make_preprocessor(Xtr)
    A = pre_final.fit_transform(Xtr)
    final = build_models(ytr)
    for name, model in final.items():
        fit_model(model, name, A, ytr)

    meta = {}
    for name in MODEL_NAMES:
        m = {"oof_roc_auc": float(roc_auc_score(ytr, oof[name])),
             "oof_pr_auc": float(average_precision_score(ytr, oof[name]))}
        # Threshold comes from the TRUE-base-rate holdout. Tuning it on the
        # training split would tune against a 17% positive rate that does not
        # exist in the world (negatives were subsampled 5:1).
        if len(base) and base[TARGET].nunique() > 1:
            pb = final[name].predict_proba(pre_final.transform(select_features(base, source)))[:, 1]
            thr = best_f1_threshold(base[TARGET].to_numpy(), pb)
            m["threshold"] = thr
            m["threshold_tuned_on"] = "holdout_baserate"
            # NB: the threshold was fitted on this same set, so these numbers are
            # optimistic. The clean comparison is webcam_human below, which uses
            # this threshold on data that had no say in choosing it.
            m["baserate"] = scores(base[TARGET].to_numpy(), pb, thr)
            m["baserate"]["note"] = "threshold fitted on this set -- optimistic"
            m["baserate"]["roc_auc"] = float(roc_auc_score(base[TARGET], pb))
            m["baserate"]["pr_auc"] = float(average_precision_score(base[TARGET], pb))
        else:
            thr = best_f1_threshold(ytr, oof[name])
            m["threshold"] = thr
            m["threshold_tuned_on"] = "train OOF (no base-rate holdout available)"

        # Headline: human webcam truth, never the remark screen.
        if len(web):
            pw = final[name].predict_proba(pre_final.transform(select_features(web, source)))[:, 1]
            yw = (web["hand_label"] >= 0.5).astype(int).to_numpy()
            if yw.sum():
                m["webcam_human"] = scores(yw, pw, thr)
                m["webcam_human"]["roc_auc"] = float(roc_auc_score(yw, pw))
                by = {}
                for lead, g in web.groupby("target_lead_h"):
                    pg = final[name].predict_proba(
                        pre_final.transform(select_features(g, source)))[:, 1]
                    yg = (g["hand_label"] >= 0.5).astype(int).to_numpy()
                    if yg.sum():
                        by[int(lead)] = scores(yg, pg, thr)
                m["webcam_human_by_lead"] = by
        meta[name] = m

    # by-lead on the base-rate holdout too -- more positives, so more stable.
    # Also tune a SEPARATE threshold per lead: the model is deliberately
    # lead-agnostic (lead is not a feature), but its calibration is not -- a
    # 48 h forecast is less sharp than a 1 h one, so one global cut either
    # over-fires at long lead or under-fires at short. The live page knows which
    # horizon each point is and can pick the matching threshold.
    if len(base):
        for name in MODEL_NAMES:
            by, thr_by = {}, {}
            for lead, g in base.groupby("target_lead_h"):
                if g[TARGET].nunique() < 2:
                    continue
                pg = final[name].predict_proba(
                    pre_final.transform(select_features(g, source)))[:, 1]
                yg = g[TARGET].to_numpy()
                npos = int(yg.sum())
                if npos >= MIN_POS_FOR_LEAD_THRESHOLD:
                    t, src = best_f1_threshold(yg, pg), "own lead"
                else:
                    # Too few positives to fit a threshold here; a per-lead cut
                    # fitted to ~15 events is noise dressed as precision.
                    t, src = meta[name]["threshold"], "global (too few positives)"
                thr_by[int(lead)] = t
                by[int(lead)] = {
                    **scores(yg, pg, t),
                    "threshold": t,
                    "threshold_source": src,
                    "n_positives": npos,
                    "roc_auc": float(roc_auc_score(yg, pg)),
                    "pr_auc": float(average_precision_score(yg, pg)),
                }
            meta[name]["baserate_by_lead"] = by
            meta[name]["threshold_by_lead"] = thr_by

    for name in MODEL_NAMES:
        m = meta[name]
        b = m.get("baserate", {})
        w = m.get("webcam_human", {})
        print(f"         {SHORT_NAME[name]:3s} thr={m['threshold']:.2f} "
              f"| OOF AUC={m['oof_roc_auc']:.3f} PR={m['oof_pr_auc']:.3f} "
              f"| baserate P={b.get('precision', float('nan')):.2f} "
              f"R={b.get('recall', float('nan')):.2f} AUC={b.get('roc_auc', float('nan')):.3f} "
              f"| webcam P={w.get('precision', float('nan')):.2f} "
              f"R={w.get('recall', float('nan')):.2f}")
        bl = m.get("baserate_by_lead", {})
        if bl:
            print("             by lead: " + "  ".join(
                f"{k}h thr={v['threshold']:.2f}{'*' if 'global' in v['threshold_source'] else ''} "
                f"AUC={v['roc_auc']:.3f} P={v['precision']:.2f} R={v['recall']:.2f} "
                f"(n+={v['n_positives']})"
                for k, v in sorted(bl.items())))

    if report_only:
        return
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pre_final, os.path.join(out_dir, f"preprocessor_{source}.pkl"))
    final["XGBoost"].save_model(os.path.join(out_dir, f"xgboost_best_f1_{source}.json"))
    joblib.dump(final["Random Forest"],
                os.path.join(out_dir, f"random_forest_best_f1_{source}.pkl"))
    joblib.dump(final["Gradient Boosting"],
                os.path.join(out_dir, f"gradient_boosting_best_f1_{source}.pkl"))
    meta.update({
        "feature_columns": list(Xtr.columns),
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
        "target_column": TARGET,
        "random_state": RANDOM_STATE,
        "source": source,
        "n_train_rows": int(len(tr)),
        "n_train_positives": int(ytr.sum()),
        "n_train_weeks": int(tr["week"].nunique()),
        "year_range": [int(tr["year"].min()), int(tr["year"].max())],
        "imputation": f"constant fill {FILL_VALUE} + missingness indicators; "
                      "cloud-geometry columns carry an explicit _no_cloud flag "
                      "(nan in source = model reports no cloud = clear)",
        "split_separation": "by DATE with a 1-day buffer, not by observation",
        "evaluation": "trained on split=train only; threshold tuned on "
                      "holdout_baserate (true 2.7% base rate); headline metrics on "
                      "holdout_webcam scored against HUMAN webcam labels; "
                      "week-grouped 5-fold CV for OOF AUC",
    })
    with open(os.path.join(out_dir, f"model_metadata_{source}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)


if __name__ == "__main__":
    main()
