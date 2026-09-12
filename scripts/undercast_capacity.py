#!/usr/bin/env python3
"""Does the combined undercast model have more capacity than its data supports?

The combined model reads 213 features. That number invites a fair objection: the
training split holds 13,305 rows, so the model looks wide relative to its data.
The objection is even stronger than the row count suggests, because rows are not
independent observations here -- each undercast hour is sampled once per forecast
lead, and consecutive hours of one cloud deck are one weather event. Counted in
events the training set holds 175 undercast days, which is under one event per
feature.

So this script asks whether that costs anything measurable, along three axes:

  1. EFFECTIVE SAMPLE SIZE -- rows, distinct valid hours, episodes, days.
  2. CAPACITY -- holdout skill as a function of how many features the model is
     given and how deep its trees are allowed to grow.
  3. WHETHER ANY OF IT IS REAL -- a paired day-block bootstrap on the
     differences, and, just as important, a NOISE FLOOR: the deployed config
     refit under several seeds. Without that floor the capacity table invites
     over-reading, because its spread turns out to be about the same size as the
     seed-to-seed spread of one unchanged configuration.

It also reports the SERVING cost of each feature budget: every retained feature
is a GRIB field the live job downloads every six hours, and derived features are
not free (``dRH_925_850_ecmwf`` needs two ECMWF humidity fields,
``t_summit_minus_t925_rap`` needs four RAP fields).

Nothing here refits the deployed artifacts or writes into the model directory --
it loads the fitted model to rank features, then trains throwaway variants. The
figures and metadata the site serves are untouched.

    python3 scripts/undercast_capacity.py                 # everything
    python3 scripts/undercast_capacity.py --quick         # skip the bootstraps

Runtime is dominated by sklearn's single-threaded GradientBoostingClassifier;
the full run is roughly 45 minutes.
"""
import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import undercast_eval as ue  # noqa: E402
from train_undercast_obs import (  # noqa: E402
    RANDOM_STATE, TARGET, make_preprocessor, select_features,
)

SPLITS = ("holdout_baserate", "holdout_webcam")
FEATURE_BUDGETS = (8, 15, 25, 40, 70, 120)
DEPTHS = (1, 2, 3, 4, 6)
DEPLOYED_DEPTH = 3
SEEDS = (23, 7, 101, 555, 9001)
BOOT_N = 2000
# Pressure levels that stack into a profile; mirrors train_undercast_obs.
LEVELS = (1000, 925, 850, 700)


# --------------------------------------------------------------------------- #
# effective sample size
# --------------------------------------------------------------------------- #
def effective_sample_size(df, n_features):
    """Rows overstate the information available; count events instead."""
    rows = ue.split_rows(df, "all", "train")
    y = rows[TARGET].astype(int)
    pos = rows[y == 1].copy()
    pos["valid_utc"] = pd.to_datetime(pos["valid_utc"], utc=True)

    hours = np.sort(pos["valid_utc"].unique())
    # Bridge gaps of up to 2 h: a deck that thins for an hour and returns is one
    # event meteorologically, and counting it twice would flatter the total.
    gaps = np.diff(hours).astype("timedelta64[h]").astype(int)
    episodes = 1 + int((gaps > 2).sum()) if len(hours) else 0

    counts = {
        "positive rows": int(y.sum()),
        "distinct positive valid hours": int(pos["valid_utc"].nunique()),
        "distinct undercast episodes": episodes,
        "distinct undercast days": int(pos["valid_utc"].dt.date.nunique()),
    }
    print(f"\ncombined-source train split: {len(rows):,} rows, "
          f"leads {sorted(rows['target_lead_h'].unique())}")
    print(f"{'counting unit':34s} {'n':>7}  per feature")
    for label, n in counts.items():
        print(f"  {label:32s} {n:>7,}  {n / n_features:>10.2f}")
    return counts


# --------------------------------------------------------------------------- #
# serving cost of a feature budget
# --------------------------------------------------------------------------- #
def grib_dependencies(col):
    """Raw fetched columns one model feature is built from."""
    c = col[:-len("_no_cloud")] if col.endswith("_no_cloud") else col
    if c in ("month_sin", "month_cos", "hour_sin", "hour_cos"):
        return set()
    for m in ue.MODEL_SOURCES:
        suf = f"_{m}"
        if not c.endswith(suf):
            continue
        stem = c[: -len(suf)]
        if stem == "dRH_925_850":
            return {f"rh_925mb{suf}", f"rh_850mb{suf}"}
        if stem == "dewpt_dep_2m":
            return {f"tmp_2m{suf}", f"dpt_2m{suf}"}
        mm = re.fullmatch(r"dT_(\d+)_(\d+)", stem)
        if mm:
            return {f"tmp_{mm.group(1)}mb{suf}", f"tmp_{mm.group(2)}mb{suf}"}
        mm = re.fullmatch(r"lapse_(\d+)_(\d+)", stem)
        if mm:
            lo, hi = mm.groups()
            return {f"tmp_{lo}mb{suf}", f"tmp_{hi}mb{suf}",
                    f"hgt_{lo}mb{suf}", f"hgt_{hi}mb{suf}"}
        if stem == "max_inversion":
            return {f"tmp_{p}mb{suf}" for p in LEVELS}
        if stem == "t_summit":
            return {f"tmp_850mb{suf}", f"tmp_700mb{suf}",
                    f"hgt_850mb{suf}", f"hgt_700mb{suf}"}
        mm = re.fullmatch(r"t_summit_minus_t(\d+)", stem)
        if mm:
            return {f"tmp_850mb{suf}", f"tmp_700mb{suf}", f"hgt_850mb{suf}",
                    f"hgt_700mb{suf}", f"tmp_{mm.group(1)}mb{suf}"}
        break
    return {c}  # a raw passthrough column


def fetch_cost(ranked, budgets):
    """GRIB fields the live job would still need at each feature budget."""
    print("\n=== serving cost: GRIB fields needed per run ===")
    print(f"{'features':>9} {'fields':>7}   per source")
    out = {}
    for k in list(budgets) + [len(ranked)]:
        need = set()
        for c in ranked[:k]:
            need |= grib_dependencies(c)
        by = {}
        for f in need:
            by[f.rsplit("_", 1)[-1]] = by.get(f.rsplit("_", 1)[-1], 0) + 1
        out[k] = len(need)
        per = " ".join(f"{m}={by.get(m, 0)}" for m in ue.MODEL_SOURCES)
        print(f"{k:>9} {len(need):>7}   {per}")
    return out


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _fit(cols, depth, Xtr, ytr, holds, seed=RANDOM_STATE):
    """Train one throwaway variant; return its probabilities on each holdout."""
    pre, _, _ = make_preprocessor(Xtr[cols])
    A = pre.fit_transform(Xtr[cols])
    model = GradientBoostingClassifier(
        random_state=seed, n_estimators=300, max_depth=depth,
        learning_rate=0.05, subsample=0.8,
    )
    model.fit(A, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return {s: model.predict_proba(pre.transform(v[0][cols]))[:, 1]
            for s, v in holds.items()}


def rank_features(models_dir=ue.MODELS_DIR):
    """Raw columns of the deployed model, most important first.

    Importances come from the deployed fit, which saw only the training split --
    so the holdouts play no part in the ranking and the numbers below are not
    optimistic in the leakage sense. They ARE optimistic in a subtler way, noted
    in the summary: the best budget is chosen after seeing holdout results.
    """
    pre, models, meta = ue.load_artifacts("all", models_dir)
    raw = meta["feature_columns"]
    index = {c: i for i, c in enumerate(raw)}
    acc = np.zeros(len(raw))
    for name, v in zip(pre.get_feature_names_out(),
                       models["Gradient Boosting"].feature_importances_):
        name = name.split("__")[-1].replace("missingindicator_", "")
        if name in index:
            acc[index[name]] += v
    return [raw[i] for i in np.argsort(acc)[::-1]], int((acc == 0).sum())


def score(probs, holds):
    return {s: (roc_auc_score(holds[s][1], probs[s]),
                average_precision_score(holds[s][1], probs[s])) for s in holds}


def sweep(label, configs, ranked, Xtr, ytr, holds):
    """One table: each config scored on both holdouts. Returns probabilities."""
    print(f"\n=== {label} ===")
    head = f"{'config':>28} |"
    for s in SPLITS:
        head += f" {s.replace('holdout_', ''):>9} AUC     PR |"
    print(head)
    probs = {}
    for name, (k, depth) in configs.items():
        probs[name] = _fit(ranked[:k], depth, Xtr, ytr, holds)
        sc = score(probs[name], holds)
        line = f"{name:>28} |"
        for s in SPLITS:
            line += f" {sc[s][0]:13.3f} {sc[s][1]:6.3f} |"
        print(line, flush=True)
    return probs


# --------------------------------------------------------------------------- #
# is any of it real?
# --------------------------------------------------------------------------- #
def noise_floor(ranked, Xtr, ytr, holds, seeds=SEEDS):
    """The same config, refit under several seeds.

    This is the number that decides how to read every table above. subsample=0.8
    makes training stochastic, so one configuration does not have one score -- it
    has a distribution. Any difference smaller than this spread is not a finding.
    """
    print(f"\n=== noise floor: deployed config under {len(seeds)} seeds ===")
    out = {}
    per_seed = [_fit(ranked, DEPLOYED_DEPTH, Xtr, ytr, holds, seed=sd)
                for sd in seeds]
    for s in SPLITS:
        vals = [roc_auc_score(holds[s][1], p[s]) for p in per_seed]
        out[s] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "min": float(min(vals)), "max": float(max(vals))}
        print(f"  {s:18s} ROC-AUC {np.mean(vals):.4f} +/- {np.std(vals):.4f} "
              f"(min {min(vals):.4f}, max {max(vals):.4f})", flush=True)
    return out


def boot_diff(y, dates, pa, pb, metric, n=BOOT_N, seed=11):
    """Paired day-block bootstrap of metric(b) - metric(a).

    Whole days are resampled, not rows: hours within a day are the same weather,
    so resampling rows would treat 24 correlated readings as 24 observations and
    return an interval several times too narrow.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(dates, return_inverse=True)
    by_day = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(by_day), len(by_day))
        idx = np.concatenate([by_day[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(metric(y[idx], pb[idx]) - metric(y[idx], pa[idx]))
    return float(np.mean(out)), [float(v) for v in np.percentile(out, [2.5, 97.5])]


def compare(probs, base, holds):
    """Every config against the deployed one, with intervals."""
    for s, (_, y, dates) in holds.items():
        print(f"\n=== {s} ({int(y.sum())} positives) vs deployed ===")
        for mname, metric in (("ROC-AUC", roc_auc_score),
                              ("PR-AUC", average_precision_score)):
            print(f"  {mname}")
            for name, p in probs.items():
                v = metric(y, p[s])
                if name == base:
                    print(f"    {name:32s} {v:.3f}   (reference)")
                    continue
                d, ci = boot_diff(y, dates, probs[base][s], p[s], metric)
                flag = "" if ci[0] < 0 < ci[1] else "  <- excludes zero"
                print(f"    {name:32s} {v:.3f}   {d:+.4f} "
                      f"[{ci[0]:+.4f}, {ci[1]:+.4f}]{flag}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None,
                    help="pickle path for the assembled frame (speeds up reruns)")
    ap.add_argument("--models-dir", default=ue.MODELS_DIR)
    ap.add_argument("--quick", action="store_true",
                    help="skip the bootstraps and the seed sweep")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    df = ue.load_frame(cache=a.cache)
    ranked, n_zero = rank_features(a.models_dir)
    n_feat = len(ranked)
    print(f"\ndeployed combined model: {n_feat} raw features, "
          f"{n_zero} with exactly zero importance")

    results = {"effective_sample_size": effective_sample_size(df, n_feat)}
    results["fetch_cost"] = fetch_cost(ranked, FEATURE_BUDGETS)

    tr = ue.split_rows(df, "all", "train")
    Xtr = select_features(tr, "all")
    ytr = tr[TARGET].astype(int).to_numpy()
    holds = {}
    for s in SPLITS:
        h = ue.split_rows(df, "all", s)
        holds[s] = (select_features(h, "all"), np.asarray(ue.truth(h, s)),
                    pd.to_datetime(h["valid_utc"], utc=True).dt.date.to_numpy())
        print(f"{s}: {len(h)} rows, {int(holds[s][1].sum())} positives")

    deployed = f"deployed: {n_feat} feats, depth {DEPLOYED_DEPTH}"
    by_k = {(f"{k} feats, depth {DEPLOYED_DEPTH}"): (k, DEPLOYED_DEPTH)
            for k in FEATURE_BUDGETS}
    by_k[deployed] = (n_feat, DEPLOYED_DEPTH)
    sweep("feature count (depth 3, as deployed)", by_k, ranked, Xtr, ytr, holds)

    by_d = {f"{n_feat} feats, depth {d}": (n_feat, d) for d in DEPTHS}
    sweep("tree depth (all features)", by_d, ranked, Xtr, ytr, holds)

    if a.quick:
        return

    results["noise_floor"] = noise_floor(ranked, Xtr, ytr, holds)
    combos = {
        deployed: (n_feat, DEPLOYED_DEPTH),
        "trim only: 40 feats, depth 3": (40, 3),
        "shallow only: all feats, depth 2": (n_feat, 2),
        "both: 40 feats, depth 2": (40, 2),
        "both: 40 feats, depth 1": (40, 1),
        "both: 25 feats, depth 2": (25, 2),
    }
    probs = sweep("trimmed and shallow together", combos, ranked, Xtr, ytr, holds)
    compare(probs, deployed, holds)

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
