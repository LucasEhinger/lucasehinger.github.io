#!/usr/bin/env python3
"""Prove the live inference path builds the same features the model was trained on.

This is the check that would have caught the class of bug that has cost this
project the most time. Training features are built by
``train_undercast_obs.load_obs_data`` from shard CSVs; serving features are built
by ``weather_to_json.build_current_features`` from a live Herbie fetch. If those
two ever disagree -- a renamed column, a sentinel handled on one side only, a
time feature taken from the run time instead of the valid time -- the model still
returns a confident number and nothing anywhere raises.

The test replays real training rows through the SERVING code and compares, column
by column, against what the training code produced for the same rows.

One difference is expected and allowed: a shard CSV distinguishes "" (never
fetched) from "nan" (the model reported nothing), while a live fetch has no
"never fetched" state. Rows carrying any "" in a cloud-geometry column are
therefore skipped rather than fudged -- see normalize_weather_columns.

    python3 scripts/test_serving_features.py
    python3 scripts/test_serving_features.py --rows 500
"""
import argparse
import glob
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_undercast_obs import CLOUD_GEOMETRY, load_obs_data  # noqa: E402
import weather_to_json as wj  # noqa: E402

MODEL_META = "files/weather/models/obs/model_metadata_all.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="files/weather/csv/obs")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()

    wanted = json.load(open(MODEL_META))["feature_columns"]

    # results_to_dataframe() FILTERS its output down to an explicit
    # desired_columns list. A raw column the model needs but that list omits is
    # silently dropped before the features are ever built -- and because every
    # numeric carries a missingness indicator, the model would absorb it as "not
    # reported" rather than fail. Check the list itself, not just the code path.
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "weather_to_json.py"), encoding="utf-8").read()
    block = re.search(r"    desired_columns = \[(.*?)\n    \]", src, re.S)
    desired = set(re.findall(r'"([^"]+)"', block.group(1)))
    derived = re.compile(r"^(dT_|lapse_|t_summit|dRH_|dewpt_dep_|max_inversion_)")
    time_f = {"month_sin", "month_cos", "hour_sin", "hour_cos"}
    raw_needed = {c for c in wanted
                  if c not in time_f and not derived.search(c)
                  and not c.endswith("_no_cloud")}
    # build_current_features aliases the suffixed HRRR name onto the unsuffixed
    # fetch key, which the legacy preprocessors still need under its old name.
    aliased = ({"boundary_layer_cloud_layer_hrrr"}
               if "boundary_layer_cloud_layer" in desired else set())
    dropped = sorted(raw_needed - desired - aliased)
    print(f"desired_columns covers {len(raw_needed - set(dropped))}/{len(raw_needed)} "
          f"raw columns the model needs")
    if dropped:
        print(f"  DROPPED BEFORE FEATURE BUILDING: {dropped}")
        return 1

    # --- the training side, straight from the shards -------------------------
    train = load_obs_data(a.csv_dir)
    # valid_utc alone is NOT unique: every observation appears once per forecast
    # lead, with different model values each time. Key on both.
    train.index = (train["valid_utc"].astype(str) + "@"
                   + train["target_lead_h"].astype("Int64").astype(str))
    assert train.index.is_unique, "training key is still not unique"

    # --- the serving side: same rows, rebuilt through weather_to_json --------
    paths = sorted(glob.glob(os.path.join(a.csv_dir, "*.csv")))
    raw = pd.concat([pd.read_csv(p, keep_default_na=False, dtype=str)
                     for p in paths[:6]], ignore_index=True)

    # A live fetch cannot represent "never fetched", so only rows that are fully
    # populated in the cloud-geometry columns are comparable at all.
    geo = [c for c in raw.columns if CLOUD_GEOMETRY.search(c)]
    full = raw[~(raw[geo] == "").any(axis=1)]
    sample = full.head(a.rows)
    if not len(sample):
        print("no fully-populated rows to compare")
        return 1
    print(f"comparing {len(sample):,} rows "
          f"({len(full):,} of {len(raw):,} shard rows are fully populated)")

    # Shape it the way results_to_dataframe() hands it over: numeric columns,
    # plus fxx, and the valid time expressed as run time + fxx.
    fxx = 6
    valid = pd.to_datetime(sample["valid_utc"], format="%Y-%m-%dT%H:%M", utc=True)
    weather_cols = [c for c in sample.columns
                    if not c.startswith("meta_")
                    and c not in {"valid_utc", "model_valid_utc", "target_lead_h",
                                  "split", "is_undercast", "year", "month", "hour_utc"}]
    served, keys = [], []
    for run_time, grp in sample.assign(_valid=valid).groupby(
            (valid - timedelta(hours=fxx)).dt.strftime("%Y-%m-%d %H:%M")):
        wdf = grp[weather_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        wdf["fxx"] = fxx
        X, got_valid = wj.build_current_features(wdf, run_time)
        served.append(X)
        keys.extend(grp["_valid"].dt.strftime("%Y-%m-%dT%H:%M")
                    + "@" + grp["target_lead_h"].astype(str))
        # The valid time the serving code derives must be the real one.
        assert (got_valid.dt.strftime("%Y-%m-%dT%H:%M").to_numpy()
                == grp["_valid"].dt.strftime("%Y-%m-%dT%H:%M").to_numpy()).all(), \
            f"valid time mismatch for run {run_time}"
    S = pd.concat(served, ignore_index=True)
    S.index = keys

    T = train.reindex(S.index)
    assert T.notna().any(axis=1).all(), "some sampled rows did not match the training frame"
    missing_cols = [c for c in wanted if c not in S.columns]
    print(f"features the model wants: {len(wanted)}; "
          f"built by the serving path: {len(wanted) - len(missing_cols)}")
    if missing_cols:
        print(f"  MISSING: {missing_cols}")

    bad, checked = [], 0
    for c in wanted:
        if c not in S.columns or c not in T.columns:
            continue
        checked += 1
        s, t = S[c].to_numpy(float), T[c].to_numpy(float)
        both_nan = np.isnan(s) & np.isnan(t)
        diff = np.abs(np.where(both_nan, 0.0, s - t))
        n = int(np.nansum(diff > a.tol) + np.sum(np.isnan(diff)))
        if n:
            bad.append((c, n, float(np.nanmax(diff))))

    print(f"compared {checked} columns over {len(S):,} rows")
    if bad:
        print(f"\n{len(bad)} COLUMNS DISAGREE between training and serving:")
        for c, n, mx in bad[:25]:
            print(f"   {c:<42} {n:>5} rows differ, max |diff| {mx:g}")
        return 1
    print("\nPASS: every served feature matches the trained one exactly")

    # And the end-to-end number: the same rows through the real model.
    import joblib
    pre = joblib.load("files/weather/models/obs/preprocessor_all.pkl")
    gb = joblib.load("files/weather/models/obs/gradient_boosting_best_f1_all.pkl")
    ps = gb.predict_proba(pre.transform(S.reindex(columns=wanted)))[:, 1]
    pt = gb.predict_proba(pre.transform(T.reindex(columns=wanted)))[:, 1]
    print(f"model output max |served - trained| = {np.max(np.abs(ps - pt)):.3e}")
    return 0 if np.max(np.abs(ps - pt)) < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
