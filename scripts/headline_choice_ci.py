#!/usr/bin/env python3
"""Why Gradient Boosting, and not XGBoost or the 2-of-3 vote -- with intervals.

The page's closing section claims the served model beats the alternatives it was
chosen over. That claim needs a paired interval, not two point estimates: the
holdouts are small, and F1 differences of 0.05 are routine noise on 168
positives. This prints those intervals so the sentence on the page is
reproducible from the repository rather than from a number typed once.

Paired DAY-BLOCK bootstrap, reusing undercast_capacity.boot_diff: whole days are
resampled, because hours within a day are the same weather and resampling rows
would treat 24 correlated readings as 24 observations.

Each predictor is scored at its OWN per-lead thresholds, which is how the site
serves it -- comparing a lead-aware cut against a global one would measure the
thresholding, not the algorithm.

    python3 scripts/headline_choice_ci.py
"""
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import undercast_eval as ue  # noqa: E402
from train_undercast_obs import assign_splits, load_obs_data  # noqa: E402
from undercast_capacity import boot_diff  # noqa: E402

SOURCE = "all"
SERVED = "Gradient Boosting"
MEMBERS = ("XGBoost", "Random Forest", "Gradient Boosting")


def predictions(scored, meta):
    """{name: 0/1 vector}, each algorithm at its own per-lead thresholds."""
    leads = scored["rows"]["target_lead_h"].to_numpy().astype(float)
    out = {}
    for algo in MEMBERS:
        by = meta[algo]["threshold_by_lead"]
        thr = np.array([float(by[str(int(l))]) for l in leads])
        out[algo] = (scored["proba"][algo] >= thr).astype(int)
    stacked = np.vstack([out[a] for a in MEMBERS])
    out["2 of 3 (the vote)"] = (stacked.sum(axis=0) >= 2).astype(int)
    return out


def main():
    df = assign_splits(load_obs_data(
        "files/weather/csv/obs",
        "files/weather/csv/MtWashington_undercast_orig.csv",
        "files/weather/obs/undercast_record.csv"))

    for split in ("holdout_webcam", "holdout_baserate"):
        scored = ue.score_split(df, SOURCE, split)
        y = scored["y"]
        dates = scored["rows"]["date"].to_numpy()
        preds = predictions(scored, scored["meta"])
        print(f"\n=== {split}: {len(y):,} rows, {int(y.sum())} positives, "
              f"{len(np.unique(dates))} distinct days ===")
        base = preds[SERVED]
        print(f"  {SERVED} (reference)      F1 {f1_score(y, base):.3f}")
        for name, p in preds.items():
            if name == SERVED:
                continue
            # boot_diff reports metric(b) - metric(a); we want served minus other.
            d, ci = boot_diff(y, dates, p, base, f1_score)
            flag = "" if ci[0] < 0 < ci[1] else "   <- excludes zero"
            print(f"  vs {name:22s} F1 {f1_score(y, p):.3f}   "
                  f"served {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]{flag}")


if __name__ == "__main__":
    main()
