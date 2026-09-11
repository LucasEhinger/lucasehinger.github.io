#!/usr/bin/env python3
"""Predict undercast with physics and a threshold, no machine learning.

The project's premise is that an undercast is a temperature inversion with the
summit above the deck. If that is the whole story, a couple of hand-written
rules over the forecast fields should do most of the job, and the ML is
decoration. This checks that, on exactly the same holdouts the classifiers are
scored on, so the numbers are directly comparable.

Four rules, each a statement about the atmosphere rather than a fitted model:

  A  inversion            T(850 mb) - T(925 mb) > t
                          The 925->850 layer sits just below the 1,917 m summit,
                          so a positive difference is a capping inversion there.
  B  summit above deck    no modelled ceiling, or one above 1,917 m -- the model
                          says the summit itself is not in cloud.
  C  A and B              an inversion, with the summit out of the murk.
  D  C and sees far       model surface visibility > t as well.

The first version of rule B tested "modelled cloud TOP below 1,917 m", which is
the intuitive statement and is wrong; see rule_masks for why, and for what
happened when low-cloud cover was added as a term.

Thresholds are fitted on the TRAINING split only and then applied unchanged to
the holdouts, which is the same discipline the classifiers get. A rule whose
threshold was tuned on the holdout would not be a fair comparison.

Only sources carrying the needed fields are evaluated; NBM publishes no pressure
levels, so it cannot express rule A.

    python3 scripts/baseline_undercast_rules.py
    python3 scripts/baseline_undercast_rules.py --sources hrrr rap --json out.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_undercast_obs import (  # noqa: E402
    TARGET, assign_splits, load_obs_data,
)

SUMMIT_M = 1917.2
# Candidate thresholds, swept on train only.
INV_GRID = np.arange(-4.0, 6.01, 0.25)          # K, T850 - T925
VIS_GRID = np.array([2e3, 5e3, 1e4, 1.5e4, 2e4, 3e4, 5e4])   # m


def col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def fields(df, m):
    return {
        "inv": col(df, f"dT_925_850_{m}"),
        "ceil": col(df, f"cloud_ceiling_m_{m}", f"cloud_ceiling_{m}"),
        "vis": col(df, f"vis_surface_{m}"),
    }


def prf(y, pred):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn,
            "n": int(len(y)), "fired": int(pred.sum())}


def rule_masks(f, t_inv, t_vis):
    """Boolean masks for the four rules.

    The obvious rule -- "modelled cloud TOP below the 1,917 m summit" -- was
    tried first and is actively WRONG, which is worth recording. On the training
    split it selects rows with a 9.2% undercast rate against a 16.8% base rate,
    while rows where HRRR reports no cloud at all run at 21.5%. The reason is
    resolution: HRRR's 3 km terrain puts this grid cell near 1,200 m, some 700 m
    below the real summit, so the model has no column between its own ground and
    the height the observer is standing at. It cannot represent "the deck tops
    out below me"; it can only say whether the cell is in cloud.

    So the model-space statement of an undercast is the inverted one: the summit
    level is CLEAR (no ceiling, or a ceiling above 1,917 m) while the profile
    carries an inversion capping a deck somewhere beneath. Low-cloud cover was
    tested as a third term and makes every combination worse -- at this cell it
    describes cloud around the model's own ground, not the deck below the peak.
    """
    inv, ceil, vis = f["inv"], f["ceil"], f["vis"]
    A = (inv > t_inv).fillna(False) if inv is not None else None
    # nan here means no cloud at all (sentinels are normalised to nan upstream),
    # which is the clearest possible statement that the summit is not in cloud.
    B = (ceil.isna() | (ceil > SUMMIT_M)) if ceil is not None else None
    C = (A & B) if (A is not None and B is not None) else None
    D = (C & (vis > t_vis).fillna(False)) if (C is not None and vis is not None) else None
    return {"A inversion": A,
            "B summit above the deck": B,
            "C A and B": C,
            "D C and model sees far": D}


def best_thresholds(tr, m, y):
    """Sweep on TRAIN only, then apply unchanged to the holdouts."""
    f = fields(tr, m)
    best = {"t_inv": 0.0, "t_vis": 0.0}
    if f["inv"] is not None:
        best["t_inv"] = max(
            (prf(y, ((f["inv"] > t).fillna(False)).to_numpy())["f1"], t)
            for t in INV_GRID)[1]
    if f["vis"] is not None:
        cand = []
        for t in VIS_GRID:
            mk = rule_masks(f, best["t_inv"], t)["D C and model sees far"]
            if mk is None:
                break
            cand.append((prf(y, mk.to_numpy())["f1"], t))
        if cand:
            best["t_vis"] = max(cand)[1]
    return best


def evaluate(df, sources):
    out = {}
    for m in sources:
        sub = df[df[f"lead_{m}"].notna()]
        tr = sub[(sub["split_eff"] == "train") & ~sub["screen_ambiguous"]]
        if len(tr) < 200:
            print(f"[{m:>5}] skipped: {len(tr)} train rows")
            continue
        ytr = tr[TARGET].to_numpy()
        th = best_thresholds(tr, m, ytr)
        res = {"thresholds": th, "n_train": int(len(tr))}
        print(f"\n[{m:>5}] thresholds fitted on train: "
              f"inversion > {th['t_inv']:+.2f} K, visibility > {th['t_vis']/1000:.0f} km")
        hdr = f"  {'rule':<26}{'split':<12}{'P':>6}{'R':>6}{'F1':>7}{'fires':>8}{'n':>8}"
        print(hdr)
        for split, name in (("train", "train"),
                            ("holdout_baserate", "baserate"),
                            ("holdout_webcam", "webcam")):
            part = sub[sub["split_eff"] == split]
            if split == "train":
                part = part[~part["screen_ambiguous"]]
            if not len(part):
                continue
            if split == "holdout_webcam":
                part = part[part["hand_label"].notna()]
                y = (part["hand_label"] >= 0.5).to_numpy()
            else:
                y = part[TARGET].to_numpy()
            if not len(part) or not y.sum():
                continue
            masks = rule_masks(fields(part, m), th["t_inv"], th["t_vis"])
            for rname, mk in masks.items():
                if mk is None:
                    continue
                sc = prf(y, mk.to_numpy())
                res.setdefault(rname, {})[name] = sc
                print(f"  {rname:<26}{name:<12}{sc['precision']:>6.2f}"
                      f"{sc['recall']:>6.2f}{sc['f1']:>7.2f}{sc['fired']:>8,}{sc['n']:>8,}")
        out[m] = res
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", default="files/weather/csv/obs")
    p.add_argument("--labels", default="files/weather/csv/MtWashington_undercast_orig.csv")
    p.add_argument("--record", default="files/weather/obs/undercast_record.csv")
    p.add_argument("--sources", nargs="+",
                   default=["hrrr", "nam", "gfs", "rap", "ecmwf"])
    p.add_argument("--json", default="files/weather/models/obs/baseline_rules.json")
    p.add_argument("--cache", help="read/write the assembled frame here to skip reload")
    a = p.parse_args()

    if a.cache and os.path.exists(a.cache):
        df = pd.read_pickle(a.cache)
    else:
        # load_obs_data already appends the profile block; concatenating it a
        # second time here produced duplicate column NAMES, so df[col] returned
        # a 2-column frame and every comparison silently became 2-D.
        df = load_obs_data(a.csv_dir, a.labels, a.record)
        df = assign_splits(df)
        if a.cache:
            df.to_pickle(a.cache)

    res = evaluate(df, a.sources)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
