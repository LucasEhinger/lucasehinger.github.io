#!/usr/bin/env python3
"""Confusion-matrix counts per (source, forecast lead, algorithm), as JSON.

The write-up showed one confusion matrix per source, pooled over every lead. That
hides the thing most worth knowing: the same model is far more trustworthy about
tonight than about this time next week, and a pooled matrix averages the two into
a number that describes neither.

Emitting counts rather than 56 more PNGs keeps this to a few KB and lets the page
switch leads instantly.

Thresholds follow what the site actually does. A specific lead uses that lead's own
F1-optimal threshold from `threshold_by_lead`; "all leads" uses the single global
threshold, which is the cut the pooled figure was drawn at. Mixing the two would
make the pooled matrix disagree with the sum of its parts for no good reason -- so
the row totals are reported per lead and the difference is visible rather than
hidden.

The 2-of-3 vote is a MAJORITY of the three hard calls, each at its own threshold --
not an average of probabilities. That distinction matters on this page, which also
shows a mean-of-three elsewhere.

    python3 scripts/export_confusion_by_lead.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import undercast_eval as ue  # noqa: E402
from train_undercast_obs import MODEL_NAMES, SOURCES  # noqa: E402

SPLIT = "holdout_baserate"
OUT = "files/weather/models/obs/confusion_by_lead.json"
VOTE = "2 of 3 (the vote)"


def counts(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {
        "tp": int(((y == 1) & (pred == 1)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "tn": int(((y == 0) & (pred == 0)).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=ue.MODELS_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()

    df = ue.load_frame(cache=a.cache)
    out = {"algorithms": list(MODEL_NAMES) + [VOTE], "sources": {}}
    all_leads = set()

    for source in SOURCES:
        r = ue.score_split(df, source, SPLIT, a.models_dir)
        rows, y, proba, meta = r["rows"], r["y"], r["proba"], r["meta"]
        lead = rows["target_lead_h"].astype(float).to_numpy()
        leads = sorted({int(v) for v in np.unique(lead)})
        all_leads |= set(leads)

        per_lead = {}
        for key in ["all"] + [str(v) for v in leads]:
            if key == "all":
                mask = np.ones(len(y), dtype=bool)
                thr = {n: float(meta[n]["threshold"]) for n in MODEL_NAMES}
            else:
                mask = lead == float(key)
                # Its own lead's cut where the model has one; the global cut is the
                # fallback so a model trained before the ladder still renders.
                thr = {n: float((meta[n].get("threshold_by_lead") or {}).get(
                    key, meta[n]["threshold"])) for n in MODEL_NAMES}
            if not mask.any():
                continue
            yy = np.asarray(y)[mask]
            preds = {n: (np.asarray(proba[n])[mask] >= thr[n]).astype(int)
                     for n in MODEL_NAMES}
            votes = np.sum([preds[n] for n in MODEL_NAMES], axis=0)
            preds[VOTE] = (votes >= 2).astype(int)
            entry = {"n": int(mask.sum()), "positives": int(yy.sum()), "cells": {}}
            for n in list(MODEL_NAMES) + [VOTE]:
                c = counts(yy, preds[n])
                c["threshold"] = round(thr[n], 3) if n != VOTE else None
                entry["cells"][n] = c
            per_lead[key] = entry
        out["sources"][source] = per_lead
        have = [k for k in per_lead if k != "all"]
        print(f"  {source:6s} leads {','.join(have) or '(none)':<28s} "
              f"{per_lead['all']['n']:>6,} rows, {per_lead['all']['positives']:>4} undercast")

    out["leads"] = sorted(all_leads)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\nwrote {a.out} ({os.path.getsize(a.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
