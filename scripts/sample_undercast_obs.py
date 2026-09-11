#!/usr/bin/env python3
"""Choose which observation times to download NWP forecast data for.

The undercast record holds ~253k observations (1997-2026) at a 2.7% positive
rate. Downloading forecast fields for all of them is out of the question -- even
the three years where all six models exist is ~26k observations x 6 models -- so
the negatives have to be subsampled. How they are subsampled decides what the
classifier can learn, so the scheme matters more than the ratio:

STRATIFIED ON (year, month, hour-of-day).
    Undercast is not uniform in any of those three. It peaks at 10-14 UTC and
    burns off by 18-19; it has a seasonal cycle; and its reported rate roughly
    doubles across the record as observers adopted the remark more consistently
    (1.2% in 1997 -> 5.4% in 2024), which is partly behaviour, not weather.
    Sample negatives uniformly and the model can hit good apparent scores purely
    off the clock and the calendar -- features it already has as month/day. Draw
    each positive's negatives from its OWN (year, month, hour) cell and all three
    confounds are matched out by construction, so the fields have to do the work.
    It also means any sub-window (say ECMWF's 2023+) keeps the same ratio, so one
    master sample serves every model's own date range.

AMBIGUOUS OBSERVATIONS ARE DROPPED, NOT CALLED NEGATIVE.
    A report with a SCT deck, or one that misses a cut narrowly, is where the
    screen is least reliable. Labeling those 0 puts the worst label noise exactly
    on the decision boundary. They are excluded from both classes.

TWO HOLDOUTS THE SAMPLE DOES NOT TOUCH.
    Subsampling negatives inflates the positive rate ~K/30x, so precision
    measured on the sample is meaningless for real use. Two unsampled blocks fix
    that:
      webcam   every observation nearest local noon on the 589 hand-labeled
               days -- scored against human webcam truth, not the screen.
      baserate every observation at 3-hourly UTC steps through one recent full
               year -- true 2.7% base rate, so precision/recall read honestly.

Writes a CSV of observation times with a `split` column. Usage:
    python3 scripts/sample_undercast_obs.py --neg-per-pos 5
"""
import argparse
import csv
import gzip
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Single source of truth for the archive start dates -- these were probed against
# the live buckets, and having two copies drift apart would silently mis-state
# which models a sample actually covers.
from fetch_nwp_at_obs import MODEL_START as _MS  # noqa: E402

MODEL_START = {m: d.strftime("%Y-%m-%d") for m, d in _MS.items()}
BASERATE_YEAR = 2022          # full year held out unsampled, 3-hourly
BASERATE_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return list(csv.DictReader(fh))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", default="files/weather/obs/undercast_record.csv")
    p.add_argument("--labels", default="files/weather/csv/MtWashington_undercast_orig.csv")
    p.add_argument("--out", default="files/weather/obs/nwp_sample.csv")
    p.add_argument("--neg-per-pos", type=int, default=5)
    p.add_argument("--start", default=MODEL_START["hrrr"],
                   help="ignore observations before this (no NWP archive exists)")
    p.add_argument("--seed", type=int, default=23)
    a = p.parse_args()
    rng = random.Random(a.seed)

    rows = [r for r in load(a.record) if r["valid_utc"][:10] >= a.start]
    print(f"{len(rows)} observations from {a.start} onward")

    hand_days = set()
    if os.path.exists(a.labels):
        for r in load(a.labels):
            try:
                hand_days.add(datetime.strptime(r["Short Date"], "%m/%d/%y").date())
            except (ValueError, KeyError):
                continue

    # --- holdout 1: the observation nearest local noon on each hand-labeled day
    webcam = {}
    for r in rows:
        lt = datetime.strptime(r["valid_local"], "%Y-%m-%dT%H:%M")
        if lt.date() not in hand_days:
            continue
        d = abs(lt.hour * 60 + lt.minute - 720)
        if lt.date() not in webcam or d < webcam[lt.date()][0]:
            webcam[lt.date()] = (d, r)
    webcam_ids = {r["valid_utc"] for _, r in webcam.values()}

    # --- holdout 2: one unsampled year at 3-hourly steps, true base rate
    base_ids = {
        r["valid_utc"] for r in rows
        if int(r["year"]) == BASERATE_YEAR and int(r["hour_utc"]) in BASERATE_HOURS
    }

    held = webcam_ids | base_ids
    pool = [r for r in rows if r["valid_utc"] not in held]

    # --- stratified training sample ----------------------------------------
    pos = [r for r in pool if r["label"] == "undercast"]
    negs = defaultdict(list)
    for r in pool:
        if r["label"] == "clear":
            negs[(r["year"], r["month"], r["hour_utc"])].append(r)

    chosen_neg, short = [], 0
    need = Counter((r["year"], r["month"], r["hour_utc"]) for r in pos)
    for cell, n_pos in need.items():
        want = n_pos * a.neg_per_pos
        avail = negs.get(cell, [])
        if len(avail) <= want:
            chosen_neg.extend(avail)
            short += want - len(avail)
        else:
            chosen_neg.extend(rng.sample(avail, want))
    if short:
        # Thin cells (a given month+hour holds ~25-30 observations per year) can
        # run out of negatives. Backfill from the same month+hour in adjacent
        # years so the seasonal/diurnal match survives even if the year match
        # loosens slightly.
        by_mh = defaultdict(list)
        taken = {r["valid_utc"] for r in chosen_neg}
        for r in pool:
            if r["label"] == "clear" and r["valid_utc"] not in taken:
                by_mh[(r["month"], r["hour_utc"])].append(r)
        for (y, m, h), n_pos in need.items():
            deficit = n_pos * a.neg_per_pos - sum(
                1 for r in chosen_neg
                if (r["year"], r["month"], r["hour_utc"]) == (y, m, h))
            cand = [r for r in by_mh.get((m, h), []) if r["valid_utc"] not in taken]
            if deficit > 0 and cand:
                pick = rng.sample(cand, min(deficit, len(cand)))
                chosen_neg.extend(pick)
                taken.update(r["valid_utc"] for r in pick)

    out = []
    for r in pos:
        out.append((r, "train"))
    for r in chosen_neg:
        out.append((r, "train"))
    for _, r in webcam.values():
        out.append((r, "holdout_webcam"))
    for r in rows:
        if r["valid_utc"] in base_ids:
            out.append((r, "holdout_baserate"))
    out.sort(key=lambda t: t[0]["valid_utc"])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["valid_utc", "split", "label", "year", "month", "hour_utc",
                    "max_cover", "tops_ft", "depth_below_ft", "vis_sm",
                    "overhead_ceiling_ft", "overhead_lowest_ft"])
        for r, split in out:
            w.writerow([r["valid_utc"], split, r["label"], r["year"], r["month"],
                        r["hour_utc"], r["max_cover"], r["tops_ft"],
                        r["depth_below_ft"], r["vis_sm"],
                        r["overhead_ceiling_ft"], r["overhead_lowest_ft"]])

    # --- report -------------------------------------------------------------
    print(f"\nwrote {a.out}")
    for split in ("train", "holdout_webcam", "holdout_baserate"):
        sel = [r for r, s in out if s == split]
        c = Counter(r["label"] for r in sel)
        n = len(sel)
        print(f"  {split:18s} {n:6d} obs   undercast={c['undercast']:5d} "
              f"clear={c['clear']:6d} ambiguous={c['ambiguous']:5d} "
              f"({100*c['undercast']/max(n,1):.1f}% positive)")
    print(f"\n  total observation times: {len(out)}")

    print("\n  per-model coverage of the sample (its own archive window):")
    total_units = 0
    for m, start in sorted(MODEL_START.items(), key=lambda kv: kv[1]):
        sel = [(r, s) for r, s in out if r["valid_utc"][:10] >= start]
        tr = [r for r, s in sel if s == "train"]
        c = Counter(r["label"] for r in tr)
        total_units += len(sel)
        print(f"    {m:6s} from {start}  {len(sel):6d} obs  "
              f"(train {len(tr):6d}: {c['undercast']:4d} pos / {c['clear']:5d} neg)")
    print(f"\n  GRIB download units (obs x models available) = {total_units:,} per lead")


if __name__ == "__main__":
    main()
