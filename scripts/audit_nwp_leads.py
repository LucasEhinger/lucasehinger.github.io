#!/usr/bin/env python3
"""Audit the forecast leads in the shard CSVs: are they what the rows claim?

Each row says it is a forecast made `target_lead_h` hours ahead. That is a promise
about sharpness, and nothing downstream re-checks it -- `_is_lead_col` drops every
`lead_*` column from the feature set, so a row carrying a 36 h forecast under a
48 h label teaches the model that 48 h forecasts are unusually good. In production
a 48 h row carries a genuine 48 h forecast and the skill is not there.

This is the script behind the numbers quoted in README_undercast_pipeline.md and in
prune_lead_substitutions.py's docstring. Re-run it after any fetch, and after
changing LEAD_SLACK_H or RUN_SPECS.

Four sections:

  1. ACHIEVED vs CLAIMED -- how far each model's forecast hour sits from its row's
     lead. This is where HRRR's pre-2021 era ceilings showed up as 40% of its
     cells at lead 48 carrying a 15, 18 or 36 hour forecast.

  2. FILL WITHIN ARCHIVE WINDOW -- a raw fill rate conflates "the archive did not
     exist yet" with "we asked for an hour the model does not publish". Only the
     second is a fault, and only this view separates them.

  3. AVAILABILITY -- for what share of observation hours each model can serve each
     lead, computed from RUN_SPECS rather than from the data, including the
     valid-time snapping the fetcher applies. Cross-checks against what
     train_undercast_obs expects: every source the trainer REQUIRES at a lead must
     be essentially fully available there, or rows_for_source("all") silently
     discards that lead.

  4. GUARD IMPACT -- how many cells the current LEAD_SLACK_H would remove, and
     what that costs each model and the combined one.

    python3 scripts/audit_nwp_leads.py
    python3 scripts/audit_nwp_leads.py --csv-dir files/weather/csv/obs
"""
import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_nwp_at_obs as f  # noqa: E402
from train_undercast_obs import sources_expected_at  # noqa: E402

MODELS = ["hrrr", "nam", "gfs", "rap", "ecmwf", "nbm"]


def load(csv_dir):
    need = (["valid_utc", "target_lead_h", "is_undercast"]
            + [f"lead_{m}" for m in MODELS])
    paths = sorted(glob.glob(os.path.join(csv_dir, "nwp_obs_shard_*.csv")))
    if not paths:
        raise SystemExit(f"no shards under {csv_dir}")
    df = pd.concat([pd.read_csv(p, usecols=lambda c: c in need, low_memory=False)
                    for p in paths], ignore_index=True)
    df["target_lead_h"] = pd.to_numeric(df["target_lead_h"], errors="coerce")
    df["_valid"] = pd.to_datetime(df["valid_utc"], utc=True, errors="coerce")
    print(f"{len(paths)} shards -> {len(df):,} rows, "
          f"leads {sorted(int(v) for v in df['target_lead_h'].dropna().unique())}")
    return df


def achieved_vs_claimed(df):
    print("\n=== 1. achieved forecast hour vs the lead the row claims ===")
    print(f"{'model':7}{'lead':>6} {'cells':>8} {'exact':>7} {'|d|<=2':>8} "
          f"{'|d|>6':>7} {'|d|>12':>8} {'worst':>7}")
    for m in MODELS:
        for lead in sorted(df["target_lead_h"].dropna().unique()):
            sub = df[df["target_lead_h"] == lead]
            got = pd.to_numeric(sub[f"lead_{m}"], errors="coerce").dropna()
            if got.empty:
                continue
            d = (got - lead).abs()
            print(f"{m:7}{int(lead):>5}h {len(d):>8,} {(d == 0).mean():>6.0%} "
                  f"{(d <= 2).mean():>7.0%} {(d > 6).mean():>6.0%} "
                  f"{(d > 12).mean():>7.0%} {int(d.max()):>6}h")


def fill_within_window(df):
    print("\n=== 2. fill rate WITHIN each model's own archive window ===")
    print(f"{'lead':>6} | " + " ".join(m.rjust(8) for m in MODELS))
    for lead in sorted(df["target_lead_h"].dropna().unique()):
        sel = df["target_lead_h"] == lead
        cells = []
        for m in MODELS:
            inwin = sel & (df["_valid"] >= f.MODEL_START[m])
            if not inwin.any():
                cells.append("-".rjust(8))
                continue
            got = pd.to_numeric(df.loc[inwin, f"lead_{m}"], errors="coerce")
            cells.append(f"{got.notna().mean():7.0%}".rjust(8))
        print(f"{int(lead):>5}h | " + " ".join(cells))


def availability(leads):
    """From RUN_SPECS, not from the data: what COULD be fetched."""
    print("\n=== 3. availability from RUN_SPECS, with the fetcher's snapping ===")
    print(f"{'lead':>6} | " + " ".join(m.rjust(7) for m in MODELS)
          + "   | required by the trainer")
    problems = []
    for lead in leads:
        cells = []
        for m in MODELS:
            n = 0
            for hour in range(24):
                valid = datetime(2026, 2, 10, hour, tzinfo=timezone.utc)
                if f.candidate_runs(m, f.snap_valid(m, valid, lead), lead):
                    n += 1
            cells.append(f"{n}/24".rjust(7))
            if m in sources_expected_at(lead) and n < 24:
                problems.append(f"lead {lead}h: {m} available for only {n}/24 "
                                f"observation hours, but the trainer requires it")
        print(f"{int(lead):>5}h | " + " ".join(cells)
              + f"   | {','.join(sources_expected_at(lead))}")
    print()
    if problems:
        for p in problems:
            print(f"  PROBLEM {p}")
        print("  A required source that is not fully available means "
              "rows_for_source(\"all\") discards most of that lead.")
    else:
        print("  every source the trainer requires is fully available at its lead")
    return problems


def guard_impact(df):
    """What the current slack + reachability rules would remove."""
    print(f"\n=== 4. impact of LEAD_SLACK_H = {f.LEAD_SLACK_H} h ===")
    cache = {}

    def reach(m, ym):
        if (m, ym) not in cache:
            when = datetime(int(ym[:4]), int(ym[5:7]), 15, 12, tzinfo=timezone.utc)
            cache[(m, ym)] = f.max_reachable_lead(m, when)
        return cache[(m, ym)]

    ym = df["_valid"].dt.strftime("%Y-%m")
    keep = {}
    for m in MODELS:
        got = pd.to_numeric(df[f"lead_{m}"], errors="coerce")
        reachable = np.array([lead <= reach(m, y) for lead, y
                              in zip(df["target_lead_h"], ym)])
        ok = (got.notna() & reachable
              & ((got - df["target_lead_h"]).abs() <= f.LEAD_SLACK_H))
        keep[m] = ok.fillna(False).to_numpy()
        before = int(got.notna().sum())
        print(f"  {m:6} {before:>7,} filled -> {int(keep[m].sum()):>7,} kept "
              f"({1 - keep[m].sum() / max(before, 1):>5.1%} removed)")

    print("\n  rows usable by the combined model (every expected source present):")
    for label, sel in (("before", None), ("after", keep)):
        mask = np.ones(len(df), dtype=bool)
        for lead in sorted(df["target_lead_h"].dropna().unique()):
            at = (df["target_lead_h"] == lead).to_numpy()
            ok = np.ones(len(df), dtype=bool)
            for m in sources_expected_at(lead):
                present = (sel[m] if sel is not None
                           else pd.to_numeric(df[f"lead_{m}"],
                                              errors="coerce").notna().to_numpy())
                ok &= present
            mask &= ~at | ok
        pos = int(pd.to_numeric(df.loc[mask, "is_undercast"],
                                errors="coerce").fillna(0).sum())
        print(f"    {label:6} {int(mask.sum()):>7,} rows, {pos:>5,} undercast")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="files/weather/csv/obs")
    ap.add_argument("--leads", type=int, nargs="+", default=list(f.TARGET_LEADS),
                   help="leads to check availability for (section 3)")
    a = ap.parse_args()

    df = load(a.csv_dir)
    achieved_vs_claimed(df)
    fill_within_window(df)
    problems = availability(a.leads)
    guard_impact(df)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
