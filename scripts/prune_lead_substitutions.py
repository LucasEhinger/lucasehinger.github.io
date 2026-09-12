#!/usr/bin/env python3
"""Blank model cells whose forecast hour does not match the lead their row claims.

The shards on disk were fetched before ``fetch_nwp_at_obs.LEAD_SLACK_H`` existed,
when ``candidate_runs`` would walk outward without limit to the nearest achievable
forecast hour. That is right when the gap comes from run cadence and wrong when it
comes from the model's ceiling -- see ``max_reachable_lead``. The wrong case is
real and measured: 40% of HRRR's cells at lead 48 carry a 15, 18 or 36 hour
forecast from the early archive, because that was all HRRR could produce then.

Nothing downstream can detect this. Each row does record the forecast hour it
actually got in ``lead_<model>``, but ``train_undercast_obs._is_lead_col`` drops
every ``lead_*`` column from the feature set, so the model sees a sharper forecast
than the row's label promises and learns that 48 h forecasts are unusually good.
In production a 48 h row carries a genuine 48 h forecast and the skill is not
there. That is train/serve skew of exactly the kind that hid the GFS longitude
error, and it inflates the single-source numbers the site publishes.

This applies the guard retroactively, which needs no downloads at all: the
achieved forecast hour is already in the CSV, so the offending cells can simply be
emptied in place. Measured effect, at the 12 h default:

    combined model (all six sources)   31,876 -> 31,849 rows   (0.08%)
    HRRR single-source                 73,336 -> 55,931 rows   (23.7%)

So it is nearly free for the model the site serves and a substantial, honest
correction to HRRR's. Expect HRRR's reported skill to FALL after this -- that is
the point, not a regression.

Dry run by default; pass --apply to rewrite the shards in place.

    python3 scripts/prune_lead_substitutions.py
    python3 scripts/prune_lead_substitutions.py --apply
"""
import argparse
import csv
import glob
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_nwp_at_obs import (  # noqa: E402
    LEAD_SLACK_H, MODEL_VARS, max_reachable_lead,
)


def _reach_cache():
    """max_reachable_lead is pure, and its inputs collapse to (model, month).

    Calling it per cell would be ~440k calls over a 192-iteration loop; the era
    tables only change a handful of times, so one call per (model, month) is both
    exact and fast.
    """
    cache = {}

    def reach(model, ym):
        if (model, ym) not in cache:
            when = datetime(int(ym[:4]), int(ym[5:7]), 15, 12, tzinfo=timezone.utc)
            cache[(model, ym)] = max_reachable_lead(model, when)
        return cache[(model, ym)]
    return reach


def prune_row(row, models, slack, reach):
    """Empty every cell of any model that cannot honestly fill this row.

    Returns the models that were cleared.
    """
    try:
        target = int(float(row["target_lead_h"]))
    except (TypeError, ValueError):
        return []
    ym = (row.get("valid_utc") or "")[:7]
    if len(ym) != 7:
        return []
    cleared = []
    for model in models:
        got = row.get(f"lead_{model}", "")
        if got in ("", None):
            continue            # already absent; nothing to clear
        try:
            achieved = int(float(got))
        except (TypeError, ValueError):
            continue
        reachable = target <= reach(model, ym)
        if reachable and abs(achieved - target) <= slack:
            continue            # cadence offset within tolerance: legitimate
        row[f"lead_{model}"] = ""
        row[f"meta_init_{model}"] = ""
        for label in MODEL_VARS[model]:
            if label in row:
                row[label] = ""
        cleared.append(model)
    return cleared


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="files/weather/csv/obs")
    ap.add_argument("--slack", type=int, default=LEAD_SLACK_H)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the shards; without it, only report")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.csv_dir, "nwp_obs_shard_*.csv")))
    if not shards:
        ap.error(f"no shards under {a.csv_dir}")
    models = list(MODEL_VARS)
    reach = _reach_cache()

    print(f"{len(shards)} shards, slack {a.slack} h, "
          f"{'APPLYING' if a.apply else 'dry run'}")
    totals = {m: [0, 0] for m in models}    # [filled before, cleared]
    rows_touched = n_rows = 0

    for path in shards:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames
            rows = list(reader)
        changed = 0
        for row in rows:
            n_rows += 1
            for m in models:
                if row.get(f"lead_{m}", "") not in ("", None):
                    totals[m][0] += 1
            cleared = prune_row(row, models, a.slack, reach)
            for m in cleared:
                totals[m][1] += 1
            if cleared:
                changed += 1
                rows_touched += 1
        if a.apply and changed:
            # Write beside the original and replace, so an interrupted run cannot
            # leave a half-written shard -- these files are the training set.
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, path)
        print(f"  {os.path.basename(path)}: {changed:>5} rows affected"
              + ("  [written]" if a.apply and changed else ""))

    print(f"\n{n_rows:,} rows scanned, {rows_touched:,} with at least one cell cleared")
    print(f"{'model':7s} {'filled':>9} {'cleared':>9} {'share':>7}")
    for m in models:
        before, cleared = totals[m]
        print(f"{m:7s} {before:>9,} {cleared:>9,} "
              f"{cleared / max(before, 1):>6.1%}")
    if not a.apply:
        print("\ndry run -- nothing written. Re-run with --apply to rewrite.")
    else:
        print("\nShards rewritten. Retrain before trusting any published metric: "
              "the single-source numbers in files/weather/models/obs/ were fitted "
              "to the cells this just removed.")


if __name__ == "__main__":
    main()
