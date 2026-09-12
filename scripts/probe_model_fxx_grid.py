#!/usr/bin/env python3
"""Probe which forecast hours each model ACTUALLY publishes, and at what spacing.

``RUN_SPECS`` in ``fetch_nwp_at_obs.py`` encodes, per model, the longest forecast
hour available and the spacing of the grid. Those numbers decide which (init, fxx)
pairs are even attempted, so a wrong one is expensive and silent: the fetcher asks
for an hour that does not exist, finds nothing, and leaves the column blank. It
looks exactly like a model that does not publish the field.

The published documentation is not a reliable source for this. Two values in
RUN_SPECS were wrong because they came from documentation rather than the archive:

  * NAM was declared max=84, its documented range. The product Herbie serves stops
    at 60 -- hourly 36..60, then nothing -- because the 3-hourly extension to 84 h
    lives in a different product file. Asking for 72 h returned fxx=71 candidates
    that do not exist, and NAM came back 8% filled past 48 h.
  * NBM was declared step=1 throughout. Its extended runs publish 3-hourly past
    36 h, so most requested hours did not exist and it came back 10-20% filled.

So probe. This is the script that produced the numbers in those two corrections,
and the one to re-run before changing any RUN_SPECS value or adding a model.

    python3 scripts/probe_model_fxx_grid.py
    python3 scripts/probe_model_fxx_grid.py --models nam nbm --lo 30 --hi 100

Each row reports the hours found, the distinct gaps between them, and the last
hour still reachable at 1 h spacing -- which is the number `step_long_after`
wants. An index lookup per hour, so keep the range tight.
"""
import argparse
import os
import sys
import warnings
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Defaults chosen to bracket where each model's grid is known or suspected to
# coarsen, rather than sweeping every model to its full range.
RANGES = {
    "hrrr": (1, 50),
    "rap": (1, 54),
    "nam": (30, 90),
    "gfs": (30, 130),
    "nbm": (30, 100),
    "ecmwf": (1, 150),
}


def probe(model, init, lo, hi):
    """Forecast hours present in the archive for one run, between lo and hi."""
    from fetch_nwp_at_obs import HERBIE_KWARGS, HERBIE_MODEL
    from herbie import Herbie

    found, missing = [], []
    for fxx in range(lo, hi):
        try:
            h = Herbie(init.strftime("%Y-%m-%d %H:%M"),
                       model=HERBIE_MODEL[model], fxx=fxx, verbose=False,
                       **HERBIE_KWARGS.get(model, {}))
            (found if getattr(h, "grib", None) else missing).append(fxx)
        except Exception:
            missing.append(fxx)
    return found, missing


def describe(model, found, lo, hi):
    if not found:
        print(f"  NOTHING found in {lo}..{hi - 1} -- wrong product, or this run "
              f"is not archived")
        return
    gaps = sorted({b - a for a, b in zip(found, found[1:])})
    last_hourly = None
    for a, b in zip(found, found[1:]):
        if b - a == 1:
            last_hourly = b
    print(f"  present : {found[0]}..{found[-1]} ({len(found)} hours)")
    print(f"  spacing : {gaps if gaps else 'single hour only'}")
    print(f"  hourly to: {last_hourly if last_hourly else 'never (coarse throughout)'}")
    print(f"  max found: {found[-1]}"
          + ("" if found[-1] < hi - 1 else "  (hit the search ceiling -- raise --hi)"))
    # What RUN_SPECS should say, stated explicitly so the comparison is not left
    # to the reader.
    from fetch_nwp_at_obs import RUN_SPECS
    spec = RUN_SPECS[model]
    declared_max = spec.get("long_max", spec["max"])
    declared_step = spec["step"]
    coarse_after = spec.get("step_long_after")
    print(f"  RUN_SPECS declares max={declared_max}, step={declared_step}"
          + (f", step_long={spec.get('step_long')} after {coarse_after}"
             if coarse_after else ""))
    if found[-1] < declared_max and found[-1] < hi - 1:
        print(f"  >>> MISMATCH: archive stops at {found[-1]} but RUN_SPECS says "
              f"{declared_max}. Candidates past {found[-1]} will be requested and "
              f"come back empty.")
    if gaps and gaps != [1] and declared_step == 1 and not coarse_after:
        print(f"  >>> MISMATCH: grid spacing is {gaps} but step=1 is declared, so "
              f"non-existent hours will be requested.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(RANGES), choices=list(RANGES))
    ap.add_argument("--init", default=None,
                    help="run to probe, 'YYYY-MM-DD HH' UTC. Default: a recent "
                         "00Z/12Z run old enough to have finished publishing.")
    ap.add_argument("--lo", type=int, default=None, help="override range start")
    ap.add_argument("--hi", type=int, default=None, help="override range end")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    if a.init:
        init = datetime.strptime(a.init, "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
    else:
        # Two days back at 12Z: every model has published, and it is inside every
        # archive window.
        now = datetime.now(timezone.utc)
        init = (now.replace(hour=12, minute=0, second=0, microsecond=0)
                - __import__("datetime").timedelta(days=2))
    print(f"probing run {init:%Y-%m-%d %H}Z\n")

    for model in a.models:
        lo, hi = RANGES[model]
        lo = a.lo if a.lo is not None else lo
        hi = a.hi if a.hi is not None else hi
        print(f"{model}  ({lo}..{hi - 1})")
        found, _ = probe(model, init, lo, hi)
        describe(model, found, lo, hi)
        print()


if __name__ == "__main__":
    main()
