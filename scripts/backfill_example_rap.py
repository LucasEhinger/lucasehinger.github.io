"""Backfill RAP series into the historical example snapshots.

The example weather snapshots under files/weather/examples/<date>/ were captured
before RAP was added, so they only carry HRRR/NAM/GFS. This fetches archived RAP
for each snapshot's run time + forecast hours and merges the rap_* series in, so
the historical page's RAP toggle has data. Idempotent: re-running refreshes the
rap keys. ECMWF/NBM open-data archives don't reach these older dates, so they are
not backfilled.

Reuses weather_to_json.process_forecast_data for the actual extraction, so the
sampled values match exactly what the live pipeline produces.

    python3 scripts/backfill_example_rap.py
"""

import glob
import importlib.util
import json
import math
import os
import signal

import numpy as np

PER_DATE_TIMEOUT = 300  # seconds; abort a snapshot whose fetches hang the network

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "w2j", os.path.join(_here, "weather_to_json.py")
)
w2j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w2j)

RAP_LABELS = [k for k, v in w2j.variables.items() if v.get("model") == "rap"]
LOC = w2j.LOCATIONS[0]["name"]


def _clean(val):
    try:
        f = float(val)
        return None if (math.isnan(f) or not np.isfinite(f)) else f
    except Exception:
        return None


def backfill_one(path):
    with open(path) as f:
        snap = json.load(f)
    if any(k.endswith("_rap") for k in snap):
        print("  already has rap data; skipping")
        return
    date_str = snap.get("date_str")
    ref = next(
        (snap[k]["x"] for k in snap if isinstance(snap[k], dict) and "x" in snap[k]),
        None,
    )
    if not date_str or ref is None:
        print(f"  skip: no date_str/x axis")
        return
    fxxs = [int(v) for v in ref]

    per_fxx = {}
    for fxx in fxxs:
        out = w2j.process_forecast_data(
            (fxx, date_str, "rap", w2j.LOCATIONS, w2j.variables)
        )
        vals = {}
        if out is not None:
            _, _, results = out
            for label, info in results.get(LOC, {}).items():
                if "value" in info:
                    vals[label] = _clean(info["value"])
        per_fxx[fxx] = vals

    added = 0
    for label in RAP_LABELS:
        y = [per_fxx.get(fxx, {}).get(label) for fxx in fxxs]
        if any(v is not None for v in y):
            snap[label] = {"x": fxxs, "y": y}
            added += 1
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    print(f"  {added} rap series (of {len(RAP_LABELS)}) across {len(fxxs)} fxx")


def main():
    paths = sorted(glob.glob("files/weather/examples/*/weather_data_*.json"))
    print(f"Backfilling RAP into {len(paths)} snapshots ({len(RAP_LABELS)} rap params)")
    have_alarm = hasattr(signal, "SIGALRM")
    for p in paths:
        print(f"- {p}", flush=True)
        if have_alarm:
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(PER_DATE_TIMEOUT)
        try:
            backfill_one(p)
        except TimeoutError:
            print(f"  TIMEOUT after {PER_DATE_TIMEOUT}s; leaving this snapshot unchanged")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
        finally:
            if have_alarm:
                signal.alarm(0)


if __name__ == "__main__":
    main()
