"""Backfill a model's series into the historical example snapshots.

The example weather snapshots under files/weather/examples/<date>/ were captured
before RAP and NBM were added, so they only carry HRRR/NAM/GFS. This fetches the
archived model for each snapshot's run time + forecast hours and merges the
<model>_* series in, so the historical page's toggles have data.

Reuses weather_to_json.process_forecast_data for the actual extraction, so the
sampled values match exactly what the live pipeline produces -- and, critically,
the y arrays land on the snapshot's existing forecast-hour axis, which the plot
JS reuses for every model.

    python3 scripts/backfill_example_model.py --model nbm
    python3 scripts/backfill_example_model.py --model rap --force

ECMWF is not backfilled: its open-data archive doesn't reach these older dates.
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import signal

import numpy as np

PER_DATE_TIMEOUT = 900  # seconds; abort a snapshot whose fetches hang the network

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "w2j", os.path.join(_here, "weather_to_json.py")
)
w2j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w2j)

LOC = w2j.LOCATIONS[0]["name"]

# ECMWF is fetched under Herbie's model key "ifs", but its variables are labelled
# *_ecmwf, so the fetch key and the label suffix differ for it alone.
MODEL_ALIASES = {"ecmwf": "ifs"}
LABEL_SUFFIX = {"ifs": "ecmwf"}


def suffix_for(model):
    return LABEL_SUFFIX.get(model, model)


def _base(label, model):
    """Strip the trailing _<suffix> to get the bare parameter name."""
    suf = suffix_for(model)
    return label[: -(len(suf) + 1)] if label.endswith(f"_{suf}") else label


def labels_for(model, shared_only=False):
    """Variable labels for a model.

    With shared_only, keep just the parameters at least one *other* model also
    provides (e.g. tmp_925mb, which HRRR/NAM/GFS/RAP all carry), so a backfill
    adds comparable series to the existing plots rather than introducing
    single-source parameters that nothing else can be compared against.
    """
    labels = [k for k, v in w2j.variables.items() if v.get("model") == model]
    if not shared_only:
        return labels
    others = set()
    for k, v in w2j.variables.items():
        m = v.get("model")
        if m and m != model:
            others.add(_base(k, m))
    return [k for k in labels if _base(k, model) in others]


def _clean(val):
    try:
        f = float(val)
        return None if (math.isnan(f) or not np.isfinite(f)) else f
    except Exception:
        return None


def backfill_one(path, model, labels, force):
    with open(path) as f:
        snap = json.load(f)
    if not force and any(k.endswith(f"_{suffix_for(model)}") for k in snap):
        print(f"  already has {model} data; skipping (use --force to refresh)")
        return
    date_str = snap.get("date_str")
    ref = next(
        (snap[k]["x"] for k in snap if isinstance(snap[k], dict) and "x" in snap[k]),
        None,
    )
    if not date_str or ref is None:
        print("  skip: no date_str/x axis")
        return
    fxxs = [int(v) for v in ref]

    per_fxx = {}
    for fxx in fxxs:
        out = w2j.process_forecast_data(
            (fxx, date_str, model, w2j.LOCATIONS, w2j.variables)
        )
        vals = {}
        if out is not None:
            _, _, results = out
            for label, info in results.get(LOC, {}).items():
                if "value" in info:
                    vals[label] = _clean(info["value"])
        per_fxx[fxx] = vals

    added, allzero = 0, []
    for label in labels:
        y = [per_fxx.get(fxx, {}).get(label) for fxx in fxxs]
        if any(v is not None for v in y):
            snap[label] = {"x": fxxs, "y": y}
            added += 1
            # An all-zero series means the archive didn't populate that field on
            # this date. Worth reporting so the plots aren't trusted blindly.
            if all((v or 0) == 0 for v in y):
                allzero.append(label)
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    note = f"; all-zero: {', '.join(allzero)}" if allzero else ""
    print(f"  {added} {model} series (of {len(labels)}) across {len(fxxs)} fxx{note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nbm", help="model key, e.g. nbm or rap")
    ap.add_argument(
        "--force", action="store_true", help="refresh snapshots that already have it"
    )
    ap.add_argument(
        "--shared-only",
        action="store_true",
        help="only backfill parameters that at least one other model also provides",
    )
    args = ap.parse_args()

    model = MODEL_ALIASES.get(args.model, args.model)
    labels = labels_for(model, args.shared_only)
    if not labels:
        raise SystemExit(f"No variables defined for model {model!r}")

    paths = sorted(glob.glob("files/weather/examples/*/weather_data_*.json"))
    print(
        f"Backfilling {args.model.upper()} into {len(paths)} snapshots ({len(labels)} params)"
    )
    have_alarm = hasattr(signal, "SIGALRM")
    for p in paths:
        print(f"- {p}", flush=True)
        if have_alarm:
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(PER_DATE_TIMEOUT)
        try:
            backfill_one(p, model, labels, args.force)
        except TimeoutError:
            print(f"  TIMEOUT after {PER_DATE_TIMEOUT}s; leaving this snapshot unchanged")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
        finally:
            if have_alarm:
                signal.alarm(0)


if __name__ == "__main__":
    main()
