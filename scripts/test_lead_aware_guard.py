#!/usr/bin/env python3
"""Check that the serving guard accepts partial-source hours -- but only once the
model has actually been trained to handle them.

Two rules were relaxed together so the combined model can forecast past 48 h:

  * ``train_undercast_obs.rows_for_source("all")`` now keeps a row if every source
    that COULD reach its lead is present, instead of requiring all six.
  * ``weather_to_json.predict_current_model`` asks the same question per forecast
    hour, so it stops nulling an hour merely because HRRR cannot reach it.

Relaxing a guard is exactly the kind of change that can quietly start publishing
nonsense, so two properties are pinned here, and they pull in opposite directions:

  1. An hour missing only the sources that cannot reach its lead IS usable. This is
     the point of the change -- without it every row past 48 h is discarded.
  2. An hour past the longest lead the DEPLOYED model was trained at is NOT
     published, however complete it looks. The model on disk today knows leads
     1/24/48; applying it at 96 h would be extrapolation, and the threshold would
     have to be invented. Long leads must stay nulled until a retrain on the
     ladder makes them real, and then start working with no further code change.

Property 2 is why this file exists. It is the failure that would otherwise ship
silently: the relaxed source rule alone would happily hand the 48 h model a 144 h
row and publish a confident number.

    python3 scripts/test_lead_aware_guard.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import undercast_eval as ue  # noqa: E402
import weather_to_json as wj  # noqa: E402
from train_undercast_obs import sources_expected_at  # noqa: E402

ALL = ("hrrr", "nam", "gfs", "rap", "ecmwf", "nbm")
LEADS = (6, 48, 72, 96, 144, 200)
PER_LEAD = 6


def simulated_frame(rows):
    """A serving-shaped frame: raw download columns, one block per lead, with the
    sources that cannot reach each lead blanked as a live run would leave them."""
    raw = [c for c in wj.variables if c in rows.columns]
    if "boundary_layer_cloud_layer" in rows.columns:
        raw = raw + ["boundary_layer_cloud_layer"]
    rows = rows[raw].copy()
    out = []
    for i, lead in enumerate(LEADS):
        chunk = rows.iloc[i * PER_LEAD:(i + 1) * PER_LEAD].copy()
        if not len(chunk):
            raise SystemExit("not enough all-source rows to build the fixture")
        chunk["fxx"] = lead
        keep = set(sources_expected_at(lead))
        for src in ALL:
            if src in keep:
                continue
            for c in chunk.columns:
                if c.endswith(f"_{src}") or (
                        src == "hrrr" and c == "boundary_layer_cloud_layer"):
                    chunk[c] = np.nan
        out.append(chunk)
    return pd.concat(out, ignore_index=True)


def published(frame, max_fxx=400):
    out = wj.predict_current_model(frame, "2026-02-10 12:00", max_fxx=max_fxx)
    seen = {}
    for h, y in zip(out["x"], out["y"]):
        seen.setdefault(int(h), y is not None)
    return seen


def main():
    warnings.filterwarnings("ignore")
    df = ue.load_frame(cache=os.environ.get("UNDERCAST_FRAME_CACHE") or None)
    rows = ue.split_rows(df, "all", "holdout_baserate").head(len(LEADS) * PER_LEAD)
    frame = simulated_frame(rows)
    print(f"fixture: {len(frame)} rows at leads {sorted(set(frame['fxx']))}\n")

    # --- property 0: the two copies of the reach table still agree ------------
    # SOURCE_MAX_LEAD_H lives in train_undercast_obs rather than being derived from
    # fetch_nwp_at_obs.RUN_SPECS, because training should not have to import Herbie
    # to know how far GFS runs. That is a deliberate duplication, and duplication is
    # what hid the GFS longitude error for months -- so it is asserted, not trusted.
    from fetch_nwp_at_obs import RUN_SPECS
    from train_undercast_obs import SOURCE_MAX_LEAD_H
    for model, reach in SOURCE_MAX_LEAD_H.items():
        spec = RUN_SPECS[model]
        expect = spec.get("long_max", spec["max"])
        assert reach == expect, (
            f"{model}: train_undercast_obs says it reaches {reach} h, "
            f"fetch_nwp_at_obs.RUN_SPECS says {expect} h"
        )
    print(f"reach table agrees with RUN_SPECS for all "
          f"{len(SOURCE_MAX_LEAD_H)} sources")

    # --- property 1: the source requirement is lead-aware ---------------------
    # Checked directly, because the trained-lead cap below would otherwise mask it.
    for lead in LEADS:
        expect = set(sources_expected_at(lead))
        assert ("hrrr" in expect) == (lead <= 48), lead
        assert ("ecmwf" in expect) == (lead <= 144), lead
    assert sources_expected_at(200) == [], "nothing reaches 200 h"
    print("sources_expected_at: lead-aware as intended")

    # --- property 2: no extrapolation past the trained leads ------------------
    trained = sorted(float(k) for k in
                     wj.json.load(open(
                         f"{wj.CURRENT_MODEL_DIR}/"
                         f"model_metadata_{wj.CURRENT_SOURCE}.json"
                     ))[wj.CURRENT_ALGO]["threshold_by_lead"])
    cap = max(trained) if trained else 48.0
    print(f"deployed model trained at leads {trained} -> cap {cap:.0f} h")

    seen = published(frame)
    print("\nlead | published")
    bad = []
    for lead in LEADS:
        want = lead <= cap and bool(sources_expected_at(lead))
        got = seen.get(lead)
        ok = got == want
        print(f"{lead:>4}h | {str(got):>5}   expected {want}   {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(lead)
    if bad:
        raise SystemExit(f"FAIL: wrong publish decision at leads {bad}")

    # The forward-looking half: once a retrain adds ladder thresholds, the long
    # leads must start publishing with no further code change. Simulated by
    # raising the cap the same way a retrain would.
    import json as _json
    meta_path = f"{wj.CURRENT_MODEL_DIR}/model_metadata_{wj.CURRENT_SOURCE}.json"
    meta = _json.load(open(meta_path))
    thr = meta[wj.CURRENT_ALGO]["threshold_by_lead"]
    patched = dict(thr)
    for lead in (72, 96, 144):
        patched[str(lead)] = thr[max(thr, key=lambda k: float(k))]
    meta[wj.CURRENT_ALGO]["threshold_by_lead"] = patched
    tmp = meta_path + ".leadtest"
    os.rename(meta_path, tmp)
    try:
        with open(meta_path, "w") as fh:
            _json.dump(meta, fh)
        seen2 = published(frame)
    finally:
        os.replace(tmp, meta_path)

    print("\nwith ladder thresholds present (simulating a retrain):")
    bad = []
    for lead in LEADS:
        want = lead <= 144 and bool(sources_expected_at(lead))
        got = seen2.get(lead)
        ok = got == want
        print(f"{lead:>4}h | {str(got):>5}   expected {want}   {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(lead)
    if bad:
        raise SystemExit(f"FAIL: after retrain, wrong decision at leads {bad}")

    print("\nPASS: partial-source hours are usable, and nothing is extrapolated "
          "past the model's trained leads")


if __name__ == "__main__":
    main()
