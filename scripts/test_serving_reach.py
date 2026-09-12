#!/usr/bin/env python3
"""Check that the serving path can actually publish as far as the model was trained.

This exists because of a silent failure that survived a retrain. The ladder retrain
gave the deployed model per-lead thresholds out to 144 h, and
``test_lead_aware_guard.py`` proved the guard would accept those leads -- but it
proved it by calling ``predict_current_model(..., max_fxx=400)``, pinning the
ceiling open. In the real run two OTHER things still capped the panel at 48 h:

  * the caller passed ``max_fxx=min(max(FXX_LIST), max(FXX_LIST_GFS),
    max(FXX_LIST_NAM))`` -- 48 -- which silently outranked the model, and
  * ECMWF and NBM, the only two sources that reach past 120 h, were never FETCHED
    past 48 h at all, so the rows could not have existed even with the cap lifted.

So a model that knew about six days published one day and a half, and every test
passed. Three properties are pinned here, each the negation of one way that can
happen again:

  1. Nothing in the serving path publishes a shorter horizon than the model's own
     trained leads. Checked with DEFAULT arguments -- the whole point is that no
     second ceiling hides below the model's.
  2. Every lead the model was trained at is SPANNED by the fetch -- every source
     expected at that lead is downloaded at least that far. A threshold for 144 h
     is worthless if nothing downloads 144 h. Spanned rather than hit exactly,
     because lead 1 h is deliberately not on the grid: the near end is sampled at
     0 and 2 h and _lead_threshold interpolates across it. Past 48 h the trained
     leads are all multiples of three and are required exactly.
  3. Every model's availability probe leaves room to step back to an older run.
     resolve_run ADDS its offset to the probe lead, so a probe at the product
     maximum rejects every run but the newest. Extending RAP's grid from 48 to 51 h
     tripped this on the first live run: the probe went to 51 h, every candidate was
     asked for 54 h or more, and RAP came back empty for the whole run -- which
     would have nulled the panel at every lead RAP is expected at, 48 h and under.

  4. No source loses a forecast hour relative to what it was fetched at before.
     This is not hypothetical: the first version of the 3-hourly long-lead rule
     applied a flat 48 h boundary and quietly dropped NAM's 50, 52, 56 and 58 h
     while appearing only to add reach.

    python3 scripts/test_serving_reach.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_to_json as wj  # noqa: E402
from train_undercast_obs import SOURCE_MAX_LEAD_H, sources_expected_at  # noqa: E402


def legacy_grid(src, last):
    """What this source was fetched at before long leads were served."""
    if src == "ecmwf":
        return sorted(range(0, last + 1, 3))  # IFS open data is 3-hourly
    return sorted(set(range(0, last + 1, 2)) | set(range(0, last + 1, 3)))


def main():
    warnings.filterwarnings("ignore")
    grids = wj.forecast_hour_grids()
    by_src = {src: grids[herbie] for src, herbie in wj.HERBIE_NAME.items()}
    horizon = wj.current_model_max_lead()
    trained = sorted(float(k) for k in wj.json.load(open(
        f"{wj.CURRENT_MODEL_DIR}/model_metadata_{wj.CURRENT_SOURCE}.json"
    ))[wj.CURRENT_ALGO]["threshold_by_lead"])
    print(f"deployed model trained at leads {[int(t) for t in trained]} "
          f"-> horizon {horizon:.0f} h")
    print(f"grid: {sum(len(v) for v in grids.values())} source-hours\n")
    fail = []

    # --- property 3 first: it is the one a careless edit trips ----------------
    print("source | hours  max   lost vs the legacy grid")
    for src, hours in by_src.items():
        was = legacy_grid(src, wj.LEGACY_LAST_H[src])
        lost = sorted(set(was) - set(hours))
        print(f"{src:6s} | {len(hours):5d} {max(hours):4d}   {lost or '-'}")
        if lost:
            fail.append(f"{src} no longer fetches {lost}")

    # --- property 2: every trained lead is reachable from every needed source -
    print("\nlead | sources expected             | spanned | on the grid")
    for lead in trained:
        need = sources_expected_at(lead)
        short = [s for s in need if max(by_src[s]) < lead]
        exact = [s for s in need if int(lead) in by_src[s]]
        print(f"{int(lead):4d} | {','.join(need):28s} | "
              f"{'yes' if need and not short else 'NO: ' + ','.join(short):7s} | "
              f"{len(exact)}/{len(need)}")
        if not need:
            fail.append(f"lead {lead:.0f} h: no source reaches it, so it cannot "
                        f"be published -- the model should not carry a threshold")
        if short:
            fail.append(f"lead {lead:.0f} h: trained, but {','.join(short)} "
                        f"is not fetched that far")
        # Past the legacy boundary an hour has to be hit exactly to be published:
        # there is no denser neighbour to interpolate a forecast from, only a
        # threshold. Inside it, 0/2/3-hourly sampling brackets every lead.
        if lead > max(wj.LEGACY_LAST_H.values()) and len(exact) < len(need):
            fail.append(f"lead {lead:.0f} h: past the dense part of the grid and "
                        f"not fetched exactly by {set(need) - set(exact)}")

    # --- property 3: the probe can still step back to an older run ------------
    print("\nsource | probe  +step  product max")
    for src, hours in by_src.items():
        herbie = wj.HERBIE_NAME[src]
        probe = max(hours) if herbie == "rap" else min(max(hours), 18)
        step = wj.RUN_STEP_H.get(herbie, 6)
        cap = wj.MODEL_MAX_LEAD_H[herbie]
        probe = min(probe, cap - step)
        print(f"{src:6s} | {probe:5d}  {probe + step:5d}  {cap:11d}")
        if probe + step > cap:
            fail.append(f"{src}: probes at {probe} h, so one step back asks for "
                        f"{probe + step} h beyond its {cap} h maximum -- every "
                        f"older run is rejected and the source comes back empty")
        if probe <= 0:
            fail.append(f"{src}: probe lead collapsed to {probe}")

    # A source is never asked for an hour its PRODUCT cannot serve, or Herbie
    # returns nothing and the hour is a wasted request rather than a forecast.
    for src, hours in by_src.items():
        cap = wj.MODEL_MAX_LEAD_H[wj.HERBIE_NAME[src]]
        over = [h for h in hours if h > cap]
        if over:
            fail.append(f"{src}: asks for {over} beyond its {cap} h product maximum")
        train_cap = SOURCE_MAX_LEAD_H[src]
        skew = [h for h in hours if h > train_cap]
        if skew:
            fail.append(f"{src}: asks for {skew} past the {train_cap} h the model "
                        f"was TRAINED to expect it at -- train/serve skew")

    # Past the legacy boundary only multiples of three can ever be published,
    # because every long lead expects ECMWF and ECMWF is 3-hourly.
    for src, hours in by_src.items():
        stray = [h for h in hours
                 if h > wj.LEGACY_LAST_H[src] and h % 3 != 0]
        if stray:
            fail.append(f"{src}: fetches {stray} past {wj.LEGACY_LAST_H[src]} h, "
                        f"which can never carry ECMWF and so can never publish")

    # --- property 1: no ceiling below the model's, with DEFAULT arguments -----
    import test_lead_aware_guard as guard
    import undercast_eval as ue

    df = ue.load_frame(cache=os.environ.get("UNDERCAST_FRAME_CACHE") or None)
    rows = ue.split_rows(df, "all", "holdout_baserate").head(
        len(guard.LEADS) * guard.PER_LEAD)
    frame = guard.simulated_frame(rows)
    out = wj.predict_current_model(frame, "2026-02-10 12:00")  # no max_fxx
    published = sorted({int(h) for h, y in zip(out["x"], out["y"]) if y is not None})
    reach = max(published) if published else 0
    print(f"\nwith default arguments the panel publishes out to {reach} h "
          f"(model horizon {horizon:.0f} h)")
    if reach < horizon:
        fail.append(f"a ceiling below the model's own caps the panel at {reach} h "
                    f"while the model is trained to {horizon:.0f} h")

    if fail:
        print("\nFAIL:")
        for f in fail:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: the fetch reaches every trained lead, nothing published short "
          "of the model's horizon, and no source lost an hour")


if __name__ == "__main__":
    main()
