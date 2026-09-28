#!/usr/bin/env python3
"""Fit the KMWN remark screen's cuts to the reviewed webcam labels.

The screen in ``build_undercast_record.py`` turns each hourly observer remark into
'undercast' / 'ambiguous' / 'clear'. Its thresholds were originally tuned against
a hand-labeled set in which a day scored >= 0.5 on the two-camera average counted
as undercast. Those labels were re-examined image by image; this script re-fits
the cuts to the reviewed verdicts.

Two things make that harder than it sounds, and both are reported rather than
hidden:

  * There are only 20 positive days. A grid search over six thresholds will
    happily find a configuration that fits them and nothing else, so every
    candidate is also scored out of sample by repeated stratified 5-fold CV over
    DAYS, and the search reports the plateau -- how many configurations sit within
    noise of the best -- rather than just an argmax.
  * The remarks are not always sufficient. 2024-09-06 and 2024-08-29 have
    identical values in every field the screen can see (BKN, tops 5,000 ft,
    1,288 ft below the summit, 80 SM visibility, summit clear) and the cameras
    show a solid deck to the horizon on one and scattered cumulus over open
    valleys on the other. No threshold can separate those two, and the ceiling
    that imposes is the main result here.

    python3 scripts/tune_undercast_screen.py
    python3 scripts/tune_undercast_screen.py --json out.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RECORD = "files/weather/obs/undercast_record.csv"
REVIEWED = "files/weather/csv/undercast_labels_reviewed.csv"
ORIG = "files/weather/csv/MtWashington_undercast_orig.csv"

# Search space. Deliberately coarse: with 20 positives, a finer grid buys
# precision in the fit and nothing out of sample.
COVERS = {"BKN/OVC": ("BKN", "OVC"), "SCT/BKN/OVC": ("SCT", "BKN", "OVC")}
VIS = (0, 10, 20, 30, 40, 50, 60, 70, 80)
LID = (0, 3000, 4000, 5000, 6000, 8000)
DEPTH = (0, 250, 500, 750, 1000, 1500)
LOWEST = (0, 250, 500, 1000, 1500, 2000)


def noon_frame(record=RECORD, reviewed=REVIEWED, orig=ORIG):
    """One row per hand-labeled day: the observation nearest local noon, plus the
    reviewed verdict. The webcam frames are local-noon captures, so noon is the
    matching time, not a convenience."""
    rec = pd.read_csv(record, keep_default_na=False, low_memory=False)
    rec["lt"] = pd.to_datetime(rec["valid_local"], format="%Y-%m-%dT%H:%M")
    rec["date"] = rec["lt"].dt.strftime("%Y-%m-%d")

    lab = pd.read_csv(orig, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    lab["date"] = pd.to_datetime(lab["Short Date"], format="%m/%d/%y").dt.strftime("%Y-%m-%d")
    old = {}
    for r in lab.to_dict("records"):
        try:
            old[r["date"]] = int(float(r["Avg"]) >= 0.5)
        except ValueError:
            old[r["date"]] = 0

    rev = pd.read_csv(reviewed, keep_default_na=False)
    new = dict(zip(rev["date"], (rev["updated"] == "Yes").astype(int)))

    day = rec[rec["date"].isin(old)].copy()
    day["dmin"] = (day["lt"].dt.hour * 60 + day["lt"].dt.minute - 720).abs()
    noon = day.sort_values("dmin").groupby("date", as_index=False).first()
    # Only the days that were re-examined get a new verdict; every other labeled
    # day scored zero on both cameras and stays negative.
    noon["y_new"] = noon["date"].map(lambda d: new.get(d, old.get(d, 0)))
    noon["y_old"] = noon["date"].map(lambda d: old.get(d, 0))
    return noon


def columns(noon):
    def num(c):
        return pd.to_numeric(noon[c], errors="coerce").to_numpy(dtype=float)

    def flag(c):
        return pd.to_numeric(noon[c], errors="coerce").fillna(0).to_numpy()

    return dict(vis=num("vis_sm"), depth=num("depth_below_ft"),
                lid=num("overhead_ceiling_ft"), low=num("overhead_lowest_ft"),
                tops_below=flag("all_tops_below_summit"),
                in_cloud=flag("summit_in_cloud"),
                cover=noon["max_cover"].to_numpy())


def predict(c, cfg):
    """The screen's positive test, vectorised over days."""
    ok = np.isin(c["cover"], list(cfg["cover"])) & (c["vis"] > cfg["vis"])
    if cfg["req_tops_below"]:
        ok &= c["tops_below"] == 1
    if cfg["req_summit_clear"]:
        ok &= c["in_cloud"] == 0
    ok &= np.isnan(c["lid"]) | (c["lid"] >= cfg["lid"])
    if cfg["req_depth"]:
        ok &= ~np.isnan(c["depth"]) & (c["depth"] >= cfg["depth"])
    else:
        ok &= np.isnan(c["depth"]) | (c["depth"] >= cfg["depth"])
    ok &= np.isnan(c["low"]) | (c["low"] >= cfg["lowest"])
    return ok


def prf(pred, y):
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(y.sum() - tp)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9), tp, fp, fn


def cv_f1(c, cfg, y, seeds=20, folds=5):
    """Out-of-sample F1: the cuts are FIXED, so this measures how well a screen
    chosen on one set of days scores on days it did not see. Repeated because with
    20 positives a single split is mostly luck."""
    rng = np.random.default_rng(0)
    pred = predict(c, cfg)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(seeds):
        rng.shuffle(pos)
        rng.shuffle(neg)
        for k in range(folds):
            idx = np.concatenate([pos[k::folds], neg[k::folds]])
            out.append(prf(pred[idx], y[idx])[2])
    return float(np.mean(out)), float(np.std(out))


def search(c, y):
    rows = []
    for cname, cset in COVERS.items():
        for rt in (0, 1):
            for rc in (0, 1):
                for v in VIS:
                    for l in LID:
                        for rd in (0, 1):
                            for d in DEPTH:
                                for lo in LOWEST:
                                    cfg = dict(cover=cset, req_tops_below=rt,
                                               req_summit_clear=rc, vis=v, lid=l,
                                               req_depth=rd, depth=d, lowest=lo)
                                    P, R, F1, tp, fp, fn = prf(predict(c, cfg), y)
                                    rows.append((F1, P, R, tp, fp, fn, cname, rt, rc,
                                                 v, l, rd, d, lo))
    return pd.DataFrame(rows, columns=["F1", "P", "R", "tp", "fp", "fn", "cover",
                                       "req_tops_below", "req_summit_clear", "vis",
                                       "lid", "req_depth", "depth", "lowest"])


def n_cuts(r):
    """How many knobs the configuration actually turns on. Used to break ties on
    the plateau toward the simplest screen, which is the one least likely to be
    fitting these particular 20 days."""
    return (int(r.req_tops_below) + int(r.req_summit_clear) + int(r.vis > 0)
            + int(r.lid > 0) + int(r.req_depth) + int(r.depth > 0) + int(r.lowest > 0))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", default=RECORD)
    p.add_argument("--reviewed", default=REVIEWED)
    p.add_argument("--json", default=None, help="write the chosen cuts here")
    a = p.parse_args()

    noon = noon_frame(a.record, a.reviewed)
    c = columns(noon)
    y_new = noon["y_new"].to_numpy()
    y_old = noon["y_old"].to_numpy()
    print(f"{len(noon)} hand-labeled days with a noon observation")
    print(f"  old labels (two-camera average >= 0.5): {y_old.sum()} undercast")
    print(f"  reviewed labels:                        {y_new.sum()} undercast "
          f"({int((y_old != y_new).sum())} days changed)\n")

    # What the shipped screen scores, against both label sets.
    shipped = (noon["label"] == "undercast").to_numpy()
    for name, y in (("old", y_old), ("reviewed", y_new)):
        P, R, F1, tp, fp, fn = prf(shipped, y)
        print(f"  shipped cuts vs {name:8s} labels: P={P:.2f} R={R:.2f} F1={F1:.3f} "
              f"(tp={tp} fp={fp} fn={fn})")

    df = search(c, y_new)
    best = df["F1"].max()
    plateau = df[df["F1"] >= best - 0.02].copy()
    print(f"\nsearched {len(df):,} configurations against the reviewed labels")
    print(f"best in-sample F1 {best:.3f}; {len(plateau)} configurations within 0.02")

    plateau["n_cuts"] = plateau.apply(n_cuts, axis=1)
    # Simplest screen on the plateau, then highest precision among equals.
    plateau = plateau.sort_values(["n_cuts", "P", "F1"], ascending=[True, False, False])
    print("\n=== the plateau, simplest first ===")
    print(plateau.head(12).to_string(index=False))

    pick = plateau.iloc[0]
    cfg = dict(cover=list(COVERS[pick["cover"]]), req_tops_below=int(pick.req_tops_below),
               req_summit_clear=int(pick.req_summit_clear), vis=int(pick.vis),
               lid=int(pick.lid), req_depth=int(pick.req_depth), depth=int(pick.depth),
               lowest=int(pick.lowest))
    cfg_run = dict(cfg, cover=tuple(cfg["cover"]))
    m, s = cv_f1(c, cfg_run, y_new)
    P, R, F1, tp, fp, fn = prf(predict(c, cfg_run), y_new)
    print(f"\n=== chosen ===\n{json.dumps(cfg, indent=2)}")
    print(f"in-sample  P={P:.2f} R={R:.2f} F1={F1:.3f}  (tp={tp} fp={fp} fn={fn})")
    print(f"5-fold CV over days, 20 repeats: F1={m:.3f} +/- {s:.3f}")

    # The ceiling: days the screen cannot distinguish however the cuts are set.
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"cuts": cfg, "in_sample": dict(P=P, R=R, F1=F1, tp=tp, fp=fp, fn=fn),
                       "cv_f1": m, "cv_sd": s, "n_positive": int(y_new.sum()),
                       "n_days": int(len(noon))}, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
