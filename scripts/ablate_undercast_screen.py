#!/usr/bin/env python3
"""What each cut in the undercast screen is actually worth.

The write-up lists five conditions and asserts that the last one is the best
addition. This measures all of them, two ways, against the 589 hand-labeled
webcam days -- the only place the screen can be scored against human judgement
of actual photographs.

  Cumulative   start from "the observer reported a deck below the summit" and
               add one cut at a time, in the order the write-up presents them.
               Shows how precision is bought and what recall it costs.

  Leave-one-out  take the finished screen and drop each cut on its own. This is
               the honest measure of a cut's value: a cut can look worthless
               cumulatively just because an earlier one already removed the same
               days, while still being the only thing standing between you and a
               pile of false alarms once the others are in place.

Writes a figure for /weather/details/ and prints the table behind it.

    python3 scripts/ablate_undercast_screen.py
"""
import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "files/weather/examples/model_training_images"
RECORD = "files/weather/obs/undercast_record.csv"
LABELS = "files/weather/csv/MtWashington_undercast_orig.csv"
INK, RED, BLUE, GREEN, GREY = "#2f3337", "#C44E52", "#4C72B0", "#55A868", "#9aa0a6"

# Thresholds mirror build_undercast_record.py. Kept as literals rather than
# imported so this stays a check ON that screen, not a restatement of it -- if
# the two drift apart the numbers here stop matching and that is the signal.
MIN_VIS_SM = 40
MIN_LID_FT = 5000
MIN_DEPTH_FT = 500
MIN_LOWEST_ABOVE_FT = 1000

CUTS = [
    ("deck is broken or overcast", lambda d: d["max_cover"].isin(["BKN", "OVC"])),
    ("visibility > 40 SM", lambda d: d["vis_sm"] > MIN_VIS_SM),
    ("no low lid overhead", lambda d: d["overhead_ceiling_ft"].isna()
     | (d["overhead_ceiling_ft"] >= MIN_LID_FT)),
    ("deck ≥ 500 ft below summit", lambda d: d["depth_below_ft"] >= MIN_DEPTH_FT),
    ("lowest layer overhead ≥ 1,000 ft", lambda d: d["overhead_lowest_ft"].isna()
     | (d["overhead_lowest_ft"] >= MIN_LOWEST_ABOVE_FT)),
]


def prf(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def load(record, labels):
    hand = pd.read_csv(labels)
    hand.columns = [c.strip().lstrip("﻿") for c in hand.columns]
    hand["date"] = pd.to_datetime(hand["Short Date"], format="%m/%d/%y").dt.strftime("%Y-%m-%d")
    hand = hand[["date", "Avg"]].dropna()

    rec = pd.read_csv(record)
    rec["date"] = rec["valid_local"].str[:10]
    hhmm = rec["valid_local"].str[11:16]
    rec["delta"] = (hhmm.str[:2].astype(int) * 60 + hhmm.str[3:].astype(int) - 720).abs()
    # One observation per day: the one nearest local noon, which is the moment
    # the webcam frame was taken.
    #
    # NOT groupby().first(): that returns the first NON-NULL value in each
    # column independently, so a row with a blank ceiling silently borrows one
    # from a different hour of the same day. It made eight days look like the
    # screen rejected them when the real screen accepts them.
    rec = rec.sort_values("delta").drop_duplicates(subset="date", keep="first")

    df = hand.merge(rec, on="date", how="inner")
    for c in ("vis_sm", "depth_below_ft", "overhead_ceiling_ft", "overhead_lowest_ft"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["y"] = df["Avg"] >= 0.5
    # The base population: the observer said something about a deck below the
    # summit. Without that there is nothing for any cut to act on.
    df["base"] = df["n_layers"] > 0
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", default=RECORD)
    p.add_argument("--labels", default=LABELS)
    p.add_argument("--out-dir", default=OUT)
    a = p.parse_args()

    df = load(a.record, a.labels)
    y = df["y"].to_numpy()
    print(f"{len(df)} hand-labeled days joined to the record, "
          f"{int(y.sum())} scored undercast\n")

    # --- cumulative ---------------------------------------------------------
    mask = df["base"].to_numpy()
    rows = [("observer reports a deck below", prf(y, mask))]
    for name, fn in CUTS:
        mask = mask & fn(df).fillna(False).to_numpy()
        rows.append((name, prf(y, mask)))
    final = mask

    print(f"{'cumulative screen':<38}{'TP':>4}{'FP':>5}{'FN':>5}{'P':>7}{'R':>7}{'F1':>7}")
    for name, s in rows:
        print(f"{name:<38}{s['tp']:>4}{s['fp']:>5}{s['fn']:>5}"
              f"{s['precision']:>7.2f}{s['recall']:>7.2f}{s['f1']:>7.2f}")

    # --- leave-one-out ------------------------------------------------------
    print(f"\n{'dropping just this cut':<38}{'TP':>4}{'FP':>5}{'FN':>5}{'P':>7}{'R':>7}"
          f"{'F1':>7}{'ΔFP':>6}{'ΔTP':>6}")
    full = prf(y, final)
    loo = []
    for i, (name, _) in enumerate(CUTS):
        m = df["base"].to_numpy()
        for j, (_, fn) in enumerate(CUTS):
            if i == j:
                continue
            m = m & fn(df).fillna(False).to_numpy()
        s = prf(y, m)
        loo.append((name, s))
        print(f"{name:<38}{s['tp']:>4}{s['fp']:>5}{s['fn']:>5}"
              f"{s['precision']:>7.2f}{s['recall']:>7.2f}{s['f1']:>7.2f}"
              f"{s['fp']-full['fp']:>+6}{s['tp']-full['tp']:>+6}")
    print(f"\n{'FINISHED SCREEN':<38}{full['tp']:>4}{full['fp']:>5}{full['fn']:>5}"
          f"{full['precision']:>7.2f}{full['recall']:>7.2f}{full['f1']:>7.2f}")

    # --- figure -------------------------------------------------------------
    os.makedirs(a.out_dir, exist_ok=True)
    labels = [r[0] for r in rows]
    tp = [r[1]["tp"] for r in rows]
    fp = [r[1]["fp"] for r in rows]
    prec = [r[1]["precision"] for r in rows]
    rec = [r[1]["recall"] for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=160,
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    yy = np.arange(len(rows))
    a1.barh(yy, fp, color="#e3b7b9", label="false alarms")
    a1.barh(yy, tp, left=0, color=RED, height=0.55, label="real undercasts caught")
    a1.set_yticks(yy)
    a1.set_yticklabels([f"+ {l}" if i else l for i, l in enumerate(labels)], fontsize=9.5)
    a1.invert_yaxis()
    a1.set_xlabel("days (of 589 hand-labeled)", fontsize=10.5, color=INK)
    a1.set_title("Each cut applied in turn", fontsize=12.5, loc="left", color=INK, pad=10)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.grid(axis="x", color="#e6e8eb", lw=0.8)
    a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    for i, (t, f) in enumerate(zip(tp, fp)):
        a1.text(max(t, f) + 1.2, i, f"{t} right / {f} wrong", va="center",
                fontsize=8.5, color="#6b7178")

    a2.plot(rec, prec, "-o", color=BLUE, ms=5, lw=1.8)
    # Steps 2 and 3 land almost on top of each other, so alternate the label
    # side rather than letting the digits overlap.
    for i, l in enumerate(labels):
        off = (7, -3) if i % 2 == 0 else (-13, 5)
        a2.annotate(str(i), (rec[i], prec[i]), textcoords="offset points",
                    xytext=off, fontsize=9, color=BLUE)
    a2.set_xlabel("recall", fontsize=10.5, color=INK)
    a2.set_ylabel("precision", fontsize=10.5, color=INK)
    a2.set_title("The trade it buys", fontsize=12.5, loc="left", color=INK, pad=10)
    a2.set_xlim(0, 1.02)
    a2.set_ylim(0, max(prec) * 1.25)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    a2.grid(color="#e6e8eb", lw=0.8)
    a2.set_axisbelow(True)
    a2.text(0.99, -0.30, "0 = no cuts; 5 = the finished screen. "
            "Numbers match the rows on the left.",
            transform=a2.transAxes, ha="right", fontsize=8.5, color=GREY)

    out = os.path.join(a.out_dir, "screen_cut_ablation.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
