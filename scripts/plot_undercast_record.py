#!/usr/bin/env python3
"""Figures about the observation RECORD itself (not the models).

Two panels, both for /weather/details/:

  undercast_rate_drift.png
      Top panel: the share of observations passing the screen, ~1.2% (1997) to
      ~5.4% (2024). Bottom panel: what that decomposes into -- observers
      MENTIONING a deck below the summit more often (x2.6), and those mentions
      more often being complete enough to clear the screen (x1.7). 2.6 x 1.7 =
      the 4.5x rise in the top panel, and both factors are reporting behaviour.
      Note 2025-26 fall back toward the long-run level, so this is a noisy
      upward drift rather than a steady climb.

  training_data_scale.png
      Why the record exists. Hand-labeling a year of webcam stills yields 589
      days and 24 positives; reading the observer's own words yields 253,317
      hourly observations and 6,843 positives, from which the training sample is
      drawn. Log scale, because the point is the order of magnitude.

    python3 scripts/plot_undercast_record.py
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "files/weather/examples/model_training_images"
RECORD = "files/weather/obs/undercast_record.csv"
INK = "#2f3337"
BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
GREY = "#9aa0a6"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="y", color="#e6e8eb", linewidth=0.8)
    ax.set_axisbelow(True)


def drift(df, out):
    """Two panels: the drift, then what it decomposes into.

    A single panel putting both series on one axis buries the screen rate (1-5%)
    under the mention rate (7-19%). Splitting them also lets the second panel
    make the actual argument: the rise is the product of observers MENTIONING a
    lower deck more often (x2.6 from 1997 to 2024) and those mentions more often
    being complete enough to clear the screen (x1.7). Both are reporting
    behaviour. Neither is weather.
    """
    g = df.groupby("year")
    rate = g["is_pos"].mean() * 100
    mention = g["has_tps"].mean() * 100
    cond = (g["is_pos"].sum() / g["has_tps"].sum()) * 100
    n = g.size()
    years = rate.index.to_numpy()
    lo, hi = int(years.min()), int(years.max())

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.0), dpi=160, sharex=True)

    a1.plot(years, rate, "-o", ms=4, lw=2, color=RED)
    a1.set_ylabel("passes the screen (%)", color=INK, fontsize=10.5)
    a1.set_title("The share of observations reported as undercast drifts upward",
                 color=INK, fontsize=12.5, pad=10, loc="left")
    _style(a1)
    for i, off in ((0, (8, -18)), (int(np.argmax(rate.to_numpy())), (-16, 10))):
        a1.annotate(f"{rate.iloc[i]:.1f}%  ({int(years[i])})", (years[i], rate.iloc[i]),
                    textcoords="offset points", xytext=off, fontsize=9, color=RED)
    a1.set_ylim(0, max(rate) * 1.35)

    a2.plot(years, mention, "-o", ms=4, lw=2, color=BLUE,
            label="observer mentions a deck below the summit at all")
    a2.plot(years, cond, "-o", ms=4, lw=2, color=GREEN,
            label="of those mentions, share clearing the screen")
    a2.set_ylabel("share (%)", color=INK, fontsize=10.5)
    a2.set_xlabel("year", color=INK, fontsize=11)
    a2.set_title("Both components are reporting behaviour, not weather",
                 color=INK, fontsize=12.5, pad=10, loc="left")
    _style(a2)
    a2.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=1)
    a2.set_ylim(0, max(max(mention), max(cond)) * 1.45)
    a2.annotate(f"x{mention.loc[2024]/mention.loc[1997]:.1f} since 1997",
                (2024, mention.loc[2024]), textcoords="offset points",
                xytext=(-96, 13), fontsize=9, color=BLUE)
    a2.annotate(f"x{cond.loc[2024]/cond.loc[1997]:.1f}",
                (2024, cond.loc[2024]), textcoords="offset points",
                xytext=(8, 5), fontsize=9, color=GREEN)

    a2.text(0.995, -0.26,
            f"{int(n.sum()):,} hourly observations, {lo}\u2013{hi}. "
            f"{hi} is a partial year; 2025\u201326 fall back toward the long-run level.",
            transform=a2.transAxes, ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return rate, mention


def scale(df, out, n_days=589, n_hand_pos=24, n_sample=None):
    total = len(df)
    pos = int(df["is_pos"].sum())
    labels = ["Hand-labeled\nwebcam stills", "Observer remarks\n(1997–2026)"]
    totals = [n_days, total]
    positives = [n_hand_pos, pos]

    fig, ax = plt.subplots(figsize=(8.8, 3.9), dpi=160)
    y = np.arange(len(labels))
    ax.barh(y + 0.15, totals, height=0.26, color="#d7dce3",
            label="observations available")
    ax.barh(y - 0.15, positives, height=0.26, color=RED,
            label="undercast examples")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(8, total * 12)
    ax.margins(y=0.30)
    ax.set_xlabel("count (log scale)", color=INK, fontsize=11)
    ax.set_title("What reading the observer's own words buys",
                 color=INK, fontsize=12.5, pad=12, loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e8eb", linewidth=0.8)
    ax.invert_yaxis()

    for yi, v in zip(y + 0.15, totals):
        ax.text(v * 1.12, yi, f"{v:,}", va="center", fontsize=9.5, color="#6b7178")
    for yi, v in zip(y - 0.15, positives):
        ax.text(v * 1.12, yi, f"{v:,}", va="center", fontsize=9.5,
                color=RED, fontweight="bold")
    # Above the axes: inside, the long 253,317 bar runs under any corner.
    ax.legend(frameon=False, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.text(0.995, -0.42,
            f"{pos:,} / {n_hand_pos} = {pos/n_hand_pos:.0f}x more positive examples"
            + (f", from which {n_sample:,} observations are sampled for training."
               if n_sample else "."),
            transform=ax.transAxes, ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", default=RECORD)
    p.add_argument("--sample", default="files/weather/obs/nwp_sample.csv")
    p.add_argument("--out-dir", default=OUT)
    a = p.parse_args()

    df = pd.read_csv(a.record, usecols=["year", "label", "n_layers"])
    df["is_pos"] = (df["label"] == "undercast").astype(int)
    # n_layers > 0 means a TPS LWR group was present and parseable -- the
    # observer noted a deck below the summit, whatever its coverage or height.
    df["has_tps"] = (df["n_layers"] > 0).astype(int)

    n_sample = None
    if os.path.exists(a.sample):
        n_sample = sum(1 for _ in open(a.sample)) - 1

    os.makedirs(a.out_dir, exist_ok=True)
    rate, mention = drift(df, os.path.join(a.out_dir, "undercast_rate_drift.png"))
    scale(df, os.path.join(a.out_dir, "training_data_scale.png"), n_sample=n_sample)

    print("\nby year (screen rate % / mention rate %):")
    for y in rate.index:
        print(f"  {y}  {rate[y]:5.2f}   {mention[y]:5.2f}")


if __name__ == "__main__":
    main()
