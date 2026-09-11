#!/usr/bin/env python3
"""What the model fields actually look like on undercast days, and what they don't.

The historical-conditions page used to argue, from a handful of hand-picked
dates, that the models simply do not predict inversions during undercasts. With
29 years of observations joined to the forecasts that is now checkable, and the
answer is more interesting than the original claim: the inversion signal is
real and it is the strongest single thing in the dataset, but the distributions
overlap so heavily that no threshold separates them.

Four panels, each a field the rule-based attempt leaned on, plotted as
undercast against not-undercast with the fitted rule threshold marked:

    inversion strength   T(850 mb) - T(925 mb)
    cloud ceiling        where the model puts the lowest solid layer
    surface visibility   how far the model thinks you can see
    boundary layer depth how deep the mixed layer is

    python3 scripts/plot_inversion_evidence.py
    python3 scripts/plot_inversion_evidence.py --cache /tmp/obs.pkl
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_undercast_obs import TARGET, assign_splits, load_obs_data  # noqa: E402

OUT = "files/weather/examples/model_training_images"
INK, RED, BLUE, GREY = "#2f3337", "#C44E52", "#4C72B0", "#9aa0a6"
SUMMIT_M = 1917.2

PANELS = [
    # A DIFFERENCE in kelvin equals a difference in degrees Celsius, so the
    # familiar unit is used -- "K" reads as an absolute temperature to most
    # people and makes a -3 look nonsensical.
    ("dT_925_850_hrrr", "Inversion strength  T(850 mb) − T(925 mb)",
     "temperature difference (°C)", (-12, 8), -2.5, "warmer aloft →  inversion"),
    ("cloud_ceiling_m_hrrr", "Modelled cloud ceiling", "height above sea level (m)",
     (0, 8000), SUMMIT_M, "summit height"),
    ("vis_surface_hrrr", "Modelled surface visibility", "visibility (m)",
     (0, 40000), 10000, "rule threshold"),
    ("hpbl_surface_hrrr", "Boundary layer depth", "depth (m)",
     (0, 2500), None, None),
]


def panel(ax, d, y, col, title, unit, xlim, thr, thr_label):
    v = pd.to_numeric(d[col], errors="coerce")
    ok = v.notna()
    a, b = v[ok & y], v[ok & ~y]
    bins = np.linspace(xlim[0], xlim[1], 46)
    # Density, not counts: negatives outnumber positives ~5:1 even after
    # subsampling, so raw counts would show only the shape of the negatives.
    ax.hist(b, bins=bins, density=True, color=BLUE, alpha=.45, label="not undercast")
    ax.hist(a, bins=bins, density=True, color=RED, alpha=.55, label="undercast")
    if thr is not None:
        ax.axvline(thr, color=INK, ls="--", lw=1.2)
        ax.text(thr, ax.get_ylim()[1] * .97, "  " + thr_label, fontsize=8,
                color=INK, va="top")
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=8)
    ax.set_xlabel(unit, fontsize=9.5, color=INK)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(*xlim)
    # How much of the two distributions actually sit apart.
    miss = float(np.mean(a > thr)) if thr is not None else None
    return len(a), len(b), miss


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", default="files/weather/csv/obs")
    p.add_argument("--labels", default="files/weather/csv/MtWashington_undercast_orig.csv")
    p.add_argument("--record", default="files/weather/obs/undercast_record.csv")
    p.add_argument("--cache")
    p.add_argument("--out-dir", default=OUT)
    a = p.parse_args()

    if a.cache and os.path.exists(a.cache):
        df = pd.read_pickle(a.cache)
    else:
        df = load_obs_data(a.csv_dir, a.labels, a.record)
        df = assign_splits(df)
        if a.cache:
            df.to_pickle(a.cache)

    # Near-analysis only: at 1 h the model is describing the atmosphere that
    # exists, so this is about whether the fields CAN see an undercast, not
    # about forecast error.
    d = df[(df["target_lead_h"] == 1) & df["lead_hrrr"].notna()
           & ~df["screen_ambiguous"]]
    y = d[TARGET].astype(bool).to_numpy()
    print(f"{len(d):,} HRRR observations at ~1 h lead, {int(y.sum()):,} undercast")

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.6), dpi=160)
    for ax, (col, title, unit, xlim, thr, lab) in zip(axes.ravel(), PANELS):
        if col not in d.columns:
            ax.set_visible(False)
            continue
        na, nb, miss = panel(ax, d, y, col, title, unit, xlim, thr, lab)
        note = f"  ({miss*100:.0f}% of undercasts on the far side)" if miss is not None else ""
        print(f"  {col:<24} undercast n={na:,} other n={nb:,}{note}")
    axes.ravel()[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle("HRRR fields on undercast hours against everything else",
                 fontsize=13, color=INK, x=0.005, ha="left", y=1.0)
    fig.text(0.995, -0.02,
             "Near-analysis (~1 h lead), 2014–2026. Densities, so the two groups "
             "are comparable despite very different counts.",
             ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "inversion_evidence.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
