#!/usr/bin/env python3
"""Static figures for the weather write-up PDF (files/weather/report/).

The model details page draws several of its charts in the browser from JSON
(feature importances, skill against lead, the redundancy histogram, the
trimming and depth checks). A PDF cannot run that JavaScript, so this script
redraws them from the same JSON the page reads. Figures the page already
serves as PNGs are referenced directly by the LaTeX source, not copied.

    python3 scripts/build_weather_report_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OBS = "files/weather/models/obs"
CUTS = "files/weather/examples/screen_cuts"
OUT = "files/weather/report/figures"
INK, RED, BLUE, ORANGE, GREY = "#2f3337", "#C44E52", "#4C72B0", "#DD8452", "#9aa0a6"

SOURCES = ["all", "hrrr", "rap", "nam", "gfs", "ecmwf", "nbm"]
SOURCE_LABEL = {"all": "Combined", "hrrr": "HRRR", "rap": "RAP", "nam": "NAM",
                "gfs": "GFS", "ecmwf": "ECMWF", "nbm": "NBM"}
SOURCE_COLOR = {"all": INK, "hrrr": BLUE, "rap": "#55A868", "nam": ORANGE,
                "gfs": "#8172B3", "ecmwf": RED, "nbm": "#937860"}
ALGOS = [("xgboost", "XGBoost"), ("random_forest", "Random Forest"),
         ("gradient_boosting", "Gradient Boosting")]

# Copied from the trim and depth charts in _pages/weather-details.html:
# [budget, held-out ROC-AUC, held-out PR-AUC, webcam ROC-AUC, webcam PR-AUC].
WIDTH_ROWS = [
    [8, 0.813, 0.177, 0.829, 0.163],
    [15, 0.828, 0.204, 0.827, 0.170],
    [25, 0.827, 0.171, 0.846, 0.208],
    [40, 0.847, 0.176, 0.875, 0.259],
    [70, 0.861, 0.210, 0.882, 0.262],
    [120, 0.863, 0.223, 0.887, 0.262],
    [213, 0.861, 0.219, 0.883, 0.271],
]
DEPTH_ROWS = [
    [1, 0.851, 0.179, 0.868, 0.228],
    [2, 0.865, 0.207, 0.881, 0.244],
    [3, 0.861, 0.219, 0.883, 0.271],
    [4, 0.847, 0.203, 0.882, 0.267],
    [6, 0.834, 0.187, 0.839, 0.221],
]
NOISE = 0.005


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.bbox": "tight",
    })


def load(name):
    with open(os.path.join(OBS, name)) as f:
        return json.load(f)


def bars(ax, rows, title, n=15):
    rows = rows[:n][::-1]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=BLUE)
    ax.set_title(title, fontsize=9, color=INK)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("importance")


def features(fi):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    for ax, (key, label) in zip(axes, ALGOS):
        bars(ax, fi["all"][key], label)
    fig.tight_layout()
    fig.savefig(f"{OUT}/features_combined.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 8))
    for ax, src in zip(axes.flat, SOURCES[1:]):
        bars(ax, fi[src]["gradient_boosting"], f"{SOURCE_LABEL[src]} (Gradient Boosting)", n=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/features_by_source.pdf")
    plt.close(fig)


def skill_vs_lead(metas):
    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
               ("precision", "Precision"), ("recall", "Recall")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    for ax, (key, label) in zip(axes.flat, metrics):
        for src in SOURCES:
            by = metas[src]["Gradient Boosting"]["baserate_by_lead"]
            leads = sorted(int(k) for k in by)
            ys = [by[str(h)].get(key) for h in leads]
            ax.plot(leads, ys, marker="o", ms=3, lw=2.2 if src == "all" else 1.2,
                    color=SOURCE_COLOR[src], label=SOURCE_LABEL[src])
        ax.set_title(label, fontsize=9)
        ax.set_xticks([1, 24, 48, 72, 96, 120, 144])
    for ax in axes[1]:
        ax.set_xlabel("forecast lead (h)")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/skill_vs_lead.pdf")
    plt.close(fig)


def redundancy(h):
    edges = h["edges"]
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    w = h["bin_width"]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.bar(mids, h["same"]["share"], width=w * 0.9, color=BLUE,
           label=f"same quantity, different source (n={h['same']['n']})")
    ax.step(edges, [h["baseline"]["share"][0]] + h["baseline"]["share"], where="pre",
            color=INK, ls="--", lw=1,
            label=f"random pairs of different quantities (n={h['baseline']['n']:,})")
    ax.set_xlabel("|Spearman r|")
    ax.set_ylabel("share of pairs")
    ax.legend(fontsize=7, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(f"{OUT}/redundancy_hist.pdf")
    plt.close(fig)


def deltas():
    series = [(1, "Held-out year, ROC-AUC", BLUE, "-"),
              (2, "Held-out year, PR-AUC", ORANGE, "-"),
              (3, "Webcam days, ROC-AUC", BLUE, "--"),
              (4, "Webcam days, PR-AUC", ORANGE, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, rows, base, xlabel, logx in [
            (axes[0], WIDTH_ROWS, 6, "features kept (top k by importance)", True),
            (axes[1], DEPTH_ROWS, 2, "maximum tree depth", False)]:
        full = rows[base]
        xs = [r[0] for r in rows]
        ax.axhspan(-NOISE, NOISE, color=GREY, alpha=0.25, lw=0)
        ax.axhline(0, color=GREY, lw=0.8)
        for i, label, color, ls in series:
            ax.plot(xs, [r[i] - full[i] for r in rows], ls=ls, color=color,
                    marker="o", ms=3, label=label)
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            ax.minorticks_off()
        else:
            ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("change vs. current model")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/trim_depth.pdf")
    plt.close(fig)


def webcam_pairs():
    days = [("2024-09-28", "passes every cut; hand-labelled undercast"),
            ("2024-09-21", "rejected only by the coverage cut (SCT); hand-labelled undercast")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.6))
    for row, (date, caption) in zip(axes, days):
        for ax, cam in zip(row, ["tower", "observatory"]):
            ax.imshow(Image.open(f"{CUTS}/{date}_{cam}.jpg"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{date}, {cam} camera", fontsize=8)
        row[0].set_ylabel(caption.split(";")[0], fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/webcam_examples.pdf", dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    style()
    features(load("feature_importances.json"))
    skill_vs_lead({s: load(f"model_metadata_{s}.json") for s in SOURCES})
    redundancy(load("redundancy_histogram.json"))
    deltas()
    webcam_pairs()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
