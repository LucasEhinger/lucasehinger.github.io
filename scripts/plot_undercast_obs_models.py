#!/usr/bin/env python3
"""Performance figures for /weather/details/, from the per-observation pipeline.

Replaces ``plot_undercast_models.py``, which drew its figures from the 589-day
first pass: hand-labeled dates, forecast fields sampled at a fixed hour, 24
positives, and out-of-fold predictions on a subsampled training set.

Two things changed about what gets plotted, both because the old choice was
misleading rather than merely dated:

  SCORED ON THE BASE-RATE HOLDOUT, NOT OUT-OF-FOLD.  Negatives were subsampled
  5:1 for training, so an out-of-fold confusion matrix is drawn against a ~17%
  positive rate. Reality is 2.7%. Precision computed there is inflated roughly
  sixfold and means nothing. The holdout year is unsampled, so its confusion
  matrices are the ones a reader can actually interpret.

  FEATURE NAMES INCLUDE THE MISSINGNESS INDICATORS.  Every numeric column is
  paired with an indicator, so the model has two features per column and the
  importance vector is twice as long as the column list. The old script assumed
  otherwise and would have silently mislabeled the second half.

Nothing is refit: the artifacts written by train_undercast_obs.py are loaded and
applied, so these figures show the deployed models, not near-identical refits.

    python3 scripts/plot_undercast_obs_models.py
    python3 scripts/plot_undercast_obs_models.py --sources all hrrr
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import undercast_eval as ue  # noqa: E402
from train_undercast_obs import MODEL_NAMES, SHORT_NAME, SOURCES  # noqa: E402

OUT = "files/weather/examples/model_training_images"
INK, GREY = "#2f3337", "#9aa0a6"
COLORS = {"XGBoost": "#4C72B0", "Random Forest": "#55A868",
          "Gradient Boosting": "#C44E52"}
CLASS_LABELS = ["not undercast", "undercast"]
TITLE = {s: ("combined" if s == "all" else s.upper()) for s in SOURCES}
SPLIT = "holdout_baserate"


def tidy(names):
    """Strip ColumnTransformer prefixes and name the indicator columns."""
    out = []
    for n in names:
        n = n.split("__", 1)[-1]
        if n.startswith("missingindicator_"):
            n = n[len("missingindicator_"):] + "  (not reported)"
        out.append(n)
    return out


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        if s not in keep:
            ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------
def per_source_performance(df, sources, out_dir, models_dir):
    """Every source on the same rows, on the holdout where precision means something.

    Scored on the rows where EVERY source has data -- which is exactly the row
    set the combined model is defined on -- rather than on each source's own
    holdout. HRRR's archive starts in 2014 and ECMWF's in 2022, so per-source
    row sets would have the bars comparing different years' weather as much as
    different models. It moves things: restricted to shared rows HRRR climbs
    from 0.868 to 0.875 and RAP falls from 0.857 to 0.853. ECMWF is the source
    that starts latest, so the shared set IS its own set and its bar does not
    move -- everyone else is being brought onto ECMWF's years.

    Per-source numbers therefore differ slightly from the results table, which
    quotes each source on its own full holdout.
    """
    common = ue.split_rows(df, "all", SPLIT)
    y = ue.truth(common, SPLIT)
    rows = []
    for s in sources:
        pre, models, _ = ue.load_artifacts(s, models_dir)
        proba = ue.probabilities(common, s, pre, models)
        best = max(MODEL_NAMES, key=lambda n: roc_auc_score(y, proba[n]))
        rows.append((s, roc_auc_score(y, proba[best]),
                     max(average_precision_score(y, proba[n]) for n in MODEL_NAMES),
                     best))
    rows.sort(key=lambda t: -t[1])
    labels = [TITLE[s] for s, *_ in rows]
    base_rate = float(np.mean(y))
    x = np.arange(len(rows))

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=160)
    ax[0].bar(x, [r[1] for r in rows], color="#4C72B0")
    ax[0].axhline(0.5, ls="--", c=GREY, lw=1, label="no skill = 0.5")
    ax[0].set_ylim(0.5, 1.0)
    ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title("Ranking ability", fontsize=12, loc="left", color=INK, pad=8)
    ax[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    ax[1].bar(x, [r[2] for r in rows], color="#55A868")
    ax[1].axhline(base_rate, ls="--", c=GREY, lw=1,
                  label=f"base rate = {base_rate:.3f}")
    ax[1].set_ylim(0, max(r[2] for r in rows) * 1.35)
    ax[1].set_ylabel("PR-AUC")
    ax[1].set_title("Precision–recall area", fontsize=12, loc="left", color=INK, pad=8)
    ax[1].legend(loc="upper right", fontsize=8.5, frameon=False)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=30, ha="right")
        a.grid(axis="y", color="#e6e8eb", lw=0.8)
        a.set_axisbelow(True)
        despine(a)
    fig.suptitle("Per-source undercast skill on the held-out year (best of three algorithms)",
                 fontsize=12.5, color=INK, x=0.005, ha="left")
    fig.text(0.995, -0.06,
             f"Unsampled holdout, {len(y):,} hours every source covers, "
             f"{int(y.sum())} of them undercast ({100*base_rate:.1f}%). "
             "ROC-AUC starts at 0.5 because that is where skill starts.",
             ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    p = os.path.join(out_dir, "per_source_performance.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {p}")


def roc_pr_curves(df, source, out_dir, models_dir):
    r = ue.score_split(df, source, SPLIT, models_dir)
    y = r["y"]
    base_rate = float(np.mean(y))
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=160)
    for n in MODEL_NAMES:
        p = r["proba"][n]
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=COLORS[n], lw=1.8,
                   label=f"{SHORT_NAME[n]} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax[1].plot(rec, prec, color=COLORS[n], lw=1.8,
                   label=f"{SHORT_NAME[n]} (AP {average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", c=GREY, lw=1)
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC", fontsize=12, loc="left", color=INK, pad=8)
    ax[0].legend(loc="lower right", fontsize=9, frameon=False)
    ax[1].axhline(base_rate, ls="--", c=GREY, lw=1,
                  label=f"base rate = {base_rate:.3f}")
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Precision–recall", fontsize=12, loc="left", color=INK, pad=8)
    ax[1].legend(loc="upper right", fontsize=9, frameon=False)
    for a in ax:
        a.grid(color="#e6e8eb", lw=0.8)
        a.set_axisbelow(True)
        despine(a)
    fig.suptitle(f"Undercast discrimination on the held-out year — {TITLE[source]} model",
                 fontsize=12.5, color=INK, x=0.005, ha="left")
    fig.text(0.995, -0.06,
             f"The precision axis is the honest one: at a {100*base_rate:.1f}% base rate, "
             "high recall costs precision quickly.", ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    p = os.path.join(out_dir, "roc_pr_curves.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {p}")


def draw_cm(ax, y, pred, title):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    # Row-normalised colour: the negative row is ~36x the positive one, so raw
    # counts would paint one cell dark and the other three white regardless of
    # how the model did.
    shade = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11,
                color="white" if shade[i, j] > 0.5 else INK)
    ax.set_xticks([0, 1], CLASS_LABELS, fontsize=8.5)
    ax.set_yticks([0, 1], CLASS_LABELS, rotation=90, va="center", fontsize=8.5)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("actual", fontsize=9)
    ax.set_title(title, fontsize=9.5, color=INK)


def confusion_panels(df, source, out_dir, models_dir):
    r = ue.score_split(df, source, SPLIT, models_dir)
    y, thr = r["y"], r["thresholds"]
    preds = {n: (r["proba"][n] >= thr[n]).astype(int) for n in MODEL_NAMES}
    votes = np.sum([preds[n] for n in MODEL_NAMES], axis=0)
    preds["2 of 3 (deployed)"] = (votes >= 2).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=160)
    for ax, name in zip(axes, list(MODEL_NAMES) + ["2 of 3 (deployed)"]):
        p = precision_score(y, preds[name], zero_division=0)
        rc = recall_score(y, preds[name])
        head = name if name.startswith("2 of") else f"{SHORT_NAME[name]}  (thr {thr[name]:.2f})"
        draw_cm(ax, y, preds[name], f"{head}\nP {p:.2f}  ·  R {rc:.2f}")
    fig.suptitle(f"{TITLE[source]} model on the held-out year, at deployed thresholds",
                 fontsize=12.5, color=INK, x=0.005, ha="left")
    fig.text(0.995, -0.04,
             f"{len(y):,} observations, {int(y.sum())} undercast ({100*y.mean():.1f}%). "
             "Cells are counts; shading is share of the row.",
             ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    p = os.path.join(out_dir, f"confusion_matrices_{source}.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {p}")


# The model the site actually serves. Kept in step with weather_to_json.py.
HEADLINE_SOURCE, HEADLINE_ALGO = "all", "Gradient Boosting"


def headline_confusion(df, out_dir, models_dir):
    """The served model, at the thresholds it is served with, by forecast lead.

    Split by lead because the page publishes a call for every hour out to 48, and
    a single pooled matrix would hide the thing a reader most needs to know: the
    same model is far more trustworthy about tonight than about the day after
    tomorrow. Thresholds are the per-lead ones the live page uses, not a global
    cut, so these matrices are what the site does.

    The last panel is the honest one -- hand-labeled days, scored against a person
    looking at a photograph, using a threshold those days had no part in choosing.
    """
    base = ue.score_split(df, HEADLINE_SOURCE, "holdout_baserate", models_dir)
    web = ue.score_split(df, HEADLINE_SOURCE, "holdout_webcam", models_dir)
    meta = base["meta"][HEADLINE_ALGO]
    by_lead = meta["threshold_by_lead"]

    panels = []
    for lead in (1, 24, 48):
        m = base["rows"]["target_lead_h"].to_numpy() == lead
        thr = float(by_lead[str(lead)])
        panels.append((f"{lead} h ahead", base["y"][m],
                       (base["proba"][HEADLINE_ALGO][m] >= thr).astype(int), thr))
    wl = web["rows"]["target_lead_h"].to_numpy()
    wthr = np.array([float(by_lead[str(int(l))]) for l in wl])
    panels.append(("hand-labeled webcam days", web["y"],
                   (web["proba"][HEADLINE_ALGO] >= wthr).astype(int), None))

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 4.3), dpi=160)
    for ax, (title, y, pred, thr) in zip(axes, panels):
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred)
        head = f"{title}" + (f"   (cut {thr:.2f})" if thr is not None else "")
        draw_cm(ax, y, pred, f"{head}\nP {p:.2f}  ·  R {r:.2f}")
    fig.suptitle("The forecast the site publishes — combined model, Gradient Boosting",
                 fontsize=12.5, color=INK, x=0.005, ha="left")
    fig.text(0.995, -0.04,
             "First three panels: the held-out year at its true 2.4% base rate, "
             "at the per-lead thresholds the live page uses. Last panel: every "
             "hand-labeled day, scored against human webcam labels.",
             ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    out = os.path.join(out_dir, "headline_model_confusion.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out}")


def top_features(source, out_dir, models_dir, n=15):
    pre, models, _ = ue.load_artifacts(source, models_dir)
    names = tidy(pre.get_feature_names_out())
    imp = np.asarray(models["XGBoost"].feature_importances_)
    assert len(names) == len(imp), f"{len(names)} names vs {len(imp)} importances"
    order = np.argsort(imp)[::-1][:min(n, len(imp))][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6), dpi=160)
    ax.barh([names[i] for i in order], [imp[i] for i in order], color="#4C72B0")
    ax.set_xlabel("XGBoost feature importance (gain share)", fontsize=10)
    ax.set_title(f"Top {len(order)} features — {TITLE[source]} model",
                 fontsize=12, loc="left", color=INK, pad=8)
    ax.grid(axis="x", color="#e6e8eb", lw=0.8)
    ax.set_axisbelow(True)
    despine(ax)
    fig.tight_layout()
    p = os.path.join(out_dir, f"top_features_{source}.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--models-dir", default=ue.MODELS_DIR)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--cache", help="pickle of the assembled frame")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = ue.load_frame(cache=args.cache)
    print("figures:")
    headline_confusion(df, args.out_dir, args.models_dir)
    per_source_performance(df, args.sources, args.out_dir, args.models_dir)
    for s in args.sources:
        confusion_panels(df, s, args.out_dir, args.models_dir)
        top_features(s, args.out_dir, args.models_dir)
    if "all" in args.sources:
        roc_pr_curves(df, "all", args.out_dir, args.models_dir)


if __name__ == "__main__":
    main()
