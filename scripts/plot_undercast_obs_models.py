#!/usr/bin/env python3
"""Performance figures for /weather/details/, from the per-observation pipeline.

Two choices about what gets plotted, both because the obvious alternative is
misleading:

  SCORED ON THE BASE-RATE HOLDOUT, NOT OUT-OF-FOLD.  Negatives were subsampled
  5:1 for training, so an out-of-fold confusion matrix is drawn against a ~17%
  positive rate. Reality is 2.7%. Precision computed there is inflated roughly
  sixfold and means nothing. The holdout year is unsampled, so its confusion
  matrices are the ones a reader can actually interpret.

  FEATURE NAMES INCLUDE THE MISSINGNESS INDICATORS.  Every numeric column is
  paired with an indicator, so the model has two features per column and the
  importance vector is twice as long as the column list. Assuming otherwise
  silently mislabels the second half.

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
from train_undercast_obs import (MODEL_NAMES, SHORT_NAME, SOURCES,  # noqa: E402
                                 SOURCE_MAX_LEAD_H)

OUT = "files/weather/examples/model_training_images"
INK, GREY = "#2f3337", "#9aa0a6"
COLORS = {"XGBoost": "#4C72B0", "Random Forest": "#55A868",
          "Gradient Boosting": "#C44E52"}
CLASS_LABELS = ["not undercast", "undercast"]
TITLE = {s: ("combined" if s == "all" else s.upper()) for s in SOURCES}
SPLIT = "holdout_baserate"
# The three algorithms plus their consensus. Kept as one ordered tuple so the bar
# layout, the palette and the legend cannot fall out of step with each other.
# "Mean of 3", not "Consensus": the confusion matrices on the same page show a
# 2-of-3 MAJORITY vote, and one page should not use one word for two different
# objects. This is the average of the three probabilities -- see the comment in
# per_source_performance for why an area axis forces that choice.
CONSENSUS = "Mean of 3"
SERIES = tuple(MODEL_NAMES) + (CONSENSUS,)


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

    Scored on the rows where EVERY source has data, rather than on each source's
    own holdout. HRRR's archive starts in 2014 and ECMWF's in 2022, so per-source
    row sets would have the bars comparing different years' weather as much as
    different models.

    TWO restrictions are needed for that, not one. Sharing the same DATES is the
    old one. The second came in with the lead ladder: `split_rows(df, "all", ...)`
    is lead-aware, so it keeps 144 h rows where only ECMWF and NBM are required --
    and HRRR, RAP and NAM have no data there at all. Scoring them on those rows
    measures the preprocessor's fill value, not the model: HRRR reads 0.630 that
    way against 0.872 on the leads it actually reaches. So the comparison is cut
    to leads every PLOTTED source can reach, which for all six is 48 h.

    Per-source numbers therefore differ from the results table, which quotes each
    source at each lead separately and carries the long leads the table above
    cannot compare fairly.
    """
    common = ue.split_rows(df, "all", SPLIT)
    # min over the plotted sources, so plotting a subset widens the window
    # instead of silently keeping the six-source one.
    # "all" is not in SOURCE_MAX_LEAD_H and is not the constraint anyway: the
    # combined model reaches 144 h, the short-range members do not.
    max_lead = min(SOURCE_MAX_LEAD_H[s] for s in sources if s in SOURCE_MAX_LEAD_H)
    keep = common["target_lead_h"].astype(float) <= max_lead
    common = common[keep]
    y = ue.truth(common, SPLIT)
    # Every algorithm, not the best of them. A "best of three" bar hides the thing
    # worth seeing: the three are within a hair of each other everywhere, which is
    # why the page stopped serving a vote between them.
    #
    # The fourth bar is the consensus of those three, and it is the MEAN of their
    # probabilities rather than the 2-of-3 majority the page used to serve. That is
    # forced by the axis, not a preference: a majority vote is a single operating
    # point, so it has one precision and one recall and no curve behind it. Both
    # metrics here are threshold-free areas under a ranking, and the ranking version
    # of "what do the three agree on" is their average score. Scored this way the
    # consensus is directly comparable to its own members, which a vote is not.
    # compare_undercast_ensembles.py scores the same mean alongside the 1/2/3-of-3
    # votes, so the two views stay consistent.
    scores = {}
    for s in sources:
        pre, models, _ = ue.load_artifacts(s, models_dir)
        proba = ue.probabilities(common, s, pre, models)
        proba[CONSENSUS] = np.mean([proba[n] for n in MODEL_NAMES], axis=0)
        scores[s] = {n: (roc_auc_score(y, proba[n]),
                         average_precision_score(y, proba[n])) for n in SERIES}
    order = sorted(sources, key=lambda s: -max(scores[s][n][0] for n in SERIES))
    labels = [TITLE[s] for s in order]
    base_rate = float(np.mean(y))
    x = np.arange(len(order))
    width = 0.82 / len(SERIES)
    palette = {"XGBoost": "#4C72B0", "Random Forest": "#DD8452",
               "Gradient Boosting": "#55A868", CONSENSUS: "#8172B3"}

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=160)
    # Explicit margins rather than tight_layout plus a tight bbox: the title sits
    # hard left and the legend above, and letting the bbox grow to contain both
    # padded the canvas instead of the panels.
    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.21, wspace=0.17)
    handles = []
    for i, n in enumerate(SERIES):
        off = (i - (len(SERIES) - 1) / 2) * width
        # The consensus is the same three models averaged, not a fourth
        # independent one, so it is drawn as an outlined bar rather than another
        # solid colour in the row -- a reader should not read it as a peer.
        kw = dict(color=palette[n], edgecolor=palette[n])
        if n == CONSENSUS:
            kw = dict(color="white", edgecolor=palette[n], hatch="///", linewidth=1.1)
        b = ax[0].bar(x + off, [scores[s][n][0] for s in order], width, **kw)
        ax[1].bar(x + off, [scores[s][n][1] for s in order], width, **kw)
        handles.append(b)
    ax[0].axhline(0.5, ls="--", c=GREY, lw=1)
    ax[0].set_ylim(0.5, 1.0)
    ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title("Ranking ability", fontsize=12, loc="left", color=INK, pad=8)
    # Annotated rather than put in the legend: with four series the legend had to
    # move out of the axes, and these two lines mean different things on each panel.
    ax[0].text(len(order) - 0.45, 0.502, "no skill", fontsize=8, color=GREY,
               ha="left", va="bottom")
    ax[1].axhline(base_rate, ls="--", c=GREY, lw=1)
    ax[1].set_ylim(0, max(scores[s][n][1] for s in order
                          for n in SERIES) * 1.22)
    ax[1].set_ylabel("PR-AUC")
    ax[1].set_title("Precision–recall area", fontsize=12, loc="left", color=INK, pad=8)
    ax[1].text(len(order) - 0.45, base_rate * 1.12, "base rate", fontsize=8,
               color=GREY, ha="left", va="bottom")
    # One legend for both panels, above them. In the three-bar version it sat
    # inside the axes and covered the right-hand source's bars.
    fig.legend(handles, list(SERIES), loc="upper center", ncol=len(SERIES),
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.52, 0.935))
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=30, ha="right")
        # Right margin so the reference-line labels have somewhere to sit that is
        # not on top of the last source's bars.
        a.set_xlim(-0.62, len(order) - 1 + 1.3)
        a.grid(axis="y", color="#e6e8eb", lw=0.8)
        a.set_axisbelow(True)
        despine(a)
    fig.suptitle("Per-source undercast skill on the held-out year, by algorithm",
                 fontsize=12.5, color=INK, x=0.005, ha="left", y=0.985)
    fig.text(0.5, 0.055,
             f"Unsampled holdout, {len(y):,} hours every source covers (leads up to "
             f"{max_lead} h), {int(y.sum())} of them undercast "
             f"({100*base_rate:.1f}%; base rate = {base_rate:.3f}). "
             "ROC-AUC starts at 0.5 because that is where skill starts.",
             ha="center", fontsize=8.5, color=GREY)
    fig.text(0.5, 0.012,
             "\u201cMean of 3\u201d is the average of the three probabilities. It is NOT the "
             "2-of-3 majority vote shown in the confusion matrices: a majority is a single "
             "operating point, so it has no area under anything.",
             ha="center", fontsize=8.5, color=GREY)
    p = os.path.join(out_dir, "per_source_performance.png")
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print(f"  {p}")


def roc_pr_curves(df, source, out_dir, models_dir):
    """One figure per algorithm, so the page can switch between them.

    These used to be three curves on one pair of axes. At this much overlap that
    is a thicket rather than a comparison -- the three algorithms are within 0.03
    AUC of each other, so the lines sit on top of one another and the only legible
    information is the legend. Separate panels let each curve be read on its own,
    and the numbers in the per-source bars above carry the comparison.
    """
    for algo in MODEL_NAMES:
        _roc_pr_one(df, source, algo, out_dir, models_dir)


def _roc_pr_one(df, source, algo, out_dir, models_dir):
    r = ue.score_split(df, source, SPLIT, models_dir)
    y = r["y"]
    base_rate = float(np.mean(y))
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=160)
    for n in [algo]:
        p = r["proba"][n]
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=COLORS[n], lw=2.0,
                   label=f"{SHORT_NAME[n]} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax[1].plot(rec, prec, color=COLORS[n], lw=2.0,
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
    fig.suptitle(f"Undercast discrimination on the held-out year — {TITLE[source]} "
                 f"model, {algo}", fontsize=12.5, color=INK, x=0.005, ha="left")
    fig.text(0.995, -0.06,
             f"The precision axis is the honest one: at a {100*base_rate:.1f}% base rate, "
             "high recall costs precision quickly.", ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    slug = algo.lower().replace(" ", "_")
    p = os.path.join(out_dir, f"roc_pr_curves_{source}_{slug}.png")
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
    preds["2 of 3 (the vote)"] = (votes >= 2).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=160)
    for ax, name in zip(axes, list(MODEL_NAMES) + ["2 of 3 (the vote)"]):
        p = precision_score(y, preds[name], zero_division=0)
        rc = recall_score(y, preds[name])
        head = name if name.startswith("2 of") else f"{SHORT_NAME[name]}  (thr {thr[name]:.2f})"
        draw_cm(ax, y, preds[name], f"{head}\nP {p:.2f}  ·  R {rc:.2f}")
    fig.suptitle(f"{TITLE[source]} model on the held-out year, at F1-optimal thresholds",
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

    Split by lead because the page publishes a call for every hour out to 144, and
    a single pooled matrix would hide the thing a reader most needs to know: the
    same model is far more trustworthy about tonight than about this time next
    week. Thresholds are the per-lead ones the live page uses, not a global cut,
    so these matrices are what the site does.

    Every lead the model was trained at gets a panel, read off the metadata rather
    than hard-coded -- when the ladder grew from three leads to seven this figure
    was the one place still quietly showing three, which made the page look like
    it published two days ahead when it publishes six.

    The last panel is the honest one -- hand-labeled days, scored against a person
    looking at a photograph, using a threshold those days had no part in choosing.
    """
    base = ue.score_split(df, HEADLINE_SOURCE, "holdout_baserate", models_dir)
    web = ue.score_split(df, HEADLINE_SOURCE, "holdout_webcam", models_dir)
    meta = base["meta"][HEADLINE_ALGO]
    by_lead = meta["threshold_by_lead"]

    panels = []
    for lead in sorted(int(k) for k in by_lead):
        m = base["rows"]["target_lead_h"].to_numpy() == lead
        thr = float(by_lead[str(lead)])
        panels.append((f"{lead} h ahead", base["y"][m],
                       (base["proba"][HEADLINE_ALGO][m] >= thr).astype(int), thr))
    wl = web["rows"]["target_lead_h"].to_numpy()
    wthr = np.array([float(by_lead[str(int(l))]) for l in wl])
    panels.append(("hand-labeled webcam days", web["y"],
                   (web["proba"][HEADLINE_ALGO] >= wthr).astype(int), None))

    # One row of eight is unreadable; wrap into a grid instead.
    ncol = 4 if len(panels) <= 4 else (len(panels) + 1) // 2
    nrow = 1 if len(panels) <= 4 else 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.65 * ncol, 4.3 * nrow), dpi=160)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax, (title, y, pred, thr) in zip(axes, panels):
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred)
        head = f"{title}" + (f"   (cut {thr:.2f})" if thr is not None else "")
        draw_cm(ax, y, pred, f"{head}\nP {p:.2f}  ·  R {r:.2f}")
    fig.suptitle("The forecast the site publishes — combined model, Gradient Boosting",
                 fontsize=12.5, color=INK, x=0.005, ha="left")
    fig.text(0.995, -0.04,
             "All but the last panel: the held-out year at its true 2.5% base "
             "rate, at the per-lead thresholds the live page uses. Last panel: "
             "every hand-labeled day, scored against human webcam labels.",
             ha="right", fontsize=8.5, color=GREY)
    fig.tight_layout()
    out = os.path.join(out_dir, "headline_model_confusion.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out}")


def top_features(source, out_dir, models_dir, n=15):
    """Importances for every algorithm, not just XGBoost.

    All three expose feature_importances_, and they do not agree -- a tree
    ensemble's importance is a statement about which splits that particular fit
    chose, not about the atmosphere. Showing only one invited reading it as the
    latter. The deployed model is Gradient Boosting, so its panel is the one that
    describes what the site actually serves.
    """
    pre, models, _ = ue.load_artifacts(source, models_dir)
    names = tidy(pre.get_feature_names_out())
    palette = {"XGBoost": "#4C72B0", "Random Forest": "#DD8452",
               "Gradient Boosting": "#55A868"}
    for algo in MODEL_NAMES:
        imp = np.asarray(models[algo].feature_importances_)
        assert len(names) == len(imp), \
            f"{source}/{algo}: {len(names)} names vs {len(imp)} importances"
        order = np.argsort(imp)[::-1][:min(n, len(imp))][::-1]
        fig, ax = plt.subplots(figsize=(8.2, 6), dpi=160)
        ax.barh([names[i] for i in order], [imp[i] for i in order],
                color=palette[algo])
        ax.set_xlabel(f"{algo} feature importance (gain share)", fontsize=10)
        ax.set_title(f"Top {len(order)} features — {TITLE[source]} model, {algo}",
                     fontsize=12, loc="left", color=INK, pad=8)
        ax.grid(axis="x", color="#e6e8eb", lw=0.8)
        ax.set_axisbelow(True)
        despine(ax)
        fig.tight_layout()
        slug = algo.lower().replace(" ", "_")
        p = os.path.join(out_dir, f"top_features_{source}_{slug}.png")
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
