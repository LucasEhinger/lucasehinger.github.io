"""Regenerate the model-performance figures shown on /weather/details/.

Reuses train_undercast_models.py for data loading, feature selection, and
grouped out-of-fold CV predictions, then renders honest performance plots into
files/weather/examples/model_training_images/. Run *after*
train_undercast_models.py so the deployed thresholds/metadata exist.

The per-source chart reads the cv_roc_auc / cv_pr_auc already stored in each
model_metadata_*.json (no recompute). The ROC/PR-curve, confusion-matrix and
top-feature figures need probability arrays / fitted models, so out-of-fold
predictions are recomputed per source (the details page lets a reader switch
between them).

    python3 scripts/plot_undercast_models.py            # every source
    python3 scripts/plot_undercast_models.py --sources all hrrr
"""

import argparse
import importlib.util
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "tr", os.path.join(_here, "train_undercast_models.py")
)
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)

OUT = "files/weather/examples/model_training_images"
MODELS = list(tr.MODEL_NAMES)
SHORT = tr.SHORT_NAME
COLORS = {"XGBoost": "#4C72B0", "Random Forest": "#55A868", "Gradient Boosting": "#C44E52"}
# The negative class is "everything that isn't undercast" -- cloudy/overcast days
# live there too, so don't call it "clear".
CLASS_LABELS = ["not undercast", "undercast"]
SOURCE_TITLE = {s: ("all-parameters" if s == "all" else s.upper()) for s in tr.SOURCES}


def per_source_performance(base_rate):
    """Best-per-source ROC-AUC and PR-AUC from the saved metadata."""
    roc, pr, labels = [], [], []
    for s in tr.SOURCES:
        m = json.load(open(f"files/weather/models/model_metadata_{s}.json"))
        roc.append(max(m[k]["cv_roc_auc"] for k in MODELS))
        pr.append(max(m[k]["cv_pr_auc"] for k in MODELS))
        labels.append(s.upper())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(x, roc, color="#4C72B0")
    ax[0].axhline(0.5, ls="--", c="gray", lw=1, label="random = 0.5")
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title("Ranking ability (ROC-AUC)")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].bar(x, pr, color="#55A868")
    ax[1].axhline(base_rate, ls="--", c="gray", lw=1, label=f"base rate = {base_rate:.3f}")
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("PR-AUC")
    ax[1].set_title("Precision–recall (PR-AUC)")
    ax[1].legend(loc="upper right", fontsize=8)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=45, ha="right")
    fig.suptitle("Per-source undercast discrimination (grouped out-of-fold CV, best of 3 models)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/per_source_performance.png", dpi=130)
    plt.close(fig)


def roc_pr_curves(y, oof, base_rate):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in MODELS:
        fpr, tpr, _ = roc_curve(y, oof[m])
        ax[0].plot(fpr, tpr, color=COLORS[m], label=f"{SHORT[m]} (AUC {roc_auc_score(y, oof[m]):.2f})")
        prec, rec, _ = precision_recall_curve(y, oof[m])
        ax[1].plot(rec, prec, color=COLORS[m], label=f"{SHORT[m]} (AP {average_precision_score(y, oof[m]):.2f})")
    ax[0].plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC — all-parameters model")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].axhline(base_rate, ls="--", c="gray", lw=1, label=f"base rate = {base_rate:.3f}")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision–recall — all-parameters model")
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Undercast discrimination, out-of-fold (all-parameters model)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/roc_pr_curves.png", dpi=130)
    plt.close(fig)


def draw_cm(axc, y, pred, title):
    """One confusion-matrix panel, counts annotated, undercast as the bottom row."""
    cm = confusion_matrix(y, pred, labels=[0, 1])
    axc.imshow(cm, cmap="Blues")
    vmax = cm.max()
    for (i, j), v in np.ndenumerate(cm):
        axc.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > vmax / 2 else "black", fontsize=11)
    axc.set_xticks([0, 1])
    axc.set_xticklabels(CLASS_LABELS)
    axc.set_yticks([0, 1])
    axc.set_yticklabels(CLASS_LABELS, rotation=90, va="center")
    axc.set_xlabel("predicted")
    axc.set_ylabel("actual")
    axc.set_title(title, fontsize=10)


def confusion_panels(y, oof, source):
    md = json.load(open(f"files/weather/models/model_metadata_{source}.json"))
    thr = {m: md[m]["threshold_best_f1"] for m in MODELS}
    preds = {m: (oof[m] >= thr[m]).astype(int) for m in MODELS}
    votes = np.sum([preds[m] for m in MODELS], axis=0)
    preds["Consensus (2 of 3)"] = (votes >= 2).astype(int)
    panels = MODELS + ["Consensus (2 of 3)"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.3))
    for axc, name in zip(axes, panels):
        title = name if name.startswith("Consensus") else f"{SHORT[name]}  (thr {thr[name]:.2f})"
        draw_cm(axc, y, preds[name], title)
    fig.suptitle(
        f"Confusion matrices at deployed thresholds — {SOURCE_TITLE[source]} model (out-of-fold)"
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrices_{source}.png", dpi=130)
    plt.close(fig)


def top_features(source, n=15):
    md = json.load(open(f"files/weather/models/model_metadata_{source}.json"))
    # No source has categorical columns, so the transformed feature order equals
    # numerical_columns and aligns with feature_importances_.
    cols = md["numerical_columns"]
    model = xgb.XGBClassifier()
    model.load_model(f"files/weather/models/xgboost_best_f1_{source}.json")
    imp = np.asarray(model.feature_importances_)
    n = min(n, len(cols))
    order = np.argsort(imp)[::-1][:n][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([cols[i] for i in order], [imp[i] for i in order], color="#4C72B0")
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title(f"Top {n} features — {SOURCE_TITLE[source]} XGBoost model")
    fig.tight_layout()
    fig.savefig(f"{OUT}/top_features_{source}.png", dpi=130)
    plt.close(fig)


def leaky_oof_predictions(X, y):
    """Out-of-fold predictions from a plain *row-wise* split — the old bug.

    Identical to tr.oof_predictions except folds ignore the date grouping, so a
    day's other forecast-hour rows sit in the training set while it is scored.
    """
    cv = StratifiedKFold(n_splits=tr.CV_SPLITS, shuffle=True, random_state=tr.RANDOM_STATE)
    oof = {name: np.full(len(y), np.nan) for name in MODELS}
    for tr_i, te_i in cv.split(X, y):
        pre, _, _ = tr.make_preprocessor(X.iloc[tr_i])
        Xtr = pre.fit_transform(X.iloc[tr_i])
        Xte = pre.transform(X.iloc[te_i])
        for name, model in tr.build_models(y.iloc[tr_i]).items():
            tr.fit_model(model, name, Xtr, y.iloc[tr_i])
            oof[name][te_i] = model.predict_proba(Xte)[:, 1]
    return oof


def leakage_comparison(X, y, groups, oof_honest):
    """Before/after figure for the row-leakage fix (all-parameters model)."""
    oof_leaky = leaky_oof_predictions(X, y)
    rows = [
        ("Before — rows split randomly (leaky)", oof_leaky),
        ("After — folds grouped by date (honest)", oof_honest),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4))
    for axrow, (label, oof) in zip(axes, rows):
        for axc, m in zip(axrow, MODELS):
            # Each panel uses its own F1-max threshold so the two rows are
            # compared at each model's own best operating point.
            thr, _ = tr.best_threshold(y, oof[m])
            pred = (oof[m] >= thr).astype(int)
            prec = precision_score(y, pred, zero_division=0)
            rec = recall_score(y, pred, zero_division=0)
            draw_cm(axc, y, pred, f"{SHORT[m]} — P {prec:.2f} / R {rec:.2f}")
        axrow[0].text(-0.42, 0.5, label, transform=axrow[0].transAxes, rotation=90,
                      va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("Effect of the row-leakage fix — all-parameters model, F1-max thresholds")
    fig.tight_layout()
    fig.savefig(f"{OUT}/leakage_before_after.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=list(tr.SOURCES), choices=list(tr.SOURCES))
    ap.add_argument("--skip-leakage", action="store_true",
                    help="skip the (slow) before/after leakage figure")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df = tr.load_data("files/weather/csv/ML/all/")
    y = (df[tr.TARGET] >= 0.5).astype(int)
    groups = df["date"]
    base_rate = float(y.mean())

    per_source_performance(base_rate)

    for source in args.sources:
        top_features(source)
        X = tr.select_features(df, source)
        oof = tr.oof_predictions(X, y, groups)
        confusion_panels(y, oof, source)
        if source == "all":
            roc_pr_curves(y, oof, base_rate)
            if not args.skip_leakage:
                leakage_comparison(X, y, groups, oof)
        print(f"  {source}: figures written")
    print(f"Wrote figures to {OUT}/ (base rate {base_rate:.3f})")


if __name__ == "__main__":
    main()
