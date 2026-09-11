#!/usr/bin/env python3
"""Is three algorithms and a majority vote the right design?

It was not. This script is what established that, and the site now serves a
single model -- combined source, Gradient Boosting, per-lead thresholds -- rather
than the 2-of-3 vote across XGBoost, Random Forest and Gradient Boosting that it
used to. That choice had never been tested; it was inherited. Three questions,
answered on the untouched holdouts using the models already on disk:

  1 DOES THE VOTE HELP?  Compare each algorithm alone, the 1/2/3-of-3 votes, and
    the mean of the three probabilities. An ensemble only pays when its members
    make DIFFERENT mistakes, so the rank correlation between them is reported
    alongside -- it is the mechanism, and it explains the answer.

  2 ARE THOSE THREE THE RIGHT ALGORITHMS?  Four alternatives are trained on the
    identical split: histogram gradient boosting (which splits on missingness
    natively instead of being handed an imputed sentinel), L2 logistic regression
    (is any of this nonlinear?), extra trees, and a deeper random forest.

  3 DO THE WEATHER MODELS DISAGREE MORE THAN THE ALGORITHMS DO?  Per-source
    skill on the rows every source covers -- not each source's own rows, which
    would compare 2014 weather against 2023 weather -- plus averaging across
    sources, and a greedy search for how many sources are actually needed.

Uncertainty is a date-block bootstrap: whole days are resampled, because hours
within a day are the same weather and resampling them independently would
manufacture confidence that is not there.

    python3 scripts/compare_undercast_ensembles.py
    python3 scripts/compare_undercast_ensembles.py --skip-alternatives
    python3 scripts/compare_undercast_ensembles.py --cache /tmp/obs.pkl
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import undercast_eval as ue  # noqa: E402
from train_undercast_obs import (  # noqa: E402
    MODEL_NAMES, MODEL_SOURCES, RANDOM_STATE, SHORT_NAME, TARGET,
    best_f1_threshold, make_preprocessor, select_features,
)

OUT_JSON = "files/weather/models/obs/ensemble_comparison.json"
FIG_DIR = "files/weather/examples/model_training_images"
SPLITS = ("holdout_baserate", "holdout_webcam")
N_BOOT = 1000


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------
def prf(y, pred):
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fired": int(np.sum(pred)),
    }


def auc_pr(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p))}


def boot_auc_ci(y, p, dates, n=N_BOOT, seed=RANDOM_STATE):
    """Percentile CI for ROC-AUC, resampling whole DAYS.

    Resampling rows would treat 03:50 and 04:50 of the same undercast as
    independent evidence. They are not, and the interval would come out perhaps
    half as wide as it should be.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(dates, return_inverse=True)
    by_day = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_day[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out \
        else (float("nan"), float("nan"))


def boot_auc_diff(y, pa, pb, dates, n=N_BOOT, seed=RANDOM_STATE):
    """Paired CI on AUC(a) - AUC(b). Paired, so shared weather cancels out."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(dates, return_inverse=True)
    by_day = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_day[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return {"diff": float(np.mean(out)),
            "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)),
            "p_worse": float(np.mean(np.asarray(out) <= 0))}


# --------------------------------------------------------------------------
# question 1 -- does the 2-of-3 vote earn its place?
# --------------------------------------------------------------------------
def algorithm_consensus(df, sources, models_dir):
    out = {}
    for src in sources:
        got = {sp: ue.score_split(df, src, sp, models_dir) for sp in SPLITS}
        base, web = got["holdout_baserate"], got["holdout_webcam"]
        if not len(base["y"]) or not len(web["y"]):
            continue
        thr = base["thresholds"]
        res = {"n": {sp: int(len(got[sp]["y"])) for sp in SPLITS},
               "positives": {sp: int(got[sp]["y"].sum()) for sp in SPLITS},
               "members": {}}

        for name in MODEL_NAMES:
            res["members"][name] = {
                "threshold": thr[name],
                "baserate": {**auc_pr(base["y"], base["proba"][name]),
                             **prf(base["y"], (base["proba"][name] >= thr[name]).astype(int))},
                "webcam": {**auc_pr(web["y"], web["proba"][name]),
                           **prf(web["y"], (web["proba"][name] >= thr[name]).astype(int))},
            }

        votes = {sp: np.sum([(got[sp]["proba"][n] >= thr[n]).astype(int)
                             for n in MODEL_NAMES], axis=0) for sp in SPLITS}
        for k, label in ((1, "any of 3"), (2, "2 of 3 (deployed)"), (3, "unanimous")):
            res["members"][label] = {
                "threshold": None,
                "baserate": prf(base["y"], (votes["holdout_baserate"] >= k).astype(int)),
                "webcam": prf(web["y"], (votes["holdout_webcam"] >= k).astype(int)),
            }

        # Soft vote. Its threshold is fitted on the base-rate holdout, exactly as
        # the members' thresholds were, so the two are equally optimistic there
        # and equally clean on the webcam split.
        mean = {sp: np.mean([got[sp]["proba"][n] for n in MODEL_NAMES], axis=0)
                for sp in SPLITS}
        t = best_f1_threshold(base["y"], mean["holdout_baserate"])
        res["members"]["mean probability"] = {
            "threshold": t,
            "baserate": {**auc_pr(base["y"], mean["holdout_baserate"]),
                         **prf(base["y"], (mean["holdout_baserate"] >= t).astype(int))},
            "webcam": {**auc_pr(web["y"], mean["holdout_webcam"]),
                       **prf(web["y"], (mean["holdout_webcam"] >= t).astype(int))},
        }

        # The mechanism: near-identical members cannot outvote each other.
        P = pd.DataFrame({n: base["proba"][n] for n in MODEL_NAMES})
        res["rank_correlation"] = P.corr(method="spearman").round(4).to_dict()
        res["dates"] = base["rows"]["date"].to_numpy()
        out[src] = res
    return out


# --------------------------------------------------------------------------
# question 2 -- is anything better than the incumbent three?
# --------------------------------------------------------------------------
def alternative_algorithms(df, sources, models_dir):
    out = {}
    for src in sources:
        tr = ue.split_rows(df, src, "train")
        Xtr, ytr = select_features(tr, src), tr[TARGET].to_numpy()
        got = {sp: ue.split_rows(df, src, sp) for sp in SPLITS}
        X = {sp: select_features(got[sp], src) for sp in SPLITS}
        y = {sp: ue.truth(got[sp], sp) for sp in SPLITS}
        if not all(len(y[sp]) and y[sp].sum() for sp in SPLITS):
            continue

        res = {}
        # Incumbents, straight off disk -- the numbers the page quotes.
        pre, models, meta = ue.load_artifacts(src, models_dir)
        for name, m in models.items():
            res[name] = {"incumbent": True}
            for sp in SPLITS:
                res[name][sp] = auc_pr(y[sp], m.predict_proba(pre.transform(X[sp]))[:, 1])

        def record(name, pb, pw, secs):
            res[name] = {"incumbent": False, "fit_seconds": round(secs, 1),
                         "holdout_baserate": auc_pr(y["holdout_baserate"], pb),
                         "holdout_webcam": auc_pr(y["holdout_webcam"], pw)}

        # Histogram GB: no imputation at all. It routes NaN down whichever branch
        # helps, which is the same information the sentinel + indicator pair
        # encodes -- worth knowing whether the hand-built version wins.
        t0 = time.time()
        h = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            class_weight="balanced", random_state=RANDOM_STATE)
        h.fit(Xtr.astype(float), ytr)
        record("HistGradientBoosting",
               h.predict_proba(X["holdout_baserate"].astype(float))[:, 1],
               h.predict_proba(X["holdout_webcam"].astype(float))[:, 1], time.time() - t0)

        # A linear model gets its OWN preprocessing. Handing it the -9999
        # constant fill the trees use would not be a fair test -- one sentinel
        # would swamp every coefficient. Median fill, standardise, keep the
        # missingness indicators.
        t0 = time.time()
        cols = list(Xtr.columns)
        lpre = make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cols),
            (MissingIndicator(features="all"), cols),
        )
        A = lpre.fit_transform(Xtr)
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
        lr.fit(A, ytr)
        record("Logistic regression",
               lr.predict_proba(lpre.transform(X["holdout_baserate"]))[:, 1],
               lr.predict_proba(lpre.transform(X["holdout_webcam"]))[:, 1], time.time() - t0)

        for name, est in (
            ("Extra Trees", ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=20,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
            ("Random Forest (deeper)", RandomForestClassifier(
                n_estimators=600, min_samples_leaf=5,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ):
            t0 = time.time()
            tpre, _, _ = make_preprocessor(Xtr)
            est.fit(tpre.fit_transform(Xtr), ytr)
            record(name,
                   est.predict_proba(tpre.transform(X["holdout_baserate"]))[:, 1],
                   est.predict_proba(tpre.transform(X["holdout_webcam"]))[:, 1],
                   time.time() - t0)
        out[src] = res
    return out


# --------------------------------------------------------------------------
# question 3 -- how different are the weather models?
# --------------------------------------------------------------------------
def source_comparison(df, split, algorithm, models_dir, n_boot=N_BOOT):
    idx, probs, y = ue.aligned_probabilities(df, MODEL_SOURCES, split,
                                             algorithm, models_dir)
    dates = idx["valid_utc"].str[:10].to_numpy()
    res = {"split": split, "algorithm": algorithm, "n": int(len(y)),
           "positives": int(y.sum()), "sources": {}}

    for s in MODEL_SOURCES:
        lo, hi = boot_auc_ci(y, probs[s], dates, n_boot)
        res["sources"][s] = {**auc_pr(y, probs[s]), "ci95": [lo, hi]}

    M = pd.DataFrame(probs)
    res["rank_correlation"] = M.corr(method="spearman").round(4).to_dict()

    mean6 = M.mean(axis=1).to_numpy()
    lo, hi = boot_auc_ci(y, mean6, dates, n_boot)
    res["mean_of_sources"] = {**auc_pr(y, mean6), "ci95": [lo, hi]}

    # The combined model is trained on every source's columns at once, on these
    # same rows -- feature-level fusion against probability-level fusion.
    pre, models, _ = ue.load_artifacts("all", models_dir)
    rows = ue.split_rows(df, "all", split).set_index(ue.KEY)
    common = rows.index.intersection(pd.MultiIndex.from_frame(idx))
    rr = rows.loc[common].reset_index()
    pa = ue.probabilities(rr, "all", pre, models)[algorithm]
    ya = ue.truth(rr, split)
    lo, hi = boot_auc_ci(ya, pa, rr["valid_utc"].str[:10].to_numpy(), n_boot)
    res["combined_model"] = {**auc_pr(ya, pa), "ci95": [lo, hi], "n": int(len(ya))}

    # Is the combined model really beating the average, or is that within noise?
    if len(ya) == len(y):
        res["combined_vs_mean"] = boot_auc_diff(y, pa, mean6, dates, n_boot)
        best = max(MODEL_SOURCES, key=lambda s: res["sources"][s]["roc_auc"])
        res["best_single_source"] = best
        res["mean_vs_best_single"] = boot_auc_diff(y, mean6, probs[best], dates, n_boot)

    # How many sources are actually needed? Greedy forward selection on AUC.
    chosen, trail = [], []
    remaining = list(MODEL_SOURCES)
    while remaining:
        scored = [(roc_auc_score(y, M[chosen + [s]].mean(axis=1)), s) for s in remaining]
        a, s = max(scored)
        chosen.append(s)
        remaining.remove(s)
        trail.append({"added": s, "set": list(chosen), "roc_auc": float(a)})
    res["greedy_forward"] = trail

    # And the honest check on that: greedy picks the best subset ON THIS SET, so
    # its curve is optimistic. Every pair, scored exhaustively, shows the spread.
    pairs = sorted(
        ({"set": list(c), "roc_auc": float(roc_auc_score(y, M[list(c)].mean(axis=1)))}
         for c in itertools.combinations(MODEL_SOURCES, 2)),
        key=lambda d: -d["roc_auc"])
    res["all_pairs"] = pairs
    return res


# --------------------------------------------------------------------------
# the figure: where the disagreement actually lives
# --------------------------------------------------------------------------
def diversity_figure(cons, src_res, out_path, source_for_algos="all"):
    """Two correlation matrices on ONE colour scale, and what each buys.

    The scale has to be shared or the figure lies: matplotlib would stretch each
    matrix over its own range and both would look equally varied, when the point
    is that one runs 0.85-0.95 and the other 0.38-0.60.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, GREY = "#2f3337", "#9aa0a6"
    A = pd.DataFrame(cons[source_for_algos]["rank_correlation"])
    A = A.loc[list(MODEL_NAMES), list(MODEL_NAMES)]
    B = pd.DataFrame(src_res["rank_correlation"])
    B = B.loc[MODEL_SOURCES, MODEL_SOURCES]

    fig = plt.figure(figsize=(13.6, 4.8), dpi=160)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 1.0, 1.35], wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    for ax, M, labs, title in (
        (axA, A, [SHORT_NAME[n] for n in MODEL_NAMES],
         "Three algorithms,\none weather model"),
        (axB, B, [s.upper() for s in MODEL_SOURCES],
         "Six weather models,\none algorithm"),
    ):
        im = ax.imshow(M.to_numpy(), cmap="RdYlBu_r", vmin=0.3, vmax=1.0)
        ax.set_xticks(range(len(labs)), labs, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(labs)), labs, fontsize=8.5)
        for (i, j), v in np.ndenumerate(M.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.8,
                    color="white" if v > 0.82 or v < 0.42 else INK)
        ax.set_title(title, fontsize=10.5, color=INK, pad=8)
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04).set_label(
        "rank correlation of predictions", fontsize=8.5)

    # Panel C: what that diversity is worth.
    items = sorted(((s, src_res["sources"][s]) for s in MODEL_SOURCES),
                   key=lambda kv: kv[1]["roc_auc"])
    items = [(s.upper(), m["roc_auc"], m["ci95"], "#9aa8bb") for s, m in items]
    items.append(("average of\nall six", src_res["mean_of_sources"]["roc_auc"],
                  src_res["mean_of_sources"]["ci95"], "#55A868"))
    items.append(("combined\nmodel", src_res["combined_model"]["roc_auc"],
                  src_res["combined_model"]["ci95"], "#4C72B0"))
    yy = np.arange(len(items))
    for i, (_, v, ci, c) in enumerate(items):
        axC.plot(ci, [i, i], color=c, lw=2.4, solid_capstyle="round", alpha=.55)
        axC.plot([v], [i], "o", color=c, ms=7, zorder=3)
        axC.text(ci[1] + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    axC.set_yticks(yy, [t[0] for t in items], fontsize=8.5)
    axC.set_xlabel("ROC-AUC on the held-out year (95% CI, days resampled)", fontsize=9.5)
    axC.set_title("What the disagreement is worth", fontsize=10.5, color=INK, pad=8)
    axC.grid(axis="x", color="#e6e8eb", lw=0.8)
    axC.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axC.spines[s].set_visible(False)

    fig.suptitle("Ensembling helps across weather models, not across algorithms",
                 fontsize=13, color=INK, x=0.005, ha="left", y=1.03)
    fig.text(0.995, -0.09,
             f"{src_res['n']:,} holdout hours every source covers, "
             f"{src_res['positives']} undercast. Left pair shares one colour scale.",
             ha="right", fontsize=8.5, color=GREY)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out_path}")


# --------------------------------------------------------------------------
def print_consensus(out):
    for src, r in out.items():
        print(f"\n=== 1. algorithm consensus -- {src.upper()} "
              f"(base-rate {r['positives']['holdout_baserate']} pos, "
              f"webcam {r['positives']['holdout_webcam']} pos)")
        print(f"  {'':26s}{'base-rate holdout':>34s}   {'webcam holdout':>28s}")
        print(f"  {'':26s}{'AUC':>7}{'P':>7}{'R':>7}{'F1':>7}{'fires':>6}   "
              f"{'AUC':>7}{'P':>7}{'R':>7}{'F1':>7}")
        for name, m in r["members"].items():
            b, w = m["baserate"], m["webcam"]
            f = lambda d, k: f"{d[k]:.3f}" if k in d else "   --"  # noqa: E731
            print(f"  {name:26s}{f(b,'roc_auc'):>7}{b['precision']:>7.2f}"
                  f"{b['recall']:>7.2f}{b['f1']:>7.2f}{b['fired']:>6}   "
                  f"{f(w,'roc_auc'):>7}{w['precision']:>7.2f}{w['recall']:>7.2f}"
                  f"{w['f1']:>7.2f}")
        c = pd.DataFrame(r["rank_correlation"])
        off = c.to_numpy()[np.triu_indices(len(c), 1)]
        print(f"  members' rank correlation: {off.min():.2f}-{off.max():.2f} "
              f"(1.00 would mean the vote can never change an answer)")


def print_alternatives(out):
    for src, r in out.items():
        print(f"\n=== 2. algorithm alternatives -- {src.upper()}")
        print(f"  {'':26s}{'base-rate':>17}{'webcam':>17}")
        print(f"  {'':26s}{'AUC':>8}{'PR':>9}{'AUC':>8}{'PR':>9}")
        for name, m in sorted(r.items(),
                              key=lambda kv: -kv[1]["holdout_baserate"]["roc_auc"]):
            tag = " *" if m["incumbent"] else "  "
            print(f"  {name+tag:26s}{m['holdout_baserate']['roc_auc']:>8.3f}"
                  f"{m['holdout_baserate']['pr_auc']:>9.3f}"
                  f"{m['holdout_webcam']['roc_auc']:>8.3f}"
                  f"{m['holdout_webcam']['pr_auc']:>9.3f}")
        print("  * = currently deployed")


def print_sources(r):
    print(f"\n=== 3. weather models -- {r['split']}, {r['algorithm']}, "
          f"{r['n']:,} shared rows, {r['positives']} positives")
    for s, m in sorted(r["sources"].items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(f"  {s:22s} AUC={m['roc_auc']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"
              f"  PR={m['pr_auc']:.3f}")
    m = r["mean_of_sources"]
    print(f"  {'mean of all six':22s} AUC={m['roc_auc']:.3f} "
          f"[{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]  PR={m['pr_auc']:.3f}")
    c = r["combined_model"]
    print(f"  {'combined model':22s} AUC={c['roc_auc']:.3f} "
          f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]  PR={c['pr_auc']:.3f}")
    if "combined_vs_mean" in r:
        d = r["combined_vs_mean"]
        print(f"  combined - mean6: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
        d = r["mean_vs_best_single"]
        print(f"  mean6 - {r['best_single_source']} (best single): "
              f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    cm = pd.DataFrame(r["rank_correlation"]).to_numpy()
    off = cm[np.triu_indices(len(cm), 1)]
    print(f"  source rank correlation: {off.min():.2f}-{off.max():.2f}")
    print("  greedy forward selection:")
    for step in r["greedy_forward"]:
        print(f"    + {step['added']:6s} -> AUC {step['roc_auc']:.3f}   "
              f"({'+'.join(step['set'])})")
    print(f"  best pair: {'+'.join(r['all_pairs'][0]['set'])} "
          f"{r['all_pairs'][0]['roc_auc']:.3f}   "
          f"worst pair: {'+'.join(r['all_pairs'][-1]['set'])} "
          f"{r['all_pairs'][-1]['roc_auc']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default=ue.MODELS_DIR)
    p.add_argument("--cache", help="pickle of the assembled frame")
    p.add_argument("--consensus-sources", nargs="+", default=["all", "hrrr", "ecmwf"])
    p.add_argument("--alt-sources", nargs="+", default=["all", "hrrr"])
    p.add_argument("--algorithm", default="XGBoost", choices=list(MODEL_NAMES))
    p.add_argument("--boot", type=int, default=N_BOOT)
    p.add_argument("--skip-alternatives", action="store_true",
                   help="skip question 2, which is the only part that trains")
    p.add_argument("--json", default=OUT_JSON)
    p.add_argument("--figure", default=os.path.join(FIG_DIR, "ensemble_diversity.png"))
    a = p.parse_args()

    df = ue.load_frame(cache=a.cache)
    report = {}

    cons = algorithm_consensus(df, a.consensus_sources, a.models_dir)
    print_consensus(cons)
    for r in cons.values():
        r.pop("dates", None)
    report["algorithm_consensus"] = cons

    if not a.skip_alternatives:
        alt = alternative_algorithms(df, a.alt_sources, a.models_dir)
        print_alternatives(alt)
        report["alternative_algorithms"] = alt

    report["source_comparison"] = {}
    for split in SPLITS:
        r = source_comparison(df, split, a.algorithm, a.models_dir, a.boot)
        print_sources(r)
        report["source_comparison"][split] = r

    if a.figure:
        # The base-rate holdout, not the webcam one: 177 positives against 72,
        # so the intervals are the narrower and more honest pair to publish.
        diversity_figure(cons, report["source_comparison"]["holdout_baserate"],
                         a.figure)

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1, default=float)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
