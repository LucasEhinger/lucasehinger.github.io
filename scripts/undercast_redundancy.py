#!/usr/bin/env python3
"""How many of the combined model's 213 columns are actually distinct?

`undercast_capacity.py` deflates the NUMERATOR of the overfitting worry: 31,009
training rows are really 175 independent undercast events, because each hour is
sampled once per forecast lead and consecutive hours of one cloud deck are one
weather event. This script deflates the DENOMINATOR, which is the other half of
the same argument and was missing from the write-up.

The combined model is six weather models' columns stacked side by side. They are
describing the same atmosphere, so `tmp_500mb_hrrr`, `tmp_500mb_gfs`,
`tmp_500mb_nam`, `tmp_500mb_rap` and `tmp_500mb_ecmwf` are not five facts. Three
measurements, in increasing order of how much they assume:

  1. NAME COLLAPSE -- strip the source suffix and count distinct quantities.
     Assumption-free, and already a large cut.
  2. CROSS-SOURCE RANK CORRELATION -- for each quantity carried by more than one
     source, how alike are the copies in practice? Spearman, because the trees
     only ever see order. The interesting result is that this is BIMODAL, not
     uniformly high, and which side a field falls on is not random.
  3. PRINCIPAL COMPONENTS -- how many orthogonal directions carry the variance.
     The strictest count, and the most caveated: it needs complete rows, so it
     can only be taken where every source is present, which is the regime with
     the MOST duplication to find. It is a lower bound on effective width, not
     an estimate of it.

Then the question that decides whether any of it matters: does the fitted model's
importance land on the duplicated columns or the diverse ones?

Read-only. Loads the shards and the fitted artifacts to rank columns; refits
nothing and writes nothing into the model directory.

    python3 scripts/undercast_redundancy.py
    python3 scripts/undercast_redundancy.py --cache /tmp/obs.pkl
"""
import argparse
import collections
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_undercast_obs import (  # noqa: E402
    MODEL_SOURCES, load_obs_data, assign_splits, rows_for_source, select_features,
)

MODELS_DIR = "files/weather/models/obs"
OUT_JSON = "files/weather/models/obs/redundancy_histogram.json"
BIN_W = 0.05      # 20 bins over [0, 1]; ~15 of the 300 pairs per bin
HIGH = 0.9      # "the same number written down twice"
MIN_PRESENT = 0.5   # a column must be this populated to enter the PCA


def base_of(col):
    """Split `tmp_500mb_hrrr` into ('tmp_500mb', 'hrrr').

    `_no_cloud` flags carry the source in the MIDDLE of the name
    (`cloud_top_hrrr_no_cloud`), so they need their own case or they would be
    read as source-less and counted as 213 distinct quantities.
    """
    for s in MODEL_SOURCES:
        if col.endswith(f"_{s}"):
            return col[: -(len(s) + 1)], s
        if col.endswith(f"_{s}_no_cloud"):
            return col[: -(len(s) + 1) - len("_no_cloud")] + "_no_cloud", s
    return col, None


def training_features(csv_dir, labels, record, cache=None):
    if cache and os.path.exists(cache):
        df = pd.read_pickle(cache)
    else:
        df = assign_splits(load_obs_data(csv_dir, labels, record), buffer_days=1)
        if cache:
            df.to_pickle(cache)
    sub = rows_for_source(df, "all")
    tr = sub[(sub["split_eff"] == "train") & ~sub["screen_ambiguous"]]
    return select_features(tr, "all")


def cross_source_correlation(X):
    """Median |Spearman| between copies of one quantity, per quantity."""
    groups = collections.defaultdict(list)
    for c in X.columns:
        b, s = base_of(c)
        if s:
            groups[b].append(c)
    C = X.select_dtypes(exclude="object").corr(method="spearman")
    per_quantity, all_pairs = {}, []
    for b, cols in groups.items():
        cols = [c for c in cols if c in C.columns]
        vs = [abs(C.loc[cols[i], cols[j]])
              for i in range(len(cols)) for j in range(i + 1, len(cols))]
        vs = [v for v in vs if np.isfinite(v)]
        if vs:
            per_quantity[b] = (len(cols), float(np.median(vs)))
            all_pairs.extend(vs)
    return C, per_quantity, np.array(all_pairs)


def unrelated_baseline(C, n=20000, seed=0):
    """|Spearman| between two columns of DIFFERENT quantities, for scale.

    Without this the same-quantity median means nothing: these are all weather
    fields at one point, so everything correlates with everything somewhat.
    """
    rng = np.random.default_rng(seed)
    cn = list(C.columns)
    out = []
    for _ in range(n):
        i, j = rng.choice(len(cn), 2, replace=False)
        if base_of(cn[i])[0] == base_of(cn[j])[0]:
            continue
        v = C.iloc[i, j]
        if np.isfinite(v):
            out.append(abs(v))
    return np.array(out)


def effective_dimension(X):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = X.select_dtypes(exclude="object")
    Z = Z.loc[:, Z.notna().mean() > MIN_PRESENT].dropna()
    if len(Z) < 200:
        return None
    A = StandardScaler().fit_transform(Z)
    ev = PCA().fit(A).explained_variance_ratio_
    cum = np.cumsum(ev)
    return {
        "rows": int(len(Z)),
        "cols": int(Z.shape[1]),
        "n90": int(np.searchsorted(cum, 0.90)) + 1,
        "n95": int(np.searchsorted(cum, 0.95)) + 1,
        "n99": int(np.searchsorted(cum, 0.99)) + 1,
        "participation_ratio": float(1 / np.sum(ev ** 2)),
    }


def importance_by_redundancy(per_quantity, models_dir=MODELS_DIR):
    """Does fitted importance avoid the near-duplicate columns?

    Aggregated to the QUANTITY, because importance split between five copies of
    one field would otherwise look like five weak features instead of one strong
    one -- which is the exact illusion this script exists to dispel.
    """
    pre = joblib.load(os.path.join(models_dir, "preprocessor_all.pkl"))
    names = list(pre.get_feature_names_out())
    out = {}
    for tag in ("gradient_boosting", "random_forest", "xgboost"):
        path = os.path.join(models_dir, f"{tag}_best_f1_all.pkl")
        if not os.path.exists(path):
            continue
        fi = getattr(joblib.load(path), "feature_importances_", None)
        if fi is None or len(fi) != len(names):
            continue
        agg = collections.defaultdict(float)
        for n, v in zip(names, fi):
            raw = n.split("__", 1)[-1].replace("missingindicator_", "")
            agg[base_of(raw)[0]] += float(v)
        tot = sum(agg.values())
        hi = sum(v for b, v in agg.items()
                 if per_quantity.get(b, (0, 0.0))[1] > HIGH)
        wm = sum(v * per_quantity[b][1] for b, v in agg.items() if b in per_quantity)
        ww = sum(v for b, v in agg.items() if b in per_quantity)
        out[tag] = {"share_on_duplicated": 100 * hi / tot,
                    "weighted_median_r": wm / ww if ww else float("nan")}
    return out


def write_histogram(same, diff, per_quantity, out_path=OUT_JSON):
    """Bin both distributions for the page's chart.

    The baseline is ~19k pairs against 300, so the two are stored as SHARES of
    their own distribution, not raw counts. Plotting raw counts would put the
    baseline off the top of the axis and make the comparison unreadable -- and
    the comparison is the entire point of drawing them together.
    """
    edges = [round(i * BIN_W, 4) for i in range(int(round(1 / BIN_W)) + 1)]

    def binned(v):
        idx = np.clip((np.asarray(v) / BIN_W).astype(int), 0, len(edges) - 2)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        return counts.tolist(), (counts / max(1, counts.sum())).tolist()

    same_counts, same_share = binned(same)
    diff_counts, diff_share = binned(diff)
    payload = {
        "bin_width": BIN_W,
        "edges": edges,
        "same": {
            "n": int(len(same)), "counts": same_counts, "share": same_share,
            "median": float(np.median(same)),
            "q25": float(np.percentile(same, 25)),
            "q75": float(np.percentile(same, 75)),
            "share_above_0_9": float(np.mean(same > HIGH)),
            "share_above_0_95": float(np.mean(same > 0.95)),
        },
        "baseline": {
            "n": int(len(diff)), "counts": diff_counts, "share": diff_share,
            "median": float(np.median(diff)),
        },
        # A few labelled quantities so the chart can point at what lives where,
        # which is what the table it replaced was actually for.
        "markers": [
            {"quantity": b, "sources": n, "r": round(v, 3)}
            for b, (n, v) in sorted(per_quantity.items(), key=lambda kv: -kv[1][1])
            if n >= 3
        ],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"\nwrote {out_path} "
          f"({len(same)} same-quantity pairs, {len(diff)} baseline pairs)")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="files/weather/csv/obs")
    ap.add_argument("--labels", default="files/weather/csv/MtWashington_undercast_orig.csv")
    ap.add_argument("--record", default="files/weather/obs/undercast_record.csv")
    ap.add_argument("--cache", default=None, help="pickle the assembled frame here")
    args = ap.parse_args()

    X = training_features(args.csv_dir, args.labels, args.record, args.cache)
    print(f"\ncombined model: {X.shape[1]} feature columns over {len(X):,} training rows")

    # --- 1. name collapse --------------------------------------------------
    groups = collections.defaultdict(list)
    for c in X.columns:
        groups[base_of(c)[0]].append(c)
    n_src = collections.Counter(
        sum(1 for c in cols if base_of(c)[1]) for cols in groups.values())
    print(f"  distinct quantities            {len(groups)}")
    print(f"  carried by >1 source           {sum(1 for c in groups.values() if sum(1 for x in c if base_of(x)[1]) > 1)}")
    print("  sources per quantity           "
          + ", ".join(f"{k}: {v}" for k, v in sorted(n_src.items())))

    # --- 2. how alike are the copies? --------------------------------------
    C, per_quantity, same = cross_source_correlation(X)
    diff = unrelated_baseline(C)
    print(f"\ncross-source |Spearman|, {len(same)} same-quantity pairs")
    print(f"  median {np.median(same):.3f}   q25 {np.percentile(same, 25):.3f}"
          f"   q75 {np.percentile(same, 75):.3f}")
    print(f"  >{HIGH}: {100 * np.mean(same > HIGH):.0f}%      "
          f">0.95: {100 * np.mean(same > 0.95):.0f}%")
    print(f"  baseline (different quantities): median {np.median(diff):.3f}")

    ranked = sorted(((v, b, n) for b, (n, v) in per_quantity.items() if n >= 3),
                    reverse=True)
    print("\n  most duplicated (>=3 sources):")
    for v, b, n in ranked[:8]:
        print(f"    {b:30s} {n} sources   {v:.3f}")
    print("  least duplicated:")
    for v, b, n in ranked[-8:]:
        print(f"    {b:30s} {n} sources   {v:.3f}")

    dup_cols = 100 * np.mean(
        [per_quantity.get(base_of(c)[0], (0, 0.0))[1] > HIGH for c in X.columns])
    print(f"\n  {dup_cols:.0f}% of the {X.shape[1]} columns belong to a quantity "
          f"duplicated at r>{HIGH}")

    # --- 3. effective dimension --------------------------------------------
    pca = effective_dimension(X)
    if pca:
        print(f"\nPCA on {pca['rows']:,} complete rows x {pca['cols']} columns "
              f"(>{MIN_PRESENT:.0%} populated)")
        print(f"  components for 90/95/99% of variance: "
              f"{pca['n90']} / {pca['n95']} / {pca['n99']}")
        print(f"  participation ratio                   {pca['participation_ratio']:.1f}")
        print("  NOTE: complete rows exist only at short leads, where all six "
              "sources report.\n        That is the regime with the most "
              "duplication, so this is a LOWER bound\n        on effective width. "
              "Past 48 h most of these copies do not exist to collapse.")

    # --- 4. does the model use the copies? ---------------------------------
    print("\nfitted importance, aggregated to the quantity:")
    for tag, r in importance_by_redundancy(per_quantity).items():
        print(f"  {tag:20s} {r['share_on_duplicated']:5.1f}% of importance on "
              f"r>{HIGH} quantities   (weighted median r {r['weighted_median_r']:.3f})")
    print(f"  {'columns, for scale':20s} {dup_cols:5.1f}%")

    # --- 5. the page's chart -----------------------------------------------
    write_histogram(same, diff, per_quantity)


if __name__ == "__main__":
    main()
