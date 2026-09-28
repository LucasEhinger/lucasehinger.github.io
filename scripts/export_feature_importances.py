#!/usr/bin/env python3
"""Export top-N feature importances per (source, algorithm) as JSON.

The write-up used to show these as 21 static PNGs, one per source and algorithm.
A reader looking at `dRH_925_850_ecmwf` cannot tell what it is, and a picture
cannot answer them. Emitting the numbers instead lets the page draw the bars
itself and attach an explanation to each one.

Importances are whatever the fitted estimator reports -- gain share for the
boosted trees, mean impurity decrease for the forest. They describe which splits
THAT fit happened to choose, not which quantity the atmosphere cares about, which
is why all three algorithms are exported rather than only the deployed one.

    python3 scripts/export_feature_importances.py
    python3 scripts/export_feature_importances.py --n 20
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import undercast_eval as ue  # noqa: E402
from plot_undercast_obs_models import tidy  # noqa: E402
from train_undercast_obs import MODEL_NAMES, SOURCES  # noqa: E402

OUT = "files/weather/models/obs/feature_importances.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--models-dir", default=ue.MODELS_DIR)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    out = {}
    for source in SOURCES:
        pre, models, _ = ue.load_artifacts(source, a.models_dir)
        names = tidy(pre.get_feature_names_out())
        per_algo = {}
        for algo in MODEL_NAMES:
            imp = np.asarray(models[algo].feature_importances_)
            assert len(names) == len(imp), \
                f"{source}/{algo}: {len(names)} names vs {len(imp)} importances"
            order = np.argsort(imp)[::-1][:min(a.n, len(imp))]
            slug = algo.lower().replace(" ", "_")
            per_algo[slug] = [[names[i], round(float(imp[i]), 5)] for i in order]
        out[source] = per_algo
        top = per_algo["gradient_boosting"][0]
        print(f"  {source:6s} {len(names):3d} features, top (GB): {top[0]} ({top[1]:.3f})")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\nwrote {a.out} ({os.path.getsize(a.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
