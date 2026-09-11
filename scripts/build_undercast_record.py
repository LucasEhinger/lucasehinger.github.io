#!/usr/bin/env python3
"""Build the 1997-present undercast record from KMWN observer remarks.

Every hour, a human observer on the summit augments the KMWN METAR with a
``TPS LWR <COVER><HEIGHT>`` group -- "tops lower" -- giving the coverage and the
top height (ft MSL) of any cloud deck BELOW the 6,288 ft summit. That, plus the
prevailing visibility and the sky groups above the station, is enough to screen
for an undercast: a continuous deck below you, clear air between you and it, and
no low lid overhead.

The screen was calibrated against 589 hand-labeled days of Observatory webcam
imagery (24 scored undercast) and tuned on the noon observation of each:

    cover below summit is BKN or OVC   (>= 5/8 sky; SCT is a broken-up deck,
                                        and including it cost more precision
                                        than the recall was worth)
    prevailing visibility > 40 SM      (you have to be able to SEE the deck;
                                        KMWN is staffed, so visibility is
                                        estimated off landmarks out to 140 SM)
    no lid, or lid >= 5,000 ft AGL     (a low ceiling overhead makes it a
                                        sandwich, not an undercast)
    deck top >= 500 ft below summit    (a deck 200 ft down is not a sea of
                                        clouds)
    lowest layer above >= 1,000 ft     (the lid test looks only at the CEILING,
                                        i.e. the lowest BKN/OVC, so it happily
                                        passes a report carrying FEW002 or
                                        SCT010 -- cloud touching the summit.
                                        You cannot look down on a deck from
                                        inside the muck. This cut alone removes
                                        9 of 16 false positives and costs no
                                        true positive; it is also a plateau,
                                        scoring the same anywhere from 500 to
                                        1,000 ft, so it is not a fitted knob.)

Against the hand labels at noon that screen scores precision 0.70, recall 0.67.
Both numbers matter downstream: the labels it produces are roughly 30% false
positive, so anything trained on them must be validated against the hand labels
rather than against the screen itself.

Recall is capped near 0.79 by the remarks themselves: of the 8 labeled undercast
days this misses, 5 report a SCT (scattered) deck -- deliberately excluded -- and
2 report the summit fogged in at 1/16 SM, where the observer could not see the
deck the webcam could.

Writes one row per observation (not per day), because the forecast features are
sampled at each observation's own valid time -- a deck that exists at 06 UTC and
burns off by 18 UTC is two different atmospheric states, not one daily label.

Usage:
    python3 scripts/build_undercast_record.py --cache <iem dir>
    python3 scripts/build_undercast_record.py --cache <iem dir> --validate
"""
import argparse
import csv
import glob
import gzip
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_summit_remarks import FEATURES, UTC, ET, extract  # noqa: E402

# --- the screen, as calibrated on the hand-labeled noon set -----------------
MIN_VIS_SM = 40
MIN_LID_FT = 5000
MIN_DEPTH_FT = 500
MIN_LOWEST_ABOVE_FT = 1000
POS_COVER = ("BKN", "OVC")

# Observations that fail the screen only narrowly are the ones the screen is
# least trustworthy about. Calling them negatives would inject label noise
# exactly at the decision boundary, so they are marked ambiguous and excluded
# from training rather than counted as clear.
AMB_COVER = ("SCT",)
AMB_VIS_SM = 25
AMB_LID_FT = 3500
AMB_DEPTH_FT = 250
AMB_LOWEST_ABOVE_FT = 500

OUT_FIELDS = ["valid_utc", "valid_local", "label", "year", "month", "hour_utc"] + FEATURES[3:]


def _f(v):
    return None if v in ("", None) else float(v)


def classify(f):
    """-> 'undercast' | 'ambiguous' | 'clear'."""
    vis, depth = _f(f["vis_sm"]), _f(f["depth_below_ft"])
    lid, lowest = _f(f["overhead_ceiling_ft"]), _f(f["overhead_lowest_ft"])
    cover = f["max_cover"]

    passes = (
        cover in POS_COVER
        and vis is not None and vis > MIN_VIS_SM
        and (lid is None or lid >= MIN_LID_FT)
        and depth is not None and depth >= MIN_DEPTH_FT
        and (lowest is None or lowest >= MIN_LOWEST_ABOVE_FT)
    )
    if passes:
        return "undercast"
    # Near-miss on any single cut, or a scattered deck, is ambiguous.
    near = (
        cover in POS_COVER + AMB_COVER
        and (vis is None or vis > AMB_VIS_SM)
        and (lid is None or lid >= AMB_LID_FT)
        and (depth is None or depth >= AMB_DEPTH_FT)
        and (lowest is None or lowest >= AMB_LOWEST_ABOVE_FT)
    )
    return "ambiguous" if near else "clear"


def iter_reports(cache_dir, start, end):
    """Yield (datetime UTC, raw metar) for every cached report in range."""
    for path in sorted(glob.glob(os.path.join(cache_dir, "metar_*.csv"))):
        yr = int(os.path.basename(path)[6:10])
        if not (start <= yr <= end):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                m = (r.get("metar") or "").strip()
                if not m:
                    continue
                try:
                    t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
                except (ValueError, KeyError):
                    continue
                yield t.replace(tzinfo=UTC), m


def build(cache_dir, start, end, out_path, keep_text=False):
    rows = []
    seen = set()
    per_year = defaultdict(Counter)
    for t, metar in iter_reports(cache_dir, start, end):
        # KMWN transmits roughly hourly; a special report in the same clock hour
        # would otherwise double-count that hour in the sample.
        key = t.strftime("%Y-%m-%dT%H")
        if key in seen:
            continue
        seen.add(key)
        local = t.astimezone(ET)
        f = extract(metar, local, require_layers=False)
        if f is None:
            continue
        label = classify(f)
        per_year[t.year][label] += 1
        row = {k: f.get(k, "") for k in FEATURES[3:]}
        row.update({
            "valid_utc": t.strftime("%Y-%m-%dT%H:%M"),
            "valid_local": local.strftime("%Y-%m-%dT%H:%M"),
            "label": label,
            "year": t.year,
            "month": t.month,
            "hour_utc": t.hour,
        })
        rows.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Write UNCOMPRESSED even though this is 26 MB. Committing .gz to git is a
    # trap: git already zlib-compresses every blob, so the initial repo size is
    # identical either way, but gzip scrambles the bytes so git cannot delta a
    # regenerated file against its predecessor. Measured on 120 shard files with
    # 2% of rows changed: +0.8 MB to re-commit as raw CSV, +33 MB as .gz. Raw
    # also stays greppable, which is how the ISD truncation bug got found.
    fields = OUT_FIELDS if keep_text else [c for c in OUT_FIELDS
                                           if c not in ("remark", "metar_body")]
    opener = gzip.open if out_path.endswith(".gz") else open
    with opener(out_path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"{'year':>5} {'obs':>7} {'undercast':>10} {'ambig':>7} {'clear':>7} {'%uc':>6}")
    tot = Counter()
    for y in sorted(per_year):
        c = per_year[y]
        n = sum(c.values())
        tot.update(c)
        print(f"{y:>5} {n:>7} {c['undercast']:>10} {c['ambiguous']:>7} "
              f"{c['clear']:>7} {100*c['undercast']/max(n,1):>5.1f}%")
    n = sum(tot.values())
    print(f"{'ALL':>5} {n:>7} {tot['undercast']:>10} {tot['ambiguous']:>7} "
          f"{tot['clear']:>7} {100*tot['undercast']/max(n,1):>5.1f}%")
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    return rows


def validate(rows, labels_path):
    """Score the screen's noon observation against the hand-labeled webcam days."""
    hand = {}
    with open(labels_path) as fh:
        for r in csv.DictReader(fh):
            try:
                d = datetime.strptime(r["Short Date"], "%m/%d/%y").date()
                hand[d] = float(r["Avg"])
            except (ValueError, KeyError):
                continue
    # the observation closest to local noon on each hand-labeled day
    best = {}
    for r in rows:
        lt = datetime.strptime(r["valid_local"], "%Y-%m-%dT%H:%M")
        d = lt.date()
        if d not in hand:
            continue
        delta = abs(lt.hour * 60 + lt.minute - 720)
        if d not in best or delta < best[d][0]:
            best[d] = (delta, r)
    pos = {d for d, v in hand.items() if v >= 0.5}
    tp = fp = fn = amb = 0
    for d, v in hand.items():
        if d not in best:
            continue
        lab = best[d][1]["label"]
        if lab == "ambiguous":
            amb += 1
            continue
        if lab == "undercast":
            tp += d in pos
            fp += d not in pos
    fn = len(pos) - tp
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    print(f"\nvs {len(hand)} hand-labeled days ({len(pos)} undercast), noon observation:")
    print(f"  matched={len(best)}  ambiguous(excluded)={amb}")
    print(f"  TP={tp} FP={fp} FN={fn}  precision={P:.2f} recall={R:.2f} "
          f"F1={2*P*R/max(P+R,1e-9):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, help="dir of IEM metar_YYYY.csv files")
    p.add_argument("--start", type=int, default=1997)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--out", default="files/weather/obs/undercast_record.csv")
    p.add_argument("--keep-text", action="store_true",
                   help="also write the raw remark and METAR body columns. Useful "
                        "for auditing a disagreement, but they are 40%% of the file "
                        "and regenerable in ~3 min, so they stay out of git.")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--labels", default="files/weather/csv/MtWashington_undercast_orig.csv")
    a = p.parse_args()
    rows = build(a.cache, a.start, a.end, a.out, keep_text=a.keep_text)
    if a.validate:
        validate(rows, a.labels)


if __name__ == "__main__":
    main()
