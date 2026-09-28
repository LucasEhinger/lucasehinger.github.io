#!/usr/bin/env python3
"""Show what the METAR screen's cuts actually accept and reject, in pictures.

The screen in ``build_undercast_record.py`` decides "undercast" from five
conditions on the summit observer's remark. This builds a gallery of the
hand-labeled days organised by what the cuts DID to them:

  passes      every cut is satisfied -- the screen calls it undercast
  rejected    the day reports a deck below the summit but fails at least one cut,
              grouped by which one, so each group answers "what does this cut
              throw out?"

Each day carries the human verdict from the webcam frame alongside, so a group is
readable as a scorecard: the rejected-by-visibility group should be full of days
that are not undercast, and where it is not, that is the cut's cost.

Only days that reported SOMETHING below the summit are shown. The ~475 days with
no deck at all are not interesting -- nothing for a cut to act on.

Source images live under files/weather/webcam/, which is gitignored (~300 MB has
no business in a Pages repo), so this copies the days shown, downscaled, into
files/weather/examples/screen_cuts/.

    python3 scripts/build_screen_cut_gallery.py
    python3 scripts/build_screen_cut_gallery.py --dry-run
"""
import argparse
import json
import os
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_undercast_record import (  # noqa: E402
    MIN_DEPTH_FT, MIN_LID_FT, MIN_LOWEST_ABOVE_FT, MIN_VIS_SM, POS_COVER,
)
from build_observation_gallery import load_blended  # noqa: E402

RECORD = "files/weather/obs/undercast_record.csv"
LABELS = "files/weather/csv/MtWashington_undercast_orig.csv"
WEBCAM = "files/weather/webcam"
OUT_IMG = "files/weather/examples/screen_cuts"
OUT_JSON = "files/weather/examples/screen_cuts/manifest.json"
MAX_W = 760
QUALITY = 76
PER_GROUP = 6

# Imported from the screen rather than restated, so the gallery cannot drift away
# from the thing it is illustrating.
CUTS = [
    ("cover", f"deck is {' or '.join(POS_COVER)}",
     lambda r: r["max_cover"] in POS_COVER),
    ("visibility", f"visibility > {MIN_VIS_SM} SM",
     lambda r: r["vis_sm"] is not None and r["vis_sm"] > MIN_VIS_SM),
    ("lid", f"no lid overhead below {MIN_LID_FT:,} ft",
     lambda r: r["overhead_ceiling_ft"] is None or r["overhead_ceiling_ft"] >= MIN_LID_FT),
    ("depth", f"deck top >= {MIN_DEPTH_FT:,} ft below the summit",
     lambda r: r["depth_below_ft"] is not None and r["depth_below_ft"] >= MIN_DEPTH_FT),
    ("lowest", f"lowest layer overhead >= {MIN_LOWEST_ABOVE_FT:,} ft",
     lambda r: r["overhead_lowest_ft"] is None or r["overhead_lowest_ft"] >= MIN_LOWEST_ABOVE_FT),
]


def num(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def noon_rows():
    """One observation per hand-labeled day: the one nearest local noon, which is
    when the webcam frame was captured."""
    rec = pd.read_csv(RECORD, keep_default_na=False, low_memory=False)
    rec["date"] = rec["valid_local"].str[:10]
    hhmm = rec["valid_local"].str[11:16]
    rec["delta"] = (hhmm.str[:2].astype(int) * 60 + hhmm.str[3:].astype(int) - 720).abs()
    # sort_values + drop_duplicates, NOT groupby().first(): the latter takes the
    # first non-null value per column independently, which silently mixes fields
    # from different hours of the same day.
    rec = rec.sort_values("delta").drop_duplicates(subset="date", keep="first")

    hand = pd.read_csv(LABELS, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    hand["date"] = pd.to_datetime(hand["Short Date"], format="%m/%d/%y").dt.strftime("%Y-%m-%d")
    hand["y"] = pd.to_numeric(hand["Avg"], errors="coerce").fillna(0) >= 0.5
    return rec.merge(hand[["date", "y", "Avg"]], on="date", how="inner")


def copy_still(date, cam, dry):
    src = f"{WEBCAM}/{cam}/{date}.jpg"
    if not os.path.exists(src):
        return None
    name = f"{date}_{cam}.jpg"
    if not dry:
        os.makedirs(OUT_IMG, exist_ok=True)
        im = Image.open(src)
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        im.convert("RGB").save(f"{OUT_IMG}/{name}", "JPEG", quality=QUALITY, optimize=True)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--per-group", type=int, default=PER_GROUP)
    a = ap.parse_args()

    df = noon_rows()
    days = []
    for r in df.to_dict("records"):
        f = {k: num(r.get(k)) for k in ("vis_sm", "depth_below_ft",
                                        "overhead_ceiling_ft", "overhead_lowest_ft",
                                        "tops_ft", "n_layers")}
        f["max_cover"] = r["max_cover"]
        failed = [key for key, _, ok in CUTS if not ok(f)]
        days.append({"date": r["date"], "undercast": bool(r["y"]),
                     "screen_label": r["label"], "failed": failed, "fields": f,
                     "reported_deck": (f["n_layers"] or 0) > 0})

    with_deck = [d for d in days if d["reported_deck"]]
    print(f"{len(days)} hand-labeled days; {len(with_deck)} report a deck below the summit")

    groups = []
    passes = [d for d in with_deck if not d["failed"]]
    groups.append({"key": "passes", "title": "Passes every cut",
                   "rule": "the screen calls these undercast",
                   "days": passes})
    for key, label, _ in CUTS:
        # Days this cut alone is responsible for rejecting. Attributing a day to
        # every cut it happens to fail would credit each cut with work the others
        # already did; "this cut and no other" is the honest attribution.
        only = [d for d in with_deck if d["failed"] == [key]]
        groups.append({"key": key, "title": f"Fails only: {label}",
                       "rule": "fails this cut and no other", "days": only})
    multi = [d for d in with_deck if len(d["failed"]) > 1]
    groups.append({"key": "multi", "title": "Rejected by more than one cut",
                   "rule": "fails two or more", "days": multi})

    blended = load_blended(WEBCAM)
    out = []
    for g in groups:
        # Undercast days first: in a rejection group those are the cut's cost, and
        # they are the ones worth looking at hardest.
        ordered = sorted(g["days"], key=lambda d: (not d["undercast"], d["date"]))
        shown, imgs = [], 0
        for d in ordered:
            if imgs >= a.per_group and not d["undercast"]:
                continue
            pics = {c: copy_still(d["date"], c, a.dry_run) for c in ("tower", "observatory")}
            pics = {k: v for k, v in pics.items() if v}
            if not pics:
                continue
            shown.append(dict(d, images=pics,
                              blended=[c for c in pics if d["date"] in blended[c]],
                              fields={k: ("" if v is None else v) for k, v in d["fields"].items()}))
            imgs += 1
        n_pos = sum(1 for d in g["days"] if d["undercast"])
        out.append({"key": g["key"], "title": g["title"], "rule": g["rule"],
                    "n_days": len(g["days"]), "n_undercast": n_pos, "days": shown})
        print(f"  {g['title']:52s} {len(g['days']):>4} days, {n_pos:>2} undercast, "
              f"{len(shown)} shown")

    if not a.dry_run:
        os.makedirs(OUT_IMG, exist_ok=True)
        cuts = [{"key": k, "label": lab} for k, lab, _ in CUTS]
        with open(OUT_JSON, "w") as fh:
            json.dump({"cuts": cuts, "groups": out}, fh, indent=1)
        n = len([f for f in os.listdir(OUT_IMG) if f.endswith(".jpg")])
        mb = sum(os.path.getsize(f"{OUT_IMG}/{f}") for f in os.listdir(OUT_IMG)) / 1e6
        print(f"\nwrote {OUT_JSON} and {n} stills ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
