#!/usr/bin/env python3
"""Build the observation viewer: webcam stills paired with the observer's words.

The point of the viewer is to make Step 2 of the write-up concrete. For each
example day it shows what the two cameras saw at local noon, what the human
scorer called it, the METAR the summit observer filed within minutes of that
frame, and whether the automated screen agreed. A reader can then see for
themselves why `TPS LWR BKN045` and a 130 SM visibility means "sea of cloud",
and why the borderline cases are genuinely hard.

Source images live under files/weather/webcam/, which is gitignored -- ~300 MB
of stills has no business in a Pages repo. This copies a curated subset,
downscaled, into files/weather/examples/observations/ where it can be served.

Selection is deliberately not "the prettiest days":
  * every hand-labeled undercast day (Avg >= 0.5)
  * a seasonal spread of clear negatives (Avg == 0)
  * the partial-agreement days (Avg == 0.25), where one camera showed undercast
    and the other did not -- the cases that set the ceiling on what any
    classifier can do

Usage:
    python3 scripts/build_observation_gallery.py --metar-cache <dir>
    python3 scripts/build_observation_gallery.py --metar-cache <dir> --dry-run
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd
from PIL import Image

WEBCAM = "files/weather/webcam"
OUT_IMG = "files/weather/examples/observations"
OUT_JSON = "files/weather/examples/observations/manifest.json"
LABELS = "files/weather/csv/MtWashington_undercast_orig.csv"
RECORD = "files/weather/obs/undercast_record.csv"
MAX_W = 1000
QUALITY = 70


def load_hand(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    out = {}
    for r in df.to_dict("records"):
        try:
            d = datetime.strptime(str(r["Short Date"]), "%m/%d/%y").strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        try:
            avg = float(r["Avg"])
        except (TypeError, ValueError):
            continue
        out[d] = {"avg": avg, "tower": str(r.get("Tower", "")),
                  "obs": str(r.get("Observatory", ""))}
    return out


def load_metar(cache_dir, dates):
    """Nearest-to-local-noon raw METAR for each wanted date.

    The cache is in UTC; local noon is 16-17 UTC depending on daylight time, and
    KMWN files at :5x, so the report that matches a noon frame is the one near
    16:5x (EDT) or 17:5x (EST). Rather than hard-code the offset, every report
    for the date is scored against 12:00 local after conversion.
    """
    from zoneinfo import ZoneInfo
    UTC, ET = ZoneInfo("UTC"), ZoneInfo("America/New_York")
    want = set(dates)
    best = {}
    for p in sorted(glob.glob(os.path.join(cache_dir, "metar_*.csv"))):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
                except (ValueError, KeyError):
                    continue
                lt = t.replace(tzinfo=UTC).astimezone(ET)
                d = lt.strftime("%Y-%m-%d")
                if d not in want:
                    continue
                delta = abs((lt.hour * 60 + lt.minute) - 720)
                if d not in best or delta < best[d][0]:
                    best[d] = (delta, r["metar"], lt.strftime("%H:%M"))
    return {d: {"metar": v[1], "local_time": v[2], "minutes_from_noon": v[0]}
            for d, v in best.items()}


def load_record(path, dates):
    """The screen's own verdict on the observation nearest noon for each date."""
    want = set(dates)
    best = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = r["valid_local"][:10]
            if d not in want:
                continue
            hhmm = r["valid_local"][11:16]
            delta = abs(int(hhmm[:2]) * 60 + int(hhmm[3:]) - 720)
            if d not in best or delta < best[d][0]:
                best[d] = (delta, r)
    return {d: {"label": v[1]["label"], "vis_sm": v[1]["vis_sm"],
                "max_cover": v[1]["max_cover"], "tops_ft": v[1]["tops_ft"],
                "summit_in_cloud": v[1]["summit_in_cloud"]}
            for d, v in best.items()}


def pick(hand, n_clear, n_partial):
    """Every undercast day, plus a seasonal spread of the other two classes."""
    und = sorted(d for d, v in hand.items() if v["avg"] >= 0.5)
    partial = sorted(d for d, v in hand.items() if 0 < v["avg"] < 0.5)
    clear = sorted(d for d, v in hand.items() if v["avg"] == 0)

    def spread(days, n):
        # Even stride through the date-ordered list: a seasonal spread without
        # hand-picking, so the negatives are not all mid-summer blue sky.
        if len(days) <= n:
            return days
        step = len(days) / n
        return [days[int(i * step)] for i in range(n)]

    return ([(d, "undercast") for d in und]
            + [(d, "partial") for d in spread(partial, n_partial)]
            + [(d, "clear") for d in spread(clear, n_clear)])


def copy_image(cam, date, out_dir, dry):
    src = os.path.join(WEBCAM, cam, f"{date}.jpg")
    if not os.path.exists(src):
        return None
    suffix = "Tower" if cam == "tower" else "Obs"
    name = f"{date}_{suffix}.jpg"
    if dry:
        return name
    im = Image.open(src)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)),
                       Image.LANCZOS)
    im.save(os.path.join(out_dir, name), "JPEG", quality=QUALITY, optimize=True)
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metar-cache", required=True)
    p.add_argument("--out-img", default=OUT_IMG)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--clear", type=int, default=12, help="how many clear days")
    p.add_argument("--partial", type=int, default=6, help="how many split-verdict days")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    hand = load_hand(LABELS)
    chosen = pick(hand, a.clear, a.partial)
    dates = [d for d, _ in chosen]
    metar = load_metar(a.metar_cache, dates)
    rec = load_record(RECORD, dates)
    print(f"selected {len(chosen)} days "
          f"({sum(1 for _, k in chosen if k=='undercast')} undercast, "
          f"{sum(1 for _, k in chosen if k=='partial')} split, "
          f"{sum(1 for _, k in chosen if k=='clear')} clear)")
    print(f"METAR found for {len(metar)}/{len(dates)}; "
          f"record row for {len(rec)}/{len(dates)}")

    if not a.dry_run:
        os.makedirs(a.out_img, exist_ok=True)
    entries, skipped = [], []
    for d, kind in chosen:
        t = copy_image("tower", d, a.out_img, a.dry_run)
        o = copy_image("observatory", d, a.out_img, a.dry_run)
        if not t and not o:
            skipped.append(d)
            continue
        m = metar.get(d, {})
        r = rec.get(d, {})
        entries.append({
            "date": d, "kind": kind,
            "hand_avg": hand[d]["avg"],
            "hand_tower": hand[d]["tower"], "hand_obs": hand[d]["obs"],
            "tower": t, "observatory": o,
            "metar": m.get("metar", ""), "local_time": m.get("local_time", ""),
            "screen": r.get("label", ""), "vis_sm": r.get("vis_sm", ""),
            "max_cover": r.get("max_cover", ""), "tops_ft": r.get("tops_ft", ""),
            "summit_in_cloud": r.get("summit_in_cloud", ""),
        })
    if skipped:
        print(f"no image on disk for {len(skipped)} days: {skipped[:6]}")

    entries.sort(key=lambda e: e["date"])
    agree = sum(1 for e in entries
                if (e["screen"] == "undercast") == (e["hand_avg"] >= 0.5))
    print(f"{len(entries)} entries; screen agrees with the human on "
          f"{agree}/{len(entries)}")
    if a.dry_run:
        print("dry run -- nothing written")
        for e in entries[:4]:
            print(f"  {e['date']} {e['kind']:9s} hand={e['hand_avg']} "
                  f"screen={e['screen']:9s} {e['metar'][:70]}")
        return
    with open(a.out_json, "w") as fh:
        json.dump({"generated": datetime.now().strftime("%Y-%m-%d"),
                   "entries": entries}, fh, indent=1)
    mb = sum(os.path.getsize(os.path.join(a.out_img, f))
             for f in os.listdir(a.out_img)) / 1e6
    print(f"wrote {a.out_json} and {len(entries)*2} images ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
