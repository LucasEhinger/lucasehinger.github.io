#!/usr/bin/env python3
"""Parse Mount Washington summit observer remarks into undercast features.

MWOBS observers hand-augment the KMWN METAR with a note describing cloud decks
BELOW the summit -- "TPS LWR" (tops lower) -- giving coverage and top height.
NOAA archives the full remark text in the ISD (Integrated Surface Database)
``REM`` field, so ~30 years of it is free.

This reads ISD global-hourly CSVs, takes the observation nearest local noon on
each day, and extracts features that might distinguish a genuine undercast (the
summit floating above a continuous deck) from the far more common "there is some
cloud somewhere below me".

Three remark eras, all handled:
    1996-present  canonical  TPS LWR BKN045
    1994-1996     spaced     TPS LWR SCT 60 ALQDS   /  TPS LWR 55 BKN
    1973-1993     AWY/SYN terse remarks -- no lower-deck reports, nothing to parse

Features per day (see FEATURES below): max coverage of the lower deck, its top
height and depth below the summit, layer count, direction/extent tokens
(ALQDS = all quadrants, DSNT = distant, BBLO = banks below), valley fog, and
whether the summit itself was in cloud -- which matters because you cannot see an
undercast from inside one.

Usage:
    python3 scripts/parse_summit_remarks.py --isd-dir <cache> --start 1994 --end 2025
    python3 scripts/parse_summit_remarks.py --isd-dir <cache> --evaluate
"""
import argparse
import csv
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

SUMMIT_FT = 6288
UTC, ET = ZoneInfo("UTC"), ZoneInfo("America/New_York")
# ISD splits KMWN across two station ids; the modern one starts ~2006.
ISD_IDS = ["72613014755", "72613099999"]
UA = "lucasehinger.github.io undercast research (lucasehinger@gmail.com)"

COVER_RANK = {"": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}
RANK_NAME = {v: k for k, v in COVER_RANK.items()}

# --- lower-deck report, three era variants -------------------------------
RX_CANON = re.compile(r"TPS LWR (FEW|SCT|BKN|OVC)(\d{3})")          # BKN045
RX_SPACED = re.compile(r"TPS LWR (FEW|SCT|BKN|OVC)\s+(\d{2,3})\b")  # SCT 60
RX_REVERSED = re.compile(r"TPS LWR (\d{2,3})\s+(FEW|SCT|BKN|OVC)")  # 55 BKN
# --- summit-in-cloud / visibility from the METAR body --------------------
RX_VV = re.compile(r"\bVV(\d{3}|///)")
# NB: KMWN reports "BKN///" (layer of unknown height) even with 130SM visibility,
# so it is NOT evidence the summit is in cloud. Do not use it for that.
RX_VIS_FRAC = re.compile(r"\b(?:(\d+) )?(\d+)/(\d+)SM\b")   # 1/16SM, 1 1/2SM
RX_VIS_WHOLE = re.compile(r"\b(\d{1,3})SM\b")
RX_FOG = re.compile(r"(?<!BC)(?<!PR)(?<!FZ)\bFG\b|\bFZFG\b")
RX_PARTIAL_FOG = re.compile(r"\b(BCFG|PRFG)\b")


def parse_vis(body):
    """Visibility in statute miles; handles 1/16SM and 1 1/2SM fractions."""
    m = RX_VIS_FRAC.search(body)
    if m:
        whole = int(m.group(1)) if m.group(1) else 0
        return round(whole + int(m.group(2)) / int(m.group(3)), 3)
    m = RX_VIS_WHOLE.search(body)
    return int(m.group(1)) if m else None

FEATURES = [
    "date", "ob_time_local", "minutes_from_noon", "era",
    "n_layers", "max_cover", "max_cover_rank", "tops_ft", "depth_below_ft",
    "lowest_tops_ft", "all_tops_below_summit",
    "overhead_max_cover", "overhead_lowest_ft", "overhead_ceiling_ft",
    "overhead_ceiling_msl", "lid_above", "acsl",
    "alqds", "dsnt", "directional", "bblo", "vly_fog", "binovc",
    "summit_in_cloud", "vis_sm", "sun_dimly_visible", "remark", "metar_body",
]

# Sky groups in the METAR BODY are heights ABOVE THE STATION (AGL), so every
# layer listed there sits above the summit -- the opposite datum to TPS LWR,
# which is MSL and below it. The "TPS LWR" prefix is what flips the reference.
RX_BODY_LAYER = re.compile(r'\b(FEW|SCT|BKN|OVC)(\d{3})\b')


def fetch_isd(year, cache_dir):
    """Return path to a cached ISD CSV for `year`, downloading if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    for sid in ISD_IDS:
        p = os.path.join(cache_dir, f"{year}_{sid}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            return p
    for sid in ISD_IDS:
        p = os.path.join(cache_dir, f"{year}_{sid}.csv")
        url = (f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{sid}.csv")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
        except Exception:
            continue
        if len(data) > 1000:
            with open(p, "wb") as f:
                f.write(data)
            time.sleep(1.5)
            return p
    return None


def parse_layers(rmk):
    """All lower-deck layers as (cover, tops_ft). Tops are ft MSL."""
    out = []
    for cov, h in RX_CANON.findall(rmk):
        out.append((cov, int(h) * 100))
    if not out:
        for cov, h in RX_SPACED.findall(rmk):
            out.append((cov, int(h) * 100))
        for h, cov in RX_REVERSED.findall(rmk):
            out.append((cov, int(h) * 100))
    return out


def era_of(year):
    if year >= 1996:
        return "canonical"
    if year >= 1994:
        return "spaced"
    return "terse"


def extract(rem, when, require_layers=True):
    """One METAR/REM string -> feature dict.

    With ``require_layers`` (the default) a report carrying no lower-deck
    ``TPS LWR`` group returns None -- the original noon-only behaviour. Pass
    False to get a row for every observation, with the lower-deck fields blank;
    the all-hours record needs those rows as negatives.
    """
    text = (rem or "").upper()
    text = re.sub(r"^[A-Z]{3}\d{3}", "", text)          # strip MET081 / AWY008
    body, _, rmk = text.partition(" RMK ")
    rmk = rmk.rstrip("=;? ")
    layers = parse_layers(rmk if rmk else text)
    if not layers and require_layers:
        return None
    if layers:
        top_layer = max(layers, key=lambda t: (COVER_RANK[t[0]], -t[1]))
        cov, tops = top_layer
    else:
        cov, tops = "", None
    vis = parse_vis(body)
    # Summit is in cloud when vertical visibility is reported or fog is at the
    # station, corroborated by visibility actually being poor. On undercast days
    # the deck top sits at summit level and the summit drifts in and out, so this
    # is a state to record -- not a reason to exclude the day.
    in_cloud = bool(RX_VV.search(body) or RX_FOG.search(body)
                    or (vis is not None and vis < 1))
    # Cloud ABOVE the summit -- what turns a clean undercast into a sandwich.
    above = RX_BODY_LAYER.findall(body)
    ov_cov = max((c for c, _ in above), key=lambda c: COVER_RANK[c], default="")
    ov_low = min((int(h) * 100 for _, h in above), default=None)
    ceil = min((int(h) * 100 for c, h in above if c in ("BKN", "OVC")), default=None)
    return {
        "ob_time_local": when.strftime("%H:%M"),
        "era": era_of(when.year),
        "n_layers": len(layers),
        "max_cover": cov,
        "max_cover_rank": COVER_RANK[cov],
        "tops_ft": tops if tops is not None else "",
        "depth_below_ft": (SUMMIT_FT - tops) if tops is not None else "",
        "lowest_tops_ft": min((t for _, t in layers), default=""),
        "all_tops_below_summit": int(bool(layers) and all(t < SUMMIT_FT for _, t in layers)),
        "overhead_max_cover": ov_cov,
        "overhead_lowest_ft": ov_low if ov_low is not None else "",
        "overhead_ceiling_ft": ceil if ceil is not None else "",
        "overhead_ceiling_msl": (ceil + SUMMIT_FT) if ceil is not None else "",
        # a broken-or-worse layer overhead = lid on the sandwich
        "lid_above": int(ceil is not None),
        # standing lenticular: mid-level by definition, so necessarily overhead
        "acsl": int(bool(re.search(r'\b[ACS]CSL\b', rmk))),
        "alqds": int("ALQDS" in rmk or re.search(r"\bAQ\b", rmk) is not None),
        "dsnt": int("DSNT" in rmk),
        "directional": int(re.search(r"\b[NS]?[EW]?-[NS]?[EW]?\b", rmk) is not None
                           or re.search(r"\b(N|S|E|W|NE|NW|SE|SW)\b", rmk) is not None),
        "bblo": int("BBLO" in rmk),
        "vly_fog": int("VLY" in rmk),
        "binovc": int("BINOVC" in rmk),
        "summit_in_cloud": int(in_cloud),
        "vis_sm": vis if vis is not None else "",
        "sun_dimly_visible": int("SUN DMLY VSBL" in rmk),
        "remark": rmk[:120],
        "metar_body": body.strip()[:90],
    }


def noon_rows(path):
    """Yield (local_date, local_dt, REM) for the ob nearest local noon each day."""
    best = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            rem = r.get("REM")
            if not rem:
                continue
            try:
                t = datetime.strptime(r["DATE"], "%Y-%m-%dT%H:%M:%S")
            except (ValueError, KeyError):
                continue
            t = t.replace(tzinfo=UTC).astimezone(ET)
            d = t.strftime("%Y-%m-%d")
            delta = abs((t.hour * 60 + t.minute) - 720)
            if d not in best or delta < best[d][0]:
                best[d] = (delta, t, rem)
    for d in sorted(best):
        delta, t, rem = best[d]
        if delta <= 60:                     # within an hour of noon, else skip
            yield d, t, rem, delta


def build(years, cache_dir, out_path):
    rows = []
    for y in years:
        p = fetch_isd(y, cache_dir)
        if not p:
            print(f"  {y}: no ISD file")
            continue
        n = 0
        for d, t, rem, delta in noon_rows(p):
            feat = extract(rem, t)
            if not feat:
                continue
            feat["date"] = d
            feat["minutes_from_noon"] = delta
            rows.append(feat)
            n += 1
        print(f"  {y}: {n} noon days with a lower-deck report")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FEATURES)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["date"]):
            w.writerow({k: r.get(k, "") for k in FEATURES})
    print(f"\nwrote {out_path}  ({len(rows)} rows)")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--isd-dir", required=True, help="cache dir for ISD CSVs")
    p.add_argument("--start", type=int, default=1994)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--out", default="files/weather/obs/summit_remarks_noon.csv")
    args = p.parse_args()
    build(range(args.start, args.end + 1), args.isd_dir, args.out)


if __name__ == "__main__":
    main()
