#!/usr/bin/env python3
"""Backfill detailed GPS tracks for activities in the White Mountains.

  python3 scripts/strava_streams.py           # fetch anything not cached yet
  python3 scripts/strava_streams.py --report  # summarize the cache, fetch nothing
  python3 scripts/strava_streams.py --refresh # re-fetch every candidate

Writes scripts/cache/wm_streams.json — committed, so trail coverage can be
recomputed in CI without ever touching the Strava API.

Why not just use the summary polylines we already have
------------------------------------------------------
Strava's summary polyline is aggressively decimated: a ten-mile hike arrives as
a couple hundred points, which cuts switchbacks into straight lines and can sit
50-100 m off the real trail. That is fine for "did this hike touch Mt Eisenhower"
(strava_peaks.py matches at 120 m) but not for "what fraction of this trail have
I walked", where the error is the same size as the measurement.

The detailed latlng stream is ~1 point/second — roughly 20k points for a long
hike, far more than is needed. Each track is simplified to DP_TOLERANCE_M with
Douglas-Peucker before being stored, which drops the point count by an order of
magnitude while staying well inside the matching threshold coverage uses.

Only activities whose summary polyline reaches the WMNF bounding box are
fetched; the other ~55 candidates are hikes elsewhere in the country and would
be a pure waste of the rate limit.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

from strava_sync import API, CACHE_DIR, TRACKS_FILE, SKI_TRACKS_FILE, get_access_token
from strava_peaks import decode_polyline
from trackutil import crop_zones, load_privacy_zones
from wm_trails import BOUNDARY_FILE

STREAMS_FILE = CACHE_DIR / "wm_streams.json"

# What counts as having covered a trail on foot (or skins). Deliberately
# narrower than the ski cache this reads from: that file also holds nordic and
# alpine days, which belong to the summit list but not here — groomed touring
# loops and lift-served resort laps aren't hiking trails, and counting them
# would credit trail mileage that mostly isn't trail.
COVERAGE_SPORTS = {"Hike", "Snowshoe", "TrailRun", "BackcountrySki"}

# Simplification tolerance. Coverage matches at 35 m, so 5 m of geometric error
# is an order of magnitude below the thing it feeds and cannot change a verdict.
DP_TOLERANCE_M = 5.0

# Pad the forest's bounding box before deciding an activity is irrelevant.
# Cheap insurance: a trail that starts at a trailhead just outside the boundary
# still belongs, and this filter only has to be conservative, not exact.
BBOX_PAD_DEG = 0.05

EARTH_RADIUS_M = 6371008.8


def encode_polyline(coords):
    """Encode [(lat, lon), ...] to a Google-encoded polyline at 5 decimals."""
    out = []
    prev_lat = prev_lon = 0

    for lat, lon in coords:
        ilat, ilon = round(lat * 1e5), round(lon * 1e5)
        for delta in (ilat - prev_lat, ilon - prev_lon):
            value = ~(delta << 1) if delta < 0 else delta << 1
            while value >= 0x20:
                out.append(chr((0x20 | (value & 0x1F)) + 63))
                value >>= 5
            out.append(chr(value + 63))
        prev_lat, prev_lon = ilat, ilon

    return "".join(out)


def simplify(coords, tolerance_m):
    """Douglas-Peucker on lat/lon, with distances measured in meters.

    Latitude and longitude are projected to a local flat plane first: at 44 N a
    degree of longitude is only ~0.72 of a degree of latitude, and skipping that
    correction would let the algorithm keep east-west detail it should drop
    while discarding north-south detail it should keep.
    """
    if len(coords) < 3:
        return list(coords)

    lat_scale = math.pi * EARTH_RADIUS_M / 180.0
    lon_scale = lat_scale * math.cos(math.radians(coords[0][0]))
    pts = [(lon * lon_scale, lat * lat_scale) for lat, lon in coords]

    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]

    while stack:
        start, end = stack.pop()
        x1, y1 = pts[start]
        x2, y2 = pts[end]
        dx, dy = x2 - x1, y2 - y1
        span = math.hypot(dx, dy)

        worst = worst_i = 0
        for i in range(start + 1, end):
            x0, y0 = pts[i]
            if span == 0:
                dist = math.hypot(x0 - x1, y0 - y1)
            else:
                dist = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / span
            if dist > worst:
                worst, worst_i = dist, i

        if worst > tolerance_m:
            keep[worst_i] = True
            stack.append((start, worst_i))
            stack.append((worst_i, end))

    return [coords[i] for i in range(len(coords)) if keep[i]]


def forest_bbox():
    if not BOUNDARY_FILE.exists():
        sys.exit("No WMNF boundary cached. Run: python3 scripts/wm_trails.py")
    boundary = json.loads(BOUNDARY_FILE.read_text())
    lats = [p[0] for ring in boundary["outer"] for p in ring]
    lons = [p[1] for ring in boundary["outer"] for p in ring]
    return (
        min(lats) - BBOX_PAD_DEG,
        min(lons) - BBOX_PAD_DEG,
        max(lats) + BBOX_PAD_DEG,
        max(lons) + BBOX_PAD_DEG,
    )


def candidates():
    """Hikes, trail runs and ski tours whose summary track reaches the Whites."""
    tracks = {}
    for path in (TRACKS_FILE, SKI_TRACKS_FILE):
        if path.exists():
            tracks.update(json.loads(path.read_text()))
    if not tracks:
        sys.exit("No track caches. Run: python3 scripts/strava_sync.py")

    south, west, north, east = forest_bbox()
    hits = {}
    for activity_id, t in tracks.items():
        if t["sport"] not in COVERAGE_SPORTS:
            continue
        coords = decode_polyline(t["polyline"])
        if any(south <= la <= north and west <= lo <= east for la, lo in coords):
            hits[activity_id] = t

    return dict(sorted(hits.items(), key=lambda kv: kv[1]["date"]))


def fetch_stream(activity_id, token):
    """Return the latlng stream, or None if the activity has no GPS data.

    Sleeps and retries on 429 rather than giving up: the cache is written after
    every activity, so a rate-limit pause costs time and nothing else.
    """
    url = f"{API}/activities/{activity_id}/streams"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"keys": "latlng", "key_by_type": "true"}

    for attempt in range(6):
        resp = requests.get(url, headers=headers, params=params, timeout=60)

        if resp.status_code == 429:
            # Strava's short window is 15 minutes; wait for the next boundary
            # plus a little slack rather than guessing at a shorter backoff.
            wait = 15 * 60 - (int(time.time()) % (15 * 60)) + 10
            print(f"    rate limited, sleeping {wait // 60}m{wait % 60}s")
            time.sleep(wait)
            continue

        # A manual entry or a track-less activity: nothing to fetch, ever.
        if resp.status_code == 404:
            return None

        if resp.status_code == 200:
            return resp.json().get("latlng", {}).get("data") or None

        wait = min(10 * (attempt + 1), 60)
        print(f"    HTTP {resp.status_code}, retrying in {wait}s")
        time.sleep(wait)

    sys.exit(f"Gave up on activity {activity_id}. Re-run — progress is cached.")


def report(cached, wanted):
    have = [v for v in cached.values() if v.get("polyline")]
    empty = [v for v in cached.values() if not v.get("polyline")]
    points = sum(v["points"] for v in have)
    raw_points = sum(v["raw_points"] for v in have)

    print(f"\nCandidates in the Whites: {len(wanted)}")
    print(f"  fetched:            {len(cached)}")
    print(f"  with GPS:           {len(have)}")
    if empty:
        print(f"  without GPS:        {len(empty)}")
    print(f"  remaining to fetch: {len(wanted) - len(cached)}")

    if have:
        print(f"\n  stream points:      {raw_points:,}")
        print(f"  after simplifying:  {points:,} ({points / raw_points:.1%})")
        print(f"  cache size:         {STREAMS_FILE.stat().st_size / 1024:,.0f} KB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-fetch everything")
    parser.add_argument(
        "--report", action="store_true", help="summarize the cache without fetching"
    )
    parser.add_argument(
        "--limit", type=int, help="stop after this many fetches (for a trial run)"
    )
    args = parser.parse_args()

    wanted = candidates()
    cached = {}
    if STREAMS_FILE.exists() and not args.refresh:
        cached = json.loads(STREAMS_FILE.read_text())

    if args.report:
        report(cached, wanted)
        return

    missing = [i for i in wanted if i not in cached]
    if args.limit:
        missing = missing[: args.limit]

    if not missing:
        print(f"All {len(wanted)} Whites activities already cached.")
        report(cached, wanted)
        return

    print(f"{len(wanted)} candidates in the Whites; fetching {len(missing)}.")
    zones = load_privacy_zones()
    token = get_access_token()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for i, activity_id in enumerate(missing, 1):
        meta = wanted[activity_id]
        print(f"  [{i}/{len(missing)}] {meta['date']} {meta['name'][:40]}")

        data = fetch_stream(activity_id, token)
        entry = {
            "name": meta["name"],
            "date": meta["date"],
            "sport": meta["sport"],
        }
        if data:
            # Zones come off before simplification so a cropped gap can't be
            # bridged by a straight line through the middle of one.
            points = crop_zones([(p[0], p[1]) for p in data], zones)
            simplified = simplify(points, DP_TOLERANCE_M)
            entry["raw_points"] = len(data)
            entry["points"] = len(simplified)
            entry["polyline"] = encode_polyline(simplified)
            print(f"    {len(data):,} -> {len(simplified):,} points")
        else:
            entry["polyline"] = None
            print("    no GPS stream")

        cached[activity_id] = entry
        STREAMS_FILE.write_text(json.dumps(cached, separators=(",", ":"), sort_keys=True) + "\n")
        time.sleep(1)  # stay well clear of the burst limit

    report(cached, wanted)


if __name__ == "__main__":
    main()
