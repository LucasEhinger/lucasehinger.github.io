#!/usr/bin/env python3
"""Work out which named summits a hike passed over.

Reads the raw Strava cache written by strava_sync.py, decodes each hike's
track, and matches it against named peaks from OpenStreetMap (fetched via
Overpass). A peak counts as climbed if the track passes within THRESHOLD_M of
it — once per activity, no matter how many times the track loops the summit.

  python3 scripts/strava_peaks.py             # uses the cached OSM peak data
  python3 scripts/strava_peaks.py --refresh   # re-query Overpass

Writes files/strava/peaks.json.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import peak_lists  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "local"

# Committed so CI can rematch without the (gitignored) raw Strava cache.
CACHE_DIR = REPO / "scripts" / "cache"
TRACKS_FILE = CACHE_DIR / "tracks.json"
SKI_TRACKS_FILE = CACHE_DIR / "ski_tracks.json"
OSM_CACHE = CACHE_DIR / "osm_peaks.json"

OUT_FILE = REPO / "files" / "strava" / "peaks.json"

# The public Overpass instances throttle hard and time out often, so rotate
# between mirrors rather than hammering one. Tile results are cached as they
# arrive, so an interrupted run just resumes where it left off.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
# Overpass returns 406 to the default python-requests User-Agent.
OVERPASS_HEADERS = {"User-Agent": "lucasehinger.github.io hike-stats/1.0"}

HIKE_TYPES = {"Hike", "Snowshoe", "TrailRun"}

# How close the track must come to a summit to count as having climbed it.
# Strava's summary polyline is decimated, so this can't be too tight; 120 m is
# loose enough to survive that but tight enough to reject a peak you merely
# walked past in the valley below.
THRESHOLD_M = 120

# Peaks are fetched in whole-degree tiles so nearby hikes share a query.
TILE = 0.5
EARTH_RADIUS_M = 6371008.8
FEET_PER_METER = 3.280839895


def decode_polyline(encoded):
    """Decode a Google-encoded polyline into [(lat, lon), ...]."""
    coords = []
    index = lat = lon = 0

    while index < len(encoded):
        for axis in range(2):
            shift = result = 0
            while True:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if axis == 0:
                lat += delta
            else:
                lon += delta
        coords.append((lat / 1e5, lon / 1e5))

    return coords


def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def tiles_for(coords):
    """The set of TILE-degree tiles a track touches, padded by one threshold."""
    tiles = set()
    for lat, lon in coords:
        tiles.add((math.floor(lat / TILE), math.floor(lon / TILE)))
    return tiles


def fetch_peaks(tiles, cached):
    """Fetch named OSM peaks for any tile we haven't already queried."""
    missing = [t for t in sorted(tiles) if f"{t[0]},{t[1]}" not in cached]
    if not missing:
        return cached

    print(f"Querying Overpass for {len(missing)} new tile(s)...")
    for i, (ty, tx) in enumerate(missing, 1):
        south, west = ty * TILE, tx * TILE
        north, east = south + TILE, west + TILE
        # Pad slightly so a summit just outside the tile still matches.
        query = (
            "[out:json][timeout:180];"
            f'node["natural"="peak"]["name"]'
            f"({south - 0.02},{west - 0.02},{north + 0.02},{east + 0.02});"
            "out;"
        )

        resp = None
        for attempt in range(9):
            url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
            try:
                resp = requests.post(
                    url,
                    data=query.encode("utf-8"),
                    headers=OVERPASS_HEADERS,
                    timeout=200,
                )
            except requests.RequestException as exc:
                print(f"  tile {ty},{tx}: {type(exc).__name__}, trying next mirror")
                time.sleep(5)
                continue

            if resp.status_code == 200:
                break

            wait = min(10 * (attempt + 1), 45)
            print(f"  tile {ty},{tx}: HTTP {resp.status_code}, retrying in {wait}s")
            time.sleep(wait)
        else:
            sys.exit(
                f"Every mirror failed for tile {ty},{tx}. "
                "Re-run later — completed tiles are cached."
            )

        elements = resp.json().get("elements", [])
        cached[f"{ty},{tx}"] = [
            {
                "id": e["id"],
                "name": e["tags"]["name"],
                "lat": e["lat"],
                "lon": e["lon"],
                "ele": e["tags"].get("ele"),
            }
            for e in elements
            if e.get("tags", {}).get("name")
        ]
        print(f"  [{i}/{len(missing)}] tile {ty},{tx}: {len(elements)} peaks")

        OSM_CACHE.write_text(json.dumps(cached) + "\n")
        time.sleep(2)  # be polite to a free public API

    return cached


def parse_elevation(raw):
    """OSM 'ele' is free text; keep it only when it's a plain number of meters."""
    if raw is None:
        return None
    try:
        return round(float(str(raw).split()[0]) * FEET_PER_METER)
    except (ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="discard the cached OSM peak data"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_M,
        help=f"match distance in meters (default {THRESHOLD_M})",
    )
    args = parser.parse_args()

    if not TRACKS_FILE.exists():
        sys.exit("No track cache. Run: python3 scripts/strava_sync.py")

    # Ski tours count as summits too — earning a peak on skins is still earning
    # it. They live in a separate cache so that route grouping doesn't see them.
    tracks = json.loads(TRACKS_FILE.read_text())
    if SKI_TRACKS_FILE.exists():
        tracks.update(json.loads(SKI_TRACKS_FILE.read_text()))

    hikes = []
    for activity_id, t in tracks.items():
        hikes.append(
            {
                "id": int(activity_id),
                "name": t["name"],
                "date": t["date"],
                "sport": t["sport"],
                "coords": decode_polyline(t["polyline"]),
            }
        )

    print(f"{len(hikes)} tracks to match (hikes, trail runs and ski tours).")

    all_tiles = set()
    for h in hikes:
        all_tiles |= tiles_for(h["coords"])

    osm = {} if args.refresh else (
        json.loads(OSM_CACHE.read_text()) if OSM_CACHE.exists() else {}
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    osm = fetch_peaks(all_tiles, osm)

    peaks_by_id = {}
    for tile_peaks in osm.values():
        for p in tile_peaks:
            peaks_by_id[p["id"]] = p
    print(f"{len(peaks_by_id)} named peaks in range of those tracks.")

    # Fold in the curated lists. Where a list peak already exists in OSM the
    # two are merged (canonical name wins); where it doesn't, the list entry
    # becomes its own candidate so the summit still counts.
    list_membership = {}   # peak id -> set of list keys
    list_rosters = {}      # list key -> [(name, peak id)] in list order

    for list_key, spec in peak_lists.LISTS.items():
        roster = []
        for item in spec["peaks"]:
            # Entries are (name, lat, lon) or (name, lat, lon, elev_ft) where
            # OSM has no node to read an elevation from.
            name, lat, lon = item[0], item[1], item[2]
            listed_elev = item[3] if len(item) > 3 else None
            match = None
            best = peak_lists.MATCH_RADIUS_M
            for p in peaks_by_id.values():
                d = haversine_m(lat, lon, p["lat"], p["lon"])
                if d <= best:
                    best, match = d, p

            if match is None:
                synthetic_id = f"{list_key}:{name}"
                peaks_by_id[synthetic_id] = {
                    "id": synthetic_id,
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "ele": None,
                    "elev_ft": listed_elev,
                }
                match = peaks_by_id[synthetic_id]
            else:
                # Prefer the canonical name: OSM calls East Osceola "East Peak".
                match["name"] = name
                if listed_elev and not match.get("ele"):
                    match["elev_ft"] = listed_elev

            list_membership.setdefault(match["id"], set()).add(list_key)
            roster.append((name, match["id"]))

        list_rosters[list_key] = roster

    # Bucket peaks by tile so each track only tests nearby summits.
    buckets = {}
    for p in peaks_by_id.values():
        key = (math.floor(p["lat"] / TILE), math.floor(p["lon"] / TILE))
        buckets.setdefault(key, []).append(p)

    ascents = {}  # peak id -> list of activities
    for h in hikes:
        candidates = {}
        for tile in tiles_for(h["coords"]):
            ty, tx = tile
            # Include neighbouring tiles so summits near an edge aren't missed.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for p in buckets.get((ty + dy, tx + dx), []):
                        candidates[p["id"]] = p

        for peak in candidates.values():
            hit = any(
                haversine_m(peak["lat"], peak["lon"], lat, lon) <= args.threshold
                for lat, lon in h["coords"]
            )
            # Recorded once per activity even if the track crosses it twice.
            if hit:
                ascents.setdefault(peak["id"], []).append(
                    {
                        "id": h["id"],
                        "name": h["name"],
                        "date": h["date"],
                        "sport": h["sport"],
                    }
                )

    peaks = []
    entry_by_id = {}
    for peak_id, climbs in ascents.items():
        peak = peaks_by_id[peak_id]
        climbs.sort(key=lambda c: c["date"], reverse=True)
        entry = {
            "name": peak["name"],
            "lat": round(peak["lat"], 5),
            "lon": round(peak["lon"], 5),
            "region": peak_lists.region_for(peak["lat"], peak["lon"]),
            "count": len(climbs),
            "first": climbs[-1]["date"],
            "last": climbs[0]["date"],
            "ascents": climbs,
        }
        elevation = parse_elevation(peak.get("ele")) or peak.get("elev_ft")
        if elevation:
            entry["elev_ft"] = elevation
        if peak_id in list_membership:
            entry["lists"] = sorted(list_membership[peak_id])
        peaks.append(entry)
        entry_by_id[peak_id] = entry

    peaks.sort(key=lambda p: (-(p.get("elev_ft") or 0), p["name"]))

    # The roster carries every list peak, climbed or not, so the page can show
    # progress against the full list rather than only what's been ticked off.
    lists_payload = {}
    for list_key, spec in peak_lists.LISTS.items():
        members = []
        for name, peak_id in list_rosters[list_key]:
            entry = entry_by_id.get(peak_id)
            member = {"name": name, "count": entry["count"] if entry else 0}
            source = entry or peaks_by_id[peak_id]
            if entry and "elev_ft" in entry:
                member["elev_ft"] = entry["elev_ft"]
            elif not entry:
                elevation = parse_elevation(source.get("ele")) or source.get("elev_ft")
                if elevation:
                    member["elev_ft"] = elevation
            if entry:
                member["last"] = entry["last"]
            members.append(member)

        climbed = sum(1 for m in members if m["count"])
        lists_payload[list_key] = {
            "label": spec["label"],
            "total": len(members),
            "climbed": climbed,
            "members": members,
        }
        print(f"  {spec['label']}: {climbed} / {len(members)}")

    # Only regions that actually contain a summit, in the order declared, so
    # the page never offers a filter that returns nothing.
    region_counts = {}
    for p in peaks:
        region_counts[p["region"]] = region_counts.get(p["region"], 0) + 1
    ordered = [key for key, _, _ in peak_lists.REGIONS] + ["other"]
    regions_payload = [
        {
            "key": key,
            "label": peak_lists.REGION_LABELS[key],
            "phrase": peak_lists.REGION_PHRASES[key],
            "count": region_counts[key],
        }
        for key in ordered
        if key in region_counts
    ]

    payload = {
        "threshold_m": args.threshold,
        "unique_peaks": len(peaks),
        "total_ascents": sum(p["count"] for p in peaks),
        "lists": lists_payload,
        "regions": regions_payload,
        "peaks": peaks,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    print(
        f"\n{len(peaks)} unique named peaks, "
        f"{payload['total_ascents']} ascents total."
    )
    print(f"Wrote {OUT_FILE.relative_to(REPO)}")
    print("\nTop by elevation:")
    for p in peaks[:12]:
        elev = f"{p['elev_ft']:,} ft" if "elev_ft" in p else "elev unknown"
        print(f"  {p['name']:<32} {elev:>14}   x{p['count']}")


if __name__ == "__main__":
    main()
