#!/usr/bin/env python3
"""Fetch the White Mountain National Forest trail network from OpenStreetMap.

  python3 scripts/wm_trails.py            # fetch anything not already cached
  python3 scripts/wm_trails.py --refresh  # re-fetch every tile from scratch
  python3 scripts/wm_trails.py --report   # summarize the cache, fetch nothing

Writes two committed caches:

  scripts/cache/wm_boundary.json    the WMNF outline, as lat/lon rings
  scripts/cache/wm_trails_osm.json  named trail ways, keyed by tile

Both are cached because Overpass is slow and unreliable, and because CI must be
able to recompute coverage without touching the network at all. Nothing here
runs on the daily job — refresh it by hand when OSM has moved on.

Why a bbox and a Python clip, rather than an Overpass area query
---------------------------------------------------------------
The obvious query is `way[highway=path](area:3600331880)`, scoping straight to
the WMNF boundary relation. That relation exists and Overpass will happily
report an area id for it, but the area itself comes back empty — area
generation has failed for it upstream, and every way query against it returns
zero. So the ways are fetched by bounding box and clipped here instead, against
the boundary geometry pulled from the relation directly.

The bbox is a lot bigger than the forest: it reaches into Waterville and over
toward Evans Notch, and it catches trails in neither. The clip is what makes
the denominator mean "in the WMNF" rather than "near it".
"""

import argparse
import json
import math
import sys
import time

from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "scripts" / "cache"
BOUNDARY_FILE = CACHE_DIR / "wm_boundary.json"
TRAILS_FILE = CACHE_DIR / "wm_trails_osm.json"

# Same mirrors and User-Agent as strava_peaks.py — see the note there about
# the 406 the default python-requests agent earns.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
OVERPASS_HEADERS = {"User-Agent": "lucasehinger.github.io hike-stats/1.0"}

# The WMNF boundary relation.
WMNF_RELATION = 331880

# Generous enough to contain the forest's NH and ME units with room to spare;
# the boundary clip does the real work of deciding what's in.
BBOX = (43.70, -71.95, 44.55, -70.85)

# Ways are fetched in tiles so that one timeout costs one tile, not the run.
# At 0.25 deg this is ~20 requests, each small enough that Overpass answers it
# without the timeouts a whole-bbox `out geom` reliably provokes.
TILE = 0.25

# What counts as a trail. `track` catches the old logging roads that a good
# number of real trails are routed along; `bridleway` is rare here but free.
TRAIL_HIGHWAYS = ("path", "footway", "track", "bridleway")

EARTH_RADIUS_M = 6371008.8
METERS_PER_MILE = 1609.344


def overpass(query, label):
    """POST a query, rotating mirrors and backing off until one answers."""
    for attempt in range(9):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            resp = requests.post(
                url,
                data=query.encode("utf-8"),
                headers=OVERPASS_HEADERS,
                timeout=300,
            )
        except requests.RequestException as exc:
            print(f"  {label}: {type(exc).__name__}, trying next mirror")
            time.sleep(5)
            continue

        # Overpass signals overload with a 200 and an HTML error body, so the
        # status code alone doesn't tell you whether this worked.
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                print(f"  {label}: mirror returned an error page, retrying")
                time.sleep(10)
                continue

        wait = min(10 * (attempt + 1), 45)
        print(f"  {label}: HTTP {resp.status_code}, retrying in {wait}s")
        time.sleep(wait)

    sys.exit(f"Every mirror failed for {label}. Re-run later — progress is cached.")


def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def way_length_m(coords):
    return sum(
        haversine_m(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1)
    )


def assemble_rings(segments):
    """Stitch relation member ways into closed rings, end to end.

    OSM stores a boundary as a pile of ways in arbitrary order and arbitrary
    direction, so they have to be walked into rings before anything can be
    tested against them. A ring that won't close (a gap in the relation) is
    still kept — treated as closed, it's a good enough polygon for a clip that
    only has to answer "roughly, is this trail in the forest".
    """
    pool = [list(map(tuple, s)) for s in segments if len(s) > 1]
    rings = []

    while pool:
        ring = pool.pop(0)
        extended = True
        while extended and ring[0] != ring[-1]:
            extended = False
            for i, seg in enumerate(pool):
                if seg[0] == ring[-1]:
                    ring.extend(seg[1:])
                elif seg[-1] == ring[-1]:
                    ring.extend(reversed(seg[:-1]))
                elif seg[-1] == ring[0]:
                    ring[:0] = seg[:-1]
                elif seg[0] == ring[0]:
                    ring[:0] = list(reversed(seg[1:]))
                else:
                    continue
                pool.pop(i)
                extended = True
                break
        rings.append(ring)

    return rings


def fetch_boundary(refresh):
    if BOUNDARY_FILE.exists() and not refresh:
        return json.loads(BOUNDARY_FILE.read_text())

    print(f"Fetching WMNF boundary (relation {WMNF_RELATION})...")
    data = overpass(
        f"[out:json][timeout:300];rel({WMNF_RELATION});out geom;",
        "boundary",
    )

    outer, inner = [], []
    for element in data.get("elements", []):
        for member in element.get("members", []):
            geometry = member.get("geometry")
            if member["type"] != "way" or not geometry:
                continue
            coords = [(p["lat"], p["lon"]) for p in geometry]
            (inner if member.get("role") == "inner" else outer).append(coords)

    if not outer:
        sys.exit("Boundary relation came back with no outer ways.")

    boundary = {
        "relation": WMNF_RELATION,
        "outer": [[list(p) for p in r] for r in assemble_rings(outer)],
        "inner": [[list(p) for p in r] for r in assemble_rings(inner)],
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BOUNDARY_FILE.write_text(json.dumps(boundary, separators=(",", ":")) + "\n")
    print(
        f"  {len(boundary['outer'])} outer ring(s), "
        f"{len(boundary['inner'])} inner ring(s)"
    )
    return boundary


def tiles_for_bbox():
    south, west, north, east = BBOX
    tiles = []
    y = math.floor(south / TILE)
    while y * TILE < north:
        x = math.floor(west / TILE)
        while x * TILE < east:
            tiles.append((y, x))
            x += 1
        y += 1
    return tiles


def fetch_trails(refresh):
    cached = {}
    if TRAILS_FILE.exists() and not refresh:
        cached = json.loads(TRAILS_FILE.read_text())

    tiles = tiles_for_bbox()
    missing = [t for t in tiles if f"{t[0]},{t[1]}" not in cached]
    if not missing:
        print(f"All {len(tiles)} tiles already cached.")
        return cached

    print(f"Fetching {len(missing)} of {len(tiles)} tiles...")
    highways = "|".join(TRAIL_HIGHWAYS)

    for i, (ty, tx) in enumerate(missing, 1):
        south, west = ty * TILE, tx * TILE
        north, east = south + TILE, west + TILE
        # Unnamed ways are skipped: they can't be grouped into a trail, named
        # or clicked on, and they're mostly driveways and social paths that
        # would pad the denominator with things nobody sets out to hike.
        query = (
            "[out:json][timeout:280];"
            f'way["highway"~"^({highways})$"]["name"]'
            f"({south},{west},{north},{east});"
            "out geom;"
        )

        data = overpass(query, f"tile {ty},{tx}")
        ways = []
        for element in data.get("elements", []):
            geometry = element.get("geometry")
            tags = element.get("tags", {})
            if not geometry or not tags.get("name"):
                continue
            ways.append(
                {
                    "id": element["id"],
                    "name": tags["name"].strip(),
                    "highway": tags.get("highway"),
                    # Five decimals is ~1 m — far finer than the 35 m matching
                    # threshold, and it halves the size of the cache.
                    "coords": [
                        [round(p["lat"], 5), round(p["lon"], 5)] for p in geometry
                    ],
                }
            )

        cached[f"{ty},{tx}"] = ways
        print(f"  [{i}/{len(missing)}] tile {ty},{tx}: {len(ways)} named ways")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TRAILS_FILE.write_text(json.dumps(cached, separators=(",", ":")) + "\n")
        time.sleep(2)  # be polite to a free public API

    return cached


def index_rings(rings):
    """Pair each ring with its bounding box.

    The WMNF relation assembles into 64 outer and 89 inner rings — the forest
    is a patchwork of parcels, not one blob. Ray casting every trail vertex
    against every ring is billions of segment tests; the bbox check rejects
    virtually all of them in four comparisons.
    """
    indexed = []
    for ring in rings:
        lats = [p[0] for p in ring]
        lons = [p[1] for p in ring]
        indexed.append((min(lats), min(lons), max(lats), max(lons), ring))
    return indexed


def point_in_rings(lat, lon, indexed):
    """Ray casting against bbox-indexed rings; True if inside an odd number."""
    inside = False
    for min_lat, min_lon, max_lat, max_lon, ring in indexed:
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            continue
        for i in range(len(ring) - 1):
            y1, x1 = ring[i]
            y2, x2 = ring[i + 1]
            if (y1 > lat) != (y2 > lat):
                if lon < x1 + (lat - y1) / (y2 - y1) * (x2 - x1):
                    inside = not inside
    return inside


def clip_to_boundary(tiles, boundary):
    """Keep ways with at least one vertex inside the forest.

    A trail that crosses the boundary is kept whole rather than cut at the
    line. Splitting it would strand the piece outside under the same name and
    make the per-trail percentages read strangely — "Trail X: 60%" when you've
    walked all of the part that's in the forest.
    """
    outer = index_rings(boundary["outer"])
    inner = index_rings(boundary["inner"])
    kept, seen = [], set()

    for ways in tiles.values():
        for way in ways:
            # Tiles overlap at their edges, and a way can be returned by more
            # than one, so dedupe on the OSM id.
            if way["id"] in seen:
                continue
            for lat, lon in way["coords"]:
                if point_in_rings(lat, lon, outer) and not point_in_rings(
                    lat, lon, inner
                ):
                    seen.add(way["id"])
                    kept.append(way)
                    break

    return kept


def report(tiles, boundary):
    total_ways = sum(len(w) for w in tiles.values())
    print(f"\nCached: {total_ways} named ways across {len(tiles)} tiles")

    kept = clip_to_boundary(tiles, boundary)
    miles = sum(way_length_m(w["coords"]) for w in kept) / METERS_PER_MILE
    names = {w["name"] for w in kept}
    points = sum(len(w["coords"]) for w in kept)

    print(f"Inside the WMNF boundary: {len(kept)} ways")
    print(f"  distinct trail names:   {len(names)}")
    print(f"  total trail mileage:    {miles:,.0f} mi")
    print(f"  total vertices:         {points:,}")

    # Rough guide to what the front end would have to download. Actual GeoJSON
    # carries per-feature overhead on top of this.
    raw_kb = points * 22 / 1024
    print(f"  ~raw coordinate bytes:  {raw_kb:,.0f} KB (before simplification)")

    by_name = {}
    for way in kept:
        by_name[way["name"]] = by_name.get(way["name"], 0) + way_length_m(way["coords"])

    print("\nLongest named trails:")
    for name, length_m in sorted(by_name.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {length_m / METERS_PER_MILE:6.1f} mi  {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="re-fetch every tile from scratch"
    )
    parser.add_argument(
        "--report", action="store_true", help="summarize the cache without fetching"
    )
    args = parser.parse_args()

    if args.report:
        if not TRAILS_FILE.exists() or not BOUNDARY_FILE.exists():
            sys.exit("Nothing cached yet — run without --report first.")
        report(json.loads(TRAILS_FILE.read_text()), json.loads(BOUNDARY_FILE.read_text()))
        return

    boundary = fetch_boundary(args.refresh)
    tiles = fetch_trails(args.refresh)
    report(tiles, boundary)


if __name__ == "__main__":
    main()
