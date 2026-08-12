#!/usr/bin/env python3
"""Match hiked tracks against the WMNF trail network and score coverage.

  python3 scripts/wm_coverage.py                # write files/strava/wm_coverage.json
  python3 scripts/wm_coverage.py --threshold 25 # try a tighter match distance
  python3 scripts/wm_coverage.py --report       # print, write nothing
  python3 scripts/wm_coverage.py --trail "Franconia Ridge"   # explain one trail

Reads only committed caches — the trail network from wm_trails.py and the
detailed tracks from strava_streams.py — so this runs in CI with no network.

How a trail is scored
---------------------
Each OSM way is densified to a point every DENSIFY_M, and a vertex counts as
walked if any track point falls within THRESHOLD_M of it. A segment between two
densified vertices scores its full length when both ends are covered and half
when only one is, which is what lets a turnaround halfway up a trail score ~50%
instead of the 0% a whole-way test would give it.

Track points are bucketed into a grid of THRESHOLD_M cells, so testing a vertex
touches only the nine cells around it rather than all 31k points. Without that
this is a 3-billion-comparison job; with it, it is a few seconds.

Choosing the threshold
----------------------
35 m is loose enough to absorb GPS error under tree cover and the ~5 m of
simplification in the stream cache, and tight enough that it will not credit a
parallel trail one drainage over. It is genuinely a tuning knob, though: run
--threshold to see how sensitive the totals are, and --trail to see which hikes
a specific trail is being credited to. Trails that run close together for real
(Tuckerman and Lion Head, the Falling Waters/Old Bridle Path pair) are the ones
to check when changing it.
"""

import argparse
import collections
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from strava_peaks import decode_polyline
from trackutil import write_json_stable
from strava_streams import simplify
from wm_trails import (
    BOUNDARY_FILE,
    TRAILS_FILE,
    clip_to_boundary,
    haversine_m,
    way_length_m,
    METERS_PER_MILE,
)

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "scripts" / "cache"
STREAMS_FILE = CACHE_DIR / "wm_streams.json"
OUT_FILE = REPO / "files" / "strava" / "wm_coverage.json"
GEOJSON_FILE = REPO / "files" / "strava" / "wm_trails.geojson"

# Display simplification for the map. Coarser than the 5 m used on the tracks,
# because this only has to look right: at the zoom where a 12 m wiggle would be
# visible you are looking at one ravine, not the forest. Scoring always runs on
# the full-resolution geometry, so this never moves a percentage.
DISPLAY_TOLERANCE_M = 12.0

# Visit counts are capped at this for display. Four separate trips over the same
# ground already reads as "often"; beyond that the ramp would need steps too
# close together to tell apart.
HEAT_MAX = 4

# Hand-drawn region polygons, keyed in the order they should be listed. The
# rings live in scripts/wm_regions.json so the map can draw the same boundary
# the scoring uses — a drawn line that isn't the rule would invite reading
# geography into something arbitrary.
REGIONS_FILE = Path(__file__).resolve().parent / "wm_regions.json"
REGIONS = json.loads(REGIONS_FILE.read_text())["regions"]

# Ways that fell outside every hand-drawn ring and had to fall back to the
# nearest one. Reported at the end of a run: a rising count means a polygon
# needs redrawing.
FALLBACK = {"ways": 0}

# How close a track has to pass to count as having walked that spot.
THRESHOLD_M = 35.0

# Spacing for resampling trail geometry. OSM ways have vertices wherever the
# mapper happened to click, sometimes hundreds of meters apart on a straight
# section; without resampling, a long straight segment is scored by its
# endpoints alone and coverage comes out quantized and wrong.
DENSIFY_M = 20.0

EARTH_RADIUS_M = 6371008.8

# Longest straight line between two consecutive track points that still counts
# as ground covered. Above this it's a dropout under tree cover or a paused
# activity resumed elsewhere, and the straight line between the two ends crosses
# terrain nobody walked. Real straight stretches — causeways, rail trails, the
# Lincoln Woods logging grade — sit well under this.
MAX_SEGMENT_M = 400.0

# A trail counts as "complete" a little short of 100%: the last few percent are
# usually a spur to a trailhead sign or a bit of geometry the track clipped.
COMPLETE_PCT = 95.0

# Least coverage that counts as having been on a trail at all. Crossing a trail
# at a junction picks up roughly two thresholds' worth of it — 70 m, plus a
# little from densification — so 160 m sits at a comfortable 2x margin above
# the noise while still admitting the genuinely short connector trails.
#
# A percentage floor can't do this job: 50 m of junction contact is 1% of a long
# trail but 11% of a quarter-mile cutoff, so any percentage strict enough to
# reject the second would throw away real hiking on the first.
TOUCHED_MIN_M = 160.0

# ...except that a flat distance floor can never be cleared by a trail shorter
# than itself. Nine trails came out 100% walked but "not hiked" under the flat
# rule — 113 m connectors and spur paths walked end to end. Walking all of a
# short trail is not the same act as crossing a long one, so a trail also counts
# as hiked when it is essentially finished.
#
# The second guard is what keeps that from readmitting the noise: a crossing
# picks up about two thresholds' worth (70 m), so anything at or under that is
# still indistinguishable from a junction brush no matter what percentage it
# works out to, and stays out.
COMPLETE_ENOUGH_PCT = 90.0
MIN_WALKED_M = 2 * THRESHOLD_M


def is_touched(covered_m, total_m):
    if covered_m >= TOUCHED_MIN_M:
        return True
    if covered_m <= MIN_WALKED_M or not total_m:
        return False
    return 100 * covered_m / total_m >= COMPLETE_ENOUGH_PCT


def projector(ref_lat):
    """Equirectangular lat/lon -> meters, good enough over a 100 km box."""
    lat_scale = math.pi * EARTH_RADIUS_M / 180.0
    lon_scale = lat_scale * math.cos(math.radians(ref_lat))

    def project(lat, lon):
        return (lon * lon_scale, lat * lat_scale)

    return project


def densify(coords, project, spacing_m):
    """Resample a way to a vertex at least every `spacing_m`, in lat/lon.

    Returns lat/lon rather than projected meters because these same points are
    what gets drawn: the map splits a trail at the boundary between walked and
    unwalked, and those boundaries only exist at this resolution. Interpolating
    in lat/lon is exact here — the added points are collinear with the segment
    they subdivide, so the drawn line still follows the original geometry.
    """
    if len(coords) < 2:
        return [tuple(c) for c in coords]

    out = [tuple(coords[0])]
    for a, b in zip(coords, coords[1:]):
        (x1, y1), (x2, y2) = project(*a), project(*b)
        steps = max(1, math.ceil(math.hypot(x2 - x1, y2 - y1) / spacing_m))
        for step in range(1, steps + 1):
            t = step / steps
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def coverage_runs(points, per_point):
    """Split a densified way into stretches of equal visit count.

    The count is how many *distinct* activities came within the threshold of
    that spot — `per_point` holds a set of activity ids, so a single hike that
    doubles back over itself still counts once, while two separate hikes over
    the same ground count twice. That is the difference between measuring how
    often ground was covered and measuring how far someone walked.

    Because the count is per point rather than per trail, hiking a loop twice by
    different halves leaves every point at 1: no stretch was actually repeated.

    Where the count changes across a segment the split lands at the segment's
    midpoint, the same split the scoring uses when it awards half credit.
    """
    runs = []

    def push(level, pair):
        if runs and runs[-1][0] == level:
            runs[-1][1].append(pair[1])
        else:
            runs.append((level, list(pair)))

    for i in range(len(points) - 1):
        a, b = len(per_point[i]), len(per_point[i + 1])
        p0, p1 = points[i], points[i + 1]
        if a == b:
            push(a, (p0, p1))
        else:
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            push(a, (p0, mid))
            push(b, (mid, p1))

    return runs


def build_track_index(streams, project, cell_m):
    """Bucket track *segments* into a grid keyed by (cell_x, cell_y).

    Segments, not points. The stored tracks are Douglas-Peucker simplified, so a
    straight stretch of trail keeps only its two endpoints — the retained points
    sit a median 36 m apart and sometimes far more. Matching trail vertices
    against track *points* therefore reported a trail as unwalked wherever it ran
    straight, purely because the nearest retained vertex was around the corner.
    Daniel Webster and Glen Boulder both scored in the 80s that way despite the
    track never leaving the trail; measured against the segments they are 100%.

    Each segment is stamped into every cell it passes through, sampling at half a
    cell so a diagonal crossing can't skip one.
    """
    index = collections.defaultdict(list)

    for activity_id, entry in streams.items():
        if not entry.get("polyline"):
            continue
        pts = [project(lat, lon) for lat, lon in decode_polyline(entry["polyline"])]

        for a, b in zip(pts, pts[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            # A segment this long is a GPS dropout or a paused activity, not a
            # walk in a straight line. Treating it as ground covered would credit
            # every trail lying under the shortcut it draws.
            if length > MAX_SEGMENT_M:
                continue

            segment = (a[0], a[1], b[0], b[1], activity_id)
            steps = max(1, int(length / (cell_m / 2)) + 1)
            seen = set()
            for i in range(steps + 1):
                t = i / steps
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                key = (int(x // cell_m), int(y // cell_m))
                if key not in seen:
                    seen.add(key)
                    index[key].append(segment)

    return index


def covering_activities(point, index, cell_m, threshold_m):
    """Activity ids whose track passes within threshold_m of `point`.

    The 3x3 cell scan is only exhaustive while cell_m == threshold_m, which is
    how main() calls it.
    """
    x, y = point
    cx, cy = int(x // cell_m), int(y // cell_m)
    limit = threshold_m * threshold_m
    hits = set()

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for x0, y0, x1, y1, activity_id in index.get((cx + dx, cy + dy), ()):
                if activity_id in hits:
                    continue
                sx, sy = x1 - x0, y1 - y0
                if sx == 0 and sy == 0:
                    d2 = (x - x0) ** 2 + (y - y0) ** 2
                else:
                    t = ((x - x0) * sx + (y - y0) * sy) / (sx * sx + sy * sy)
                    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                    d2 = (x - (x0 + t * sx)) ** 2 + (y - (y0 + t * sy)) ** 2
                if d2 <= limit:
                    hits.add(activity_id)
    return hits


def score_way(coords, index, project, cell_m, threshold_m):
    """Return (total_m, covered_m, {activity_id: meters}, runs) for one way."""
    ll = densify(coords, project, DENSIFY_M)
    if len(ll) < 2:
        return 0.0, 0.0, {}, []

    pts = [project(lat, lon) for lat, lon in ll]
    per_point = [covering_activities(p, index, cell_m, threshold_m) for p in pts]

    total = covered = 0.0
    by_activity = collections.defaultdict(float)

    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        length = math.hypot(x2 - x1, y2 - y1)
        total += length

        a, b = per_point[i], per_point[i + 1]
        if a and b:
            share = length
        elif a or b:
            # Half credit at the boundary: the true turnaround point sits
            # somewhere inside this segment, and splitting the difference is
            # unbiased where assigning all or nothing is not.
            share = length / 2
        else:
            continue

        covered += share
        # sorted(), not the raw set: iteration order over a set of strings
        # changes with the per-process hash seed, and float addition is not
        # associative, so an unsorted loop makes the output differ run to run.
        for activity_id in sorted(a | b):
            by_activity[activity_id] += share

    return total, covered, dict(by_activity), coverage_runs(ll, per_point)


def in_ring(lat, lon, ring):
    inside = False
    n = len(ring)
    for i in range(n):
        y1, x1 = ring[i]
        y2, x2 = ring[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat - y1) / (y2 - y1) * (x2 - x1):
                inside = not inside
    return inside


def regions_for(coords):
    """Every region a way passes through.

    Sampled along the whole way rather than at its midpoint: a trail that runs
    from one range into the next belongs to both, and the midpoint can only
    ever name one of them. This is also what removes the old race — with
    multiple membership there is no single winner to be decided by whichever
    OSM way happened to be read first.
    """
    step = max(1, len(coords) // 24)
    found = []
    for name, ring in REGIONS.items():
        for lat, lon in coords[::step] + [coords[-1]]:
            if in_ring(lat, lon, ring):
                found.append(name)
                break

    if found:
        return found

    # Hand-drawn rings leave slivers along the forest's ragged edge. Rather
    # than orphan a trail, fall back to the nearest ring's centre — and main()
    # reports how often this happens, because a large count means a polygon
    # needs redrawing, not that the fallback is working.
    FALLBACK["ways"] += 1
    lat, lon = coords[len(coords) // 2]
    nearest = min(
        REGIONS,
        key=lambda n: haversine_m(
            lat, lon,
            sum(p[0] for p in REGIONS[n]) / len(REGIONS[n]),
            sum(p[1] for p in REGIONS[n]) / len(REGIONS[n]),
        ),
    )
    return [nearest]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_M)
    parser.add_argument("--report", action="store_true", help="print, write nothing")
    parser.add_argument("--trail", help="show the hikes credited to one trail")
    args = parser.parse_args()

    for path in (BOUNDARY_FILE, TRAILS_FILE, STREAMS_FILE):
        if not path.exists():
            sys.exit(f"Missing {path.relative_to(REPO)} — run the fetch scripts first.")

    boundary = json.loads(BOUNDARY_FILE.read_text())
    tiles = json.loads(TRAILS_FILE.read_text())
    streams = json.loads(STREAMS_FILE.read_text())

    ways = clip_to_boundary(tiles, boundary)
    print(f"{len(ways)} ways inside the WMNF; {len(streams)} tracks to match.")

    project = projector(44.2)
    cell_m = args.threshold
    index = build_track_index(streams, project, cell_m)
    print(f"{sum(len(v) for v in index.values()):,} track points indexed.")

    trails = {}
    way_rows = []

    for way in ways:
        total, covered, by_activity, runs = score_way(
            way["coords"], index, project, cell_m, args.threshold
        )
        way_regions = regions_for(way["coords"])
        way_rows.append((way, total, covered, runs, way_regions))
        name = way["name"]
        trail = trails.setdefault(
            name,
            {
                "name": name,
                "meters": 0.0,
                "covered_m": 0.0,
                # A set, accumulated across every way — not decided by the
                # first one seen. That `setdefault` used to fix the region from
                # whichever way came out of the tile cache first, which filed
                # Kinsman Ridge Trail under Moosilauke.
                "regions": set(),
                "by_activity": collections.defaultdict(float),
            },
        )
        trail["meters"] += total
        trail["covered_m"] += covered
        trail["regions"].update(way_regions)
        for activity_id, meters in by_activity.items():
            trail["by_activity"][activity_id] += meters

    entries = []
    for trail in trails.values():
        pct = 100 * trail["covered_m"] / trail["meters"] if trail["meters"] else 0.0
        hikes = sorted(
            (
                {
                    "id": int(activity_id),
                    "name": streams[activity_id]["name"],
                    "date": streams[activity_id]["date"],
                    "sport": streams[activity_id]["sport"],
                    "miles": round(meters / METERS_PER_MILE, 2),
                }
                for activity_id, meters in trail["by_activity"].items()
                # Same test as `touched`, applied per hike: a hike that clipped
                # a trail at a junction crossed it, it did not hike it, and
                # listing it would bury the real hikes under noise. Sharing the
                # rule also keeps a short trail from being marked hiked while
                # showing an empty hike list.
                if is_touched(meters, trail["meters"])
            ),
            # Date alone leaves same-day hikes tied, and the tie was being
            # broken by dict insertion order, which traces back to set iteration
            # and therefore to the hash seed. That alone rewrote this file on
            # every run and produced a commit a day with no new hiking in it.
            key=lambda h: (h["date"], h["id"]),
        )
        entries.append(
            {
                "name": trail["name"],
                "regions": sorted(trail["regions"]),
                "miles": round(trail["meters"] / METERS_PER_MILE, 2),
                "covered_miles": round(trail["covered_m"] / METERS_PER_MILE, 2),
                "pct": round(pct, 1),
                # Mileage stays honest either way — walking 50 m of a trail is
                # still 50 m walked — but a trail you merely crossed shouldn't
                # be counted, or drawn, as one you've been on.
                "touched": is_touched(trail["covered_m"], trail["meters"]),
                "hikes": hikes,
            }
        )

    entries.sort(key=lambda e: (-e["pct"], -e["miles"]))

    total_mi = sum(e["miles"] for e in entries)
    covered_mi = sum(e["covered_miles"] for e in entries)
    walked = [e for e in entries if e["touched"]]
    crossed = [e for e in entries if e["pct"] > 0 and not e["touched"]]
    complete = [e for e in entries if e["pct"] >= COMPLETE_PCT]

    if args.trail:
        needle = args.trail.lower()
        for entry in entries:
            if needle in entry["name"].lower():
                print(f"\n{entry['name']} — {entry['pct']}% of {entry['miles']} mi")
                for hike in entry["hikes"]:
                    print(f"  {hike['date']}  {hike['miles']:5.2f} mi  {hike['name']}")
        return

    print(f"\nWMNF trail coverage (threshold {args.threshold:.0f} m)")
    print(f"  named trails:      {len(entries)}")
    print(f"  total mileage:     {total_mi:,.0f} mi")
    print(f"  walked mileage:    {covered_mi:,.1f} mi  ({100 * covered_mi / total_mi:.1f}%)")
    print(f"  trails hiked:      {len(walked)}")
    print(f"  merely crossed:    {len(crossed)}  (too little of them to count)")
    print(f"  trails >= {COMPLETE_PCT:.0f}%:    {len(complete)}")

    # Mileage per region comes from the ways, so a trail straddling a boundary
    # contributes each stretch to the region that stretch is actually in and the
    # regions still sum to the forest total. Trail counts are membership-based,
    # so that same trail is listed in both — those deliberately do not sum.
    region_meters = collections.defaultdict(float)
    region_covered = collections.defaultdict(float)
    for way, total, covered, _runs, way_regions in way_rows:
        share = len(way_regions)
        for region in way_regions:
            region_meters[region] += total / share
            region_covered[region] += covered / share

    region_rows = []
    for region in REGIONS:
        subset = [e for e in entries if region in e["regions"]]
        miles = region_meters[region] / METERS_PER_MILE
        if not subset or not miles:
            continue
        done = region_covered[region] / METERS_PER_MILE
        region_rows.append(
            {
                "name": region,
                "miles": round(miles, 1),
                "covered_miles": round(done, 1),
                "pct": round(100 * done / miles, 1),
                "trails": len(subset),
                "trails_hiked": sum(1 for e in subset if e["touched"]),
                "trails_complete": sum(1 for e in subset if e["pct"] >= COMPLETE_PCT),
                # Shipped so the map can draw the same ring the scoring used.
                "ring": REGIONS[region],
            }
        )

    print("\nBy region:")
    for row in region_rows:
        print(
            f"  {row['name']:<26} {row['covered_miles']:6.1f} / {row['miles']:6.1f} mi  "
            f"({row['pct']:4.1f}%)  {row['trails_hiked']:>3}/{row['trails']:>3} trails"
        )

    multi = [e for e in entries if len(e["regions"]) > 1]
    print(f"\n{len(multi)} trails span more than one region (listed in each).")
    if FALLBACK["ways"]:
        print(
            f'{FALLBACK["ways"]} of {len(ways)} ways fell outside every polygon '
            "and were filed with the nearest one."
        )

    print("\nMost-complete trails you've walked:")
    for entry in [e for e in walked if e["miles"] >= 1][:15]:
        print(f"  {entry['pct']:5.1f}%  {entry['covered_miles']:5.2f}/{entry['miles']:5.2f} mi  {entry['name']}")

    if args.report:
        return

    payload = {
        "updated": None,  # stamped by write_json_stable only when data moves
        "threshold_m": args.threshold,
        "totals": {
            "trails": len(entries),
            "miles": round(total_mi, 1),
            "covered_miles": round(covered_mi, 1),
            "pct": round(100 * covered_mi / total_mi, 1) if total_mi else 0.0,
            "trails_hiked": len(walked),
            "trails_crossed": len(crossed),
            "trails_complete": len(complete),
        },
        # Precomputed so the region dropdown can swap the headline stats
        # without re-aggregating 846 trails in the browser on every change.
        #
        # Mileage is summed over *ways*, which is what keeps a trail that
        # straddles two regions from being counted whole in each; the trail
        # counts are per-region membership, so such a trail does appear in both
        # lists. The regions therefore sum to the forest's mileage but not to
        # its trail count, which is the honest way round.
        "regions": region_rows,
        "trails": entries,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if write_json_stable(OUT_FILE, payload):
        print(f"\nWrote {OUT_FILE.relative_to(REPO)}")
    else:
        print(f"\nNo coverage changes; left {OUT_FILE.relative_to(REPO)} untouched.")

    # One feature per walked/unwalked stretch, not per way. Colouring a whole
    # way by its percentage drew the Daniel Webster Trail — one way, 83% — as a
    # fully highlighted line, when 0.55 mi of it has never been walked. Cutting
    # the geometry at the coverage boundary is the only way the map can show
    # *which* 83%.
    #
    # Properties are single letters and coordinates are trimmed to five decimals
    # (~1 m): this file is downloaded by every visitor, and the long-form keys
    # cost more than the geometry does.
    trail_index = {entry["name"]: i for i, entry in enumerate(entries)}
    features = []
    for way, total, covered, runs, _regions in way_rows:
        for visits, points in runs:
            coords = simplify(points, DISPLAY_TOLERANCE_M)
            if len(coords) < 2:
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "t": trail_index[way["name"]],
                        # Distinct hikes over this stretch, uncapped. The map
                        # caps it at HEAT_MAX when picking a shade, but the
                        # hover reports the real number — "7 times" is the
                        # interesting fact about a stretch, and flattening it to
                        # "4+" throws that away for no saving worth having.
                        "n": visits,
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [round(lon, 5), round(lat, 5)] for lat, lon in coords
                        ],
                    },
                }
            )

    GEOJSON_FILE.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            separators=(",", ":"),
        )
        + "\n"
    )
    kept = sum(len(f["geometry"]["coordinates"]) for f in features)
    before = sum(len(w["coords"]) for w, _, _, _, _ in way_rows)
    heat = collections.Counter(
        min(f["properties"]["n"], HEAT_MAX) for f in features
    )
    busiest = max((f["properties"]["n"] for f in features), default=0)
    print(
        f"Wrote {GEOJSON_FILE.relative_to(REPO)} — {len(features)} stretches "
        f"from {len(way_rows)} ways, "
        f"{before:,} -> {kept:,} points, "
        f"{GEOJSON_FILE.stat().st_size / 1024:,.0f} KB"
    )
    print(
        "  stretches by visit count: "
        + ", ".join(
            f"{n}{'+' if n == HEAT_MAX else ''}x {heat.get(n, 0)}"
            for n in range(0, HEAT_MAX + 1)
        )
        + f"  (busiest stretch: {busiest} hikes)"
    )


if __name__ == "__main__":
    main()
