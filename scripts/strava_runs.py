#!/usr/bin/env python3
"""Group repeated running routes into "Lucas Loops".

Reads the committed run track cache written by strava_sync.py, decides which
runs retraced the same route, and writes the loop list the /runs page reads.

  python3 scripts/strava_runs.py
  python3 scripts/strava_runs.py --threshold 0.65   # merge route variants
  python3 scripts/strava_runs.py --report           # print, write nothing

Writes files/strava/runs.json.

How two runs are judged to be the same route
--------------------------------------------
Every track is snapped to an 80 m grid and reduced to the *set* of cells it
touched; similarity is the Jaccard index |A n B| / |A u B|. Using a set rather
than a sequence buys two invariances for free, both of which matter here:

  * direction — a set has no order, so a loop run clockwise and the same loop
    run counterclockwise are identical by construction, and
  * start point — the same loop entered at a different corner still matches.

The 80 m cell absorbs GPS noise and the decimation in Strava's summary
polyline, while staying tight enough to tell apart two routes that diverge at
the next bridge over.

Runs are then clustered greedily around a medoid: take the run with the most
neighbours above the threshold, make it the loop's canonical route, absorb its
neighbours, repeat. Single-linkage was tried first and chains genuinely
distinct loops together through intermediate variants; the medoid also hands
each loop a real representative track to draw.

Direction is measured against that canonical route rather than from the
polygon's signed area. These loops run up one bank of a river and back down
the other, so their enclosed area is tiny next to their perimeter (circularity
~0.04) and the shoelace sign goes unstable on the near-out-and-back ones. So
each run is walked against the canonical point order instead, and only loops
whose shape makes handedness meaningful get labelled clockwise/counter.
"""

import argparse
import collections
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from trackutil import write_json_stable

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "scripts" / "cache"
RUN_TRACKS_FILE = CACHE_DIR / "run_tracks.json"

# Optional hand-written display names, {loop_id: "Charles River loop"}. Loop ids
# are stable across re-runs (see loop_id), so a name set here sticks.
NAMES_FILE = CACHE_DIR / "run_loop_names.json"

OUT_FILE = REPO / "files" / "strava" / "runs.json"

# Grid resolution for the cell sets, in metres. Loosening this merges route
# variants; tightening it starts splitting one route on GPS noise alone.
CELL_M = 80.0

# Jaccard similarity at or above which two runs are the same route. The river
# loop's bridge variants sit at 0.68-0.70 to each other, so this is the knob
# that decides whether those read as one loop or as siblings.
THRESHOLD = 0.70

# A route has to have been repeated this many times to earn a place on the page.
MIN_RUNS = 3

# Points used when resampling a track to compare its direction with the
# canonical route. Well above the number of vertices in a summary polyline.
RESAMPLE_N = 200

# How strongly the traversal has to trend one way before the direction is
# called at all, as a fraction of a full lap. An out-and-back covers its cells
# in both directions and lands near zero.
DIRECTION_CONFIDENCE = 0.5

# Enclosed area over that of a circle with the same perimeter. Above this the
# loop encloses enough ground for "clockwise" to mean something; below it the
# two directions are only ever described relative to the usual one.
MIN_CIRCULARITY = 0.02

# Start and finish within this distance is treated as a closed loop.
LOOP_CLOSE_M = 400

METERS_PER_MILE = 1609.344
DEG_LAT_M = 111320.0


def decode_polyline(encoded):
    """Decode a Google-encoded polyline into [(lat, lon), ...]."""
    points = []
    index = lat = lng = 0

    while index < len(encoded):
        for target in range(2):
            shift = result = 0
            while True:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if target == 0:
                lat += delta
            else:
                lng += delta
        points.append((lat / 1e5, lng / 1e5))

    return points


def project(points):
    """Flatten lat/lon to local metres. Fine at the scale of a single run."""
    lat0 = sum(p[0] for p in points) / len(points)
    scale = math.cos(math.radians(lat0))
    return [(p[1] * scale * DEG_LAT_M, p[0] * DEG_LAT_M) for p in points]


def cell_set(points, size=CELL_M):
    """The set of grid cells a track touched — the unit of comparison."""
    cells = set()
    for lat, lon in points:
        y = int(lat * DEG_LAT_M / size)
        x = int(lon * math.cos(math.radians(lat)) * DEG_LAT_M / size)
        cells.add((y, x))
    return cells


def jaccard(a, b):
    intersection = len(a & b)
    if not intersection:
        return 0.0
    return intersection / (len(a) + len(b) - intersection)


def resample(xy, count=RESAMPLE_N):
    """Evenly spaced points along a projected track, by arc length.

    Vertex spacing in a summary polyline reflects Strava's decimation, not the
    route, so comparing raw vertices would weight the wiggly parts far too
    heavily.
    """
    cumulative = [0.0]
    for i in range(1, len(xy)):
        cumulative.append(cumulative[-1] + math.dist(xy[i - 1], xy[i]))
    total = cumulative[-1]
    if total <= 0:
        return [xy[0]] * count

    out = []
    j = 0
    for i in range(count):
        target = total * i / count
        while j < len(cumulative) - 2 and cumulative[j + 1] < target:
            j += 1
        span = cumulative[j + 1] - cumulative[j]
        frac = (target - cumulative[j]) / span if span else 0.0
        x0, y0 = xy[j]
        x1, y1 = xy[j + 1]
        out.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac))
    return out


def signed_area(xy):
    """Shoelace area of the closed ring. Sign is the handedness."""
    total = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        total += x1 * y2 - x2 * y1
    return total / 2


def path_length(xy):
    return sum(math.dist(xy[i - 1], xy[i]) for i in range(1, len(xy)))


def direction_vs(canonical, sample):
    """How far `sample` advances along `canonical`, in laps.

    Each resampled point of the run is mapped to the nearest point of the
    canonical route, then the steps through those indices are summed with
    wrap-around. Retracing the canonical route forwards totals about +1 lap,
    running it backwards about -1, and an out-and-back cancels out near 0.
    """
    n = len(canonical)
    indices = []
    for point in sample:
        best = min(range(n), key=lambda i: math.dist(canonical[i], point))
        indices.append(best)

    advance = 0
    for i in range(1, len(indices)):
        step = (indices[i] - indices[i - 1]) % n
        if step > n // 2:
            step -= n  # a step backwards is shorter than the long way round
        advance += step

    return advance / n


def svg_path(xy, box=100.0):
    """Normalise a projected track into a small viewBox, as an SVG path."""
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    scale = box / max(width, height, 1e-6)

    commands = []
    for i, (x, y) in enumerate(xy):
        # SVG's y axis points down, so flip it to keep north at the top.
        px = (x - min(xs)) * scale
        py = (max(ys) - y) * scale
        commands.append(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}")

    return {
        "d": "".join(commands),
        "w": round(width * scale, 1),
        "h": round(height * scale, 1),
    }


def loop_id(members):
    """Stable identifier: the id of the earliest run on the route.

    Ties the loop to a fact that does not move as new runs arrive, unlike the
    medoid, which can shift to a more central track at any time.
    """
    earliest = min(members, key=lambda r: (r["date"], r["id"]))
    return f"L{earliest['id']}"


def load_runs(cell=CELL_M):
    if not RUN_TRACKS_FILE.exists():
        sys.exit(
            f"No run tracks at {RUN_TRACKS_FILE.relative_to(REPO)}.\n"
            "Run: python3 scripts/strava_sync.py --full"
        )

    runs = []
    for activity_id, track in json.loads(RUN_TRACKS_FILE.read_text()).items():
        points = decode_polyline(track["polyline"])
        if len(points) < 4:
            continue
        xy = project(points)
        runs.append(
            {
                "id": int(activity_id),
                "name": track["name"],
                "date": track["date"],
                "sport": track["sport"],
                "distance_mi": track["distance_mi"],
                "moving_time_s": track["moving_time_s"],
                "gain_ft": track.get("gain_ft", 0),
                "xy": xy,
                "cells": cell_set(points, cell),
                "resampled": resample(xy),
                "gap_m": math.dist(xy[0], xy[-1]),
            }
        )

    runs.sort(key=lambda r: (r["date"], r["id"]))
    return runs


def cluster(runs, threshold):
    """Greedy medoid clustering. Returns [(medoid_index, [member indices])]."""
    n = len(runs)
    neighbours = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(runs[i]["cells"], runs[j]["cells"]) >= threshold:
                neighbours[i].add(j)
                neighbours[j].add(i)

    unassigned = set(range(n))
    clusters = []
    while unassigned:
        seed = max(unassigned, key=lambda i: len(neighbours[i] & unassigned))
        members = sorted((neighbours[seed] & unassigned) | {seed})
        clusters.append((seed, members))
        unassigned -= set(members)

    clusters.sort(key=lambda c: (-len(c[1]), c[0]))
    return clusters


def describe(runs, seed, members):
    """Turn one cluster into the loop record the page renders."""
    canonical = runs[seed]
    entries = [runs[i] for i in members]

    area = signed_area(canonical["resampled"])
    perimeter = path_length(canonical["resampled"])
    circularity = abs(area) / (perimeter**2 / (4 * math.pi)) if perimeter else 0.0

    # Whether "clockwise" describes anything on this shape, and if so which way
    # the canonical route itself goes. Screen y is flipped in svg_path, but the
    # projection here is y-up, so a positive shoelace area is counterclockwise.
    oriented = circularity >= MIN_CIRCULARITY
    canonical_is_ccw = area > 0

    activities = []
    counts = collections.Counter()
    for entry in entries:
        if entry is canonical:
            laps = 1.0
        else:
            laps = direction_vs(canonical["resampled"], entry["resampled"])

        if abs(laps) < DIRECTION_CONFIDENCE:
            heading = "unknown"
        elif oriented:
            forward = laps > 0
            heading = "ccw" if forward == canonical_is_ccw else "cw"
        else:
            heading = "forward" if laps > 0 else "reverse"

        counts[heading] += 1
        activities.append(
            {
                "id": entry["id"],
                "date": entry["date"],
                "mi": entry["distance_mi"],
                "s": entry["moving_time_s"],
                "dir": heading,
            }
        )

    activities.sort(key=lambda a: a["date"], reverse=True)
    distances = sorted(e["distance_mi"] for e in entries)
    paces = [
        e["moving_time_s"] / e["distance_mi"]
        for e in entries
        if e["distance_mi"] > 0 and e["moving_time_s"] > 0
    ]

    return {
        "id": loop_id(entries),
        "runs": len(entries),
        "median_mi": round(statistics.median(distances), 2),
        "min_mi": distances[0],
        "max_mi": distances[-1],
        "median_gain_ft": round(statistics.median(e["gain_ft"] for e in entries)),
        "median_pace_s": round(statistics.median(paces)) if paces else None,
        "best_pace_s": round(min(paces)) if paces else None,
        "total_mi": round(sum(distances), 1),
        "first": min(e["date"] for e in entries),
        "last": max(e["date"] for e in entries),
        "closed": canonical["gap_m"] <= LOOP_CLOSE_M,
        "oriented": oriented,
        "circularity": round(circularity, 3),
        "directions": dict(counts),
        "canonical_id": canonical["id"],
        "shape": svg_path(canonical["resampled"] + [canonical["resampled"][0]]),
        "activities": activities,
    }


def assign_names(loops):
    """Auto-name by distance, disambiguated by suffix; overrides win."""
    overrides = {}
    if NAMES_FILE.exists():
        overrides = json.loads(NAMES_FILE.read_text())

    by_label = collections.defaultdict(list)
    for loop in loops:
        by_label[f"{loop['median_mi']:.1f} mi loop"].append(loop)

    for label, group in by_label.items():
        for index, loop in enumerate(group):
            suffix = f" {chr(ord('A') + index)}" if len(group) > 1 else ""
            loop["name"] = overrides.get(loop["id"], label + suffix)
            loop["auto_name"] = label + suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--cell", type=float, default=CELL_M)
    parser.add_argument("--min-runs", type=int, default=MIN_RUNS)
    parser.add_argument(
        "--report", action="store_true", help="print the loops without writing"
    )
    args = parser.parse_args()

    runs = load_runs(args.cell)
    print(f"{len(runs)} runs with tracks.")

    clusters = cluster(runs, args.threshold)
    kept = [c for c in clusters if len(c[1]) >= args.min_runs]
    print(
        f"{len(clusters)} distinct routes; {len(kept)} run at least "
        f"{args.min_runs} times."
    )

    loops = [describe(runs, seed, members) for seed, members in kept]
    assign_names(loops)

    assign = {}
    for loop in loops:
        for activity in loop["activities"]:
            assign[str(activity["id"])] = loop["id"]

    payload = {
        "updated": None,  # stamped by write_json_stable only when data moves
        "params": {
            "cell_m": args.cell,
            "threshold": args.threshold,
            "min_runs": args.min_runs,
        },
        "tracked_runs": len(runs),
        "distinct_routes": len(clusters),
        "loops": loops,
        "assign": assign,
    }

    print()
    for loop in loops[:20]:
        directions = " ".join(
            f"{key}={value}" for key, value in sorted(loop["directions"].items())
        )
        print(
            f"  {loop['name']:<16} x{loop['runs']:<4} "
            f"{loop['median_mi']:>5.2f} mi  {loop['first']}..{loop['last']}  "
            f"circ {loop['circularity']:.2f}  {directions}"
        )

    if args.report:
        print("\n--report: nothing written.")
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = write_json_stable(OUT_FILE, payload)
    size_kb = OUT_FILE.stat().st_size / 1024
    verb = "Wrote" if written else "Unchanged:"
    print(
        f"\n{verb} {OUT_FILE.relative_to(REPO)} ({size_kb:.0f} KB), "
        f"{sum(loop['runs'] for loop in loops)} runs on named loops."
    )


if __name__ == "__main__":
    main()
