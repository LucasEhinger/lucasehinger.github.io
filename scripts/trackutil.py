#!/usr/bin/env python3
"""Shared helpers for the Strava/OSM scripts: polylines, distance, JSON writes.

Kept in one place because the alternative — a copy of decode_polyline in every
script that needs it — is how the four copies that were already here came to
disagree about nothing in particular.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

EARTH_RADIUS_M = 6371008.8

# Radius around a track's own start point that gets removed before the track is
# committed. Runs from a front door reveal that front door; 300 m puts the
# published start somewhere in the surrounding blocks instead.
PRIVACY_RADIUS_M = 300.0


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


def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def crop_near_start(coords, radius_m=PRIVACY_RADIUS_M):
    """Drop every point within `radius_m` of the track's first point.

    Every point, not just the leading ones: a loop that returns to where it
    started would otherwise publish the same address at the far end, and an
    out-and-back would publish it twice. The cost is that a route passing its
    own start mid-run gets a straight chord across the gap — acceptable, since
    the alternative is drawing a line to a front door.

    Returns [] when the whole track sits inside the radius; the caller decides
    whether a track that short is worth keeping at all.
    """
    if not coords:
        return []

    lat0, lon0 = coords[0]
    return [
        (lat, lon)
        for lat, lon in coords
        if haversine_m(lat0, lon0, lat, lon) > radius_m
    ]


def load_privacy_zones():
    """Fixed places to erase from every track, as [[lat, lon, radius_m], ...].

    Cropping around each track's *own* start is not enough on its own: it only
    protects runs that begin at home. A run that starts across town and finishes
    at the front door keeps every one of those points, and measurement showed
    published points landing 14 m from the address that way. Worse, cropping to
    a fixed radius leaves hundreds of tracks starting on a circle whose centre is
    exactly the thing being hidden.

    The coordinates are the sensitive part, so they never live in the repo:
      * locally, local/privacy_zones.json (gitignored)
      * in CI, the STRAVA_PRIVACY_ZONES secret holding the same JSON

    On CI, having no zones is a hard failure; locally it is only a warning,
    because a fresh clone legitimately has none until someone sets them up.
    """
    zones = []
    raw = os.environ.get("STRAVA_PRIVACY_ZONES")

    if not raw:
        path = Path(__file__).resolve().parent.parent / "local" / "privacy_zones.json"
        raw = path.read_text() if path.exists() else ""

    if raw:
        try:
            zones = json.loads(raw)
        except ValueError:
            zones = []

    # Every path that yields no zones has to land here, including "the file
    # doesn't exist" — an early return for that case skipped this check and let
    # the single most likely mistake, an unset secret, straight through. In CI
    # the next step would commit un-cropped GPS to a public repo and push it,
    # recoverable only by rewriting history.
    if not zones and os.environ.get("GITHUB_ACTIONS") == "true":
        sys.exit(
            "STRAVA_PRIVACY_ZONES is unset or unparseable.\n"
            "Refusing to write track caches that would publish un-cropped GPS.\n"
            "Set the repository secret to the contents of local/privacy_zones.json."
        )

    return [(z[0], z[1], z[2] if len(z) > 2 else PRIVACY_RADIUS_M) for z in zones]


def crop_zones(coords, zones):
    """Drop every point inside any privacy zone, wherever it falls in the track."""
    if not zones:
        return list(coords)
    return [
        (lat, lon)
        for lat, lon in coords
        if all(haversine_m(zlat, zlon, lat, lon) > radius for zlat, zlon, radius in zones)
    ]


def write_json_stable(path, payload, date_key="updated"):
    """Write `payload`, but leave the file alone when only the date would move.

    These files carry a "last updated" stamp, so writing them unconditionally
    produced a commit every single day whether or not anything was hiked. The
    comparison ignores the stamp: same data, no write, no commit.

    Returns True if the file was written.
    """
    payload = dict(payload)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except ValueError:
            existing = None

        if isinstance(existing, dict):
            before = {k: v for k, v in existing.items() if k != date_key}
            after = {k: v for k, v in payload.items() if k != date_key}
            if before == after:
                return False

    payload[date_key] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return True
