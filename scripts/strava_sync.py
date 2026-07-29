#!/usr/bin/env python3
"""Pull activities from Strava and write the data the /hikes page reads.

Run scripts/strava_auth.py once first. After that:

  python3 scripts/strava_sync.py            # incremental: only new activities
  python3 scripts/strava_sync.py --full     # re-fetch everything from scratch

Writes:
  local/strava_activities.json   raw cache, gitignored, keeps re-runs cheap
  files/strava/activities.json   one trimmed entry per activity, newest first

The page filters and aggregates in the browser, so this ships per-activity rows
rather than precomputed totals. Run scripts/strava_peaks.py afterwards to
refresh the summit list.
"""

import argparse
import collections
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "local"
TOKEN_FILE = LOCAL / "strava_tokens.json"
CACHE_FILE = LOCAL / "strava_activities.json"
OUT_DIR = REPO / "files" / "strava"
ACTIVITIES_FILE = OUT_DIR / "activities.json"

# Committed caches, so CI can work incrementally without the local raw cache.
# Excluded from the built site in _config.yml.
CACHE_DIR = REPO / "scripts" / "cache"
TRACKS_FILE = CACHE_DIR / "tracks.json"

API = "https://www.strava.com/api/v3"

# Strava sport_type values grouped into the categories the page can toggle.
# Each sport belongs to exactly one category, so nothing is ever double-counted;
# trail running is split out from both hiking and running for that reason.
CATEGORIES = {
    "hiking": {"Hike", "Snowshoe"},
    "trailrunning": {"TrailRun"},
    "running": {"Run", "VirtualRun"},
    "biking": {"Ride", "MountainBikeRide", "GravelRide", "EBikeRide", "VirtualRide"},
    "backcountryski": {"BackcountrySki"},
    "nordicski": {"NordicSki"},
    "alpineski": {"AlpineSki", "Snowboard"},
    "water": {
        "Sail",
        "Windsurf",
        "Kitesurf",
        "Surfing",
        "Rowing",
        "Kayaking",
        "Canoeing",
        "StandUpPaddling",
        "Swim",
    },
}

SPORT_CATEGORY = {
    sport: category for category, sports in CATEGORIES.items() for sport in sports
}

# Activities whose tracks get scanned for summits (see strava_peaks.py).
HIKE_TYPES = CATEGORIES["hiking"] | CATEGORIES["trailrunning"]

METERS_PER_MILE = 1609.344
FEET_PER_METER = 3.280839895


def load_tokens():
    """Credentials come from the environment in CI, or local/ on a laptop."""
    env_refresh = os.environ.get("STRAVA_REFRESH_TOKEN")
    if env_refresh:
        return {
            "client_id": os.environ["STRAVA_CLIENT_ID"],
            "client_secret": os.environ["STRAVA_CLIENT_SECRET"],
            "refresh_token": env_refresh,
            "expires_at": 0,
        }, True

    if not TOKEN_FILE.exists():
        sys.exit("No tokens found. Run: python3 scripts/strava_auth.py")
    return json.loads(TOKEN_FILE.read_text()), False


def get_access_token():
    tokens, from_env = load_tokens()

    # Access tokens last 6 hours; refresh if we're within 5 minutes of expiry.
    if tokens.get("expires_at", 0) > time.time() + 300:
        return tokens["access_token"]

    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": tokens["client_id"],
            "client_secret": tokens["client_secret"],
            "refresh_token": tokens["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if resp.status_code == 400:
        sys.exit(
            "Refresh failed — the token was likely revoked in Strava's settings.\n"
            "Re-run: python3 scripts/strava_auth.py"
        )
    resp.raise_for_status()
    new = resp.json()

    if from_env:
        # Strava usually returns the same refresh token, but it is allowed to
        # rotate it. CI can't rewrite its own secret, so say so loudly instead
        # of failing silently on some later run.
        if new["refresh_token"] != tokens["refresh_token"]:
            print(
                "WARNING: Strava rotated the refresh token. Update the "
                "STRAVA_REFRESH_TOKEN secret to:\n  " + new["refresh_token"],
                file=sys.stderr,
            )
        return new["access_token"]

    tokens.update(
        {
            "refresh_token": new["refresh_token"],
            "access_token": new["access_token"],
            "expires_at": new["expires_at"],
        }
    )
    TOKEN_FILE.write_text(json.dumps(tokens, indent=2) + "\n")
    return tokens["access_token"]


def fetch_activities(token, after_epoch=None):
    """Page through the athlete's activities, newest pages first."""
    headers = {"Authorization": f"Bearer {token}"}
    activities = []
    page = 1

    while True:
        params = {"per_page": 200, "page": page}
        if after_epoch:
            params["after"] = after_epoch

        resp = requests.get(
            f"{API}/athlete/activities", headers=headers, params=params, timeout=30
        )
        if resp.status_code == 429:
            sys.exit(
                "Hit Strava's rate limit (200 requests / 15 min). "
                "Wait a few minutes and re-run — progress is cached."
            )
        resp.raise_for_status()

        batch = resp.json()
        if not batch:
            break

        activities.extend(batch)
        print(f"  fetched page {page} ({len(batch)} activities)")
        page += 1

    return activities


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    raw = json.loads(CACHE_FILE.read_text())
    return {str(a["id"]): a for a in raw}


def save_cache(by_id):
    LOCAL.mkdir(exist_ok=True)
    ordered = sorted(by_id.values(), key=lambda a: a["start_date_local"], reverse=True)
    CACHE_FILE.write_text(json.dumps(ordered, indent=1) + "\n")


def to_activity(a):
    """Trim a Strava activity down to what the site actually renders."""
    start = a["start_date_local"]
    sport = a.get("sport_type", a.get("type"))
    entry = {
        "id": a["id"],
        "name": a["name"].strip(),
        "date": start[:10],
        "sport": sport,
        "distance_mi": round(a["distance"] / METERS_PER_MILE, 2),
        "gain_ft": round(a.get("total_elevation_gain", 0) * FEET_PER_METER),
        "moving_time_s": a.get("moving_time", 0),
        "elapsed_time_s": a.get("elapsed_time", 0),
    }

    if a.get("elev_high") is not None:
        entry["high_point_ft"] = round(a["elev_high"] * FEET_PER_METER)

    # Tracks stay out of the site payload — strava_peaks.py reads them straight
    # from the local cache, so there's no reason to ship 1,600 polylines.
    return entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="re-fetch the entire history instead of only new activities",
    )
    args = parser.parse_args()

    token = get_access_token()
    cached = {} if args.full else load_cache()

    # Previously published rows and tracks. On CI the raw cache doesn't exist,
    # so these committed files are what makes an incremental run possible.
    published = {}
    tracks = {}
    if not args.full:
        if ACTIVITIES_FILE.exists():
            for e in json.loads(ACTIVITIES_FILE.read_text())["activities"]:
                published[str(e["id"])] = e
        if TRACKS_FILE.exists():
            tracks = json.loads(TRACKS_FILE.read_text())

    after_epoch = None
    if cached:
        newest = max(a["start_date"] for a in cached.values())
        after_epoch = int(
            datetime.fromisoformat(newest.replace("Z", "+00:00")).timestamp()
        )
        print(f"Cache has {len(cached)} activities; fetching anything after {newest}.")
    elif published:
        # Only local dates are published, so back off two days rather than risk
        # skipping an activity across a timezone boundary. Re-fetching a few is
        # free — everything is keyed by id.
        newest = max(e["date"] for e in published.values())
        after_epoch = int(
            (
                datetime.fromisoformat(newest).replace(tzinfo=timezone.utc)
                - timedelta(days=2)
            ).timestamp()
        )
        print(
            f"{len(published)} activities already published; "
            f"fetching anything after {newest} (minus 2 days)."
        )
    else:
        print("Fetching full activity history (this may take a minute)...")

    fetched = fetch_activities(token, after_epoch)
    print(f"Got {len(fetched)} new activit{'y' if len(fetched) == 1 else 'ies'}.")

    for a in fetched:
        cached[str(a["id"])] = a
    if cached:
        save_cache(cached)

    # Only categorized sports reach the site; workouts, weight training, and
    # soccer aren't part of the story this page tells.
    for a in fetched:
        if a.get("sport_type", a.get("type")) not in SPORT_CATEGORY:
            continue
        published[str(a["id"])] = to_activity(a)

    # Full rebuilds re-derive every row from the raw cache.
    if args.full:
        published = {
            str(a["id"]): to_activity(a)
            for a in cached.values()
            if a.get("sport_type", a.get("type")) in SPORT_CATEGORY
        }

    entries = sorted(published.values(), key=lambda e: e["date"], reverse=True)

    # Hike tracks are kept in the repo so strava_peaks.py can rematch summits
    # without needing the raw cache — which CI never has.
    source = cached.values() if args.full else fetched
    for a in source:
        if a.get("sport_type", a.get("type")) not in HIKE_TYPES:
            continue
        polyline = (a.get("map") or {}).get("summary_polyline")
        if polyline:
            tracks[str(a["id"])] = {
                "name": a["name"].strip(),
                "date": a["start_date_local"][:10],
                "sport": a.get("sport_type", a.get("type")),
                "polyline": polyline,
            }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TRACKS_FILE.write_text(json.dumps(tracks, separators=(",", ":"), sort_keys=True) + "\n")

    payload = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "categories": {c: sorted(s) for c, s in CATEGORIES.items()},
        "activities": entries,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVITIES_FILE.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    by_category = collections.Counter(SPORT_CATEGORY[e["sport"]] for e in entries)
    print(f"\nWrote {len(entries)} activities to {ACTIVITIES_FILE.relative_to(REPO)}:")
    for category in CATEGORIES:
        count = by_category.get(category, 0)
        if not count:
            continue
        subset = [e for e in entries if SPORT_CATEGORY[e["sport"]] == category]
        miles = sum(e["distance_mi"] for e in subset)
        hours = sum(e["moving_time_s"] for e in subset) / 3600
        gain = sum(e["gain_ft"] for e in subset)
        print(
            f"  {category:<13} {count:>5} activities  {miles:>9,.1f} mi  "
            f"{hours:>7,.1f} h  {gain:>9,} ft"
        )

    # Only meaningful when the raw cache holds the full history; an incremental
    # run has nothing to compare against.
    if cached:
        skipped = len(cached) - len(entries)
        if skipped > 0:
            print(f"\n  ({skipped} uncategorized activities excluded)")


if __name__ == "__main__":
    main()
