#!/usr/bin/env python3
"""Convert the BluewaterExperienceScraper event-level CSV (sailing_events.csv)
into the JSON the website's Bluewater pages consume.

The site aggregates everything client-side (for arbitrary date ranges and
per-person histories), so this script just normalizes the raw event records:

  - assigns each sailor a stable opaque id (hash of their full name) and a
    privacy-preserving display name ("Scott D."); full last names are never
    written to the output,
  - groups the flat (event, participant) rows into events with participant
    lists, de-duplicating a person who appears both as crew and skipper.

The parsing (`parse_csv_events`) and output-building (`build_output`) steps are
exposed separately so bluewater_merge.py can reuse them to splice a freshly
scraped recent window into an existing dataset.

Output schema (files/bluewater/bluewater_data.json):
  {
    "last_updated": ISO8601,
    "date_min": "YYYY-MM", "date_max": "YYYY-MM",
    "sailors": [ {"id": "<hex>", "n": "Scott D."}, ... ],   # index = position
    "events":  [ {"d":"YYYY-MM-DD","t":title,"h":hours,"r":0|1,"e":eventId,
                  "p":[[sailorIndex, role, status], ...]}, ... ]
  }
  role:   "s" skipper | "c" crew
  status: "c" counts as a sail (confirmed/skipper) | "p" pending |
          "x" cancelled | "u" unknown  (only "c" counts toward sail metrics)

Usage:
    python scripts/bluewater_to_json.py INPUT.csv [OUTPUT.json]
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os

ID_LEN = 16  # hex chars; 64 bits -> collisions negligible for ~2k sailors

# Real bluewater trips top out around a week (the longest genuine one observed is
# ~6.7 days). Anything longer is a data-entry error (a wrong end date), so cap
# durations here to keep those from inflating sail-hour totals.
MAX_TRIP_HOURS = 168.0  # 7 days


def norm(s):
    return (s or "").strip()


def sailor_id(first, last):
    key = norm(first).lower() + "|" + norm(last).lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:ID_LEN]


def display_name(first, last):
    """'SCOTT', 'DYNES' -> 'Scott D.'  (privacy-preserving)."""
    first = " ".join(w.capitalize() for w in norm(first).split())
    last = norm(last)
    initial = " " + last[0].upper() + "." if last else ""
    return (first + initial).strip()


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def classify(statuses):
    """Collapse a person's status strings for one event into (role, status)."""
    lowered = [s.lower() for s in statuses]
    is_skipper = any("skipper" in s for s in lowered)
    is_confirmed = any("confirmed" in s for s in lowered)
    is_pending = any("pending" in s for s in lowered)
    is_cancelled = any("cancel" in s for s in lowered)

    role = "s" if is_skipper else "c"
    if is_skipper or is_confirmed:
        st = "c"
    elif is_pending:
        st = "p"
    elif is_cancelled:
        st = "x"
    else:
        st = "u"
    return role, st


def parse_csv_events(input_path):
    """Read sailing_events.csv into (events, names).

    events: list of dicts {eid, d (YYYY-MM-DD), t, h, r (bool),
            participants: [(sailor_id, role, status), ...]}
    names:  dict sailor_id -> display name
    """
    events = {}
    names = {}
    with open(input_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = sailor_id(row["first name"], row["last name"])
            if sid not in names:
                names[sid] = display_name(row["first name"], row["last name"])
            eid = row["event id"]
            ev = events.get(eid)
            if ev is None:
                start = norm(row.get("start"))
                hours = round(float(row["duration"]), 2) if norm(row.get("duration")) else 0.0
                if hours < 0:
                    hours = 0.0  # source data-entry errors (end before start)
                elif hours > MAX_TRIP_HOURS:
                    hours = MAX_TRIP_HOURS  # implausibly long (bad end date); cap
                ev = {
                    "eid": eid,
                    "d": start[:10],
                    "t": norm(row.get("trip name")),
                    "h": hours,
                    "r": truthy(row.get("race")),
                    "people": {},  # sailor id -> list of status strings
                }
                events[eid] = ev
            ev["people"].setdefault(sid, []).append(norm(row.get("status")))

    out = []
    for ev in events.values():
        participants = []
        for sid, statuses in ev["people"].items():
            role, st = classify(statuses)
            participants.append((sid, role, st))
        out.append(
            {
                "eid": ev["eid"],
                "d": ev["d"],
                "t": ev["t"],
                "h": ev["h"],
                "r": ev["r"],
                "participants": participants,
            }
        )
    return out, names


def build_output(events, names):
    """Turn a list of events (+ sailor-id->name map) into the site JSON dict.

    events: iterable of {eid, d, t, h, r (bool), participants: [(sid, role, st)]}
    """
    used = sorted({sid for e in events for (sid, _r, _s) in e["participants"]})
    index_of = {sid: i for i, sid in enumerate(used)}
    sailor_list = [{"id": sid, "n": names[sid]} for sid in used]

    out_events = []
    for e in sorted(events, key=lambda e: e["d"]):
        parts = sorted(
            [[index_of[sid], role, st] for (sid, role, st) in e["participants"]],
            key=lambda p: p[0],
        )
        out_events.append(
            {
                "d": e["d"],
                "t": e["t"],
                "h": e["h"],
                "r": 1 if e["r"] else 0,
                "e": e["eid"],
                "p": parts,
            }
        )

    months = [e["d"][:7] for e in out_events if e["d"]]
    return {
        "last_updated": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "date_min": min(months) if months else None,
        "date_max": max(months) if months else None,
        "sailor_count": len(sailor_list),
        "event_count": len(out_events),
        "sailors": sailor_list,
        "events": out_events,
    }


def convert(input_path):
    events, names = parse_csv_events(input_path)
    return build_output(events, names)


def write_json(data, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to sailing_events.csv")
    parser.add_argument(
        "output",
        nargs="?",
        default="files/bluewater/bluewater_data.json",
        help="Path to write JSON (default: files/bluewater/bluewater_data.json)",
    )
    args = parser.parse_args()

    data = convert(args.input)
    write_json(data, args.output)
    print(
        "Wrote %d sailors and %d events (%s to %s) to %s"
        % (
            data["sailor_count"],
            data["event_count"],
            data["date_min"],
            data["date_max"],
            args.output,
        )
    )


if __name__ == "__main__":
    main()
