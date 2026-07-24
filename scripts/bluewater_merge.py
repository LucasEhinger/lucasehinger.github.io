#!/usr/bin/env python3
"""Merge a freshly-scraped recent window of events into the existing
bluewater_data.json, so a daily job can refresh just the last year without
re-scraping all of history.

Every existing event dated on/after --since is dropped and replaced by the
freshly scraped events (so additions, edits, and removals within the window all
take effect); events before --since are kept untouched. Sailor ids are stable
hashes, so people carry over across the merge automatically.

Usage:
    python scripts/bluewater_merge.py EXISTING.json RECENT_EVENTS.csv [OUTPUT.json] --since YYYY-MM

If OUTPUT.json is omitted, EXISTING.json is overwritten in place.
"""

import argparse
import json

import bluewater_to_json as conv  # sibling module (scripts/ is on sys.path[0])


def events_from_json(data):
    """Reconstruct (events, names) from an existing bluewater_data.json dict,
    matching the shape parse_csv_events() returns."""
    sailors = data["sailors"]
    names = {s["id"]: s["n"] for s in sailors}
    events = []
    for ev in data["events"]:
        participants = [
            (sailors[idx]["id"], role, st) for (idx, role, st) in ev["p"]
        ]
        events.append(
            {
                "eid": ev.get("e"),
                "d": ev["d"],
                "t": ev["t"],
                "h": ev["h"],
                "r": bool(ev["r"]),
                "participants": participants,
            }
        )
    return events, names


def merge(existing_data, recent_csv, since):
    ex_events, ex_names = events_from_json(existing_data)
    new_events, new_names = conv.parse_csv_events(recent_csv)

    by_id = {}
    kept = 0
    for e in ex_events:
        if e["d"][:7] < since:  # keep history before the refreshed window
            by_id[e["eid"]] = e
            kept += 1
    for e in new_events:  # recent scrape wins for the window
        by_id[e["eid"]] = e

    merged = list(by_id.values())
    names = dict(ex_names)
    names.update(new_names)
    return conv.build_output(merged, names), kept, len(new_events)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("existing", help="Existing bluewater_data.json")
    ap.add_argument("recent_csv", help="Freshly scraped recent sailing_events.csv")
    ap.add_argument("output", nargs="?", default=None,
                    help="Output path (default: overwrite EXISTING.json)")
    ap.add_argument("--since", required=True,
                    help="YYYY-MM; existing events on/after this month are replaced")
    args = ap.parse_args()

    out_path = args.output or args.existing
    with open(args.existing, encoding="utf-8") as f:
        existing = json.load(f)

    data, kept, added = merge(existing, args.recent_csv, args.since)
    conv.write_json(data, out_path)
    print(
        "Merged: kept %d events before %s + %d from recent scrape -> "
        "%d events, %d sailors (%s to %s)"
        % (kept, args.since, added, data["event_count"], data["sailor_count"],
           data["date_min"], data["date_max"])
    )


if __name__ == "__main__":
    main()
