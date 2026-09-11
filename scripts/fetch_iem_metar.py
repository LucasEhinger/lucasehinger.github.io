#!/usr/bin/env python3
"""Cache raw KMWN METAR text (body + RMK) from the Iowa Environmental Mesonet.

Why IEM and not NOAA ISD, which ``parse_summit_remarks.py`` originally used:

  * ISD's ``REM`` field only carries the FULL METAR body from 1999-10 onward.
    Before that it holds the bare remark (``MET019NEW TPS LWR FEW050;``) with no
    prevailing visibility and no sky groups -- two of the three undercast cuts
    live in the body, so the screen simply cannot be computed.
  * ISD's KMWN series stops at 2025-08-24. IEM is current to today, which
    matters disproportionately: the recent end is exactly where the NWP
    archives (ECMWF, NBM, RAP, GFS) actually have data.
  * IEM reconstructs a METAR body for the pre-METAR SA era, so the record
    reaches back to 1997 -- the first year ``TPS LWR`` is reported in the
    canonical ``TPS LWR SCT035`` form. 1995-96 use a spaced variant that often
    omits the height entirely (``TPS LWR SCT NE-SW``), which the deck-depth cut
    needs, so 1997 is the real start of a usable record.

One CSV per year, cached on disk; re-running only fetches missing years.

Usage:
    python3 scripts/fetch_iem_metar.py --cache <dir> --start 1997 --end 2026
"""
import argparse
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
UA = "lucasehinger.github.io undercast research (lucasehinger@gmail.com)"
STATION = "MWN"


def fetch_year(year, cache_dir, force=False):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"metar_{year}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 2000 and not force:
        return path, "cached"
    q = {
        "station": STATION, "data": "metar",
        "year1": year, "month1": 1, "day1": 1,
        "year2": year + 1, "month2": 1, "day2": 1,
        "tz": "UTC", "format": "onlycomma",
        "missing": "empty", "trace": "empty", "latlon": "no",
    }
    # report_type 3 = routine hourly, 4 = specials. KMWN transmits ~1/hour, so
    # including specials adds very few rows and no duplicate hours in practice.
    url = BASE + "?" + urllib.parse.urlencode(q) + "&report_type=3&report_type=4"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            break
        except Exception as e:
            if attempt == 4:
                print(f"  {year}: FAILED ({type(e).__name__})")
                return None, "failed"
            time.sleep(5 * (attempt + 1))
    if len(data) < 2000:
        print(f"  {year}: empty response ({len(data)} bytes)")
        return None, "empty"
    with open(path, "wb") as f:
        f.write(data)
    return path, "fetched"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--start", type=int, default=1997)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--force", action="store_true", help="refetch even if cached")
    a = p.parse_args()
    for y in range(a.start, a.end + 1):
        # The current year is still filling, so always refetch it.
        force = a.force or y == a.end
        path, how = fetch_year(y, a.cache, force=force)
        if path:
            n = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
            print(f"  {y}: {n:6d} reports ({how})")
        if how == "fetched":
            time.sleep(2)


if __name__ == "__main__":
    main()
