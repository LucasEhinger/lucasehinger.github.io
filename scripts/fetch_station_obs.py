#!/usr/bin/env python3
"""Fetch OBSERVED station temperatures near Mt Washington and join them to the
undercast labels, so measured conditions can be compared against the labeled days.

Three independent sources, each of which can be run on its own:

  iem       KMWN summit hourly (Mount Washington Observatory's ASOS feed) via the
            Iowa Environmental Mesonet archive. Free, no token, full history.
            Gives temp/dewpoint/RH/wind/visibility/sky-cover/ceiling hourly.

  ncei      Daily TMAX/TMIN/TOBS for the nearby NWS COOP sites -- including
            Pinkham Notch (USC00276818, 617 m, 4.3 km SE of the summit) -- via
            NCEI daily-summaries. Free, no token. DAILY ONLY: COOP sites report a
            max, a min, and one reading at the morning observation time, so they
            cannot give a noon value.

  synoptic  The Mount Washington Observatory Regional Mesonet: hourly/15-min
            Pinkham Notch plus the Auto Road Vertical Profile (ARVP, ~6 stations
            every ~300 m up the Auto Road). This is the only route to hourly
            Pinkham and Auto Road data -- mountwashington.org itself sits behind a
            Cloudflare JS challenge and cannot be scraped. Needs a free Synoptic
            Open Access token in $SYNOPTIC_TOKEN. Run --discover first to list the
            station IDs actually available to your token, then pass them to --stids.

Outputs (under files/weather/obs/):
    station_obs_hourly.csv   tidy long-format hourly obs, one row per station-hour
    station_obs_daily.csv    tidy long-format daily COOP obs

Both carry the undercast label for that date: undercast_tower / undercast_observatory
/ undercast_avg (raw 0/0.25/0.5/0.75/1 as scored) and is_undercast (avg >= 0.5,
matching train_undercast_models.py).

Usage:
    python3 scripts/fetch_station_obs.py --sources iem ncei
    python3 scripts/fetch_station_obs.py --sources synoptic --discover
    SYNOPTIC_TOKEN=... python3 scripts/fetch_station_obs.py --sources synoptic \
        --stids PKNN3,ARVP2000,ARVP3000
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

SUMMIT_LAT, SUMMIT_LON = 44.27040, -71.30327

DEFAULT_LABELS = "files/weather/csv/MtWashington_undercast_orig.csv"
DEFAULT_OUTDIR = "files/weather/obs"

# Hourly ASOS. KMWN *is* the summit observatory feed; the others are the nearest
# stations with real hourly obs, all well outside the notch (kept here so a
# summit-vs-valley delta is possible at all before the mesonet token lands).
IEM_STATIONS = {
    "MWN": ("Mt Washington summit (MWOBS/ASOS)", 1910),
    "HIE": ("Whitefield - Mt Washington Regional AP", 319),
    "BML": ("Berlin Municipal AP", 353),
    "IZG": ("Fryeburg ME - Eastern Slopes AP", 138),
}
IEM_VARS = ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby",
            "skyc1", "skyc2", "skyc3", "skyl1", "skyl2", "skyl3", "wxcodes"]

# Daily COOP / airport dailies. Elevation in m, distance from summit in km.
# "winter only" sites are snowplot stations that report roughly Dec-Apr.
NCEI_STATIONS = {
    "USC00276818": ("Pinkham Notch (AMC VC)", 617, 4.3, False),
    "USW00014755": ("Mt Washington summit (daily)", 1912, 0.0, False),
    "USC00273535": ("Gray Knob (RMC cabin)", 1333, 6.9, False),
    "USC00271800": ("Crawford Notch", 577, 10.4, False),
    "USC00273860": ("Hermit Lake Snowplot", 1143, 1.9, True),
    "USC00273856": ("Harvard Cabin Snowplot", 1062, 2.1, True),
    "USC00271190": ("Carter Notch Hut", 1011, 8.7, True),
    "USW00054728": ("Whitefield AP (daily)", 319, 22.5, False),
    "USW00094700": ("Berlin AP (daily)", 342, 35.4, False),
}

SYNOPTIC_VARS = ["air_temp", "relative_humidity", "dew_point_temperature",
                 "wind_speed", "wind_direction", "solar_radiation"]

UA = "lucasehinger.github.io undercast research (lucasehinger@gmail.com)"


def get(url, timeout=300, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            # Auth/permission problems will not fix themselves on retry.
            if exc.code in (401, 403):
                body = exc.read().decode("utf-8", "replace")[:200]
                sys.exit(f"ERROR: HTTP {exc.code} from "
                         f"{urllib.parse.urlparse(url).netloc} -- token rejected "
                         f"or not permitted for this request.\n  {body}")
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1} after {exc}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1} after {exc}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))


def load_labels(path):
    """date (YYYY-MM-DD) -> label dict. '?' and blank become None, not 0."""
    def num(v):
        v = (v or "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            short = (row.get("Short Date") or "").strip()
            if not short:
                continue
            date = datetime.strptime(short, "%m/%d/%y").strftime("%Y-%m-%d")
            avg = num(row.get("Avg"))
            out[date] = {
                "undercast_tower": row.get("Tower", "").strip(),
                "undercast_observatory": row.get("Observatory", "").strip(),
                "undercast_avg": "" if avg is None else avg,
                # Matches train_undercast_models.py: y = (is_undercast >= 0.5)
                "is_undercast": "" if avg is None else int(avg >= 0.5),
            }
    return out


# --------------------------------------------------------------------------- IEM

def fetch_iem(stations, start, end):
    """Hourly ASOS rows. report_type 3+4 = routine hourly + specials."""
    q = [("station", s) for s in stations]
    q += [("data", v) for v in IEM_VARS]
    q += [
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        # IEM's end bound is exclusive of the final day, so step past it.
        ("year2", (end + timedelta(days=1)).year),
        ("month2", (end + timedelta(days=1)).month),
        ("day2", (end + timedelta(days=1)).day),
        ("tz", "America/New_York"), ("format", "onlycomma"),
        ("latlon", "no"), ("elev", "no"), ("missing", "empty"),
        ("trace", "empty"), ("direct", "no"),
        ("report_type", "3"), ("report_type", "4"),
    ]
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + \
        urllib.parse.urlencode(q)
    print(f"  IEM: {', '.join(stations)}")
    text = get(url)
    rows = list(csv.DictReader(text.splitlines()))
    print(f"    {len(rows)} rows")

    out = []
    for r in rows:
        sid = r["station"]
        name, elev = IEM_STATIONS.get(sid, (sid, ""))
        valid = r["valid"]  # 'YYYY-MM-DD HH:MM' local
        rec = {
            "source": "IEM-ASOS", "station_id": sid, "station_name": name,
            "elev_m": elev, "date": valid[:10], "time_local": valid[11:16],
            "temp_f": r.get("tmpf", ""), "dewpoint_f": r.get("dwpf", ""),
            "rh_pct": r.get("relh", ""), "wind_dir": r.get("drct", ""),
            "wind_kt": r.get("sknt", ""), "altimeter_inhg": r.get("alti", ""),
            "visibility_mi": r.get("vsby", ""),
            "sky_cover_1": (r.get("skyc1") or "").strip(),
            "sky_cover_2": (r.get("skyc2") or "").strip(),
            "sky_cover_3": (r.get("skyc3") or "").strip(),
            "sky_level_1_ft": r.get("skyl1", ""),
            "sky_level_2_ft": r.get("skyl2", ""),
            "sky_level_3_ft": r.get("skyl3", ""),
            "present_wx": (r.get("wxcodes") or "").strip(),
        }
        out.append(rec)
    return out


# -------------------------------------------------------------------------- NCEI

def fetch_ncei(station_ids, start, end):
    """Daily TMAX/TMIN/TOBS. NCEI returns tenths of degC; converted to degF."""
    def c10_to_f(v):
        v = (v or "").strip()
        if not v:
            return ""
        return round(float(v) / 10.0 * 9 / 5 + 32, 1)

    out = []
    for sid in station_ids:
        name, elev, dist, winter = NCEI_STATIONS.get(sid, (sid, "", "", False))
        url = ("https://www.ncei.noaa.gov/access/services/data/v1"
               f"?dataset=daily-summaries&stations={sid}"
               f"&startDate={start:%Y-%m-%d}&endDate={end:%Y-%m-%d}"
               "&dataTypes=TMAX,TMIN,TOBS&format=json")
        print(f"  NCEI: {sid} {name}", end="")
        try:
            rows = json.loads(get(url, timeout=120) or "[]")
        except Exception as exc:
            print(f"  -> FAILED ({exc})")
            continue
        print(f"  -> {len(rows)} days" + ("  (winter-only site)" if winter else ""))
        for r in rows:
            out.append({
                "source": "NCEI-daily", "station_id": sid, "station_name": name,
                "elev_m": elev, "dist_km": dist, "date": r.get("DATE", ""),
                "tmax_f": c10_to_f(r.get("TMAX")),
                "tmin_f": c10_to_f(r.get("TMIN")),
                "tobs_f": c10_to_f(r.get("TOBS")),
            })
        time.sleep(0.5)
    return out


# ---------------------------------------------------------------------- SYNOPTIC

TOKEN_FILE = "local/synoptic_token.txt"  # local/ is gitignored, same as strava_tokens.json


def synoptic_token():
    tok = os.environ.get("SYNOPTIC_TOKEN", "").strip()
    if not tok and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            tok = f.read().strip()
    if not tok:
        sys.exit(f"ERROR: no token. Put it in {TOKEN_FILE} or set $SYNOPTIC_TOKEN.\n"
                 "  Open Access token: https://synopticdata.com/open-access-program/")
    # Data tokens are 32 hex chars. API *keys* use a wider alphabet and are not
    # accepted for data requests -- catch that swap before spending a request.
    if not re.fullmatch(r"[0-9a-f]{32}", tok):
        sys.exit(f"ERROR: {tok[:6]}... does not look like a Synoptic data token "
                 "(expected 32 hex chars).\n  Looks like an API key? Generate a "
                 "token FROM the key in the credentials panel and use that.")
    return tok


def synoptic_discover(radius_km=25):
    """List every station the token can see near the summit, so we can identify
    the MWOBS mesonet / Auto Road Vertical Profile IDs by name and elevation."""
    miles = radius_km / 1.609
    url = ("https://api.synopticdata.com/v2/stations/metadata?"
           f"radius={SUMMIT_LAT},{SUMMIT_LON},{miles:.0f}"
           f"&sensorvars=1&complete=1&token={synoptic_token()}")
    d = json.loads(get(url, timeout=120))
    summary = d.get("SUMMARY", {})
    if summary.get("RESPONSE_CODE") != 1:
        sys.exit(f"Synoptic error: {summary.get('RESPONSE_MESSAGE')}")
    sts = d.get("STATION") or []
    print(f"\n{len(sts)} stations within {radius_km} km:\n")
    print(f"{'STID':<12}{'elev_ft':>8}  {'network':<8} {'since':<12} name")
    rows = []
    for s in sts:
        try:
            elev = float(s.get("ELEVATION") or 0)
        except ValueError:
            elev = 0
        rows.append((-elev, s))
    for _, s in sorted(rows):
        por = (s.get("PERIOD_OF_RECORD") or {}).get("start") or ""
        print(f"{s.get('STID',''):<12}{s.get('ELEVATION') or '':>8}  "
              f"{str(s.get('MNET_ID','')):<8} {por[:10]:<12} {s.get('NAME','')}")
    print("\nPass the ones you want to --stids (comma separated).")
    return sts


def fetch_synoptic(stids, start, end):
    """Hourly-ish mesonet obs, chunked monthly to stay under the 100k
    station-hours-per-request cap and the Open Access history window."""
    token = synoptic_token()
    # Open Access grants 1 year of history. Chunks older than that will be
    # refused, so say so up front rather than burning requests on them.
    horizon = datetime.now() - timedelta(days=365)
    if start < horizon:
        print(f"  NOTE: Open Access allows ~1 year of history (back to "
              f"{horizon:%Y-%m-%d}). Chunks before that will likely be refused;"
              f" ask Synoptic support to extend the token to {start:%Y-%m-%d}.")
    out = []
    cur = start
    while cur <= end:
        nxt = min((cur.replace(day=1) + timedelta(days=32)).replace(day=1), end + timedelta(days=1))
        url = ("https://api.synopticdata.com/v2/stations/timeseries?"
               f"stid={','.join(stids)}"
               f"&start={cur:%Y%m%d}0000&end={nxt:%Y%m%d}0000"
               f"&vars={','.join(SYNOPTIC_VARS)}"
               "&units=temp|F,speed|kts&obtimezone=local"
               f"&token={token}")
        print(f"  Synoptic: {cur:%Y-%m} ", end="", flush=True)
        try:
            d = json.loads(get(url, timeout=180))
        except Exception as exc:
            print(f"-> FAILED ({exc})")
            cur = nxt
            continue
        summary = d.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE") != 1:
            print(f"-> {summary.get('RESPONSE_MESSAGE')}")
            if "token" in str(summary.get("RESPONSE_MESSAGE", "")).lower():
                sys.exit("  Token rejected -- stopping.")
            cur = nxt
            continue

        n = 0
        for st in d.get("STATION") or []:
            obs = st.get("OBSERVATIONS") or {}
            times = obs.get("date_time") or []
            series = {k: v for k, v in obs.items() if k != "date_time"}
            for i, t in enumerate(times):
                rec = {
                    "source": "Synoptic-MWOBS",
                    "station_id": st.get("STID", ""),
                    "station_name": st.get("NAME", ""),
                    "elev_m": round(float(st.get("ELEVATION") or 0) * 0.3048, 1)
                              if st.get("ELEVATION") else "",
                    "date": t[:10], "time_local": t[11:16],
                }
                for k, v in series.items():
                    rec[k] = "" if i >= len(v) or v[i] is None else v[i]
                out.append(rec)
                n += 1
        print(f"-> {n} rows")
        cur = nxt
        time.sleep(1)
    return out


# ------------------------------------------------------------------------ output

def write_csv(path, rows, labels, sort_keys):
    if not rows:
        print(f"  (nothing to write for {path})")
        return
    for r in rows:
        r.update(labels.get(r.get("date", ""), {
            "undercast_tower": "", "undercast_observatory": "",
            "undercast_avg": "", "is_undercast": "",
        }))
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in sort_keys))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    labeled = sum(1 for r in rows if r.get("is_undercast") != "")
    print(f"  wrote {path}  ({len(rows)} rows, {len(fields)} cols, "
          f"{labeled} label-matched)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=["iem", "ncei"],
                   choices=["iem", "ncei", "synoptic"])
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default: first labeled date)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: last labeled date)")
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--iem-stations", default="MWN",
                   help="comma separated; MWN is the summit. Add HIE,BML,IZG for valley hourly.")
    p.add_argument("--ncei-stations", default=",".join(NCEI_STATIONS))
    p.add_argument("--stids", default="", help="Synoptic station IDs, comma separated")
    p.add_argument("--discover", action="store_true",
                   help="Synoptic: list nearby stations visible to your token and exit")
    args = p.parse_args()

    labels = load_labels(args.labels)
    dates = sorted(labels)
    start = datetime.strptime(args.start or dates[0], "%Y-%m-%d")
    end = datetime.strptime(args.end or dates[-1], "%Y-%m-%d")
    pos = sum(1 for d in labels.values() if d["is_undercast"] == 1)
    print(f"Labels: {len(labels)} dates {dates[0]} .. {dates[-1]}, "
          f"{pos} undercast (avg>=0.5)")
    print(f"Window: {start:%Y-%m-%d} .. {end:%Y-%m-%d}\n")

    if args.discover:
        synoptic_discover()
        return

    hourly = []
    if "iem" in args.sources:
        hourly += fetch_iem([s.strip() for s in args.iem_stations.split(",") if s.strip()],
                            start, end)
    if "synoptic" in args.sources:
        stids = [s.strip() for s in args.stids.split(",") if s.strip()]
        if not stids:
            sys.exit("ERROR: --sources synoptic needs --stids (run --discover first)")
        hourly += fetch_synoptic(stids, start, end)
    if hourly:
        write_csv(os.path.join(args.outdir, "station_obs_hourly.csv"), hourly,
                  labels, ("date", "time_local", "station_id"))

    if "ncei" in args.sources:
        daily = fetch_ncei([s.strip() for s in args.ncei_stations.split(",") if s.strip()],
                           start, end)
        write_csv(os.path.join(args.outdir, "station_obs_daily.csv"), daily,
                  labels, ("date", "station_id"))


if __name__ == "__main__":
    main()
