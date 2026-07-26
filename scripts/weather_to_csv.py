from herbie import Herbie
import numpy as np
import xarray as xr
from pathlib import Path
import re
from datetime import datetime, timezone
import json
import math
from datetime import timedelta
import sys
from io import StringIO
import csv
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import socket
import warnings

# xarray emits a FutureWarning on .argmin()/.argmax() without an explicit dim
# (used by the nearest-gridpoint lookup). The current flat-index behaviour is
# exactly what we want, so silence the deprecation noise in these batch runs.
warnings.filterwarnings("ignore", category=FutureWarning)

# Herbie's GRIB/IDX downloads have no socket timeout, so a dropped connection
# leaves a worker blocked forever in the read() syscall -- unkillable even with
# SIGKILL until the OS times the socket out (can be hours). A module-level
# default timeout makes any stalled read raise instead, so the date is skipped
# and retried on the next --resume pass. Applies to spawned workers too, since
# they re-import this module.
socket.setdefaulttimeout(120)

# The real hang source: Herbie downloads each GRIB subset by shelling out to
# `curl -s --range ... > file` via os.system(), with NO --max-time. On a dead
# socket that curl blocks forever, freezing the worker (no Python timeout can
# reach a subprocess). Wrap os.system so every curl gets connect/transfer caps
# and a couple retries -- a stalled download then self-aborts in ~2 min and the
# fxx is skipped, instead of hanging the worker until the per-date alarm.
_orig_os_system = os.system


def _os_system_with_curl_timeout(cmd):
    if isinstance(cmd, str) and cmd.lstrip().startswith("curl "):
        cmd = cmd.replace(
            "curl -s ",
            "curl -s --connect-timeout 20 --max-time 150 --retry 2 --retry-delay 3 ",
            1,
        )
    return _orig_os_system(cmd)


os.system = _os_system_with_curl_timeout

LOCATIONS = [
    {"name": "MtWashington", "lat": 44.27040, "lon": -71.30327},
]

variables = {
    "cloud_top_hrrr": {"aliases": ["cloudTop", "nominalTop", "RETOP"], "model": "hrrr"},
    "boundary_layer_cloud_layer": {
        "aliases": [
            "boundaryLayerCloudLayer",
            "TCDC:boundary layer cloud layer",
            "TCDC",
        ],
        "model": "hrrr",
    },
    "low_cloud_layer_percent_hrrr": {
        "aliases": ["lowCloudLayer", "LCDC:low cloud layer", "LCDC"],
        "model": "hrrr",
    },
    "middle_cloud_layer_percent_hrrr": {
        "aliases": ["middleCloudLayer", "MCDC:middle cloud layer", "MCDC"],
        "model": "hrrr",
    },
    "high_cloud_layer_percent_hrrr": {
        "aliases": ["highCloudLayer", "HCDC:high cloud layer", "HCDC"],
        "model": "hrrr",
    },
    "cloud_ceiling_m_hrrr": {
        "aliases": ["HGT:cloud ceiling", "HGT_ceiling"],
        "model": "hrrr",
    },
    "cloud_base_m_hrrr": {
        "aliases": ["HGT:cloud base", "HGT_base"],
        "model": "hrrr",
    },
    "cloud_top_pres_hrrr": {"aliases": ["PRES:cloud top", "PRES_cloud_top"], "model": "hrrr"},
    "cloud_base_pres_hrrr": {"aliases": ["PRES:cloud base", "PRES_cloud_base"], "model": "hrrr"},
    "cloud_top_hgt_hrrr": {"aliases": ["HGT:cloud top", "HGT_cloud_top"], "model": "hrrr"},
    "wind_10m_day_max_hrrr": {
        "aliases": [
            ":WIND:10 m above ground:0-0 day max fcst",
            "WIND:10 m above ground:0-0 day max fcst",
            "WIND:10 m",
        ],
        "model": "hrrr",
    },
    "tmp_500mb_hrrr": {"aliases": [":TMP:500 mb"], "model": "hrrr"},
    "tmp_700mb_hrrr": {"aliases": [":TMP:700 mb"], "model": "hrrr"},
    "tmp_850mb_hrrr": {"aliases": [":TMP:850 mb"], "model": "hrrr"},
    "tmp_925mb_hrrr": {"aliases": [":TMP:925 mb"], "model": "hrrr"},
    "tmp_1000mb_hrrr": {"aliases": [":TMP:1000 mb"], "model": "hrrr"},
    "hgt_500mb_hrrr": {"aliases": [":HGT:500 mb"], "model": "hrrr"},
    "hgt_700mb_hrrr": {"aliases": [":HGT:700 mb"], "model": "hrrr"},
    "hgt_850mb_hrrr": {"aliases": [":HGT:850 mb"], "model": "hrrr"},
    "hgt_925mb_hrrr": {"aliases": [":HGT:925 mb"], "model": "hrrr"},
    "hgt_1000mb_hrrr": {"aliases": [":HGT:1000 mb"], "model": "hrrr"},
    "tmp_2m_hrrr": {"aliases": [":TMP:2 m above ground"], "model": "hrrr"},
    "rh_2m_hrrr": {"aliases": [":RH:2 m above ground"], "model": "hrrr"},
    "hpbl_surface_hrrr": {"aliases": [":HPBL:surface"], "model": "hrrr"},
    "hgt_0C_iso_hrrr": {"aliases": [":HGT:0C isotherm:"], "model": "hrrr"},
    "vis_surface_hrrr": {"aliases": [":VIS:surface"], "model": "hrrr"},
    "prate_surface_hrrr": {"aliases": [":PRATE:surface:%n hour"], "model": "hrrr"},
    "apcp_surface_hrrr": {
        "aliases": [":APCP:surface"],
        "model": "hrrr",
    },
    "cloud_ceiling_gfs": {
        "aliases": ["cloudCeiling", "HGT:cloud ceiling", "HGT_ceiling"],
        "model": "gfs",
    },
    "low_cloud_layer_percent_gfs": {
        "aliases": [":LCDC:low cloud layer:%n hour"],
        "model": "gfs",
    },
    "middle_cloud_layer_percent_gfs": {
        "aliases": [":MCDC:middle cloud layer:%n hour"],
        "model": "gfs",
    },
    "high_cloud_layer_percent_gfs": {
        "aliases": [":HCDC:high cloud layer:%n hour"],
        "model": "gfs",
    },
    "boundary_layer_cloud_layer_gfs": {
        "aliases": [":TCDC:boundary layer cloud layer:%n hour"],
        "model": "gfs",
    },
    "vis_surface_gfs": {"aliases": [":VIS:surface"], "model": "gfs"},
    "prate_surface_gfs": {"aliases": [":PRATE:surface:%n hour"], "model": "gfs"},
    "apcp_surface_gfs": {
        "aliases": [":APCP:surface"],
        "model": "gfs",
    },
    "tmp_500mb_gfs": {"aliases": [":TMP:500 mb"], "model": "gfs"},
    "tmp_700mb_gfs": {"aliases": [":TMP:700 mb"], "model": "gfs"},
    "tmp_850mb_gfs": {"aliases": [":TMP:850 mb"], "model": "gfs"},
    "tmp_925mb_gfs": {"aliases": [":TMP:925 mb"], "model": "gfs"},
    "tmp_1000mb_gfs": {"aliases": [":TMP:1000 mb"], "model": "gfs"},
    "hgt_500mb_gfs": {"aliases": [":HGT:500 mb"], "model": "gfs"},
    "hgt_700mb_gfs": {"aliases": [":HGT:700 mb"], "model": "gfs"},
    "hgt_850mb_gfs": {"aliases": [":HGT:850 mb"], "model": "gfs"},
    "hgt_925mb_gfs": {"aliases": [":HGT:925 mb"], "model": "gfs"},
    "hgt_1000mb_gfs": {"aliases": [":HGT:1000 mb"], "model": "gfs"},
    "tmp_2m_gfs": {"aliases": [":TMP:2 m above ground"], "model": "gfs"},
    "rh_2m_gfs": {"aliases": [":RH:2 m above ground"], "model": "gfs"},
    "rh_925mb_gfs": {"aliases": [":RH:925 mb"], "model": "gfs"},
    "hpbl_surface_gfs": {"aliases": [":HPBL:surface"], "model": "gfs"},
    "hgt_0C_iso_gfs": {"aliases": [":HGT:0C isotherm:"], "model": "gfs"},
    "cloud_ceiling_nam": {
        "aliases": ["cloudCeiling", "HGT:cloud ceiling", "HGT_ceiling"],
        "model": "nam",
    },
    "low_cloud_layer_percent_nam": {
        "aliases": [":LCDC:low cloud layer:%n hour"],
        "model": "nam",
    },
    "middle_cloud_layer_percent_nam": {
        "aliases": [":MCDC:middle cloud layer:%n hour"],
        "model": "nam",
    },
    "high_cloud_layer_percent_nam": {
        "aliases": [":HCDC:high cloud layer:%n hour"],
        "model": "nam",
    },
    "boundary_layer_cloud_layer_nam": {
        "aliases": [":TCDC:boundary layer cloud layer:%n hour"],
        "model": "nam",
    },
    "vis_surface_nam": {"aliases": [":VIS:surface"], "model": "nam"},
    "tmp_500mb_nam": {"aliases": [":TMP:500 mb"], "model": "nam"},
    "tmp_700mb_nam": {"aliases": [":TMP:700 mb"], "model": "nam"},
    "tmp_850mb_nam": {"aliases": [":TMP:850 mb"], "model": "nam"},
    "tmp_925mb_nam": {"aliases": [":TMP:925 mb"], "model": "nam"},
    "tmp_1000mb_nam": {"aliases": [":TMP:1000 mb"], "model": "nam"},
    "hgt_500mb_nam": {"aliases": [":HGT:500 mb"], "model": "nam"},
    "hgt_700mb_nam": {"aliases": [":HGT:700 mb"], "model": "nam"},
    "hgt_850mb_nam": {"aliases": [":HGT:850 mb"], "model": "nam"},
    "hgt_925mb_nam": {"aliases": [":HGT:925 mb"], "model": "nam"},
    "hgt_1000mb_nam": {"aliases": [":HGT:1000 mb"], "model": "nam"},
    "tmp_2m_nam": {"aliases": [":TMP:2 m above ground"], "model": "nam"},
    "rh_2m_nam": {"aliases": [":RH:2 m above ground"], "model": "nam"},
    "rh_925mb_nam": {"aliases": [":RH:925 mb"], "model": "nam"},
    "hpbl_surface_nam": {"aliases": [":HPBL:surface"], "model": "nam"},
    "hgt_0C_iso_nam": {"aliases": [":HGT:0C isotherm:"], "model": "nam"},
    "prate_surface_nam": {"aliases": [":PRATE:surface:%n hour"], "model": "nam"},
    "apcp_surface_nam": {
        "aliases": [":APCP:surface"],
        "model": "nam",
    },
    # --- RAP (Rapid Refresh, 13 km; Herbie model="rap"). HRRR's parent model,
    # GRIB-standard cloud/visibility/temperature/height/boundary-layer fields with
    # a full historical archive. Mirrors the HRRR/NAM field set.
    "cloud_ceiling_m_rap": {"aliases": ["cloudCeiling", "HGT:cloud ceiling", "HGT_ceiling"], "model": "rap"},
    "low_cloud_layer_percent_rap": {"aliases": [":LCDC:low cloud layer:%n hour"], "model": "rap"},
    "middle_cloud_layer_percent_rap": {"aliases": [":MCDC:middle cloud layer:%n hour"], "model": "rap"},
    "high_cloud_layer_percent_rap": {"aliases": [":HCDC:high cloud layer:%n hour"], "model": "rap"},
    "boundary_layer_cloud_layer_rap": {"aliases": [":TCDC:boundary layer cloud layer:%n hour"], "model": "rap"},
    "vis_surface_rap": {"aliases": [":VIS:surface"], "model": "rap"},
    "tmp_500mb_rap": {"aliases": [":TMP:500 mb"], "model": "rap"},
    "tmp_700mb_rap": {"aliases": [":TMP:700 mb"], "model": "rap"},
    "tmp_850mb_rap": {"aliases": [":TMP:850 mb"], "model": "rap"},
    "tmp_925mb_rap": {"aliases": [":TMP:925 mb"], "model": "rap"},
    "tmp_1000mb_rap": {"aliases": [":TMP:1000 mb"], "model": "rap"},
    "hgt_500mb_rap": {"aliases": [":HGT:500 mb"], "model": "rap"},
    "hgt_700mb_rap": {"aliases": [":HGT:700 mb"], "model": "rap"},
    "hgt_850mb_rap": {"aliases": [":HGT:850 mb"], "model": "rap"},
    "hgt_925mb_rap": {"aliases": [":HGT:925 mb"], "model": "rap"},
    "hgt_1000mb_rap": {"aliases": [":HGT:1000 mb"], "model": "rap"},
    "tmp_2m_rap": {"aliases": [":TMP:2 m above ground"], "model": "rap"},
    "rh_2m_rap": {"aliases": [":RH:2 m above ground"], "model": "rap"},
    "rh_925mb_rap": {"aliases": [":RH:925 mb"], "model": "rap"},
    "hpbl_surface_rap": {"aliases": [":HPBL:surface"], "model": "rap"},
    "hgt_0C_iso_rap": {"aliases": [":HGT:0C isotherm:"], "model": "rap"},
    "prate_surface_rap": {"aliases": [":PRATE:surface:%n hour"], "model": "rap"},
    "apcp_surface_rap": {"aliases": [":APCP:surface"], "model": "rap"},
    # --- ECMWF IFS open data (Herbie model="ifs"). Provides geopotential height,
    # temperature, humidity, vertical velocity and surface/integrated fields, but
    # NO cloud-cover/ceiling fields -- those columns stay empty (that's expected).
    # IFS open data has only 3-hourly steps, so non-multiple-of-3 fxx come back
    # empty as well. All fine: empties are imputed downstream.
    "hgt_500mb_ecmwf": {"aliases": [":gh:500:"], "model": "ifs"},
    "hgt_700mb_ecmwf": {"aliases": [":gh:700:"], "model": "ifs"},
    "hgt_850mb_ecmwf": {"aliases": [":gh:850:"], "model": "ifs"},
    "hgt_925mb_ecmwf": {"aliases": [":gh:925:"], "model": "ifs"},
    "hgt_1000mb_ecmwf": {"aliases": [":gh:1000:"], "model": "ifs"},
    "tmp_500mb_ecmwf": {"aliases": [":t:500:"], "model": "ifs"},
    "tmp_700mb_ecmwf": {"aliases": [":t:700:"], "model": "ifs"},
    "tmp_850mb_ecmwf": {"aliases": [":t:850:"], "model": "ifs"},
    "tmp_925mb_ecmwf": {"aliases": [":t:925:"], "model": "ifs"},
    "tmp_1000mb_ecmwf": {"aliases": [":t:1000:"], "model": "ifs"},
    "rh_700mb_ecmwf": {"aliases": [":r:700:"], "model": "ifs"},
    "rh_850mb_ecmwf": {"aliases": [":r:850:"], "model": "ifs"},
    "rh_925mb_ecmwf": {"aliases": [":r:925:"], "model": "ifs"},
    "rh_1000mb_ecmwf": {"aliases": [":r:1000:"], "model": "ifs"},
    "vvel_700mb_ecmwf": {"aliases": [":w:700:"], "model": "ifs"},
    "vvel_850mb_ecmwf": {"aliases": [":w:850:"], "model": "ifs"},
    "vvel_925mb_ecmwf": {"aliases": [":w:925:"], "model": "ifs"},
    "tmp_2m_ecmwf": {"aliases": [":2t:"], "model": "ifs"},
    "dpt_2m_ecmwf": {"aliases": [":2d:"], "model": "ifs"},
    "mslp_ecmwf": {"aliases": [":msl:"], "model": "ifs"},
    "sp_surface_ecmwf": {"aliases": [":sp:"], "model": "ifs"},
    "cape_ecmwf": {"aliases": [":cape:"], "model": "ifs"},
    "tcwv_ecmwf": {"aliases": [":tcwv:"], "model": "ifs"},
    # --- NBM (National Blend of Models), CONUS "co" product. Statistical blend
    # rich in sensible-weather elements: total cloud cover, ceiling, visibility
    # (deterministic), plus 2 m temp/dewpoint/RH, wind and precip. Hourly, so all
    # fxx populate. No upper-air fields, so pressure-level columns stay empty.
    "tcdc_surface_nbm": {"aliases": [":TCDC:surface:%n hour fcst:nan:nan", ":TCDC:surface"], "model": "nbm"},
    "tcdc_high_cloud_nbm": {"aliases": [":TCDC:high cloud layer"], "model": "nbm"},
    "cdcb_high_cloud_nbm": {"aliases": [":CDCB:high cloud layer"], "model": "nbm"},
    "cloud_ceiling_m_nbm": {"aliases": [":CEIL:cloud ceiling:%n hour fcst:nan:nan", ":CEIL:cloud ceiling"], "model": "nbm"},
    "cloud_base_m_nbm": {"aliases": [":CEIL:cloud base"], "model": "nbm"},
    "vis_surface_nbm": {"aliases": [":VIS:surface:%n hour fcst:nan:nan", ":VIS:surface"], "model": "nbm"},
    # Probabilistic ceiling/visibility-below-threshold (%) -- directly relevant to
    # undercast (low ceiling / restricted visibility). Regex-anchored to a threshold.
    "ceil_prob_below_152m_nbm": {"aliases": [":CEIL:cloud ceiling:.*prob <152.4:"], "model": "nbm"},
    "ceil_prob_below_305m_nbm": {"aliases": [":CEIL:cloud ceiling:.*prob <304.8:"], "model": "nbm"},
    "ceil_prob_below_610m_nbm": {"aliases": [":CEIL:cloud ceiling:.*prob <609.6:"], "model": "nbm"},
    "ceil_prob_below_914m_nbm": {"aliases": [":CEIL:cloud ceiling:.*prob <914.5:"], "model": "nbm"},
    "ceil_prob_below_2012m_nbm": {"aliases": [":CEIL:cloud ceiling:.*prob <2011.68:"], "model": "nbm"},
    "vis_prob_below_1609m_nbm": {"aliases": [":VIS:surface:.*prob <1609.34:"], "model": "nbm"},
    "vis_prob_below_3219m_nbm": {"aliases": [":VIS:surface:.*prob <3218.69:"], "model": "nbm"},
    "vis_prob_below_4828m_nbm": {"aliases": [":VIS:surface:.*prob <4828.03:"], "model": "nbm"},
    "vis_prob_below_8047m_nbm": {"aliases": [":VIS:surface:.*prob <8046.73:"], "model": "nbm"},
    "cape_surface_nbm": {"aliases": [":CAPE:surface:%n hour fcst:nan:nan", ":CAPE:surface"], "model": "nbm"},
    "mixing_height_nbm": {"aliases": [":MIXHT:entire atmosphere"], "model": "nbm"},
    "tmp_2m_nbm": {"aliases": [":TMP:2 m above ground"], "model": "nbm"},
    "dpt_2m_nbm": {"aliases": [":DPT:2 m above ground"], "model": "nbm"},
    "rh_2m_nbm": {"aliases": [":RH:2 m above ground"], "model": "nbm"},
    "apcp_surface_nbm": {"aliases": [":APCP:surface"], "model": "nbm"},
    "wind_10m_nbm": {"aliases": [":WIND:10 m above ground"], "model": "nbm"},
    "gust_surface_nbm": {"aliases": [":GUST:10 m above ground", ":GUST:surface"], "model": "nbm"},
}


def try_load(candidates, H):
    """Try each candidate name with H.xarray and return the first successful DataArray."""
    for name in candidates:
        try:
            da = H.xarray(name)
            if da is not None:
                return da, name
        except Exception:
            continue
    return None, None


def sample_nearest(da, lat, lon):
    """Select nearest value from DataArray using common coordinate name variants."""
    sel_opts = [
        {"lat": lat, "lon": lon},
        {"latitude": lat, "longitude": lon},
        {"y": lat, "x": lon},
        {"grid_latitude": lat, "grid_longitude": lon},
    ]
    for opts in sel_opts:
        try:
            point = da.sel(method="nearest", **opts)
            val = point.squeeze()
            try:
                return float(val[list(val.data_vars)[0]].values)
            except Exception:
                return val
        except Exception:
            continue
    try:
        lat_dim = next(d for d in da.coords if "lat" in d.lower())
        lon_dim = next(d for d in da.coords if "lon" in d.lower())
        ilat = abs(da[lat_dim] - lat).argmin().item()
        ilon = abs(da[lon_dim] - lon).argmin().item()
        val = da.isel({lat_dim: ilat, lon_dim: ilon}).squeeze()
        try:
            return float(val.values)
        except Exception:
            return val
    except Exception:
        try:
            np_point, iy, ix, dkm = find_nearest_by_geodetic(da, lat, lon)
            try:
                return float(np_point[list(np_point.data_vars)[0]].values)
            except Exception:
                return np_point
        except Exception:
            raise


def find_nearest_by_geodetic(da, lat0, lon0, lat_name_hint="lat", lon_name_hint="lon"):
    """Find the nearest grid point in `da` to (lat0, lon0) using great-circle distance."""
    lat_da = None
    lon_da = None
    for name in da.coords:
        nl = name.lower()
        if lat_da is None and lat_name_hint in nl:
            lat_da = da.coords[name]
        if lon_da is None and lon_name_hint in nl:
            lon_da = da.coords[name]
    if lat_da is None and "latitude" in da.coords:
        lat_da = da.coords["latitude"]
    if lon_da is None and "longitude" in da.coords:
        lon_da = da.coords["longitude"]

    if lat_da is None or lon_da is None:
        raise ValueError("Could not find 2D latitude/longitude coordinates in DataArray")

    lat_vals = np.asarray(lat_da.values)
    lon_vals = np.asarray(lon_da.values)

    lon_max = float(np.nanmax(lon_vals))
    if lon_max > 180:
        lon0 = lon0 % 360
    else:
        if lon0 > 180:
            lon0 = ((lon0 + 180) % 360) - 180

    def haversine_km(lat1, lon1, lat2, lon2):
        lat1r = np.deg2rad(lat1)
        lon1r = np.deg2rad(lon1)
        lat2r = np.deg2rad(lat2)
        lon2r = np.deg2rad(lon2)
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))

    dist_grid = haversine_km(lat_vals, lon_vals, lat0, lon0)
    flat_idx = np.nanargmin(dist_grid.ravel())
    iy, ix = divmod(int(flat_idx), dist_grid.shape[1])

    lat_dims = lat_da.dims
    if len(lat_dims) != 2:
        dims = tuple(da.dims[:2])
    else:
        dims = lat_dims

    sel = {dims[0]: iy, dims[1]: ix}
    nearest_point = da.isel(sel)
    distance_km = float(dist_grid[iy, ix])

    return nearest_point, iy, ix, distance_km


def process_undercast_row(args):
    """Process one date with a hard wall-clock cap.

    Herbie downloads via requests/urllib3, which ignore socket.setdefaulttimeout,
    so a dropped connection can block a worker in recv() forever (unkillable).
    A per-date SIGALRM interrupts the blocked syscall, aborts the date (returns
    None so --resume retries it next pass) and frees the worker. Runs in the
    worker's main thread, where signal handlers are allowed.
    """
    import signal

    def _on_timeout(signum, frame):
        raise TimeoutError("per-date wall-clock timeout")

    have_alarm = hasattr(signal, "SIGALRM")
    if have_alarm:
        old = signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(1800)  # 30 min backstop; curl --max-time catches real hangs
    try:
        return _process_undercast_row_impl(args)
    except TimeoutError:
        print(f"\nTimeout: aborting {args[1][1] if len(args[1]) > 1 else '?'} after 1800s")
        return None
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _process_undercast_row_impl(args):
    """Process a single row of undercast data"""
    index, row, next_row, LOCATIONS, variables = args

    if len(row) < 5:
        return None

    short_date = row[1]
    avg = row[4]

    avg_next = None
    if next_row and len(next_row) >= 5:
        avg_next = next_row[4]

    # Hourly forecast-hour sampling. Base covers the first 8 hours; when the
    # next day carries the same undercast label the condition is stable across
    # the day boundary, so it's safe to extend the (identically-labeled) samples
    # out to a full 24 hours.
    FXX_LIST = list(range(1, 8 + 1))
    if avg is not None and avg_next is not None:
        if float(avg_next) == float(avg):
            FXX_LIST = list(range(1, 24 + 1))

    date_str = datetime.strptime(short_date, "%m/%d/%y").strftime("%Y-%m-%d %H:%M")

    try:
        with tempfile.TemporaryDirectory() as tmp:

            class HsCollection:
                def __init__(self):
                    self._entries = []

                def add(self, fxx, hobj):
                    self._entries.append((fxx, hobj))

                def items(self):
                    return iter(self._entries)

                def keys(self):
                    return [fxx for fxx, _ in self._entries]

                def __len__(self):
                    return len(self._entries)

                def __getitem__(self, key):
                    for fxx, h in self._entries:
                        if fxx == key:
                            return h
                    raise KeyError(key)

            Hs = HsCollection()

            successful_fxxs = []
            for fxx in FXX_LIST:
                try:
                    h = Herbie(date_str, model="nam", fxx=fxx, save_dir=tmp)
                    Hs.add(fxx, h)
                    successful_fxxs.append(fxx)
                except (ConnectionResetError, ConnectionError, Exception) as e:
                    print(f"\nSkipping {date_str} fxx={fxx} due to error: {type(e).__name__}")
                    continue
                try:
                    h = Herbie(date_str, model="hrrr", fxx=fxx, save_dir=tmp)
                    Hs.add(fxx, h)
                    successful_fxxs.append(fxx)
                except (ConnectionResetError, ConnectionError, Exception) as e:
                    print(f"\nSkipping {date_str} fxx={fxx} due to error: {type(e).__name__}")
                    continue
                try:
                    h = Herbie(date_str, model="gfs", fxx=fxx, save_dir=tmp)
                    Hs.add(fxx, h)
                    successful_fxxs.append(fxx)
                except (ConnectionResetError, ConnectionError, Exception) as e:
                    print(f"\nSkipping {date_str} fxx={fxx} due to error: {type(e).__name__}")
                    continue
                # RAP, ECMWF IFS and NBM are added independently (don't let one's
                # absence skip the others, and empties are acceptable).
                try:
                    h = Herbie(date_str, model="rap", fxx=fxx, save_dir=tmp)
                    Hs.add(fxx, h)
                    successful_fxxs.append(fxx)
                except (ConnectionResetError, ConnectionError, Exception) as e:
                    print(f"\nSkipping {date_str} rap fxx={fxx} due to error: {type(e).__name__}")
                try:
                    h = Herbie(date_str, model="ifs", product="oper", fxx=fxx, save_dir=tmp)
                    Hs.add(fxx, h)
                    successful_fxxs.append(fxx)
                except (ConnectionResetError, ConnectionError, Exception) as e:
                    print(f"\nSkipping {date_str} ecmwf fxx={fxx} due to error: {type(e).__name__}")
                try:
                    h = Herbie(date_str, model="nbm", product="co", fxx=fxx, save_dir=tmp)
                    Hs.add(fxx, h)
                    successful_fxxs.append(fxx)
                except (ConnectionResetError, ConnectionError, Exception) as e:
                    print(f"\nSkipping {date_str} nbm fxx={fxx} due to error: {type(e).__name__}")


            if len(successful_fxxs) == 0:
                print(f"\nSkipping {date_str} - no successful data downloads")
                return None

            H = Hs[successful_fxxs[0]]

            results = {}
            for loc in LOCATIONS:
                lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}".replace(
                    " ", "_"
                )
                results[lname] = {label: {} for label in variables.keys()}

            for i, (fxx, H) in enumerate(Hs.items()):
                for label, candidates in variables.items():
                    req_model = candidates.get("model")
                    candidates = dict(candidates)
                    orig_aliases = candidates.get("aliases", [])
                    processed = []
                    for a in orig_aliases:
                        if isinstance(a, str) and "%n" in a:
                            processed.append(a.replace("%n", str(fxx)))
                        else:
                            processed.append(a)
                    candidates["aliases"] = processed

                    if req_model is not None:
                        h_model = getattr(H, "model", None)
                        if str(h_model).lower() != str(req_model).lower():
                            continue

                    try:
                        da, used_name = try_load(candidates["aliases"], H)
                    except (ConnectionResetError, ConnectionError, Exception) as e:
                        for loc in LOCATIONS:
                            lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}"
                            results[lname][label][fxx] = {
                                "error": f"Connection error: {type(e).__name__}",
                                "tried": candidates["aliases"],
                            }
                        continue

                    if da is None:
                        for loc in LOCATIONS:
                            lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}"
                            results[lname][label][fxx] = {
                                "error": "could not load variable",
                                "tried": candidates["aliases"],
                            }
                        continue

                    for loc in LOCATIONS:
                        lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}"
                        lat = loc.get("lat")
                        lon = loc.get("lon")
                        try:
                            value = sample_nearest(da, lat, lon)
                            entry = {"variable": used_name, "value": value}
                            results[lname][label][fxx] = entry
                        except Exception as exc:
                            results[lname][label][fxx] = {"variable": used_name, "error": str(exc)}

                    try:
                        if hasattr(da, "close"):
                            da.close()
                    except Exception:
                        pass

            return (date_str, avg, results, sorted(set(Hs.keys())))

    except (ConnectionResetError, ConnectionError, Exception) as e:
        print(f"\nSkipping {date_str} entirely due to error: {type(e).__name__}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build per-date undercast training CSVs from a labeled-dates file."
    )
    parser.add_argument(
        "--labels",
        default="files/weather/csv/MtWashington_undercast.csv",
        help="CSV of labeled dates (columns: Date, Short Date, Tower, Observatory, Avg).",
    )
    parser.add_argument(
        "--output-dir",
        default="files/weather/csv",
        help="Directory for the per-date {date}_{location}.csv output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent worker processes. Downloads are network-bound, so it can "
        "help to oversubscribe past the core count (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dates whose output CSV already exists AND contains the newest "
        "schema columns, so an interrupted run can be re-run cheaply.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the date list into this many shards (for parallel CI jobs). "
        "Each shard processes dates where index %% num_shards == shard.",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Which shard (0-based, < --num-shards) this run should process.",
    )
    args = parser.parse_args()

    if not (0 <= args.shard < args.num_shards):
        parser.error("--shard must satisfy 0 <= shard < num-shards")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.labels)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        undercast_data = list(reader)

    def _already_complete(row):
        """True if every location's CSV for this date exists and has hgt columns."""
        try:
            date_str = datetime.strptime(row[1], "%m/%d/%y").strftime("%Y-%m-%d %H:%M")
        except (ValueError, IndexError):
            return False
        stem = date_str.replace(" ", "_").replace(":", "-")
        for loc in LOCATIONS:
            p = output_dir / f"{stem}_{loc['name']}.csv"
            if not p.exists():
                return False
            with open(p) as fh:
                if "gust_surface_nbm" not in fh.readline():
                    return False
        return True

    tasks = []
    skipped = 0
    for index, row in enumerate(undercast_data[1:]):
        next_row = undercast_data[index + 2] if index + 1 < len(undercast_data[1:]) else None
        # Sharding: split dates across parallel CI jobs by round-robin on index.
        if args.num_shards > 1 and index % args.num_shards != args.shard:
            continue
        if args.resume and _already_complete(row):
            skipped += 1
            continue
        tasks.append((index, row, next_row, LOCATIONS, variables))

    if args.resume:
        print(f"Resume: {skipped} dates already complete, {len(tasks)} to (re)process.")
    if not tasks:
        print("Nothing to do.")
        raise SystemExit(0)

    max_workers = min(args.workers or (os.cpu_count() or 4), len(tasks))
    print(f"Processing {len(tasks)} dates with {max_workers} workers -> {output_dir}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_undercast_row, task): i for i, task in enumerate(tasks)}

        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            result = future.result()
            if result:
                date_str, avg, results, fxxs = result

                for location_name, location_data in results.items():
                    csv_path = (
                        output_dir
                        / f"{date_str.replace(' ', '_').replace(':', '-')}_{location_name}.csv"
                    )

                    with open(csv_path, "w") as f:
                        header = ["fxx"] + list(location_data.keys()) + ["is_undercast"]
                        f.write(",".join(header) + "\n")

                        for fxx in fxxs:
                            row = [str(fxx)]
                            for label in location_data.keys():
                                entry = location_data[label].get(fxx, {})
                                if isinstance(entry, dict) and "value" in entry:
                                    row.append(str(entry["value"]))
                                else:
                                    row.append("")
                            row.append(str(avg))
                            f.write(",".join(row) + "\n")

            completed += 1
            pct = int(round(completed / total * 100))
            print(f"\rProgress: {pct:3d}%", end="", flush=True)

        print()
