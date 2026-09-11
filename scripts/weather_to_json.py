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
import pandas as pd
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Grid-point sampling lives in grib_sample.py, NOT here. It used to be copied
# into both this file and the other fetcher, and a longitude fix landed in only
# one copy -- see that module's docstring.
from grib_sample import (  # noqa: E402
    _match_lon_convention, _to_scalar, find_nearest_by_geodetic, sample_nearest,
)


# xarray emits a FutureWarning on .argmin()/.argmax() without an explicit dim
# (used by the nearest-gridpoint lookup). The current flat-index behaviour is
# exactly what we want, so silence the deprecation noise.
warnings.filterwarnings("ignore", category=FutureWarning)


# Helper to convert NaN/numpy types to JSON-serializable values (NaN -> None)
def _clean_for_json(obj):
    # unwrap numpy scalars
    try:
        if isinstance(obj, np.generic):
            obj = obj.item()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj
    if isinstance(obj, (int, bool)) or obj is None:
        return obj
    return str(obj)


LOCATIONS = [
    {"name": "MtWashington", "lat": 44.27040, "lon": -71.30327},
]

# Two models are served from this file at once, and the variable names below have
# to satisfy both.
#
# The LEGACY models (files/weather/models/) were fitted on the unsuffixed name
# "boundary_layer_cloud_layer", so renaming it here would make them throw on a
# missing column. The CURRENT model (files/weather/models/obs/) was trained on
# "boundary_layer_cloud_layer_hrrr". Rather than rename and break one side, the
# fetch key stays as it is and build_current_features() aliases the suffixed name
# onto it -- the two are the same field, fetched with identical GRIB aliases.
#
# "hgt_925mb_hrrr" (HRRR "prs" product only) and "boundary_layer_cloud_layer_nam"
# (NAM never publishes it) are kept purely because the legacy preprocessors expect
# the columns to exist. The current model does not use either.
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
    "tmp_2m_hrrr": {"aliases": [":TMP:2 m above ground"], "model": "hrrr"},
    "hgt_500mb_hrrr": {"aliases": [":HGT:500 mb"], "model": "hrrr"},
    "hgt_700mb_hrrr": {"aliases": [":HGT:700 mb"], "model": "hrrr"},
    "hgt_850mb_hrrr": {"aliases": [":HGT:850 mb"], "model": "hrrr"},
    "hgt_925mb_hrrr": {"aliases": [":HGT:925 mb"], "model": "hrrr"},
    "hgt_1000mb_hrrr": {"aliases": [":HGT:1000 mb"], "model": "hrrr"},
    "hgt_surface_hrrr": {"aliases": [":HGT:surface"], "model": "hrrr"},
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
        # NOT the ":%n hour" form: GFS indexes this field with no forecast-hour
        # qualifier, so that alias matched nothing and the column came back 100%
        # empty. The training fetcher was corrected first; this is the mirror.
        "aliases": [":TCDC:boundary layer cloud layer"],
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
    "tmp_2m_gfs": {"aliases": [":TMP:2 m above ground"], "model": "gfs"},
    "hgt_500mb_gfs": {"aliases": [":HGT:500 mb"], "model": "gfs"},
    "hgt_700mb_gfs": {"aliases": [":HGT:700 mb"], "model": "gfs"},
    "hgt_850mb_gfs": {"aliases": [":HGT:850 mb"], "model": "gfs"},
    "hgt_925mb_gfs": {"aliases": [":HGT:925 mb"], "model": "gfs"},
    "hgt_1000mb_gfs": {"aliases": [":HGT:1000 mb"], "model": "gfs"},
    "hgt_surface_gfs": {"aliases": [":HGT:surface"], "model": "gfs"},
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
    "tmp_2m_nam": {"aliases": [":TMP:2 m above ground"], "model": "nam"},
    "hgt_500mb_nam": {"aliases": [":HGT:500 mb"], "model": "nam"},
    "hgt_700mb_nam": {"aliases": [":HGT:700 mb"], "model": "nam"},
    "hgt_850mb_nam": {"aliases": [":HGT:850 mb"], "model": "nam"},
    "hgt_925mb_nam": {"aliases": [":HGT:925 mb"], "model": "nam"},
    "hgt_1000mb_nam": {"aliases": [":HGT:1000 mb"], "model": "nam"},
    "hgt_surface_nam": {"aliases": [":HGT:surface"], "model": "nam"},
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
    # --- ECMWF IFS open data (Herbie model="ifs"). Geopotential height,
    # temperature, humidity, vertical velocity and surface/integrated fields; NO
    # cloud fields (those columns stay empty). Only 3-hourly steps, so fxx that
    # aren't multiples of 3 come back empty too. Empties are expected/fine.
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
    # --- NBM (National Blend of Models), CONUS "co" product. Total cloud cover,
    # ceiling, visibility (deterministic), 2 m temp/dewpoint/RH, wind and precip.
    # No upper-air fields (pressure-level columns stay empty).
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


def try_load(candidates, hobj=None):
    """Try each candidate name with H.xarray and return the first successful DataArray."""
    for name in candidates:
        try:
            da = hobj.xarray(name)
            if da is not None:
                return da, name
        except Exception:
            continue
    return None, None


# How far back to look for a run that has actually published, and the maximum
# lead any model is asked for. ECMWF open data is the reason this exists: IFS
# lags the wall clock by well over six hours, so the most recent 6-hourly slot is
# reliably absent. Asking for it and taking the miss meant every ECMWF column was
# empty in every live run -- silently, because a missing column is just a column
# of nulls. The combined model then ran on five sources where it was trained on
# six.
MAX_RUN_LOOKBACK_H = 30
MODEL_MAX_LEAD_H = {"hrrr": 48, "rap": 51, "nam": 84, "gfs": 384, "ifs": 144, "nbm": 264}
# How far back to step while hunting for a usable run, per model. Six hours is
# the sensible default because the base grid is 6-hourly -- but RAP is the
# exception that matters: its 00/06/12/18Z cycles stop at 21 h and only the
# 03/09/15/21Z cycles reach 51. Stepping in sixes can therefore never find a RAP
# run that covers a 48 h forecast, and the combined model would silently lose
# every hour past 21. Stepping in threes lands on the extended cycles. The
# offset need not be a multiple of six: alignment is by valid time, and every
# forecast hour is shifted by whatever the offset turns out to be.
RUN_STEP_H = {"rap": 3}


def resolve_run(model, date_str, probe_fxx, max_back_h=MAX_RUN_LOOKBACK_H, step_h=None):
    """The most recent run of `model` that has actually published IN FULL.

    Returns (run_date_str, offset_hours). An offset of 6 means "this model's
    latest available run is 6 h older than the common base time", and every
    forecast hour asked of it must be 6 h longer to land on the same valid time.

    `probe_fxx` should be a middling lead. Runs upload in lead order, so probing
    at F06 accepts a run that started minutes ago and then most of the real
    fetches miss -- measured: a run started just after 18Z passed an F06 probe and
    returned nothing for three of six models. Probing at the LONGEST lead is the
    opposite error, because not every cycle reaches it: RAP's 00/06/12/18Z cycles
    stop at 21 h, so an F48 probe rejected every RAP run outright.

    Availability is decided by whether Herbie can locate the GRIB, not by
    downloading it -- one cheap index lookup per model rather than per hour.
    """
    base = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    _products = {"hrrr": "sfc", "ifs": "oper", "nbm": "co"}
    step = step_h or RUN_STEP_H.get(model, 6)
    for offset in range(0, max_back_h + 1, step):
        run = base - timedelta(hours=offset)
        try:
            h = Herbie(
                run.strftime("%Y-%m-%d %H:%M"),
                model=model,
                product=_products.get(model),
                fxx=probe_fxx + offset,
                verbose=False,
            )
            if getattr(h, "grib", None):
                if offset:
                    print(f"  {model}: latest published run is {offset} h back "
                          f"({run:%Y-%m-%d %H}Z); leads shifted to match valid times")
                return run.strftime("%Y-%m-%d %H:%M"), offset
        except Exception:
            continue
    print(f"  {model}: no published run found within {max_back_h} h; "
          f"falling back to the base time and accepting the misses")
    return date_str, 0


def process_forecast_data(args):
    """Fetch one forecast hour.

    `fxx` is the hour this result is REPORTED at -- hours since the common base
    time that every model is aligned to. `run_date_str` and `run_fxx` are what is
    actually asked of Herbie, and they can differ: a model whose latest run has
    not published yet is fetched from an earlier run at a longer lead, which is
    the same valid time from a staler forecast. Keying the result by `fxx` keeps
    every model on one time axis, which is what the combined model needs.
    """
    fxx, run_date_str, run_fxx, model, LOCATIONS, variables = args
    date_str = run_date_str

    try:
        with tempfile.TemporaryDirectory() as tmp:
            _products = {"hrrr": "sfc", "ifs": "oper", "nbm": "co"}
            try:
                h = Herbie(
                    run_date_str,
                    model=model,
                    product=_products.get(model),
                    fxx=run_fxx,
                    save_dir=tmp,
                )
            except Exception as e:
                return None

            results = {}
            for loc in LOCATIONS:
                lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}".replace(
                    " ", "_"
                )
                results[lname] = {}

                for label, candidates in variables.items():
                    req_model = candidates.get("model")
                    if req_model is not None and str(req_model).lower() != str(model).lower():
                        continue

                    candidates = dict(candidates)
                    orig_aliases = candidates.get("aliases", [])
                    processed = []
                    for a in orig_aliases:
                        if isinstance(a, str) and "%n" in a:
                            # run_fxx, NOT fxx. Several aliases pin the forecast
                            # hour inside the GRIB message name (":APCP:surface:6
                            # hour fcst"), and that hour belongs to the file being
                            # read. When a model is fetched from an older run at a
                            # longer lead, the two differ, and substituting the
                            # reported hour builds a name that matches nothing --
                            # losing the variable silently.
                            processed.append(a.replace("%n", str(run_fxx)))
                        else:
                            processed.append(a)
                    candidates["aliases"] = processed

                    try:
                        da, used_name = try_load(candidates["aliases"], h)
                    except Exception as e:
                        results[lname][label] = {
                            "error": f"Connection error: {type(e).__name__}",
                            "tried": candidates["aliases"],
                        }
                        continue

                    if da is None:
                        results[lname][label] = {
                            "error": "could not load variable",
                            "tried": candidates["aliases"],
                        }
                        continue

                    lat = loc.get("lat")
                    lon = loc.get("lon")
                    try:
                        value = sample_nearest(da, lat, lon)
                        results[lname][label] = {"variable": used_name, "value": value}
                    except Exception as exc:
                        results[lname][label] = {"variable": used_name, "error": str(exc)}

                    try:
                        if hasattr(da, "close"):
                            da.close()
                    except Exception:
                        pass

            return (fxx, model, results)

    except Exception as e:
        print(f"\nSkipping fxx={fxx} model={model} due to error: {type(e).__name__}")
        return None


CURRENT_MODEL_DIR = "files/weather/models/obs"
# Combined source, Gradient Boosting, per-lead thresholds. Chosen by measurement,
# not preference: on the hand-labeled webcam days -- the only holdout scored
# against a human looking at a photograph -- it beats the deployed 2-of-3 vote by
# F1 +0.090 [+0.010, +0.175] on a paired day-block bootstrap, and beats XGBoost
# by +0.084 [+0.002, +0.181]. On the base-rate holdout it ties both. See
# scripts/compare_undercast_ensembles.py.
CURRENT_SOURCE = "all"
CURRENT_ALGO = "Gradient Boosting"
# Below this share of the model's features actually present, refuse to publish a
# forecast at all. Every numeric carries a missingness indicator, so absent
# columns do not crash -- they quietly become "not reported", and the model would
# return a confident-looking number built on almost nothing. A blank panel is a
# better answer than a wrong one.
MIN_FEATURE_COVERAGE = 0.80
# ...and the same argument one level down. Column PRESENCE is not enough:
# results_to_dataframe emits a column for every variable whether or not the
# download succeeded, so an entire model going offline leaves its columns present
# and empty. The combined model is defined only on rows where all six sources
# reported -- that is how it was trained -- so a source that is wholly absent puts
# it out of distribution while nothing raises. Checked per source, not just in
# aggregate, because one missing source out of six is easy to lose in an average.
MIN_SOURCE_POPULATED = 0.30
CURRENT_REQUIRES_SOURCES = ("hrrr", "nam", "gfs", "rap", "ecmwf", "nbm")


def build_current_features(weather_df, date_str):
    """The 213 columns the current combined model was trained on.

    Every step here has a counterpart in train_undercast_obs.load_obs_data, and
    the shared helpers are imported from that module rather than reimplemented --
    this is the train/serve boundary, and it is where skew would be invisible.

    Returns (X, valid_times) or raises.
    """
    import numpy as np
    import pandas as pd

    from train_undercast_obs import (
        add_profile_features, normalize_weather_columns, time_features,
    )

    base = datetime.strptime(date_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    fxx = pd.to_numeric(weather_df["fxx"], errors="coerce")
    valid = pd.Series([base + timedelta(hours=int(h)) if pd.notna(h) else pd.NaT
                       for h in fxx], index=weather_df.index)
    valid = pd.to_datetime(valid, utc=True)

    raw = weather_df.drop(columns=["fxx", "month", "day"], errors="ignore").copy()
    # Same field, two names: the legacy preprocessors want the unsuffixed one,
    # this model was trained on the suffixed one. Fetched with identical aliases.
    if "boundary_layer_cloud_layer" in raw.columns:
        raw["boundary_layer_cloud_layer_hrrr"] = raw["boundary_layer_cloud_layer"]

    out = pd.DataFrame(time_features(valid), index=raw.index)
    built = normalize_weather_columns(raw, list(raw.columns))
    out = pd.concat([out, pd.DataFrame(built, index=raw.index)], axis=1)
    out = pd.concat([out, add_profile_features(out)], axis=1)
    return out, valid


def _lead_threshold(meta_algo, lead_h):
    """Threshold for an arbitrary forecast hour, interpolated between the trained ones.

    Thresholds were fitted at leads 1, 24 and 48 only; the page forecasts every
    hour in between. Interpolating rather than snapping avoids a visible step in
    the published series at an arbitrary hour. For this model the three values
    are 0.775 / 0.780 / 0.725, so the choice barely moves anything -- it is about
    not introducing an artefact.
    """
    import numpy as np

    by = meta_algo.get("threshold_by_lead") or {}
    if not by:
        return float(meta_algo["threshold"])
    leads = np.array(sorted(int(k) for k in by))
    vals = np.array([float(by[str(k)]) for k in leads])
    return float(np.interp(float(lead_h), leads, vals))


def predict_current_model(weather_df, date_str, max_fxx):
    """Undercast probability and call per forecast hour, from the current model.

    Deliberately additive: the legacy per-source, per-algorithm outputs are left
    exactly as they were, so nothing that exists today can break if this path
    fails. It raises on any problem and the caller drops the key.
    """
    import joblib
    import numpy as np
    import pandas as pd

    meta_path = f"{CURRENT_MODEL_DIR}/model_metadata_{CURRENT_SOURCE}.json"
    with open(meta_path) as fh:
        meta = json.load(fh)
    wanted = list(meta["feature_columns"])

    X_all, valid = build_current_features(weather_df, date_str)
    present = [c for c in wanted if c in X_all.columns]
    coverage = len(present) / len(wanted)
    populated = float(X_all.reindex(columns=wanted).notna().to_numpy().mean())
    print(f"[current model] {len(present)}/{len(wanted)} feature columns present "
          f"({coverage:.1%}), {populated:.1%} of cells populated")
    if coverage < MIN_FEATURE_COVERAGE:
        missing = [c for c in wanted if c not in X_all.columns]
        raise RuntimeError(
            f"only {coverage:.1%} of the model's features could be built "
            f"(need {MIN_FEATURE_COVERAGE:.0%}); missing e.g. {missing[:8]}"
        )
    X = X_all.reindex(columns=wanted)

    # Per ROW, not per frame. The forecast hours in this table are the union of
    # six different cadences: HRRR/RAP/NBM stop at 48 h, NAM at 60, GFS runs to
    # 120, and ECMWF is 3-hourly where the rest are 2-hourly. So an hour like
    # fxx=3 exists only because ECMWF reported it, and every other source is
    # blank on that row -- while fxx=50 has GFS and nothing else.
    #
    # The combined model was trained only on rows where all six sources reported
    # (rows_for_source("all") requires it), and every numeric carries a
    # missingness indicator, so a row missing five sources does not fail: it
    # produces a calm, confident-looking number out of almost nothing. The first
    # live run published exactly that -- a repeated 0.095 at every odd hour.
    # Those rows are nulled, the same as hours past the model's reach.
    usable = pd.Series(True, index=X.index)
    for src in CURRENT_REQUIRES_SOURCES:
        cols = [c for c in wanted if c.endswith(f"_{src}")]
        if not cols:
            continue
        filled_row = X[cols].notna().mean(axis=1)
        usable &= filled_row >= MIN_SOURCE_POPULATED
        print(f"[current model]   {src:6s} {len(cols):3d} columns, "
              f"{float(X[cols].notna().to_numpy().mean()):.0%} populated overall, "
              f"{int((filled_row >= MIN_SOURCE_POPULATED).sum())}/{len(X)} rows usable")
    print(f"[current model] {int(usable.sum())}/{len(X)} forecast hours have every "
          f"source reporting")
    if not usable.any():
        raise RuntimeError(
            "no forecast hour has all six sources reporting; the combined model "
            "cannot be applied to any of them. Publishing nothing instead."
        )

    pre = joblib.load(f"{CURRENT_MODEL_DIR}/preprocessor_{CURRENT_SOURCE}.pkl")
    model = joblib.load(
        f"{CURRENT_MODEL_DIR}/gradient_boosting_best_f1_{CURRENT_SOURCE}.pkl"
    )
    proba = model.predict_proba(pre.transform(X))[:, 1]

    fxx = pd.to_numeric(weather_df["fxx"], errors="coerce")
    xs, ys, ps, ts = [], [], [], []
    for i, h in enumerate(fxx):
        if pd.isna(h):
            continue
        h = int(h)
        xs.append(h)
        if h > max_fxx or not bool(usable.iloc[i]):
            # Either past where every source still has data, or an hour only some
            # of them reported. Both leave the combined model without its inputs.
            # Null, not a guess.
            ys.append(None)
            ps.append(None)
            ts.append(None)
            continue
        thr = _lead_threshold(meta[CURRENT_ALGO], h)
        ys.append(int(proba[i] >= thr))
        ps.append(round(float(proba[i]), 4))
        ts.append(round(thr, 4))

    m = meta[CURRENT_ALGO]
    by_lead = m.get("baserate_by_lead", {})
    return {
        "status": "ok",
        "x": xs,
        "y": ys,
        "probability": ps,
        "threshold": ts,
        "valid_utc": [v.strftime("%Y-%m-%dT%H:%M") if pd.notna(v) else None
                      for v in valid],
        "model": {
            "source": CURRENT_SOURCE,
            "algorithm": CURRENT_ALGO,
            "label": "Combined (Gradient Boosting)",
            "trained_on": meta.get("year_range"),
            "n_train_rows": meta.get("n_train_rows"),
            # Skill at the three leads it was measured at, so the page can say
            # how much to trust a call at the hour being looked at.
            "skill_by_lead": {
                k: {"precision": round(v["precision"], 3),
                    "recall": round(v["recall"], 3),
                    "roc_auc": round(v["roc_auc"], 3)}
                for k, v in by_lead.items()
            },
        },
    }


def results_to_dataframe(results, locations, date_str):
    """Convert the results dictionary to a pandas DataFrame."""
    rows = []

    try:
        base_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except Exception:
        base_date = None

    for loc in locations:
        lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}"
        location_results = results.get(lname, {})

        all_fxx = set()
        for label_data in location_results.values():
            if isinstance(label_data, dict):
                all_fxx.update(label_data.keys())

        fxx_sorted = sorted(
            all_fxx, key=lambda v: int(v) if isinstance(v, (int, str)) and str(v).isdigit() else 0
        )

        for fxx in fxx_sorted:
            try:
                fxx_int = int(fxx) if isinstance(fxx, (int, str)) and str(fxx).isdigit() else None
            except Exception:
                fxx_int = None

            row = {"fxx": fxx_int if fxx_int is not None else fxx}

            # month/day come from the model-run (init) date, constant across fxx,
            # to match training (train_undercast_models.load_data derives them from
            # the per-date filename). Using the per-fxx valid date here instead
            # would skew the feature by up to a day vs. what the models saw.
            if base_date is not None:
                row["month"] = base_date.month
                row["day"] = base_date.day
            else:
                row["month"] = None
                row["day"] = None

            for label in variables.keys():
                fxx_entry = location_results.get(label, {}).get(fxx, {})
                value = fxx_entry.get("value")
                if value is not None:
                    try:
                        row[label] = float(value)
                    except (ValueError, TypeError):
                        row[label] = None
                else:
                    row[label] = None

            rows.append(row)

    df = pd.DataFrame(rows)

    desired_columns = [
        "fxx",
        "cloud_top_hrrr",
        "boundary_layer_cloud_layer",
        "low_cloud_layer_percent_hrrr",
        "middle_cloud_layer_percent_hrrr",
        "high_cloud_layer_percent_hrrr",
        "cloud_ceiling_m_hrrr",
        "cloud_base_m_hrrr",
        "cloud_top_pres_hrrr",
        "cloud_base_pres_hrrr",
        "cloud_top_hgt_hrrr",
        "wind_10m_day_max_hrrr",
        "tmp_500mb_hrrr",
        "tmp_700mb_hrrr",
        "tmp_850mb_hrrr",
        "tmp_925mb_hrrr",
        "tmp_1000mb_hrrr",
        "hgt_500mb_hrrr",
        "hgt_700mb_hrrr",
        "hgt_850mb_hrrr",
        "hgt_925mb_hrrr",
        "hgt_1000mb_hrrr",
        "tmp_2m_hrrr",
        "rh_2m_hrrr",
        "hpbl_surface_hrrr",
        "hgt_0C_iso_hrrr",
        "vis_surface_hrrr",
        "prate_surface_hrrr",
        "apcp_surface_hrrr",
        "cloud_ceiling_gfs",
        "low_cloud_layer_percent_gfs",
        "middle_cloud_layer_percent_gfs",
        "high_cloud_layer_percent_gfs",
        "boundary_layer_cloud_layer_gfs",
        "vis_surface_gfs",
        "prate_surface_gfs",
        "apcp_surface_gfs",
        "tmp_500mb_gfs",
        "tmp_700mb_gfs",
        "tmp_850mb_gfs",
        "tmp_925mb_gfs",
        "tmp_1000mb_gfs",
        "hgt_500mb_gfs",
        "hgt_700mb_gfs",
        "hgt_850mb_gfs",
        "hgt_925mb_gfs",
        "hgt_1000mb_gfs",
        "tmp_2m_gfs",
        "rh_2m_gfs",
        "rh_925mb_gfs",
        "hpbl_surface_gfs",
        "hgt_0C_iso_gfs",
        "cloud_ceiling_nam",
        "low_cloud_layer_percent_nam",
        "middle_cloud_layer_percent_nam",
        "high_cloud_layer_percent_nam",
        "boundary_layer_cloud_layer_nam",
        "vis_surface_nam",
        "tmp_500mb_nam",
        "tmp_700mb_nam",
        "tmp_850mb_nam",
        "tmp_925mb_nam",
        "tmp_1000mb_nam",
        "hgt_500mb_nam",
        "hgt_700mb_nam",
        "hgt_850mb_nam",
        "hgt_925mb_nam",
        "hgt_1000mb_nam",
        "tmp_2m_nam",
        "rh_2m_nam",
        "rh_925mb_nam",
        "hpbl_surface_nam",
        "hgt_0C_iso_nam",
        "prate_surface_nam",
        "apcp_surface_nam",
        "cloud_ceiling_m_rap",
        "low_cloud_layer_percent_rap",
        "middle_cloud_layer_percent_rap",
        "high_cloud_layer_percent_rap",
        "boundary_layer_cloud_layer_rap",
        "vis_surface_rap",
        "tmp_500mb_rap",
        "tmp_700mb_rap",
        "tmp_850mb_rap",
        "tmp_925mb_rap",
        "tmp_1000mb_rap",
        "hgt_500mb_rap",
        "hgt_700mb_rap",
        "hgt_850mb_rap",
        "hgt_925mb_rap",
        "hgt_1000mb_rap",
        "tmp_2m_rap",
        "rh_2m_rap",
        "rh_925mb_rap",
        "hpbl_surface_rap",
        "hgt_0C_iso_rap",
        "prate_surface_rap",
        "apcp_surface_rap",
        "hgt_500mb_ecmwf",
        "hgt_700mb_ecmwf",
        "hgt_850mb_ecmwf",
        "hgt_925mb_ecmwf",
        "hgt_1000mb_ecmwf",
        "tmp_500mb_ecmwf",
        "tmp_700mb_ecmwf",
        "tmp_850mb_ecmwf",
        "tmp_925mb_ecmwf",
        "tmp_1000mb_ecmwf",
        "rh_700mb_ecmwf",
        "rh_850mb_ecmwf",
        "rh_925mb_ecmwf",
        "rh_1000mb_ecmwf",
        "vvel_700mb_ecmwf",
        "vvel_850mb_ecmwf",
        "vvel_925mb_ecmwf",
        "tmp_2m_ecmwf",
        "dpt_2m_ecmwf",
        "mslp_ecmwf",
        "sp_surface_ecmwf",
        "cape_ecmwf",
        "tcwv_ecmwf",
        "tcdc_surface_nbm",
        "tcdc_high_cloud_nbm",
        "cdcb_high_cloud_nbm",
        "cloud_ceiling_m_nbm",
        "cloud_base_m_nbm",
        "vis_surface_nbm",
        "ceil_prob_below_152m_nbm",
        "ceil_prob_below_305m_nbm",
        "ceil_prob_below_610m_nbm",
        "ceil_prob_below_914m_nbm",
        "ceil_prob_below_2012m_nbm",
        "vis_prob_below_1609m_nbm",
        "vis_prob_below_3219m_nbm",
        "vis_prob_below_4828m_nbm",
        "vis_prob_below_8047m_nbm",
        "cape_surface_nbm",
        "mixing_height_nbm",
        "tmp_2m_nbm",
        "dpt_2m_nbm",
        "rh_2m_nbm",
        "apcp_surface_nbm",
        "wind_10m_nbm",
        "gust_surface_nbm",
        "month",
        "day",
    ]

    existing_cols = [col for col in desired_columns if col in df.columns]
    df = df[existing_cols]

    return df


if __name__ == "__main__":
    # The most recent 6-hourly slot, lagged by two hours before rounding down.
    #
    # Without the lag this lands on a run that started minutes ago. The index
    # files appear early, so every availability check passes, and then most of
    # the actual subset downloads miss because the run is still uploading --
    # measured, at 18:50 UTC against the 18Z run: HRRR, NAM and RAP all returned
    # nothing. The workflow fires exactly on the synoptic hours, so this is the
    # normal case for it, not an edge case.
    #
    # Two hours costs the forecast a little reach at the near end and buys a run
    # that has finished publishing. Per-model staleness is handled separately by
    # resolve_run, which walks each model back to its own newest complete run.
    now = datetime.now().astimezone(timezone.utc) - timedelta(hours=2)
    hours = (now.hour // 6) * 6
    if hours == 24:
        hours = 0
        now = now + timedelta(days=1)
    date_str = now.replace(hour=hours, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            h = Herbie(date_str, model="hrrr", product="sfc", fxx=48, save_dir=tmp)

            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                h = Herbie(date_str, model="hrrr", product="sfc", fxx=48, save_dir=tmp)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            if "Did not find" in output:
                raise Exception(f"Herbie initialization failed: {output}")

    except Exception as e:
        hours = ((now.hour // 6) * 6 - 6) % 24
        if hours > now.hour:
            now = now - timedelta(days=1)
        date_str = now.replace(hour=hours, minute=0, second=0, microsecond=0).strftime(
            "%Y-%m-%d %H:%M"
        )

    # Every model is sampled on the union of a 2-hourly and a 3-hourly grid.
    #
    # The 3-hourly part is not cosmetic. ECMWF open data publishes 3-hourly and
    # everything else was on a 2-hourly grid, so the only hours where ALL SIX
    # reported were multiples of 6 -- and the combined model, which is defined
    # only where every source reports, could honestly be evaluated at just nine
    # points across two days. Adding the odd multiples of three to the others
    # brings that to seventeen, at the cost of about 17% more downloads.
    def _grid(last):
        return sorted(set(range(0, last + 1, 2)) | set(range(0, last + 1, 3)))

    FXX_LIST = _grid(48)
    FXX_LIST_GFS = _grid(120)
    FXX_LIST_NAM = _grid(60)
    FXX_LIST_ECMWF = list(range(0, 48 + 1, 3))  # IFS open data is 3-hourly
    FXX_LIST_NBM = _grid(48)
    FXX_LIST_RAP = _grid(48)  # RAP hourly; standard product runs to 21 h

    # One index lookup per model to find its newest published run, then every
    # forecast hour for that model is asked of THAT run at a correspondingly
    # longer lead. The reported hour stays relative to the common base time, so
    # all six models remain on one valid-time axis.
    print("Resolving the latest published run for each model:")
    wanted = [
        ("hrrr", FXX_LIST),
        ("gfs", FXX_LIST_GFS),
        ("nam", FXX_LIST_NAM),
        ("rap", FXX_LIST_RAP),
        ("ifs", FXX_LIST_ECMWF),
        ("nbm", FXX_LIST_NBM),
    ]
    tasks = []
    for model, fxx_list in wanted:
        # Probe at a MIDDLING lead, not the longest one. Long enough that a run
        # which only started minutes ago has not reached it (that is the whole
        # point of the check), short enough that every cycle publishes it: RAP's
        # 00/06/12/18Z cycles stop at 21 h and only 03/09/15/21Z reach 51, so
        # probing at F48 rejected every RAP run and fell back 30 h for nothing.
        # RAP is probed at the lead we actually need from it, because the point
        # of stepping back in threes is to skip the short cycles. Everything else
        # is probed at a middling lead: long enough to reject a run that is still
        # uploading, short enough that every cycle has it.
        probe = max(fxx_list) if model == "rap" else min(max(fxx_list), 18)
        probe = min(probe, MODEL_MAX_LEAD_H.get(model, 48))
        run_date, offset = resolve_run(model, date_str, probe_fxx=probe)
        cap = MODEL_MAX_LEAD_H.get(model, 48)
        kept = [f for f in fxx_list if f + offset <= cap]
        if len(kept) < len(fxx_list):
            print(f"  {model}: dropped {len(fxx_list) - len(kept)} hours that would "
                  f"exceed its {cap} h maximum lead once shifted")
        for fxx in kept:
            tasks.append((fxx, run_date, fxx + offset, model, LOCATIONS, variables))

    results = {}
    for loc in LOCATIONS:
        lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}".replace(" ", "_")
        results[lname] = {label: {} for label in variables.keys()}

    max_workers = min(os.cpu_count() or 4, len(tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_forecast_data, task): i for i, task in enumerate(tasks)}

        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            result = future.result()
            if result:
                fxx, model, task_results = result

                for lname, location_data in task_results.items():
                    for label, data in location_data.items():
                        results[lname][label][fxx] = data

            completed += 1
            pct = int(round(completed / total * 100))
            print(f"\rProgress: {pct:3d}%", end="", flush=True)

        print()

    # Save JSON files
    json_outdir = Path("files/weather")
    json_outdir.mkdir(parents=True, exist_ok=True)

    fxx_sorted = sorted(
        set(
            fxx
            for loc_data in results.values()
            for var_data in loc_data.values()
            for fxx in var_data.keys()
        )
    )

    for loc in LOCATIONS:
        lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}"
        json_results = {}

        for label, fxx_entries in results[lname].items():
            req_model = variables.get(label, {}).get("model")
            out_label = label
            if req_model and not (
                label.endswith("_gfs")
                or label.endswith("_hrrr")
                or label.endswith("_nam")
                or label.endswith("_rap")
                or label.endswith("_ecmwf")
                or label.endswith("_nbm")
            ):
                out_label = f"{label}_{req_model}"

            try:
                fxx_sorted_local = sorted(fxx_entries.keys(), key=lambda v: int(v))
            except Exception:
                fxx_sorted_local = sorted(fxx_entries.keys())

            xs = []
            ys = []
            for fxx in fxx_sorted_local:
                xs.append(int(fxx) if not isinstance(fxx, str) or fxx.isdigit() else fxx)
                info = fxx_entries.get(fxx, {})
                if "value" in info:
                    val = info["value"]
                    try:
                        yval = float(val)
                        if math.isnan(yval) or not np.isfinite(yval):
                            yval = None
                    except Exception:
                        yval = None
                else:
                    yval = None
                ys.append(yval)

            json_results[out_label] = {"x": xs, "y": ys}

        json_results["date_str"] = date_str
        json_results["run_time"] = (
            datetime.now().astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        )

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(lname))
        json_path = json_outdir / f"weather_data_{safe_name}.json"
        with open(json_path, "w") as f:
            json.dump(_clean_for_json(json_results), f, indent=2)
        print(f"Saved results for {lname} to {json_path}")

    # ML model loading and prediction
    import joblib
    import xgboost as xgb

    ml_models = {"gfs", "hrrr", "nam", "rap", "ecmwf", "nbm", "all"}
    # ml_models = {"nam", "all"}

    # ml_models = {"all"}  # For testing purposes, only use "All" model
    # Accumulate all predictions in a single output
    predictions_output = {
        "date_str": date_str,
        "run_time": datetime.now().astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }

    for ml_model in ml_models:

        preprocess = joblib.load(f"files/weather/models/preprocessor_{ml_model}.pkl")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(f"files/weather/models/xgboost_best_f1_{ml_model}.json")
        rf_model = joblib.load(f"files/weather/models/random_forest_best_f1_{ml_model}.pkl")
        gb_model = joblib.load(f"files/weather/models/gradient_boosting_best_f1_{ml_model}.pkl")

        with open(f"files/weather/models/model_metadata_{ml_model}.json") as f:
            metadata = json.load(f)

        weather_df = results_to_dataframe(results, [LOCATIONS[0]], date_str)
        X_new = weather_df.drop(columns=["fxx"], errors="ignore")
        if ml_model != "all":
            model_suffix = ml_model.lower()
            X_new = X_new[
                [
                    col
                    for col in X_new.columns
                    if col.endswith(f"_{model_suffix}") or col in ["month", "day"]
                ]
            ]
        # Align to exactly the features the preprocessor was fit on. This makes the
        # prediction robust whether the loaded model predates or postdates the
        # ECMWF/NBM columns: unknown extras are dropped, features the model expects
        # but that are absent become NaN (and are imputed downstream).
        if hasattr(preprocess, "feature_names_in_"):
            X_new = X_new.reindex(columns=list(preprocess.feature_names_in_))
        X_new_preprocessed = preprocess.transform(X_new)

        # Make predictions with all three models using their optimal thresholds
        def _proba_from_model(model, X):
            """Return probability-like scores for binary classification."""
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                return 1 / (1 + np.exp(-scores))
            # Fallback: use predict outputs directly (assumed to be probability/regression scores)
            preds = model.predict(X)
            return np.clip(preds, 0, 1)

        predictions = {}

        # XGBoost
        xgb_proba = _proba_from_model(xgb_model, X_new_preprocessed)
        predictions["XGBoost"] = (xgb_proba >= metadata["XGBoost"]["threshold_best_f1"]).astype(int)

        # Random Forest
        rf_proba = _proba_from_model(rf_model, X_new_preprocessed)
        predictions["Random Forest"] = (
            rf_proba >= metadata["Random Forest"]["threshold_best_f1"]
        ).astype(int)

        # Gradient Boosting
        gb_proba = _proba_from_model(gb_model, X_new_preprocessed)
        predictions["Gradient Boosting"] = (
            gb_proba >= metadata["Gradient Boosting"]["threshold_best_f1"]
        ).astype(int)

        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        results_df["consensus"] = (results_df.sum(axis=1) >= 2).astype(
            int
        )  # Majority vote (2+ models agree)

        print(results_df.head())

        xgboost_x = []
        xgboost_y = []
        rf_x = []
        rf_y = []
        gb_x = []
        gb_y = []
        consensus_x = []
        consensus_y = []

        # Determine max FXX for this model
        if ml_model == "hrrr":
            max_fxx = max(FXX_LIST)
        elif ml_model == "gfs":
            max_fxx = max(FXX_LIST_GFS)
        elif ml_model == "nam":
            max_fxx = max(FXX_LIST_NAM)
        else:  # "all"
            max_fxx = min(max(FXX_LIST), max(FXX_LIST_GFS), max(FXX_LIST_NAM))

        for idx, row in results_df.iterrows():
            fxx = weather_df.iloc[idx]["fxx"] if idx < len(weather_df) else idx
            fxx_int = int(fxx)

            # Check if fxx exceeds the model's max forecast hour
            if fxx_int > max_fxx:
                xgboost_val = None
                rf_val = None
                gb_val = None
                consensus_val = None
            else:
                xgboost_val = int(row["XGBoost"])
                rf_val = int(row["Random Forest"])
                gb_val = int(row["Gradient Boosting"])
                consensus_val = int(row["consensus"])

            xgboost_x.append(fxx_int)
            xgboost_y.append(xgboost_val)

            rf_x.append(fxx_int)
            rf_y.append(rf_val)

            gb_x.append(fxx_int)
            gb_y.append(gb_val)

            consensus_x.append(fxx_int)
            consensus_y.append(consensus_val)

        predictions_output[f"XGBoost_{ml_model}"] = {"x": xgboost_x, "y": xgboost_y}
        predictions_output[f"Random Forest_{ml_model}"] = {"x": rf_x, "y": rf_y}
        predictions_output[f"Gradient Boosting_{ml_model}"] = {"x": gb_x, "y": gb_y}
        predictions_output[f"consensus_{ml_model}"] = {"x": consensus_x, "y": consensus_y}

    # The headline forecast: one model, chosen by measurement (see
    # predict_current_model). Added ALONGSIDE the legacy outputs above rather
    # than replacing them, and wrapped, so that a failure here -- a missing
    # artifact, a renamed column, an unreadable GRIB -- costs the page its
    # headline panel and nothing else. The front end hides the panel when the
    # key is absent, which is the correct behaviour for "we do not know".
    try:
        weather_df = results_to_dataframe(results, [LOCATIONS[0]], date_str)
        predictions_output["current"] = predict_current_model(
            weather_df, date_str,
            max_fxx=min(max(FXX_LIST), max(FXX_LIST_GFS), max(FXX_LIST_NAM)),
        )
        print("[current model] published as predictions_all.json['current']")
    except Exception as exc:
        import traceback
        print(f"[current model] NOT published: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        # Record the outage rather than leaving the key absent. A panel that
        # vanishes is indistinguishable from a panel that never existed, and
        # both are indistinguishable from "no undercast expected" to a reader.
        # Saying "unavailable, and why" is the only one of the three that is
        # true when the model could not run.
        predictions_output["current"] = {
            "status": "unavailable",
            "reason": str(exc)[:300],
        }

    # Save all predictions to a single JSON file
    pred_json_path = json_outdir / "predictions_all.json"
    with open(pred_json_path, "w") as f:
        json.dump(_clean_for_json(predictions_output), f, indent=2)
    print(f"Saved all predictions to {pred_json_path}")
