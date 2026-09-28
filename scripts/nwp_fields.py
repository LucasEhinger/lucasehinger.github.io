"""What the TRAINING fetch downloads, and the guards that keep it from hanging.

Imported by fetch_nwp_at_obs.py, which samples every forecast field at each
observation's own valid time. Three things live here:

  * `variables` -- every forecast field the model is trained on, keyed by column
    name, with the GRIB search strings that find it in each source. This is the
    training contract: a field missing here is missing from every shard.
  * `try_load` -- the first alias that loads, or (None, None).
  * Two hang guards, installed on import: a default socket timeout, and an
    os.system wrapper that adds --max-time to Herbie's curl calls. Herbie shells
    out to curl with no timeout, so without them a dead socket blocks a worker
    forever, unkillable.

The live page's fetch (weather_to_json.py) keeps its own list, because it also
downloads fields that are only plotted, never fed to the model.
test_serving_features.py checks that it still carries every column the model needs.
"""
import os
import socket
import sys
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
    # NB: every key MUST end in _<source>. train_undercast_obs assigns a column
    # to its source by that suffix, so an unsuffixed name is silently dropped
    # from the per-source models and only ever reaches the combined one.
    "boundary_layer_cloud_layer_hrrr": {
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
