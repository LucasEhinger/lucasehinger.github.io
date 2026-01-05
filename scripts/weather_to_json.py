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


# Set date_str to the most recent date at the nearest 6-hour increment
now = datetime.now().astimezone(timezone.utc)
hours = (now.hour // 6) * 6
if hours == 24:
    hours = 0
    now = now + timedelta(days=1)
date_str = now.replace(hour=hours, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")
try:
    # validate by creating a Herbie object
    h = Herbie(date_str, model="hrrr", product="sfc", fxx=48)
    # Check if Herbie printed "Did not find" by capturing output

    # Re-run with output capture to check for warning
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        h = Herbie(date_str, model="hrrr", product="sfc", fxx=48)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    if "Did not find" in output:
        raise Exception(f"Herbie initialization failed: {output}")

except Exception as e:
    # If the most recent 6-hour increment fails, fall back to 6 hours prior
    # print(f"Warning: Failed to load data for {date_str}. Falling back to 6 hours prior.")
    hours = ((now.hour // 6) * 6 - 6) % 24
    # Adjust date if we wrapped around midnight
    if hours > now.hour:
        now = now - timedelta(days=1)
    date_str = now.replace(hour=hours, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")
    # print(f"Using date_str: {date_str}")

# Hard code date string for testing
# date_str = "2025-11-02 00:00"


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


# Create a Herbie object for HRRR model data
# Example: create Herbie objects for multiple forecast hours and keep a default H for
# backward compatibility with the rest of the script.
FXX_LIST = list(range(0, 48 + 1, 1))  # every 1 hour
FXX_LIST_GFS = list(range(0, 120 + 1, 2))  # every 2 hours
FXX_LIST_NAM = list(range(0, 60 + 1, 1))  # every 1 hour
# FXX_LIST = list(range(0, 49))
# Replace these with the latitude/longitude you want to sample.
# Provide a list of locations (name, lat, lon). The script will save a separate
# JSON for each location in the `jsons` directory.
LOCATIONS = [
    {"name": "MtWashington", "lat": 44.27040, "lon": -71.30327},
    {"name": "MtLafayette", "lat": 44.16083, "lon": -71.64446},
    {"name": "MtMoosilauke", "lat": 44.02343, "lon": -71.83139},
    {"name": "MtMonadnock", "lat": 42.86137, "lon": -72.10800},
]

# Create a single flat Hs dict containing Herbie objects for both HRRR and GFS fxx lists.
# If a forecast hour appears in both lists, the GFS entry will overwrite the HRRR one.
Hs = {}


# HRRR entries
# Use a small helper container that preserves insertion order and allows
# multiple Herbie objects for the same fxx (no overwriting).
class HsCollection:
    def __init__(self):
        # internal list of (fxx, Herbie) tuples in insertion order
        self._entries = []

    def add(self, fxx, hobj):
        self._entries.append((fxx, hobj))

    def update_from_iterable(self, iterable):
        for fxx, hobj in iterable:
            self.add(fxx, hobj)

    def items(self):
        # behave like dict.items() but keep duplicates and insertion order
        return iter(self._entries)

    def keys(self):
        return [fxx for fxx, _ in self._entries]

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, key):
        # return the first Herbie for this fxx (keeps compatibility with code that expects Hs[fxx])
        for fxx, h in self._entries:
            if fxx == key:
                return h
        raise KeyError(key)

    def get_all(self, key):
        # helper to fetch all Herbie objects for a given fxx
        return [h for fxx, h in self._entries if fxx == key]


# create the container
Hs = HsCollection()

# HRRR entries (preserve insertion order)
Hs.update_from_iterable(
    (
        (
            fxx,
            Herbie(
                date_str,
                model="hrrr",
                product="sfc",
                fxx=fxx,
            ),
        )
        for fxx in FXX_LIST
    )
)

# GFS entries (added into the same ordered collection; will NOT overwrite HRRR entries)
Hs.update_from_iterable(
    (
        (
            fxx,
            Herbie(
                date_str,
                model="gfs",
                # product="pgrb2.0p25",
                fxx=fxx,
            ),
        )
        for fxx in FXX_LIST_GFS
    )
)

# NAM entries (added into the same ordered collection; will NOT overwrite HRRR/GFS entries)
Hs.update_from_iterable(
    (
        (
            fxx,
            Herbie(
                date_str,
                model="nam",
                # product="conusnest.hiresf",
                fxx=fxx,
            ),
        )
        for fxx in FXX_LIST_NAM
    )
)

# Keep a default H so the rest of the script (which expects `H`) continues to work.
# Use Hs[f] to access a specific forecast-hour Herbie object when you need to iterate.
H = Hs[FXX_LIST[0]]


# Map friendly labels to candidate Herbie variable names (try each until one loads)
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
}

graphs = {
    "Low Cloud Cover": {
        "graphs": ["low_cloud_layer_percent", "low_cloud_layer_percent_gfs"],
        "title": "Low Cloud Cover (%)",
    },
    "Mid Cloud Cover": {
        "graphs": ["middle_cloud_layer_percent", "middle_cloud_layer_percent_gfs"],
        "title": "Mid Cloud Cover (%)",
    },
    "High Cloud Cover": {
        "graphs": ["high_cloud_layer_percent", "high_cloud_layer_percent_gfs"],
        "title": "High Cloud Cover (%)",
    },
    "Cloud Heights": {
        "graphs": [
            "cloud_base_m",
            "cloud_ceiling_m",
            "cloud_top",
            "cloud_ceiling_gfs",
        ],
        "title": "Cloud Heights (m)",
    },
    "Pressure Levels": {
        "graphs": [
            "cloud_base_pres",
            "cloud_top_pres",
        ],
        "title": "Cloud Pressure Levels (Pa)",
    },
    "10m Wind": {
        "graphs": [
            "wind_10m_day_max",
        ],
        "title": "10m Wind Day Max (m/s)",
    },
}


def try_load(candidates):
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
            # If result still has extra dims, reduce them (e.g., time/height)
            # Use .squeeze() to remove length-1 dimensions then extract scalar
            val = point.squeeze()
            # If xarray DataArray, get Python scalar if possible
            try:
                return float(val[list(val.data_vars)[0]].values)
            except Exception:
                return val
        except Exception:
            continue
    # as a last resort try indexing by nearest integer indices
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
        # If integer-index fallback fails (e.g. coords are 2D), use geodetic nearest
        try:
            np_point, iy, ix, dkm = find_nearest_by_geodetic(da, lat, lon)
            try:
                return float(
                    np_point[list(np_point.data_vars)[0]].values
                )  # if Dataset, pick first var.
            except Exception:
                return np_point
        except Exception:
            raise


def find_nearest_by_geodetic(da, lat0, lon0, lat_name_hint="lat", lon_name_hint="lon"):
    """
    Find the nearest grid point in `da` to (lat0, lon0) using great-circle distance.
    Returns: (nearest_point, idx_y, idx_x, distance_km)
    - nearest_point: DataArray at the nearest y,x (other dims preserved)
    - idx_y, idx_x: integer indices for the two spatial dims
    - distance_km: float distance in kilometers
    The function attempts to find latitude/longitude coordinate arrays in `da.coords`.
    """
    # try to find lat/lon coords by common names
    lat_da = None
    lon_da = None
    for name in da.coords:
        nl = name.lower()
        if lat_da is None and lat_name_hint in nl:
            lat_da = da.coords[name]
        if lon_da is None and lon_name_hint in nl:
            lon_da = da.coords[name]
    # fallbacks: explicit names
    if lat_da is None and "latitude" in da.coords:
        lat_da = da.coords["latitude"]
    if lon_da is None and "longitude" in da.coords:
        lon_da = da.coords["longitude"]

    if lat_da is None or lon_da is None:
        raise ValueError("Could not find 2D latitude/longitude coordinates in DataArray")

    # Ensure we have numpy arrays
    lat_vals = np.asarray(lat_da.values)
    lon_vals = np.asarray(lon_da.values)

    # Adjust lon0 to dataset convention
    lon_max = float(np.nanmax(lon_vals))
    if lon_max > 180:
        lon0 = lon0 % 360
    else:
        if lon0 > 180:
            lon0 = ((lon0 + 180) % 360) - 180

    # Haversine (vectorized)
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

    # Find index of minimum distance
    flat_idx = np.nanargmin(dist_grid.ravel())
    iy, ix = divmod(int(flat_idx), dist_grid.shape[1])

    # Map indices back to the DataArray dims: assume lat_da has two dims (e.g., ('y','x'))
    lat_dims = lat_da.dims
    if len(lat_dims) != 2:
        # if not 2D, try to index by the first two dims of da
        dims = tuple(da.dims[:2])
    else:
        dims = lat_dims

    sel = {dims[0]: iy, dims[1]: ix}
    nearest_point = da.isel(sel)
    distance_km = float(dist_grid[iy, ix])
    # dx.gh[iy,ix]
    # If nearest_point is a Dataset (multiple data variables like "si10"), pick one to inspect
    # if isinstance(nearest_point, xr.Dataset):
    #     vars = list(nearest_point.data_vars)
    #     varname = "si10" if "si10" in vars else (vars[0] if vars else None)
    #     arr = nearest_point[varname] if varname is not None else nearest_point.to_array().squeeze()
    # else:
    #     varname = None
    #     arr = nearest_point

    # # Print info and the underlying values (use .item() for scalars)
    # print(f"nearest_point type={type(nearest_point).__name__}, selected_var={varname}")
    # try:
    #     val = arr.item() if getattr(arr, "size", 1) == 1 else arr.values
    #     print("values:", val)
    # except Exception:
    #     print("values (fallback):", getattr(arr, "values", arr))

    return nearest_point, iy, ix, distance_km


for i, (fxx, H) in enumerate(Hs.items()):
    # initialize per-location results structure on first iteration
    if i == 0:
        # results will be a dict: results[location_name][label][fxx] = {..}
        results = {}
        for loc in LOCATIONS:
            lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}".replace(" ", "_")
            results[lname] = {label: {} for label in variables.keys()}

    for label, candidates in variables.items():
        # Skip this label if it's for a different model than the current H
        req_model = candidates.get("model")
        # make a per-iteration copy so we don't mutate the global `variables` mapping
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

        da, used_name = try_load(candidates["aliases"])
        if da is None:
            # record error for every location
            for loc in LOCATIONS:
                lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}"
                results[lname][label][fxx] = {
                    "error": "could not load variable",
                    "tried": candidates["aliases"],
                }
            continue

        # # create output directory (per-model subfolder)
        # outdir = Path("plots") / (str(getattr(H, "model", "model")) or "model")
        # outdir.mkdir(parents=True, exist_ok=True)

        # # create plot and save to file once for this fxx+variable
        # try:
        #     ax = get_lat_lon_plot(da)
        #     try:
        #         fig = ax.get_figure()
        #     except Exception:
        #         fig = plt.gcf()

        #     # sanitize parts for filename
        #     def _safe(s):
        #         return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))

        #     fname = outdir / f"{_safe(label)}_fxx{fxx}_{_safe(used_name)}.png"
        #     fig.savefig(fname, dpi=300, bbox_inches="tight")
        #     try:
        #         fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
        #     except Exception:
        #         pass
        #     plt.close(fig)
        # except Exception:
        #     fname = None

        # Now sample for each location and record results (include plot path)
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

    total = len(Hs)
    pct = int(round((i + 1) / total * 100))
    # Overwrite previous percentage using carriage return and padding to clear longer text
    print("\r" + f"Progress: {pct:3d}%" + " " * 10, end="", flush=True)
    if i == total - 1:
        print()  # newline when done


# Plot based on the `graphs` mapping
fxxs = sorted(Hs.keys())

# After sampling, write per-location JSON files and create a combined results
# object (from the first location) so the existing plotting code can continue
# to operate with minimal changes.
json_outdir = Path("files/weather")
json_outdir.mkdir(parents=True, exist_ok=True)

first_loc_name = LOCATIONS[0].get("name") if LOCATIONS else None
for loc in LOCATIONS:
    lname = loc.get("name") or f"loc_{loc.get('lat')}_{loc.get('lon')}"
    # Build JSON with 'x' (fxx list) and 'y' (values list) per variable for this location
    json_results = {}
    for label, fxx_entries in results[lname].items():
        # determine output label: append model if not already present
        req_model = variables.get(label, {}).get("model")
        out_label = label
        if req_model and not (
            label.endswith("_gfs") or label.endswith("_hrrr") or label.endswith("_nam")
        ):
            out_label = f"{label}_{req_model}"

        # sort forecast hours numerically
        try:
            fxx_sorted = sorted(fxx_entries.keys(), key=lambda v: int(v))
        except Exception:
            fxx_sorted = sorted(fxx_entries.keys())

        xs = []
        ys = []
        for fxx in fxx_sorted:
            xs.append(int(fxx) if not isinstance(fxx, str) or fxx.isdigit() else fxx)
            info = fxx_entries.get(fxx, {})
            # prefer stored numeric 'value'; if missing or error, use null in JSON (None)
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

    # Compute undercast probability
    undercast_probs = {}
    for model_name in ["hrrr", "gfs", "nam"]:
        undercast_probs[model_name] = {"x": [], "y": []}

        for fxx in fxx_sorted:
            n = 0

            # Condition 1: TMP(850 mb) - TMP(2 m) >= 5°C
            tmp_850_key = f"tmp_850mb_{model_name}"
            tmp_2m_key = f"tmp_2m_{model_name}"
            if tmp_850_key in results[lname] and tmp_2m_key in results[lname]:
                tmp_850 = results[lname][tmp_850_key].get(fxx, {}).get("value")
                tmp_2m = results[lname][tmp_2m_key].get(fxx, {}).get("value")
                if tmp_850 is not None and tmp_2m is not None:
                    if float(tmp_850) - float(tmp_2m) >= 5:
                        n += 1

            # Condition 2: TCDC(BL) >= 0.7
            tcdc_key = f"boundary_layer_cloud_layer_{model_name}"
            if tcdc_key in results[lname]:
                tcdc = results[lname][tcdc_key].get(fxx, {}).get("value")
                if tcdc is not None and float(tcdc) >= 0.7:
                    n += 1

            # Condition 3: RH(2 m) >= 90%
            rh_2m_key = f"rh_2m_{model_name}"
            if rh_2m_key in results[lname]:
                rh_2m = results[lname][rh_2m_key].get(fxx, {}).get("value")
                if rh_2m is not None and float(rh_2m) >= 90:
                    n += 1

            # Condition 4: PBL height <= 500 m
            hpbl_key = f"hpbl_surface_{model_name}"
            if hpbl_key in results[lname]:
                hpbl = results[lname][hpbl_key].get(fxx, {}).get("value")
                if hpbl is not None and float(hpbl) <= 500:
                    n += 1

            undercast_probs[model_name]["x"].append(
                int(fxx) if not isinstance(fxx, str) or fxx.isdigit() else fxx
            )
            undercast_probs[model_name]["y"].append(n / 5.0)

        json_results[f"undercast_prob_{model_name}"] = undercast_probs[model_name]

    # Save run time info
    try:
        json_results["date_str"] = _clean_for_json(globals().get("date_str", None))
        json_results["run_time"] = (
            datetime.now().astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        )
    except Exception:
        json_results["date_str"] = None
        json_results["run_time"] = None

    # attach model name(s) used in Hs (best-effort)
    try:
        models = list({str(getattr(h, "model", "")) for _, h in Hs.items()})
        json_results["models"] = models
    except Exception:
        pass

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(lname))
    json_path = json_outdir / f"weather_data_{safe_name}.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results for {lname} to {json_path}")

# For backwards compatibility with subsequent code that expects `results[label][fxx]`,
# set `results` to the first location's results (if available).
if first_loc_name and first_loc_name in results:
    results = results[first_loc_name]

for graph_name, ginfo in graphs.items():
    labels = ginfo.get("graphs", [])
    title = ginfo.get("title", graph_name)

    for label in labels:
        if label not in results:
            # no data collected for this label
            continue

        y = []
        used_names = []
        for fxx in fxxs:
            entry = results[label].get(fxx, {})
            used_names.append(entry.get("variable"))
            # prefer stored numeric 'value'; if missing, treat as NaN
            if "value" in entry:
                val = entry["value"]
            else:
                val = np.nan
            try:
                yval = float(val)
            except Exception:
                try:
                    yval = float(np.asarray(val).squeeze())
                except Exception:
                    yval = np.nan
            y.append(yval)

        if not any(np.isfinite(y)):
            # skip plotting series that are entirely NaN
            continue

        # choose the most common variable name seen for this label (if any)
        varname = ""
        try:
            varname = max([n for n in used_names if n], key=used_names.count)
        except Exception:
            pass

        # (results.json write removed)


# combine per-label PNG series into animated GIFs
# gif_outdir = Path("plots") / "gifs"
# gif_outdir.mkdir(parents=True, exist_ok=True)

# for label, fxx_entries in results.items():
#     # collect (fxx, Path(plot)) pairs for this label
#     pairs = []
#     for fxx, info in fxx_entries.items():
#         p = info.get("plot")
#         if not p:
#             continue
#         try:
#             pairs.append((int(fxx), Path(p)))
#         except Exception:
#             # fallback if fxx isn't an int-like string
#             pairs.append((fxx, Path(p)))

#     if not pairs:
#         continue

#     # sort by forecast hour (numeric when possible)
#     try:
#         pairs.sort(key=lambda t: int(t[0]))
#     except Exception:
#         pairs.sort(key=lambda t: str(t[0]))

#     frames = []
#     for fxx, p in pairs:
#         if not p.exists():
#             print(f"Warning: missing file {p} for label {label}, fxx={fxx}; skipping")
#             continue
#         try:
#             img = imageio.imread(str(p))
#             frames.append(img)
#         except Exception as e:
#             print(f"Warning: failed to read {p} ({e}); skipping")

#     if not frames:
#         continue

#     # sanitize label for filename
#     safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label))
#     gif_path = gif_outdir / f"{safe_label}.gif"

#     try:
#         # duration=0.8 seconds per frame; change as desired
#         imageio.mimsave(str(gif_path), frames, duration=0.8)
#         print(f"Saved GIF for '{label}' -> {gif_path}")
#     except Exception as e:
#         print(f"Error saving GIF for '{label}': {e}")
