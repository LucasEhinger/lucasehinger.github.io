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
    """Process a single row of undercast data"""
    index, row, next_row, LOCATIONS, variables = args

    if len(row) < 5:
        return None

    short_date = row[1]
    avg = row[4]

    avg_next = None
    if next_row and len(next_row) >= 5:
        avg_next = next_row[4]

    FXX_LIST_NAM = list(range(2, 8 + 1, 2))
    if avg is not None and avg_next is not None:
        if float(avg_next) == float(avg):
            FXX_LIST_NAM = list(range(2, 8+1, 2)) + list(range(12, 27, 4))

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
            for fxx in FXX_LIST_NAM:
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
    csv_path = Path("files/weather/csv/MtWashington_undercast.csv")
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        undercast_data = list(reader)

    tasks = []
    for index, row in enumerate(undercast_data[1:]):
        next_row = undercast_data[index + 2] if index + 1 < len(undercast_data[1:]) else None
        tasks.append((index, row, next_row, LOCATIONS, variables))

    max_workers = min(os.cpu_count() or 4, len(tasks))
    output_dir = Path("files/weather/csv")
    output_dir.mkdir(parents=True, exist_ok=True)

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
