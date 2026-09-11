#!/usr/bin/env python3
"""Read one grid point out of a GRIB field, in the grid's own coordinate system.

This module exists because it used to be duplicated. `weather_to_csv.py` (which
builds the training data) and `weather_to_json.py` (which serves the live page)
each carried their own copy of `sample_nearest`, and when the longitude bug was
found and fixed, only one copy got the fix. The consequence was that the models
were trained on one set of numbers and served another -- the exact failure the
duplication made invisible.

The bug itself is worth knowing about: GFS publishes longitude on a 0..360 grid.
Mount Washington is at -71.30, and xarray's `sel(method="nearest")` does NOT
raise for an out-of-range request -- it silently returns the closest value it
has, which is 0.0. Every GFS field was therefore read at 44.25N, 0E, in
southwestern France, about 5,400 km from the mountain, with no error anywhere.

Import from here. Do not copy these functions into a caller again.
"""
import math

import numpy as np


def _to_scalar(x):
    """Coerce an xarray Dataset/DataArray, ndarray, or number to a plain float.

    Returns nan when a single value can't be recovered. This guarantees the CSV
    writer never receives a raw xarray object -- its multi-line repr would
    otherwise be str()'d into a cell and corrupt the row structure.
    """
    try:
        if hasattr(x, "data_vars"):  # xarray Dataset -> its single data variable
            dvs = list(x.data_vars)
            if len(dvs) != 1:
                return float("nan")
            x = x[dvs[0]]
        vals = getattr(x, "values", x)
        arr = np.asarray(vals).ravel()
        return float(arr[0]) if arr.size else float("nan")
    except Exception:
        return float("nan")


def _match_lon_convention(da, lon, lon_name):
    """Express `lon` in the same convention the grid uses.

    GFS publishes longitude on 0..360; Mount Washington is -71.30. Asking
    xarray for -71.30 with method="nearest" does NOT raise -- it silently
    clamps to the closest value in range, which is 0.0, i.e. southwestern
    France, ~5,400 km away. That is exactly how this went unnoticed: every GFS
    column was populated, plausibly-valued, and wrong. Detected by comparing
    850 mb temperature across models -- GFS correlated 0.596 with HRRR and ran
    +6.8 K warm, where NAM, RAP and ECMWF all correlate 0.99+.
    """
    try:
        coord = da[lon_name].values
        hi, lo = float(np.nanmax(coord)), float(np.nanmin(coord))
    except Exception:
        return lon
    if hi > 180.0 and lon < 0.0:        # grid is 0..360, request is signed
        return lon % 360.0
    if lo < 0.0 and lon > 180.0:        # grid is -180..180, request is 0..360
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def sample_nearest(da, lat, lon):
    """Select nearest value from DataArray using common coordinate name variants.

    Longitude is converted to the grid's own convention first -- see
    _match_lon_convention for why a silent 5,400 km error is possible otherwise.
    """
    sel_opts = [
        {"lat": lat, "lon": lon},
        {"latitude": lat, "longitude": lon},
        {"y": lat, "x": lon},
        {"grid_latitude": lat, "grid_longitude": lon},
    ]
    for opts in sel_opts:
        try:
            lat_name, lon_name = list(opts)
            # Only for genuinely longitude-named coordinates. On Lambert grids
            # "x" is a projection coordinate in metres, where a ">180 means
            # 0..360" test would misfire badly.
            if "lon" in lon_name.lower() and lon_name in da.coords:
                opts = dict(opts)
                opts[lon_name] = _match_lon_convention(da, lon, lon_name)
            point = da.sel(method="nearest", **opts)
            return _to_scalar(point.squeeze())
        except Exception:
            continue
    try:
        lat_dim = next(d for d in da.coords if "lat" in d.lower())
        lon_dim = next(d for d in da.coords if "lon" in d.lower())
        ilat = abs(da[lat_dim] - lat).argmin().item()
        lon_q = _match_lon_convention(da, lon, lon_dim) if "lon" in lon_dim.lower() else lon
        ilon = abs(da[lon_dim] - lon_q).argmin().item()
        val = da.isel({lat_dim: ilat, lon_dim: ilon}).squeeze()
        return _to_scalar(val)
    except Exception:
        try:
            np_point, iy, ix, dkm = find_nearest_by_geodetic(da, lat, lon)
            return _to_scalar(np_point)
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
