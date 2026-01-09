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


def process_forecast_data(args):
    """Process forecast data for a single fxx value"""
    fxx, date_str, model, LOCATIONS, variables = args

    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                h = Herbie(
                    date_str,
                    model=model,
                    product="sfc" if model == "hrrr" else None,
                    fxx=fxx,
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
                            processed.append(a.replace("%n", str(fxx)))
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

            if base_date is not None and fxx_int is not None:
                forecast_date = base_date + timedelta(hours=fxx_int)
                row["month"] = forecast_date.month
                row["day"] = forecast_date.day
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
        "tmp_2m_nam",
        "rh_2m_nam",
        "rh_925mb_nam",
        "hpbl_surface_nam",
        "hgt_0C_iso_nam",
        "prate_surface_nam",
        "apcp_surface_nam",
        "month",
        "day",
    ]

    existing_cols = [col for col in desired_columns if col in df.columns]
    df = df[existing_cols]

    return df


if __name__ == "__main__":
    # Set date_str to the most recent date at the nearest 6-hour increment
    now = datetime.now().astimezone(timezone.utc)
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

    FXX_LIST = list(range(0, 48 + 1, 2))  # every 2 hour
    FXX_LIST_GFS = list(range(0, 120 + 1, 2))  # every 2 hours
    FXX_LIST_NAM = list(range(0, 60 + 1, 2))  # every 2 hour

    tasks = []
    for fxx in FXX_LIST:
        tasks.append((fxx, date_str, "hrrr", LOCATIONS, variables))
    for fxx in FXX_LIST_GFS:
        tasks.append((fxx, date_str, "gfs", LOCATIONS, variables))
    for fxx in FXX_LIST_NAM:
        tasks.append((fxx, date_str, "nam", LOCATIONS, variables))

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
                label.endswith("_gfs") or label.endswith("_hrrr") or label.endswith("_nam")
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

    ml_models = {"gfs", "hrrr", "nam", "all"}
    # ml_models = {"nam", "all"}
    
    # ml_models = {"all"}  # For testing purposes, only use "All" model
    # Accumulate all predictions in a single output
    predictions_output = {
        "date_str": date_str,
        "run_time": datetime.now().astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }

    for ml_model in ml_models:

        preprocess = joblib.load(
            f"/files/weather/models/preprocessor_{ml_model}.pkl"
        )
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(
            f"/files/weather/models/xgboost_best_f1_{ml_model}.json"
        )
        rf_model = joblib.load(
            f"/files/weather/models/random_forest_best_f1_{ml_model}.pkl"
        )
        gb_model = joblib.load(
            f"/files/weather/models/gradient_boosting_best_f1_{ml_model}.pkl"
        )

        with open(
            f"/files/weather/models/model_metadata_{ml_model}.json"
        ) as f:
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
        results_df["consensus"] = (results_df.sum(axis=1) >= 2).astype(int)  # Majority vote (2+ models agree)

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

        # Save all predictions to a single JSON file
        pred_json_path = json_outdir / "predictions_all.json"
        with open(pred_json_path, "w") as f:
            json.dump(_clean_for_json(predictions_output), f, indent=2)
        print(f"Saved all predictions to {pred_json_path}")
