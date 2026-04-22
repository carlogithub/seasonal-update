"""
download.py — CDS API downloads for ERA5 and C3S seasonal forecasts.

All functions are generic: they take Location, Variable, and Season objects.

File naming convention
----------------------
ERA5:              era5_{variable.slug}_{location.slug}.nc
Postproc forecast: forecast_postproc_{model}_{location.slug}_{variable.slug}
                       _{init_year}_m{init_month:02d}.grib
Monthly forecast:  forecast_monthly_{model}_{location.slug}_{variable.slug}
                       _{init_year}_m{init_month:02d}.grib
Hindcast clim:     hindcast_monthly_{model}_{location.slug}_{variable.slug}
                       _m{init_month:02d}.grib

Debiasing approach
------------------
Primary path: seasonal-postprocessed-single-levels delivers bias-corrected
anomalies directly.  No separate hindcast download needed.

Fallback (if the primary path fails, e.g. for models not yet in the
postprocessed dataset): download monthly_mean + multi-year hindcast and
subtract the model's own climatological mean in process.py.
"""

import os
import zipfile
import cdsapi
import config
from location import Location
from variable import Variable
from season   import Season


def _cds_client():
    return cdsapi.Client()


def _ensure_dirs():
    for d in [config.DATA_DIR, config.ERA5_DIR, config.FORECAST_DIR, config.OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ERA5
# ─────────────────────────────────────────────────────────────────────────────

def download_era5(location: Location, variable: Variable, season: Season,
                  start_year: int, end_year: int) -> str:
    """
    Download ERA5 for a location+variable, all 12 months, start_year to end_year+1.

    All months are downloaded so one file covers both JJA and DJF seasons.
    end_year+1 ensures DJF Jan/Feb of the final forecast year are present.
    CDS delivers ERA5 as plain NetCDF or ZIP — both handled.
    """
    _ensure_dirs()

    nc_path  = os.path.join(config.ERA5_DIR, f"era5_{variable.slug}_{location.slug}.nc")
    zip_path = nc_path.replace(".nc", "_download.tmp")

    if os.path.exists(nc_path):
        print(f"[ERA5] Already downloaded: {nc_path}")
        return nc_path

    all_years = list(range(start_year, end_year + 2))
    print(f"[ERA5] Downloading {variable.name} for {location.name}, "
          f"all months, years {start_year}–{all_years[-1]} …")

    c = _cds_client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable":     variable.era5_name,
            "year":         [str(y) for y in all_years],
            "month":        [str(m).zfill(2) for m in range(1, 13)],
            "day":          [str(d).zfill(2) for d in range(1, 32)],
            "time":         "12:00",
            "area":         location.cds_area,
            "data_format":  "netcdf",
        },
        zip_path,
    )

    if zipfile.is_zipfile(zip_path):
        print("[ERA5] Unzipping …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise RuntimeError("ERA5 ZIP contains no .nc file")
            zf.extract(nc_names[0], config.ERA5_DIR)
            os.rename(os.path.join(config.ERA5_DIR, nc_names[0]), nc_path)
        os.remove(zip_path)
    else:
        print("[ERA5] File is plain NetCDF (no unzip needed).")
        os.rename(zip_path, nc_path)

    print(f"[ERA5] Saved to {nc_path}")
    return nc_path


# ─────────────────────────────────────────────────────────────────────────────
# C3S MONTHLY FORECAST ENSEMBLE  (monthly_mean)
# ─────────────────────────────────────────────────────────────────────────────

def download_forecast_monthly(location: Location, variable: Variable,
                               season: Season, model_name: str,
                               init_year: int) -> str | None:
    """
    Download C3S per-member monthly-mean forecast for one model.

    Uses product_type='monthly_mean'.  Raw values are in absolute units
    (Kelvin for T/SST, m or m/s for TP).  Debiasing is done in process.py
    by subtracting the hindcast climatological mean downloaded separately.
    """
    _ensure_dirs()

    model    = config.MODELS[model_name]
    out_path = os.path.join(
        config.FORECAST_DIR,
        f"forecast_monthly_{model_name}_{location.slug}_{variable.slug}"
        f"_{init_year}_m{season.init_month:02d}.grib",
    )

    if os.path.exists(out_path):
        print(f"[FORECAST] Already downloaded: {out_path}")
        return out_path

    print(f"[FORECAST] Downloading {variable.name} for {location.name} — "
          f"{model_name} {init_year}-{season.init_month:02d} …")

    try:
        c = _cds_client()
        c.retrieve(
            "seasonal-monthly-single-levels",
            {
                "data_format":        "grib",
                "originating_centre": model["originating_centre"],
                "system":             model["system"],
                "variable":           variable.c3s_name,
                "product_type":       "monthly_mean",
                "year":               str(init_year),
                "month":              str(season.init_month).zfill(2),
                "leadtime_month":     [str(lm) for lm in season.leadtime_months()],
                "area":               location.cds_area,
            },
            out_path,
        )
        print(f"[FORECAST] Saved to {out_path}")
        return out_path

    except Exception as exc:
        print(f"[FORECAST] WARNING: {model_name} failed — {exc}")
        print(f"[FORECAST]   This model will be skipped.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 POINT TIME SERIES  (ARCO/ZARR-backed — fast for single points)
# ─────────────────────────────────────────────────────────────────────────────

def download_era5_timeseries(location: Location, variable: Variable,
                               start_year: int, end_year: int) -> str:
    """
    Download ERA5 as a single-point hourly time series using the
    reanalysis-era5-single-levels-timeseries dataset.

    This endpoint is backed by ARCO (Analysis-Ready Cloud-Optimized) ZARR
    storage on the CDS side, making single-point requests much faster than
    the standard ERA5 download (~30–60 s vs several minutes).

    Delivers hourly NetCDF inside a ZIP. process.load_era5() resamples to
    daily automatically when it detects sub-daily resolution.

    Only valid for point locations (location.is_point must be True).
    """
    _ensure_dirs()

    nc_path  = os.path.join(config.ERA5_DIR, f"era5_{variable.slug}_{location.slug}.nc")
    zip_path = nc_path.replace(".nc", "_download.tmp")

    if os.path.exists(nc_path):
        print(f"[ERA5-TS] Already downloaded: {nc_path}")
        return nc_path

    from datetime import date as _date
    end_date   = _date(end_year + 1, 12, 31)
    actual_end = min(end_date, _date.today()).strftime("%Y-%m-%d")
    start_str  = f"{start_year}-01-01"

    print(f"[ERA5-TS] Downloading {variable.name} time series for {location.name} "
          f"{start_year}–{end_year} (ARCO endpoint) …")

    c = _cds_client()
    c.retrieve(
        "reanalysis-era5-single-levels-timeseries",
        {
            "variable":     [variable.era5_name],
            "location":     {"latitude": location.lat, "longitude": location.lon},
            "date":         f"{start_str}/{actual_end}",
            "data_format":  "netcdf",
        },
        zip_path,
    )

    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise RuntimeError("ERA5 timeseries ZIP contains no .nc file")
            zf.extract(nc_names[0], config.ERA5_DIR)
            os.rename(os.path.join(config.ERA5_DIR, nc_names[0]), nc_path)
        os.remove(zip_path)
    else:
        os.rename(zip_path, nc_path)

    print(f"[ERA5-TS] Saved to {nc_path}")
    return nc_path


# ─────────────────────────────────────────────────────────────────────────────
# C3S POST-PROCESSED FORECAST  (bias-corrected anomalies — preferred path)
# ─────────────────────────────────────────────────────────────────────────────

def download_forecast_postprocessed(location: Location, variable: Variable,
                                     season: Season, model_name: str,
                                     init_year: int) -> str | None:
    """
    Download from seasonal-postprocessed-single-levels (bias-corrected anomalies).

    This dataset delivers pre-debiased anomalies directly, eliminating the need
    for separate hindcast downloads and manual debiasing.  Available for years
    2017+ only; covers all major C3S models including ECMWF system=51.

    Returns None if the variable has no postprocessed CDS name or the download fails.
    """
    if not variable.c3s_postproc_name:
        return None

    _ensure_dirs()

    model    = config.MODELS[model_name]
    out_path = os.path.join(
        config.FORECAST_DIR,
        f"forecast_postproc_{model_name}_{location.slug}_{variable.slug}"
        f"_{init_year}_m{season.init_month:02d}.grib",
    )

    if os.path.exists(out_path):
        print(f"[POSTPROC] Already downloaded: {out_path}")
        return out_path

    print(f"[POSTPROC] Downloading {variable.name} postprocessed anomaly for "
          f"{location.name} — {model_name} {init_year}-{season.init_month:02d} …")

    try:
        c = _cds_client()
        c.retrieve(
            "seasonal-postprocessed-single-levels",
            {
                "data_format":        "grib",
                "originating_centre": model["originating_centre"],
                "system":             model["system"],
                "variable":           variable.c3s_postproc_name,
                "product_type":       "monthly_mean",
                "year":               str(init_year),
                "month":              str(season.init_month).zfill(2),
                "leadtime_month":     [str(lm) for lm in season.leadtime_months()],
                "area":               location.cds_area,
            },
            out_path,
        )
        print(f"[POSTPROC] Saved to {out_path}")
        return out_path

    except Exception as exc:
        print(f"[POSTPROC] WARNING: {model_name} failed — {exc}")
        print(f"[POSTPROC]   Will fall back to monthly_mean + hindcast debiasing.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# C3S HINDCAST MONTHLY MEANS  (all years, for computing per-model climatology)
# ─────────────────────────────────────────────────────────────────────────────

def download_hindcast_monthly(location: Location, variable: Variable,
                               season: Season, model_name: str) -> str | None:
    """
    Download monthly_mean hindcast data for all hindcast years in one request.

    The result is a single GRIB containing all members × all lead months × all
    hindcast years (1993–2016).  process.py averages over years to derive the
    model's own climatological mean for each (init_month, lead_month) pair,
    which is then subtracted from the operational forecast to produce anomalies
    in the model's own reference frame.

    This approach is used because hindcast_climate_mean (the pre-computed
    climatology product) is not available for ECMWF SEAS5.1 system=51 via MARS.

    The file is independent of init_year and cached without one in its name.
    """
    _ensure_dirs()

    model    = config.MODELS[model_name]
    out_path = os.path.join(
        config.FORECAST_DIR,
        f"hindcast_monthly_{model_name}_{location.slug}_{variable.slug}"
        f"_m{season.init_month:02d}.grib",
    )

    if os.path.exists(out_path):
        print(f"[HINDCAST] Already downloaded: {out_path}")
        return out_path

    print(f"[HINDCAST] Downloading {variable.name} hindcast (all years) for "
          f"{location.name} — {model_name} init month {season.init_month:02d} …")

    try:
        c = _cds_client()
        c.retrieve(
            "seasonal-monthly-single-levels",
            {
                "data_format":        "grib",
                "originating_centre": model["originating_centre"],
                "system":             model["system"],
                "variable":           variable.c3s_name,
                "product_type":       "monthly_mean",
                "year":               [str(y) for y in range(
                                           config.HINDCAST_START_YEAR,
                                           config.HINDCAST_END_YEAR + 1)],
                "month":              str(season.init_month).zfill(2),
                "leadtime_month":     [str(lm) for lm in season.leadtime_months()],
                "area":               location.cds_area,
            },
            out_path,
        )
        print(f"[HINDCAST] Saved to {out_path}")
        return out_path

    except Exception as exc:
        print(f"[HINDCAST] WARNING: {model_name} failed — {exc}")
        print(f"[HINDCAST]   Will fall back to ERA5-based debiasing.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE
# ─────────────────────────────────────────────────────────────────────────────

def download_all(location: Location, variable: Variable, season: Season,
                 init_year: int) -> dict:
    """
    Download ERA5 and forecast ensembles for all models.

    For each model, try seasonal-postprocessed-single-levels first (delivers
    bias-corrected anomalies directly).  If that fails, fall back to
    monthly_mean + multi-year hindcast for manual debiasing in process.py.

    Returns:
      {
        "era5":          path_to_nc,
        "hindcast_clim": { model_name: path_or_None, … },
        "forecast":      { model_name: path_or_None, … },
      }
    Files in "forecast" may be postprocessed (prefix forecast_postproc_) or
    monthly_mean (prefix forecast_monthly_); process.py detects which by name.
    """
    results = {"era5": None, "hindcast_clim": {}, "forecast": {}}

    if location.is_point:
        results["era5"] = download_era5_timeseries(
            location, variable,
            start_year=config.HINDCAST_START_YEAR,
            end_year=init_year,
        )
    else:
        results["era5"] = download_era5(
            location, variable, season,
            start_year=config.HINDCAST_START_YEAR,
            end_year=init_year,
        )

    for model_name in config.MODELS:
        postproc_path = download_forecast_postprocessed(
            location, variable, season, model_name, init_year,
        )
        if postproc_path:
            results["forecast"][model_name]      = postproc_path
            results["hindcast_clim"][model_name] = None   # not needed
        else:
            results["hindcast_clim"][model_name] = download_hindcast_monthly(
                location, variable, season, model_name,
            )
            results["forecast"][model_name] = download_forecast_monthly(
                location, variable, season, model_name, init_year,
            )

    return results
