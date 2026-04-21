"""
download.py — CDS API download functions for ERA5 and C3S seasonal forecasts.

Three types of data are needed:
  1. ERA5 SST (historical reanalysis) — used as proxy observations and to
     build the regression that links partial-month observations to seasonal outcomes.
  2. C3S hindcast climatological mean (per model) — used to debias each model's
     forecast by subtracting its own systematic average behaviour.
  3. C3S forecast ensemble members (daily) — the actual ensemble plume for the
     current season, which forms our probabilistic prior.

Files are cached on disk: if the output path already exists the download is
skipped, so re-running the script is cheap.
"""

import os
import zipfile
import cdsapi
import config


def _cds_client():
    """Return a CDS API client. Reads credentials from ~/.cdsapirc automatically."""
    return cdsapi.Client()


# ─────────────────────────────────────────────────────────────────────────────
# ERA5
# ─────────────────────────────────────────────────────────────────────────────

def download_era5_sst(start_year: int, end_year: int) -> str:
    """
    Download ERA5 daily sea-surface temperature over the Niño 3.4 box for
    every April, May, and June from start_year to end_year (inclusive).

    We request one snapshot per day at 12:00 UTC. SST is a slowly-varying
    boundary condition in ERA5, so a single daily snapshot is a good proxy
    for the daily mean.

    CDS delivers ERA5 as a ZIP archive containing a NetCDF file. The function
    unzips it and returns the path to the NetCDF.

    Parameters
    ----------
    start_year : first year of the historical period (e.g. 1993)
    end_year   : last year of the historical period (e.g. 2026)

    Returns
    -------
    Path to the unzipped NetCDF file.
    """
    nc_path  = os.path.join(config.ERA5_DIR, "era5_sst_nino34.nc")
    zip_path = nc_path.replace(".nc", ".zip")

    # Skip if already downloaded
    if os.path.exists(nc_path):
        print(f"[ERA5] Already downloaded: {nc_path}")
        return nc_path

    print(f"[ERA5] Downloading SST for {start_year}–{end_year}, months "
          f"{config.TARGET_MONTHS} …")

    c = _cds_client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "sea_surface_temperature",
            # All years in the requested range
            "year":  [str(y) for y in range(start_year, end_year + 1)],
            # Only the target-season months (AMJ by default)
            "month": [str(m).zfill(2) for m in config.TARGET_MONTHS],
            # All calendar days (CDS ignores day 31 for short months)
            "day":   [str(d).zfill(2) for d in range(1, 32)],
            # One snapshot per day at midday
            "time":  "12:00",
            # Spatial subsetting to the Niño 3.4 box — keeps the file small
            "area":  config.NINO34_AREA,
            "format": "netcdf",
        },
        zip_path,
    )

    # CDS wraps the NetCDF in a ZIP archive — unpack it
    print("[ERA5] Unzipping …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The archive should contain exactly one .nc file
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            raise RuntimeError("ERA5 ZIP contains no .nc file")
        zf.extract(nc_names[0], config.ERA5_DIR)
        extracted = os.path.join(config.ERA5_DIR, nc_names[0])

    # Rename to our canonical filename
    os.rename(extracted, nc_path)
    os.remove(zip_path)

    print(f"[ERA5] Saved to {nc_path}")
    return nc_path


# ─────────────────────────────────────────────────────────────────────────────
# C3S HINDCAST CLIMATOLOGICAL MEAN  (seasonal-monthly-single-levels)
# ─────────────────────────────────────────────────────────────────────────────

def download_hindcast_climatology(model_name: str) -> str | None:
    """
    Download the hindcast climatological mean for one C3S model.

    The 'hindcast_climate_mean' product type in seasonal-monthly-single-levels
    gives the ensemble-mean forecast averaged over all hindcast years (1993–2016).
    This is the model's "average behaviour" for a given initialisation month and
    lead time — i.e. its climatology. We subtract it from each forecast member
    to obtain the forecast anomaly, removing per-model systematic biases before
    pooling all models together.

    Data are returned at monthly resolution for the target-season lead months.

    Parameters
    ----------
    model_name : key in config.MODELS (e.g. 'ecmwf')

    Returns
    -------
    Path to the GRIB file, or None if the download failed.
    """
    model = config.MODELS[model_name]

    out_path = os.path.join(
        config.HINDCAST_DIR,
        f"hindcast_clim_{model_name}_m{config.INIT_MONTH:02d}.grib",
    )

    if os.path.exists(out_path):
        print(f"[HINDCAST CLIM] Already downloaded: {out_path}")
        return out_path

    print(f"[HINDCAST CLIM] Downloading climatology for {model_name} …")

    # Lead months: how many months after the initialisation month do we cover?
    # For an April init targeting AMJ: lead month 1 = April, 2 = May, 3 = June
    n_target = len(config.TARGET_MONTHS)
    leadtime_months = [str(lm) for lm in range(1, n_target + 1)]

    try:
        c = _cds_client()
        c.retrieve(
            "seasonal-monthly-single-levels",
            {
                "format": "grib",
                "originating_centre": model["originating_centre"],
                "system": model["system"],
                "variable": "sea_surface_temperature",
                "product_type": "hindcast_climate_mean",
                # Initialisation month (day is always '01' for seasonal forecasts)
                "month": str(config.INIT_MONTH).zfill(2),
                "leadtime_month": leadtime_months,
                "area": config.NINO34_AREA,
            },
            out_path,
        )
        print(f"[HINDCAST CLIM] Saved to {out_path}")
        return out_path

    except Exception as exc:
        print(f"[HINDCAST CLIM] WARNING: {model_name} failed — {exc}")
        print(f"[HINDCAST CLIM]   This model will be skipped.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# C3S FORECAST ENSEMBLE MEMBERS  (seasonal-original-single-levels)
# ─────────────────────────────────────────────────────────────────────────────

def download_forecast_members(model_name: str, init_year: int) -> str | None:
    """
    Download all ensemble members of a C3S seasonal forecast at daily resolution.

    'seasonal-original-single-levels' provides the raw model output before any
    post-processing. We request all ensemble members so we can reconstruct the
    full probability distribution (plume) of daily Niño 3.4 trajectories.

    Lead times are expressed as hours from the initialisation date (day 1 = 24 h,
    day 2 = 48 h, …). We cover the full target season plus a small buffer.

    Parameters
    ----------
    model_name : key in config.MODELS
    init_year  : the forecast initialisation year (e.g. 2026)

    Returns
    -------
    Path to the GRIB file, or None if the download failed.
    """
    model = config.MODELS[model_name]

    out_path = os.path.join(
        config.FORECAST_DIR,
        f"forecast_{model_name}_{init_year}_m{config.INIT_MONTH:02d}.grib",
    )

    if os.path.exists(out_path):
        print(f"[FORECAST] Already downloaded: {out_path}")
        return out_path

    print(f"[FORECAST] Downloading {model_name} {init_year}-{config.INIT_MONTH:02d} …")

    # Build the list of lead-time hours: one value per day, starting at 24 h
    leadtime_hours = [
        str(h)
        for h in range(24, (config.FORECAST_LEADTIME_DAYS + 1) * 24, 24)
    ]

    # Member numbers as strings: '0', '1', …, 'n_members-1'
    members = [str(n) for n in range(model["n_members"])]

    try:
        c = _cds_client()
        c.retrieve(
            "seasonal-original-single-levels",
            {
                "format": "grib",
                "originating_centre": model["originating_centre"],
                "system": model["system"],
                "variable": "sea_surface_temperature",
                "product_type": "forecast",
                "year":  str(init_year),
                "month": str(config.INIT_MONTH).zfill(2),
                "day":   "01",
                "leadtime_hour": leadtime_hours,
                "number": members,
                "area":  config.NINO34_AREA,
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
# CONVENIENCE: download everything for one run
# ─────────────────────────────────────────────────────────────────────────────

def download_all(init_year: int) -> dict:
    """
    Download ERA5, hindcast climatologies, and forecast ensembles for all models.

    Returns a dict with keys 'era5', 'hindcast_clim', 'forecast' whose values
    are file paths (or None where a model download failed).
    """
    results = {
        "era5": None,
        "hindcast_clim": {},
        "forecast": {},
    }

    # ERA5 — one file covering the full hindcast period + current year
    results["era5"] = download_era5_sst(
        config.HINDCAST_START_YEAR,
        init_year,
    )

    # Per-model hindcast climatology and forecast ensemble
    for model_name in config.MODELS:
        results["hindcast_clim"][model_name] = download_hindcast_climatology(model_name)
        results["forecast"][model_name]      = download_forecast_members(model_name, init_year)

    return results
