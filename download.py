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

    # CDS sometimes delivers a ZIP archive containing a NetCDF, and sometimes
    # delivers a plain NetCDF directly — handle both cases.
    if zipfile.is_zipfile(zip_path):
        print("[ERA5] Unzipping …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise RuntimeError("ERA5 ZIP contains no .nc file")
            zf.extract(nc_names[0], config.ERA5_DIR)
            extracted = os.path.join(config.ERA5_DIR, nc_names[0])
        os.rename(extracted, nc_path)
        os.remove(zip_path)
    else:
        # Already a plain NetCDF — just rename it
        print("[ERA5] File is plain NetCDF (no unzip needed).")
        os.rename(zip_path, nc_path)

    print(f"[ERA5] Saved to {nc_path}")
    return nc_path


# ─────────────────────────────────────────────────────────────────────────────
# C3S FORECAST ENSEMBLE MEMBERS  (seasonal-monthly-single-levels)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: We use seasonal-monthly-single-levels (monthly_mean product) rather
# than seasonal-original-single-levels (daily members) because the latter
# carries a MARS AccessError for all operational forecast years — individual
# raw model contributions are restricted under C3S data policy.
# Monthly aggregated data (monthly_mean) is publicly available.
# Consequence: we lose the daily plume and work at monthly resolution.
# The Bayesian update and tercile probabilities are unaffected.
# Per-model debiasing uses ERA5 climatology as the reference (see process.py).
# ─────────────────────────────────────────────────────────────────────────────

def download_forecast_monthly(model_name: str, init_year: int) -> str | None:
    """
    Download monthly-mean individual member SST forecasts for one C3S model.

    Uses seasonal-monthly-single-levels with product_type='monthly_mean', which
    returns the per-member monthly mean SST for each lead month. This gives us
    the ensemble distribution at monthly resolution — enough to build the prior
    and compute tercile probabilities.

    Parameters
    ----------
    model_name : key in config.MODELS
    init_year  : forecast initialisation year (e.g. 2026)

    Returns
    -------
    Path to the GRIB file, or None if the download failed.
    """
    model = config.MODELS[model_name]

    out_path = os.path.join(
        config.FORECAST_DIR,
        f"forecast_monthly_{model_name}_{init_year}_m{config.INIT_MONTH:02d}.grib",
    )

    if os.path.exists(out_path):
        print(f"[FORECAST] Already downloaded: {out_path}")
        return out_path

    print(f"[FORECAST] Downloading monthly members for {model_name} {init_year}-{config.INIT_MONTH:02d} …")

    n_target = len(config.TARGET_MONTHS)
    leadtime_months = [str(lm) for lm in range(1, n_target + 1)]

    try:
        c = _cds_client()
        c.retrieve(
            "seasonal-monthly-single-levels",
            {
                "data_format": "grib",
                "originating_centre": model["originating_centre"],
                "system": model["system"],
                "variable": "sea_surface_temperature",
                "product_type": "monthly_mean",
                "year":  str(init_year),
                "month": str(config.INIT_MONTH).zfill(2),
                "leadtime_month": leadtime_months,
                "area": config.NINO34_AREA,
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

    # Per-model monthly forecast ensemble (hindcast climatology now derived from ERA5)
    for model_name in config.MODELS:
        results["forecast"][model_name] = download_forecast_monthly(model_name, init_year)

    return results
