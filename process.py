"""
process.py — Read downloaded files and compute the Niño 3.4 index.

Responsibilities:
  1. Load ERA5 NetCDF → daily Niño 3.4 SST anomaly time series.
  2. Load C3S hindcast climatology GRIBs → per-model Niño 3.4 climatological mean.
  3. Load C3S forecast GRIBs → per-model ensemble of daily Niño 3.4 trajectories,
     expressed as anomalies relative to each model's own climatology.
  4. Pool all model anomaly members into one grand ensemble array.

The Niño 3.4 index is simply the area-weighted mean SST anomaly over the
Niño 3.4 box (5°S–5°N, 120°W–170°W). Because the box is small in latitude,
the area weighting reduces to a cosine(latitude) weighting.
"""

import os
import zipfile
import warnings
import numpy as np
import xarray as xr
import cfgrib
import pandas as pd
from datetime import date, timedelta
import config


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_weights(lats: np.ndarray) -> np.ndarray:
    """
    Return cosine(latitude) weights normalised to sum to 1.
    Used for area-averaging: grid cells at the equator count more than at the poles.
    """
    w = np.cos(np.deg2rad(lats))
    return w / w.sum()


def _area_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Compute the cosine-latitude weighted spatial mean of a DataArray that has
    'latitude' and 'longitude' dimensions. Returns a DataArray with those
    dimensions collapsed.
    """
    weights = np.cos(np.deg2rad(da.latitude))
    # xarray's weighted averaging handles broadcasting automatically
    return da.weighted(weights).mean(dim=["latitude", "longitude"])


# ─────────────────────────────────────────────────────────────────────────────
# ERA5
# ─────────────────────────────────────────────────────────────────────────────

def load_era5_nino34(nc_path: str) -> xr.DataArray:
    """
    Load ERA5 SST from a NetCDF file and return a daily Niño 3.4 time series.

    The ERA5 download covers the Niño 3.4 box already, so we just need to
    spatially average and return a 1-D time series.

    ERA5 from CDS names its time coordinate 'valid_time' — we rename it to
    'time' for consistency with the rest of the code.

    Returns
    -------
    xr.DataArray with dimension 'time', values in Kelvin (raw SST).
    Anomalies are computed separately in bayesian_update.py using this series.
    """
    print(f"[PROCESS] Loading ERA5 from {nc_path} …")

    ds = xr.open_dataset(nc_path)

    # CDS ERA5 downloads use 'valid_time' as the time coordinate name
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # The SST variable may be called 'sst' or 'sea_surface_temperature'
    sst_name = "sst" if "sst" in ds else "sea_surface_temperature"
    sst = ds[sst_name]

    # Area-average over the Niño 3.4 box (already spatially subsetted by CDS)
    nino34 = _area_mean(sst)
    nino34.name = "sst"

    # Sort chronologically (CDS sometimes returns unordered time)
    nino34 = nino34.sortby("time")

    print(f"[PROCESS] ERA5 Niño 3.4 series: {nino34.time.values[0]} → "
          f"{nino34.time.values[-1]}, n={len(nino34)}")
    return nino34


def compute_era5_anomaly(nino34_raw: xr.DataArray,
                          clim_start: int,
                          clim_end: int) -> xr.DataArray:
    """
    Convert raw ERA5 SST (in Kelvin) to a Niño 3.4 *anomaly* by subtracting
    the daily climatological mean computed over the hindcast period.

    We use a simple day-of-year climatology: for each calendar day (1–366) we
    compute the mean SST across all years in [clim_start, clim_end] and subtract it.

    Parameters
    ----------
    nino34_raw  : raw SST time series from load_era5_nino34()
    clim_start  : first year of the climatological base period (e.g. 1993)
    clim_end    : last  year of the climatological base period (e.g. 2016)

    Returns
    -------
    xr.DataArray of SST anomalies (Kelvin, but since we subtract the mean the
    units effectively become °C-equivalent anomalies).
    """
    # Select the climatology period
    clim = nino34_raw.sel(
        time=slice(str(clim_start), str(clim_end))
    )

    # Group by day-of-year and take the mean across years
    # This gives one value per calendar day (1–366)
    doy_clim = clim.groupby("time.dayofyear").mean("time")

    # Subtract the climatological mean from every day in the full series
    anomaly = nino34_raw.groupby("time.dayofyear") - doy_clim
    anomaly.name = "nino34_anomaly"

    return anomaly


# ─────────────────────────────────────────────────────────────────────────────
# C3S FORECAST ENSEMBLE  (monthly resolution)
# ─────────────────────────────────────────────────────────────────────────────

def load_forecast_ensemble_monthly(grib_path: str,
                                    era5_raw: xr.DataArray,
                                    init_year: int) -> xr.DataArray:
    """
    Load C3S monthly-mean forecast members and return per-member Niño 3.4
    *anomaly* values for each target month.

    Debiasing strategy: we subtract the ERA5 climatological monthly mean
    (computed over the hindcast period) from each model member's monthly SST.
    This removes the model's absolute SST level and expresses the forecast as
    an anomaly in the same observational reference frame as the ERA5 regression,
    making the prior and the regression directly comparable.

    Note: using ERA5 as the reference rather than a per-model hindcast mean
    means the anomaly includes each model's warm/cold bias. This inflates the
    ensemble spread slightly, but for a first version it is acceptable. Per-
    model debiasing can be added later if hindcast data becomes accessible.

    Parameters
    ----------
    grib_path    : path to the monthly-mean forecast GRIB file
    era5_anomaly : ERA5 Niño 3.4 anomaly time series (from compute_era5_anomaly)
    init_year    : forecast initialisation year

    Returns
    -------
    xr.DataArray with dimensions ('member', 'month'), where 'month' holds the
    actual calendar month number (e.g. [4, 5, 6] for AMJ).
    """
    print(f"[PROCESS] Loading monthly forecast ensemble from {grib_path} …")

    datasets = cfgrib.open_datasets(grib_path, squeeze=False)
    if not datasets:
        raise RuntimeError(f"Could not read GRIB file: {grib_path}")

    ds = datasets[0]
    for d in datasets[1:]:
        try:
            ds = xr.merge([ds, d])
        except Exception:
            pass

    # Find SST variable
    sst_candidates = [v for v in ds.data_vars
                      if any(k in v.lower() for k in ["sst", "skt", "skin"])]
    if not sst_candidates:
        raise RuntimeError(f"No SST variable found in {grib_path}. "
                           f"Available: {list(ds.data_vars)}")
    sst = ds[sst_candidates[0]]

    # Spatial average over the Niño 3.4 box
    sst_spatial = _area_mean(sst)

    # The 'step' dimension here is a monthly lead time offset (e.g. 1 month, 2 months…)
    # Convert to actual calendar months
    target_months = config.TARGET_MONTHS
    n_leads = len(sst_spatial.step.values)

    # Build ERA5 monthly climatology for debiasing:
    # For each target month, compute the mean ERA5 anomaly over the hindcast period.
    # Since era5_anomaly is already anomaly (mean removed), its climatological mean
    # over the hindcast period is ~0 by construction. So we debias against ERA5
    # absolute SST climatology instead.
    #
    # Practical approach: compute ERA5 monthly mean over the hindcast years for
    # each target month, then subtract from the model forecast.
    # Compute ERA5 raw SST monthly climatology over the hindcast period.
    # We use the RAW SST (in Kelvin) so that subtracting it from the model's
    # raw SST forecast gives a proper anomaly in the same reference frame as
    # the ERA5 regression (which operates on ERA5 anomalies).
    era5_raw_monthly_clim = {}
    for cal_month in target_months:
        vals = []
        for yr in range(config.HINDCAST_START_YEAR, config.HINDCAST_END_YEAR + 1):
            era5_month = era5_raw.sel(
                time=(
                    (era5_raw.time.dt.year  == yr) &
                    (era5_raw.time.dt.month == cal_month)
                )
            )
            if len(era5_month) > 0:
                vals.append(float(era5_month.mean()))
        era5_raw_monthly_clim[cal_month] = float(np.mean(vals)) if vals else 0.0
        print(f"[PROCESS]   ERA5 SST climatology {cal_month:02d}: "
              f"{era5_raw_monthly_clim[cal_month]:.2f} K")

    # Extract member values and convert lead index → calendar month
    sst_values = sst_spatial.values   # shape: (member, step) or (step,) if 1 member

    if sst_values.ndim == 1:
        sst_values = sst_values[np.newaxis, :]   # add member axis

    n_members, n_steps = sst_values.shape

    # Build output: one value per member per calendar month
    month_labels = []
    anomaly_cols = []

    for step_idx in range(min(n_steps, len(target_months))):
        cal_month = target_months[step_idx]
        month_labels.append(cal_month)

        # Debias: subtract ERA5 climatological monthly mean of anomaly (≈0)
        col = sst_values[:, step_idx] - era5_raw_monthly_clim[cal_month]
        anomaly_cols.append(col)

    anomaly_array = np.column_stack(anomaly_cols)   # shape: (member, n_months)

    result = xr.DataArray(
        anomaly_array,
        dims=["member", "month"],
        coords={
            "member": np.arange(n_members),
            "month":  month_labels,
        },
    )
    result.attrs["units"] = "K (anomaly relative to ERA5 climatology)"

    print(f"[PROCESS]   Monthly ensemble shape: {result.shape}  "
          f"({n_members} members × {len(month_labels)} months)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRAND ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def build_grand_ensemble(model_ensembles: dict) -> xr.DataArray:
    """
    Pool daily anomaly trajectories from all models into one grand ensemble.

    Parameters
    ----------
    model_ensembles : dict mapping model_name → xr.DataArray (member × time)
                      as returned by load_forecast_ensemble().

    Returns
    -------
    xr.DataArray with dimensions ('member', 'time'), where the member axis
    now spans all models concatenated. A 'model' coordinate records which
    model each member came from.
    """
    all_members   = []
    model_labels  = []

    for model_name, da in model_ensembles.items():
        if da is None:
            continue
        all_members.append(da)
        model_labels.extend([model_name] * da.sizes["member"])

    if not all_members:
        raise RuntimeError("No model ensembles available — all downloads failed.")

    # Align on the month axis before concatenating
    combined = xr.concat(all_members, dim="member")

    # Store which model each member came from as metadata
    combined = combined.assign_coords(
        member=np.arange(len(model_labels)),
        model=("member", model_labels),
    )

    print(f"[PROCESS] Grand ensemble: {combined.sizes['member']} total members "
          f"from {len(model_ensembles)} models.")
    return combined
