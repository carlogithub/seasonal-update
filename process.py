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
# C3S HINDCAST CLIMATOLOGY
# ─────────────────────────────────────────────────────────────────────────────

def load_hindcast_climatology(grib_path: str) -> dict:
    """
    Load the C3S hindcast climatological mean for one model from a GRIB file.

    The file contains the model's average forecast for each target month
    (April, May, June), expressed as monthly-mean SST over the Niño 3.4 box.
    We spatially average it to get one SST value per lead month.

    Returns
    -------
    dict mapping lead_month (int, 1-based) → area-mean SST (float, Kelvin)

    Example: {1: 301.2, 2: 301.5, 3: 301.4}
    where 1 = first target month (April for April init), 2 = May, 3 = June.
    """
    print(f"[PROCESS] Loading hindcast climatology from {grib_path} …")

    # cfgrib reads GRIB as xarray Datasets; squeeze=False keeps all dimensions
    datasets = cfgrib.open_datasets(grib_path, squeeze=False)

    if not datasets:
        raise RuntimeError(f"Could not read GRIB file: {grib_path}")

    # There may be multiple GRIB messages; combine and look for SST
    ds = datasets[0]
    for d in datasets[1:]:
        try:
            ds = xr.merge([ds, d])
        except Exception:
            pass

    # Find the SST variable — GRIB short names: 'sst' or 'skt' (skin temp) or 'sstk'
    sst_candidates = [v for v in ds.data_vars
                      if any(k in v.lower() for k in ["sst", "skt", "skin"])]
    if not sst_candidates:
        raise RuntimeError(f"No SST variable found in {grib_path}. "
                           f"Available: {list(ds.data_vars)}")
    sst_var = sst_candidates[0]
    sst = ds[sst_var]

    # Spatial average over the Niño 3.4 box
    # The 'step' dimension represents lead time (months ahead)
    clim_by_lead = {}
    for i, step_val in enumerate(sst.step.values):
        lead_month = i + 1   # 1-based lead month index
        sst_step = sst.isel(step=i)
        # Average over latitude and longitude
        spatial_mean = float(_area_mean(sst_step).values)
        clim_by_lead[lead_month] = spatial_mean

    print(f"[PROCESS]   Climatology lead months: {list(clim_by_lead.keys())}")
    return clim_by_lead


# ─────────────────────────────────────────────────────────────────────────────
# C3S FORECAST ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def load_forecast_ensemble(grib_path: str,
                            hindcast_clim: dict,
                            init_year: int) -> xr.DataArray:
    """
    Load C3S forecast ensemble members from a GRIB file and return a DataArray
    of daily Niño 3.4 *anomaly* trajectories for each member.

    The anomaly is computed by subtracting the model's hindcast climatological
    mean for the appropriate target month. This removes the model's systematic
    bias before we pool members from different models into a grand ensemble.

    Parameters
    ----------
    grib_path      : path to the forecast GRIB file
    hindcast_clim  : dict from load_hindcast_climatology(), keyed by lead month
    init_year      : initialisation year (needed to compute actual dates)

    Returns
    -------
    xr.DataArray with dimensions ('member', 'time'), values in °C-equivalent
    anomaly units. The 'time' coordinate holds actual calendar dates.
    """
    print(f"[PROCESS] Loading forecast ensemble from {grib_path} …")

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

    # Spatial average → shape (member, step) where step is lead time in hours
    sst_spatial = _area_mean(sst)

    # Convert the 'step' coordinate (lead-time offsets) to actual calendar dates.
    # The initialisation date is always the 1st of INIT_MONTH.
    init_date = date(init_year, config.INIT_MONTH, 1)
    actual_dates = pd.to_datetime([
        init_date + timedelta(hours=int(s / np.timedelta64(1, 'h')))
        for s in sst_spatial.step.values
    ])

    # Determine which lead-month index (1-based) each date belongs to,
    # so we can subtract the right climatological monthly mean.
    # lead_month 1 = same month as INIT_MONTH, 2 = next month, etc.
    target_month_list = config.TARGET_MONTHS  # e.g. [4, 5, 6]

    # Build anomaly array: for each timestep subtract the climatological mean
    # of the corresponding target month.
    sst_values = sst_spatial.values  # shape: (member, time)

    # Make sure we have a 'number' dimension (ensemble member axis)
    if "number" not in sst_spatial.dims:
        # If only one member, add a dimension
        sst_values = sst_values[np.newaxis, :]
        n_members = 1
    else:
        n_members = sst_values.shape[sst_spatial.dims.index("number")]

    anomaly_values = sst_values.copy()
    for t_idx, dt in enumerate(actual_dates):
        cal_month = dt.month
        # Find which lead-month index this calendar month corresponds to
        if cal_month in target_month_list:
            lm_index = target_month_list.index(cal_month) + 1  # 1-based
            if lm_index in hindcast_clim:
                # Subtract model climatology for this lead month from all members
                anomaly_values[:, t_idx] -= hindcast_clim[lm_index]

    # Build output DataArray with clear dimension names
    member_dim = np.arange(n_members)
    result = xr.DataArray(
        anomaly_values,
        dims=["member", "time"],
        coords={"member": member_dim, "time": actual_dates},
    )
    result.attrs["units"] = "K (anomaly)"

    # Keep only dates that fall within the target season months
    mask = np.isin(actual_dates.month, config.TARGET_MONTHS)
    result = result.isel(time=mask)

    print(f"[PROCESS]   Ensemble shape: {result.shape}  "
          f"({n_members} members × {result.sizes['time']} days)")
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

    # Align on the time axis before concatenating (models may have slightly
    # different date ranges due to different calendar conventions)
    combined = xr.concat(all_members, dim="member")

    # Store which model each member came from as metadata
    combined = combined.assign_coords(
        member=np.arange(len(model_labels)),
        model=("member", model_labels),
    )

    print(f"[PROCESS] Grand ensemble: {combined.sizes['member']} total members "
          f"from {len(model_ensembles)} models.")
    return combined
