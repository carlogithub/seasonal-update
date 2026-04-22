"""
process.py — Load downloaded files and compute variable anomalies.

Responsibilities
----------------
  1. Load ERA5 NetCDF → daily time series for the chosen variable and location.
  2. Compute ERA5 anomalies relative to the hindcast climatological period.
  3. Load C3S monthly-mean forecast GRIB → per-member anomaly values for each
     target month (with per-model debiasing using the ERA5 climatology).
  4. Pool all model ensembles into one grand ensemble DataArray.

Generic design
--------------
All functions accept Variable, Season, and Location objects so they work
equally for SST (Niño 3.4), T2M (Terrassa), and TP (Terrassa).

The key differences between variables are:
  - Aggregation: temperature/SST uses mean(), precipitation uses sum().
  - Units:       T2M/SST raw values are in Kelvin (subtract kelvin_offset for °C).
                 TP raw values are in metres (multiply by mm_scale for mm).
  - Transform:   TP applies a cube-root before regression to stabilise the
                 distribution.  SST and T2M are untransformed.

DJF year boundary
-----------------
When the target season crosses a year boundary (e.g. DJF: Dec of init_year,
Jan/Feb of init_year+1) the time selections must use season.target_year() to
find the correct calendar year for each target month.
"""

import os
import numpy as np
import xarray as xr
import cfgrib
import pandas as pd
import config
from location import Location
from variable import Variable
from season   import Season


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _area_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Cosine-latitude weighted spatial mean over 'latitude' and 'longitude' dims.

    For large regions (Niño 3.4) these are proper spatial dimensions.
    For a small point-location box, CDS may return a single grid cell where
    lat/lon are scalar coordinates rather than dimensions — in that case there
    is nothing to average and we return the DataArray as-is.
    """
    lat_is_dim = "latitude" in da.dims and "longitude" in da.dims
    if not lat_is_dim:
        # Single grid point — drop the scalar coords so downstream code is clean
        return da.squeeze(drop=False)
    weights = np.cos(np.deg2rad(da.latitude))
    return da.weighted(weights).mean(dim=["latitude", "longitude"])


def _period_aggregate(da: xr.DataArray, variable: Variable) -> xr.DataArray:
    """
    Aggregate a time-slice DataArray to a scalar according to the variable's
    observation type: mean for temperature/SST, sum for precipitation.
    """
    if variable.obs_type == "mean":
        return float(da.mean())
    elif variable.obs_type == "total":
        return float(da.sum())
    else:
        raise ValueError(f"Unknown obs_type '{variable.obs_type}'")


# ─────────────────────────────────────────────────────────────────────────────
# ERA5
# ─────────────────────────────────────────────────────────────────────────────

def load_era5(nc_path: str, variable: Variable) -> xr.DataArray:
    """
    Load ERA5 data from a NetCDF file and return a daily area-averaged time series.

    ERA5 from CDS may name its time coordinate 'valid_time' — we rename it to
    'time' for consistency.  For precipitation, raw values are in metres; for
    temperature/SST, values are in Kelvin.  No unit conversion is done here —
    that happens in compute_era5_anomaly() and in the forecast loading.

    Parameters
    ----------
    nc_path  : path to the ERA5 NetCDF file
    variable : Variable object (determines which variable to extract)

    Returns
    -------
    xr.DataArray with dimension 'time', daily values in the raw CDS units.
    """
    print(f"[PROCESS] Loading ERA5 {variable.name} from {nc_path} …")
    ds = xr.open_dataset(nc_path)

    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # The variable may be stored under slightly different names — try the
    # standard ERA5 short name first, then the full CDS name.
    candidates = [variable.short_name, variable.era5_name,
                  "sst", "sea_surface_temperature", "t2m", "tp"]
    var_name = next((v for v in candidates if v in ds.data_vars), None)
    if var_name is None:
        raise RuntimeError(
            f"Could not find {variable.era5_name} in {nc_path}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[var_name]

    # Spatial average over the download box
    da = _area_mean(da)
    da.name = variable.short_name

    # For precipitation: convert m → mm
    if variable.mm_scale > 0:
        da = da * variable.mm_scale

    da = da.sortby("time")

    print(f"[PROCESS] ERA5 {variable.short_name}: "
          f"{da.time.values[0]} → {da.time.values[-1]}, n={len(da)}")
    return da


def compute_era5_anomaly(era5_raw: xr.DataArray,
                          variable: Variable,
                          season: Season,
                          clim_start: int,
                          clim_end: int) -> xr.DataArray:
    """
    Convert raw ERA5 values to anomalies by subtracting the climatological mean.

    For temperature/SST: we use a day-of-year climatology (one value per
    calendar day, averaged over the hindcast period).

    For precipitation: we use a monthly climatology (one value per calendar
    month, summed over the hindcast period), because we are interested in
    monthly totals rather than individual daily values.

    Parameters
    ----------
    era5_raw   : raw ERA5 time series from load_era5()
    variable   : Variable object
    season     : Season object (needed for DJF year-boundary handling)
    clim_start : first year of the climatological base period
    clim_end   : last  year of the climatological base period

    Returns
    -------
    xr.DataArray of anomalies.
    """
    clim = era5_raw.sel(time=slice(str(clim_start), str(clim_end)))

    if variable.obs_type == "mean":
        # Day-of-year climatology for temperature/SST
        doy_clim = clim.groupby("time.dayofyear").mean("time")
        anomaly  = era5_raw.groupby("time.dayofyear") - doy_clim

    elif variable.obs_type == "total":
        # Monthly climatology for precipitation:
        # compute mean monthly total over hindcast years, then subtract day-scaled value.
        # We work at daily resolution but express anomalies as:
        #   daily_value - (monthly_clim / days_in_month)
        # This keeps the time series at daily granularity while removing the seasonal cycle.
        monthly_clim = clim.groupby("time.month").mean("time")
        anomaly      = era5_raw.groupby("time.month") - monthly_clim
    else:
        raise ValueError(f"Unknown obs_type '{variable.obs_type}'")

    anomaly.name = f"{variable.short_name}_anomaly"
    return anomaly


# ─────────────────────────────────────────────────────────────────────────────
# C3S FORECAST ENSEMBLE  (monthly resolution)
# ─────────────────────────────────────────────────────────────────────────────

def _load_grib_values(grib_path: str, variable: Variable) -> tuple:
    """
    Load a C3S monthly GRIB and return (values_array, n_steps).

    Handles variable name detection and spatial averaging.
    Returns values as a 2-D numpy array (member, step) — scalar-member GRIBs
    (e.g. hindcast climatology) come back as shape (1, step).
    """
    datasets = cfgrib.open_datasets(grib_path, squeeze=False)
    if not datasets:
        raise RuntimeError(f"Could not read GRIB file: {grib_path}")

    ds = datasets[0]
    for d in datasets[1:]:
        try:
            ds = xr.merge([ds, d])
        except Exception:
            pass

    sst_keys = ["sst", "skt", "skin"]
    t2m_keys = ["t2m", "2t"]
    tp_keys  = ["tp", "tprate", "lsp", "cp"]

    if variable.short_name == "nino34":
        search_keys = sst_keys
    elif variable.short_name == "t2m":
        search_keys = t2m_keys
    elif variable.short_name == "tp":
        search_keys = tp_keys
    else:
        search_keys = [variable.short_name, variable.c3s_name]

    var_in_grib = next(
        (v for v in ds.data_vars if any(k in v.lower() for k in search_keys)),
        None,
    )
    if var_in_grib is None:
        raise RuntimeError(
            f"No {variable.name} variable found in {grib_path}. "
            f"Available: {list(ds.data_vars)}"
        )

    fc = ds[var_in_grib]
    fc_spatial = _area_mean(fc)
    values = fc_spatial.values

    if values.ndim == 1:
        values = values[np.newaxis, :]   # (1, step) for climatology / single-member

    return values   # shape: (member_or_1, step)


def _convert_tp_units(col: np.ndarray, variable: Variable) -> np.ndarray:
    """
    Ensure precipitation values are in mm.

    C3S GRIBs may deliver TP as m/s (tprate flux), m/month, or mm.
    We detect the unit by magnitude and convert to mm.
    """
    if variable.short_name != "tp" or variable.mm_scale == 0:
        return col
    median = np.nanmedian(np.abs(col))
    if median < 0.01:
        # Almost certainly m/s (tprate): convert to mm/month
        # 30.5 days/month × 86400 s/day × 1000 mm/m
        return col * 30.5 * 86400 * variable.mm_scale
    elif median < 1.0:
        # Likely in metres — convert to mm
        return col * variable.mm_scale
    return col   # already in mm


def _compute_hindcast_clim(hc_grib_path: str, variable: Variable,
                            n_use: int) -> np.ndarray | None:
    """
    Load a multi-year hindcast GRIB and return the mean over years for each
    lead step.

    The hindcast GRIB contains monthly_mean data for all hindcast years
    (1993–2016) stacked along a time/step dimension.  We average over all
    year-realisations to get the model's climatological mean for each
    (init_month, lead_month) pair.  Shape of result: (n_use,).

    Returns None if the file cannot be read.
    """
    try:
        datasets = cfgrib.open_datasets(hc_grib_path, squeeze=False)
        if not datasets:
            return None

        ds = datasets[0]
        for d in datasets[1:]:
            try:
                ds = xr.merge([ds, d])
            except Exception:
                pass

        sst_keys = ["sst", "skt", "skin"]
        t2m_keys = ["t2m", "2t"]
        tp_keys  = ["tp", "tprate", "lsp", "cp"]

        if variable.short_name == "nino34":
            search_keys = sst_keys
        elif variable.short_name == "t2m":
            search_keys = t2m_keys
        elif variable.short_name == "tp":
            search_keys = tp_keys
        else:
            search_keys = [variable.short_name, variable.c3s_name]

        var_name = next(
            (v for v in ds.data_vars if any(k in v.lower() for k in search_keys)),
            None,
        )
        if var_name is None:
            return None

        da = ds[var_name]
        da = _area_mean(da)
        values = da.values   # may have dims: (number, time, step) or (time, step) etc.

        # Flatten to 2-D: (all_year_member_combinations, step)
        values = values.reshape(-1, values.shape[-1])

        # Mean over all year×member combinations → shape (step,)
        clim = np.nanmean(values, axis=0)
        return clim[:n_use]

    except Exception as exc:
        print(f"[PROCESS]   Could not compute hindcast climatology: {exc}")
        return None


def load_forecast_ensemble_monthly(grib_path: str,
                                    hindcast_path: str | None,
                                    era5_raw: xr.DataArray,
                                    variable: Variable,
                                    season: Season) -> xr.DataArray:
    """
    Load a C3S monthly_mean forecast GRIB and return per-member anomalies.

    Debiasing strategy (preferred → fallback)
    ------------------------------------------
    1. If hindcast_path exists: load the multi-year hindcast GRIB, compute the
       mean over all years for each lead step → model climatological mean.
       Subtract from each forecast member to get anomalies in the model's own
       reference frame.  This correctly removes bias for each (init, lead) pair.

    2. Fallback (no hindcast file): subtract ERA5 climatological monthly mean.
       Leaves the model's absolute bias uncorrected — use only when hindcast
       data is unavailable.

    Parameters
    ----------
    grib_path    : path to the operational forecast monthly_mean GRIB
    hindcast_path: path to the multi-year hindcast monthly_mean GRIB, or None
    era5_raw     : raw ERA5 time series (for ERA5 fallback debiasing)
    variable     : Variable object
    season       : Season object

    Returns xr.DataArray (member × month) of anomalies.
    """
    print(f"[PROCESS] Loading monthly forecast from {grib_path} …")

    fc_values = _load_grib_values(grib_path, variable)
    n_members, n_steps = fc_values.shape
    target_months = season.target_months
    n_use = min(n_steps, len(target_months))

    # ── Debiasing ─────────────────────────────────────────────────────────────
    clim_per_step = None
    debias_source = "ERA5 climatology (fallback)"

    if hindcast_path and os.path.exists(hindcast_path):
        clim_per_step = _compute_hindcast_clim(hindcast_path, variable, n_use)
        if clim_per_step is not None:
            debias_source = "model hindcast climatology (1993–2016)"
            print(f"[PROCESS]   Debiasing with model hindcast climatology …")
            for i, cal_month in enumerate(target_months[:n_use]):
                raw_clim = _convert_tp_units(
                    np.array([clim_per_step[i]]), variable
                )[0]
                print(f"[PROCESS]   Model clim month {cal_month:02d}: {raw_clim:.4f}")

    if clim_per_step is None:
        # ERA5 fallback
        print(f"[PROCESS]   Using ERA5 fallback debiasing.")
        era5_clim = []
        for cal_month in target_months[:n_use]:
            vals = []
            for yr in range(config.HINDCAST_START_YEAR, config.HINDCAST_END_YEAR + 1):
                cal_year = season.target_year(yr, cal_month)
                m = era5_raw.sel(time=(
                    (era5_raw.time.dt.year  == cal_year) &
                    (era5_raw.time.dt.month == cal_month)
                ))
                if len(m) > 0:
                    vals.append(_period_aggregate(m, variable))
            era5_clim.append(float(np.mean(vals)) if vals else 0.0)
        clim_per_step = np.array(era5_clim)

    # ── Build anomaly array ───────────────────────────────────────────────────
    month_labels = []
    anomaly_cols = []
    for step_idx in range(n_use):
        cal_month = target_months[step_idx]
        month_labels.append(cal_month)
        col      = _convert_tp_units(fc_values[:, step_idx], variable)
        clim_val = _convert_tp_units(np.array([clim_per_step[step_idx]]), variable)[0]
        anomaly_cols.append(col - clim_val)

    result = xr.DataArray(
        np.column_stack(anomaly_cols),
        dims=["member", "month"],
        coords={"member": np.arange(n_members), "month": month_labels},
    )
    result.attrs["units"]         = f"{variable.units} (anomaly)"
    result.attrs["debias_source"] = debias_source

    print(f"[PROCESS]   Ensemble shape: {result.shape}  "
          f"({n_members} members × {len(month_labels)} months)  "
          f"[debias: {debias_source}]")
    return result


def load_forecast_postprocessed(grib_path: str,
                                variable: Variable,
                                season: Season) -> xr.DataArray:
    """
    Load a seasonal-postprocessed-single-levels GRIB (bias-corrected anomalies).

    No debiasing is performed — the values are already anomalies.
    For TP, applies unit conversion (m/s → mm) and the cube-root transform
    is NOT applied here; process.build_grand_ensemble callers handle transforms.

    Returns xr.DataArray (member × month).
    """
    print(f"[PROCESS] Loading postprocessed forecast from {grib_path} …")

    datasets = cfgrib.open_datasets(grib_path, squeeze=False)
    if not datasets:
        raise RuntimeError(f"Could not read postprocessed GRIB: {grib_path}")

    ds = datasets[0]
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    da = _area_mean(da)

    values = da.values   # shape: (number, step) — already anomalies

    if variable.short_name == "tp":
        for i in range(values.shape[1] if values.ndim > 1 else 1):
            col = values[:, i] if values.ndim > 1 else values
            values[:, i] = _convert_tp_units(col, variable)

    n_members = values.shape[0]
    target_months = season.target_months
    n_use = min(values.shape[1] if values.ndim > 1 else 1, len(target_months))

    result = xr.DataArray(
        values[:, :n_use],
        dims=["member", "month"],
        coords={"member": np.arange(n_members), "month": target_months[:n_use]},
    )
    result.attrs["units"]         = f"{variable.units} (anomaly)"
    result.attrs["debias_source"] = "seasonal-postprocessed-single-levels"

    print(f"[PROCESS]   Postprocessed ensemble: {result.shape}  "
          f"({n_members} members × {n_use} months)  [no debiasing needed]")
    return result


def load_model_forecast(fc_path: str,
                        hc_path: str | None,
                        era5_raw: xr.DataArray,
                        variable: Variable,
                        season: Season) -> xr.DataArray:
    """
    Dispatch to the correct loader based on the forecast file type.

    Postprocessed files (prefix forecast_postproc_) are loaded directly as
    bias-corrected anomalies.  Monthly-mean files (prefix forecast_monthly_)
    are loaded with per-model hindcast debiasing or ERA5 fallback.
    """
    if os.path.basename(fc_path).startswith("forecast_postproc_"):
        return load_forecast_postprocessed(fc_path, variable, season)
    else:
        return load_forecast_ensemble_monthly(fc_path, hc_path, era5_raw, variable, season)


# ─────────────────────────────────────────────────────────────────────────────
# GRAND ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def build_grand_ensemble(model_ensembles: dict) -> xr.DataArray:
    """
    Concatenate per-model ensembles into one grand ensemble.

    Parameters
    ----------
    model_ensembles : dict mapping model_name → xr.DataArray (member × month)

    Returns
    -------
    xr.DataArray (member × month) with a 'model' coordinate recording the
    originating model for each member.
    """
    all_members  = []
    model_labels = []

    for model_name, da in model_ensembles.items():
        if da is None:
            continue
        all_members.append(da)
        model_labels.extend([model_name] * da.sizes["member"])

    if not all_members:
        raise RuntimeError("No model ensembles available — all downloads failed.")

    combined = xr.concat(all_members, dim="member")
    combined = combined.assign_coords(
        member=np.arange(len(model_labels)),
        model=("member", model_labels),
    )

    print(f"[PROCESS] Grand ensemble: {combined.sizes['member']} total members "
          f"from {len(model_ensembles)} models.")
    return combined
