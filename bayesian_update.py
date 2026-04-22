"""
bayesian_update.py — Bayesian regression update of seasonal forecast probabilities.

The update logic follows three steps:

  STEP 1 — ERA5 regression
    Using historical ERA5 anomalies (1993–2016), we fit a linear regression
    that predicts the seasonal mean anomaly Y from the partial-month running
    mean/total x observed up to the cutoff day.

    For precipitation, both x and Y are cube-root transformed before fitting
    the regression. The cube-root stabilises the distribution and makes the
    Gaussian assumption more defensible.

  STEP 2 — Gaussian prior from the C3S ensemble
    The grand ensemble of seasonal-mean anomalies forms the prior P(Y | forecast).
    We approximate it as Gaussian.  For precipitation, ensemble values are also
    cube-root transformed before fitting the Gaussian.

  STEP 3 — Bayesian fusion (Gaussian conjugate update)
    The likelihood P(x_obs | Y) from the regression is Gaussian.
    The posterior is also Gaussian (closed-form conjugate update):

        posterior precision  = 1/σ²_f + 1/σ²_ε
        posterior mean       = (μ_f/σ²_f + μ_r/σ²_ε) / posterior_precision

    where μ_r = α + β·x_obs_transformed  and  σ²_ε is the regression residual variance.

    Tercile probabilities follow from the posterior Gaussian CDF,
    with thresholds derived from the same transformed ERA5 climatology.
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List
import config
from variable import Variable
from season   import Season


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionParams:
    """Fitted parameters of the ERA5-based linear regression."""
    alpha: float   # intercept
    beta:  float   # slope
    sigma: float   # residual standard deviation
    r2:    float   # coefficient of determination


@dataclass
class GaussianDist:
    """A Gaussian distribution described by its mean and standard deviation."""
    mean: float
    std:  float


@dataclass
class TercileProbs:
    """Probabilities for the three standard seasonal forecast categories."""
    below_normal: float
    near_normal:  float
    above_normal: float


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ERA5 REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def fit_era5_regression(era5_anomaly: xr.DataArray,
                         variable: Variable,
                         season: Season,
                         cutoff_day: int,
                         hindcast_start: int,
                         hindcast_end: int) -> RegressionParams:
    """
    Fit Y = α + β·x + ε using historical ERA5 data.

    x = partial-month aggregate of init_month up to cutoff_day
        (mean for T/SST, total for TP) — cube-root transformed for TP
    Y = seasonal aggregate over target_months
        (mean for T/SST, total for TP) — cube-root transformed for TP

    Parameters
    ----------
    era5_anomaly   : daily ERA5 anomaly time series from process.py
    variable       : Variable object (determines aggregation and transform)
    season         : Season object (init_month, target_months, year boundary)
    cutoff_day     : how many days of init_month are observed
    hindcast_start : first year of the training period
    hindcast_end   : last year  of the training period

    Returns
    -------
    RegressionParams with fitted α, β, σ, R².
    """
    x_vals = []
    y_vals = []

    for year in range(hindcast_start, hindcast_end + 1):
        try:
            # ── Predictor x: partial init-month aggregate ─────────────────────
            partial = era5_anomaly.sel(
                time=(
                    (era5_anomaly.time.dt.year  == year) &
                    (era5_anomaly.time.dt.month == season.init_month) &
                    (era5_anomaly.time.dt.day   <= cutoff_day)
                )
            )
            if len(partial) < max(1, cutoff_day // 2):
                continue
            x = _aggregate(partial, variable)

            # ── Predictand Y: full seasonal aggregate ─────────────────────────
            # We concatenate the values for each target month, respecting year boundaries
            seasonal_vals = []
            for cal_month in season.target_months:
                cal_year = season.target_year(year, cal_month)
                month_da = era5_anomaly.sel(
                    time=(
                        (era5_anomaly.time.dt.year  == cal_year) &
                        (era5_anomaly.time.dt.month == cal_month)
                    )
                )
                if len(month_da) > 0:
                    seasonal_vals.extend(month_da.values.tolist())

            if len(seasonal_vals) < 30:
                continue

            # Seasonal aggregate: mean for T/SST, sum for TP
            if variable.obs_type == "mean":
                y = float(np.mean(seasonal_vals))
            else:
                y = float(np.sum(seasonal_vals))

            # Apply cube-root transform for precipitation
            x = float(variable.apply_transform(x))
            y = float(variable.apply_transform(y))

            x_vals.append(x)
            y_vals.append(y)

        except Exception:
            continue

    if len(x_vals) < 5:
        raise RuntimeError(
            f"Only {len(x_vals)} valid historical samples for regression — "
            f"need at least 5. Check ERA5 data coverage."
        )

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)

    slope, intercept, r_value, _, _ = stats.linregress(x_arr, y_arr)

    y_pred    = intercept + slope * x_arr
    residuals = y_arr - y_pred
    sigma_eps = float(np.std(residuals, ddof=2))

    print(f"[BAYES] ERA5 regression ({variable.short_name}): "
          f"α={intercept:.3f}, β={slope:.3f}, σ={sigma_eps:.3f}, "
          f"R²={r_value**2:.2f}  (n={len(x_vals)} years)")

    return RegressionParams(
        alpha=float(intercept),
        beta=float(slope),
        sigma=sigma_eps,
        r2=r_value ** 2,
    )


def _aggregate(da: xr.DataArray, variable: Variable) -> float:
    """Aggregate a DataArray slice using the variable's obs_type."""
    if variable.obs_type == "mean":
        return float(da.mean())
    else:
        return float(da.sum())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GAUSSIAN PRIOR FROM ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def compute_prior_monthly(ensemble: xr.DataArray,
                           variable: Variable) -> GaussianDist:
    """
    Derive the Gaussian prior from a monthly ensemble (member × month).

    Takes the mean across all target months for each member (or sum for TP),
    applies the variable's transform, then fits a Gaussian.

    Parameters
    ----------
    ensemble : xr.DataArray (member × month) from process.build_grand_ensemble()
    variable : Variable object (determines aggregation and transform)
    """
    if variable.obs_type == "mean":
        seasonal_means = ensemble.mean(dim="month").values
    else:
        seasonal_means = ensemble.sum(dim="month").values

    # Apply transform (cube-root for TP, no-op for others)
    seasonal_means = np.array([float(variable.apply_transform(v)) for v in seasonal_means])

    mu    = float(np.mean(seasonal_means))
    sigma = float(np.std(seasonal_means, ddof=1))
    print(f"[BAYES] Prior ({variable.short_name} monthly ensemble): "
          f"μ={mu:.3f}, σ={sigma:.3f}, n_members={len(seasonal_means)}")
    return GaussianDist(mean=mu, std=sigma)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BAYESIAN FUSION
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_update(prior: GaussianDist,
                     reg: RegressionParams,
                     x_obs: float) -> GaussianDist:
    """
    Gaussian conjugate update.

    Likelihood: P(x_obs | Y) ∝ N(Y; α + β·x_obs, σ²_ε)
    Prior:      P(Y | fc)  = N(μ_f, σ²_f)
    Posterior:  precision = 1/σ²_f + 1/σ²_ε
                mean      = σ²_post · (μ_f/σ²_f + μ_r/σ²_ε)

    All quantities are in transform space (cube-root for TP).
    x_obs must already be transformed before calling this function.
    """
    var_f = prior.std ** 2
    var_e = reg.sigma ** 2

    mu_r = reg.alpha + reg.beta * x_obs

    prec_f    = 1.0 / var_f
    prec_e    = 1.0 / var_e
    prec_post = prec_f + prec_e

    mu_post    = (prec_f * prior.mean + prec_e * mu_r) / prec_post
    sigma_post = np.sqrt(1.0 / prec_post)

    print(f"[BAYES] x_obs={x_obs:.3f} → μ_r={mu_r:.3f} | "
          f"Posterior: μ={mu_post:.3f}, σ={sigma_post:.3f}")

    return GaussianDist(mean=mu_post, std=sigma_post)


# ─────────────────────────────────────────────────────────────────────────────
# TERCILE PROBABILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_tercile_thresholds(era5_anomaly: xr.DataArray,
                                variable: Variable,
                                season: Season,
                                hindcast_start: int,
                                hindcast_end: int) -> tuple:
    """
    Compute the lower and upper tercile thresholds from ERA5 climatology.

    Thresholds are the 33rd and 67th percentiles of the ERA5 seasonal aggregate
    over the hindcast period.  For precipitation they are in cube-root space
    (matching the transform applied to the regression and prior).

    Returns
    -------
    (t_lower, t_upper) both in the same transform space as the prior/regression.
    """
    seasonal_vals = []
    for year in range(hindcast_start, hindcast_end + 1):
        vals = []
        for cal_month in season.target_months:
            cal_year = season.target_year(year, cal_month)
            month_da = era5_anomaly.sel(
                time=(
                    (era5_anomaly.time.dt.year  == cal_year) &
                    (era5_anomaly.time.dt.month == cal_month)
                )
            )
            if len(month_da) > 0:
                vals.extend(month_da.values.tolist())

        if len(vals) < 30:
            continue

        if variable.obs_type == "mean":
            agg = float(np.mean(vals))
        else:
            agg = float(np.sum(vals))

        seasonal_vals.append(float(variable.apply_transform(agg)))

    arr     = np.array(seasonal_vals)
    t_lower = float(np.percentile(arr, 100 * config.TERCILE_LOWER))
    t_upper = float(np.percentile(arr, 100 * config.TERCILE_UPPER))

    print(f"[BAYES] ERA5 tercile thresholds ({variable.short_name}): "
          f"BN < {t_lower:.3f} ≤ NN ≤ {t_upper:.3f} < AN")
    return t_lower, t_upper


def compute_tercile_probs(posterior: GaussianDist,
                           t_lower: float,
                           t_upper: float) -> TercileProbs:
    """Compute tercile probabilities from a Gaussian posterior."""
    norm    = stats.norm(loc=posterior.mean, scale=posterior.std)
    p_below = float(norm.cdf(t_lower))
    p_above = float(1.0 - norm.cdf(t_upper))
    p_normal = 1.0 - p_below - p_above
    return TercileProbs(below_normal=p_below,
                        near_normal=p_normal,
                        above_normal=p_above)


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_probability_evolution(era5_anomaly: xr.DataArray,
                                   ensemble: xr.DataArray,
                                   variable: Variable,
                                   season: Season,
                                   init_year: int,
                                   max_cutoff_day: int) -> pd.DataFrame:
    """
    Repeat the Bayesian update for cutoff days 1, 2, …, max_cutoff_day.

    Returns a DataFrame recording how tercile probabilities evolve as more
    observations accumulate during the initialisation month.

    Parameters
    ----------
    era5_anomaly   : full ERA5 anomaly time series
    ensemble       : grand C3S ensemble (member × month)
    variable       : Variable object
    season         : Season object
    init_year      : forecast year
    max_cutoff_day : last day to compute (e.g. 21)

    Returns
    -------
    pd.DataFrame with columns:
      cutoff_day, x_obs, mu_prior, mu_posterior, sigma_posterior,
      prob_below, prob_normal, prob_above
    """
    # Prior and thresholds are fixed across all cutoff days
    prior    = compute_prior_monthly(ensemble, variable)
    t_lower, t_upper = compute_tercile_thresholds(
        era5_anomaly, variable, season,
        config.HINDCAST_START_YEAR, config.HINDCAST_END_YEAR,
    )

    rows = []
    for cutoff_day in range(1, max_cutoff_day + 1):
        try:
            reg = fit_era5_regression(
                era5_anomaly, variable, season, cutoff_day,
                config.HINDCAST_START_YEAR, config.HINDCAST_END_YEAR,
            )
        except RuntimeError as e:
            print(f"[BAYES] Skipping day {cutoff_day}: {e}")
            continue

        # Current year's partial-month observation
        obs_partial = era5_anomaly.sel(
            time=(
                (era5_anomaly.time.dt.year  == init_year) &
                (era5_anomaly.time.dt.month == season.init_month) &
                (era5_anomaly.time.dt.day   <= cutoff_day)
            )
        )
        if len(obs_partial) == 0:
            continue

        x_raw = _aggregate(obs_partial, variable)
        x_obs = float(variable.apply_transform(x_raw))   # transform before Bayesian update

        posterior = bayesian_update(prior, reg, x_obs)
        probs     = compute_tercile_probs(posterior, t_lower, t_upper)

        rows.append({
            "cutoff_day":      cutoff_day,
            "x_obs":           x_obs,
            "mu_prior":        prior.mean,
            "mu_posterior":    posterior.mean,
            "sigma_posterior": posterior.std,
            "prob_below":      probs.below_normal,
            "prob_normal":     probs.near_normal,
            "prob_above":      probs.above_normal,
        })

    df = pd.DataFrame(rows)
    print(f"[BAYES] Evolution computed for days 1–{max_cutoff_day} "
          f"({len(df)} valid points).")
    return df
