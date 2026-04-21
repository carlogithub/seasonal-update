"""
bayesian_update.py — Bayesian regression update of seasonal forecast probabilities.

The update logic follows three steps:

  STEP 1 — ERA5 regression
    Using historical ERA5 Niño 3.4 anomalies (1993–2016), we fit a linear
    regression that predicts the seasonal (e.g. AMJ) mean anomaly Y from the
    partial-month running mean anomaly x observed up to the cutoff day.
    This regression quantifies how informative the partial observation is about
    the seasonal outcome.

  STEP 2 — Gaussian prior from the C3S ensemble
    The grand ensemble of seasonal-mean anomalies from all C3S models forms our
    prior: P(Y | forecast).  We approximate it as Gaussian with mean μ_f and
    variance σ²_f.

  STEP 3 — Bayesian fusion (Gaussian conjugate update)
    The likelihood P(x_obs | Y) derived from the regression is also Gaussian.
    The posterior is therefore Gaussian and has a closed-form solution:

        Posterior precision   = 1/σ²_f  +  1/σ²_ε
        Posterior mean        = (μ_f/σ²_f  +  μ_r/σ²_ε) / posterior_precision

    where μ_r = α + β·x_obs is the regression's prediction and σ²_ε is the
    residual variance of the regression.

    Tercile probabilities then follow directly from the posterior Gaussian CDF.
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List
import config


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES (plain containers — no fancy OOP)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionParams:
    """Slope, intercept and residual std of the ERA5-based regression."""
    alpha: float   # intercept
    beta:  float   # slope (sensitivity of seasonal mean to partial obs)
    sigma: float   # residual standard deviation (square root of σ²_ε)
    r2:    float   # coefficient of determination (how much variance is explained)


@dataclass
class GaussianDist:
    """A Gaussian distribution described by its mean and standard deviation."""
    mean: float
    std:  float


@dataclass
class TercileProbs:
    """Probabilities for the three standard seasonal forecast categories."""
    below_normal: float   # probability of being in the lower tercile
    near_normal:  float   # probability of being in the middle tercile
    above_normal: float   # probability of being in the upper tercile


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ERA5 REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def fit_era5_regression(era5_anomaly: xr.DataArray,
                         cutoff_day: int,
                         init_month: int,
                         target_months: List[int],
                         hindcast_start: int,
                         hindcast_end: int) -> RegressionParams:
    """
    Fit a linear regression: Y = α + β·x + ε  using ERA5 historical data.

    For each historical year (hindcast_start to hindcast_end):
      x = mean of daily Niño 3.4 anomaly from day 1 to cutoff_day of init_month
      Y = mean of daily Niño 3.4 anomaly over the full target season (e.g. AMJ)

    The regression captures the statistical relationship between what we have
    already observed (partial month x) and the seasonal outcome Y we are trying
    to forecast.

    Parameters
    ----------
    era5_anomaly   : daily Niño 3.4 anomaly time series (from process.py)
    cutoff_day     : how many days of the init month are observed (e.g. 21)
    init_month     : calendar month of the forecast initialisation (e.g. 4 = April)
    target_months  : list of calendar months defining the target season (e.g. [4,5,6])
    hindcast_start : first year of the regression training period
    hindcast_end   : last  year of the regression training period

    Returns
    -------
    RegressionParams with fitted α, β, σ, R².
    """
    x_vals = []   # partial-month running means (predictors)
    y_vals = []   # seasonal means (predictands)

    for year in range(hindcast_start, hindcast_end + 1):
        try:
            # --- predictor x: running mean up to cutoff_day ---
            # Select daily anomalies for the initialisation month in this year
            partial = era5_anomaly.sel(
                time=(
                    (era5_anomaly.time.dt.year  == year) &
                    (era5_anomaly.time.dt.month == init_month) &
                    (era5_anomaly.time.dt.day   <= cutoff_day)
                )
            )
            if len(partial) < cutoff_day // 2:
                # Too few observations for this year — skip
                continue
            x = float(partial.mean())

            # --- predictand Y: seasonal mean over target months ---
            seasonal = era5_anomaly.sel(
                time=(
                    (era5_anomaly.time.dt.year  == year) &
                    (era5_anomaly.time.dt.month.isin(target_months))
                )
            )
            if len(seasonal) < 30:
                # Less than ~30 days in the target season — skip
                continue
            y = float(seasonal.mean())

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

    # Ordinary least squares via scipy
    slope, intercept, r_value, _, std_err = stats.linregress(x_arr, y_arr)

    # Residual standard deviation: how much Y varies around the regression line
    y_pred  = intercept + slope * x_arr
    residuals = y_arr - y_pred
    sigma_eps = float(np.std(residuals, ddof=2))   # unbiased estimate

    print(f"[BAYES] ERA5 regression: α={intercept:.3f}, β={slope:.3f}, "
          f"σ={sigma_eps:.3f}, R²={r_value**2:.2f}  (n={len(x_vals)} years)")

    return RegressionParams(
        alpha=float(intercept),
        beta=float(slope),
        sigma=sigma_eps,
        r2=r_value ** 2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GAUSSIAN PRIOR FROM ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def compute_prior_monthly(ensemble: xr.DataArray) -> GaussianDist:
    """
    Derive the Gaussian prior from a monthly ensemble DataArray (member × month).

    Takes the mean across all target months for each member, then fits a Gaussian
    to the resulting distribution of seasonal means.
    """
    seasonal_means = ensemble.mean(dim="month").values   # shape: (member,)
    mu    = float(np.mean(seasonal_means))
    sigma = float(np.std(seasonal_means, ddof=1))
    print(f"[BAYES] Prior (C3S monthly ensemble): μ={mu:.3f} K, σ={sigma:.3f} K, "
          f"n_members={len(seasonal_means)}")
    return GaussianDist(mean=mu, std=sigma)


def compute_prior(ensemble: xr.DataArray,
                   target_months: List[int]) -> GaussianDist:
    """
    Derive the Gaussian prior P(Y | forecast) from the C3S grand ensemble.

    Y is the seasonal-mean Niño 3.4 anomaly across all target months.
    For each ensemble member we compute Y, then fit a Gaussian to the
    resulting distribution.

    Parameters
    ----------
    ensemble      : (member × time) DataArray from process.build_grand_ensemble()
    target_months : list of calendar months for the target season

    Returns
    -------
    GaussianDist representing the prior distribution over Y.
    """
    # Seasonal mean for each member: average over all target-season days
    mask = np.isin(ensemble.time.dt.month.values, target_months)
    seasonal_means = ensemble.isel(time=mask).mean(dim="time").values  # shape: (member,)

    mu    = float(np.mean(seasonal_means))
    sigma = float(np.std(seasonal_means, ddof=1))

    print(f"[BAYES] Prior (C3S ensemble): μ={mu:.3f} K, σ={sigma:.3f} K, "
          f"n_members={len(seasonal_means)}")

    return GaussianDist(mean=mu, std=sigma)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BAYESIAN FUSION
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_update(prior: GaussianDist,
                     reg: RegressionParams,
                     x_obs: float) -> GaussianDist:
    """
    Compute the Gaussian posterior given the prior and the current observation.

    The likelihood is derived from the ERA5 regression:
        P(x_obs | Y)  ∝  N(Y;  α + β·x_obs,  σ²_ε)

    Combined with the Gaussian prior P(Y | forecast) = N(μ_f, σ²_f), the
    posterior is:

        1/σ²_post = 1/σ²_f + 1/σ²_ε
        μ_post    = σ²_post · (μ_f/σ²_f  +  μ_r/σ²_ε)

    where μ_r = α + β·x_obs  (regression prediction from the observation).

    If the regression R² is very low (< 0.05) the observation carries little
    information and the posterior stays close to the prior — this is handled
    naturally by the maths (large σ_ε → low likelihood weight).

    Parameters
    ----------
    prior  : Gaussian prior from compute_prior()
    reg    : regression parameters from fit_era5_regression()
    x_obs  : observed partial-month Niño 3.4 running mean (scalar, Kelvin)

    Returns
    -------
    GaussianDist representing the posterior.
    """
    # Variance of prior and likelihood
    var_f = prior.std ** 2
    var_e = reg.sigma ** 2

    # Regression prediction given the observation
    mu_r = reg.alpha + reg.beta * x_obs

    # Posterior precision = sum of precisions (Gaussian conjugate formula)
    prec_f    = 1.0 / var_f
    prec_e    = 1.0 / var_e
    prec_post = prec_f + prec_e

    # Posterior mean = precision-weighted average of prior mean and regression prediction
    mu_post    = (prec_f * prior.mean + prec_e * mu_r) / prec_post
    sigma_post = np.sqrt(1.0 / prec_post)

    print(f"[BAYES] x_obs={x_obs:.3f} → μ_r={mu_r:.3f} | "
          f"Posterior: μ={mu_post:.3f}, σ={sigma_post:.3f}")

    return GaussianDist(mean=mu_post, std=sigma_post)


# ─────────────────────────────────────────────────────────────────────────────
# TERCILE PROBABILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_tercile_thresholds(era5_anomaly: xr.DataArray,
                                target_months: List[int],
                                hindcast_start: int,
                                hindcast_end: int) -> tuple:
    """
    Compute the lower and upper tercile thresholds from ERA5 climatology.

    The thresholds are the 33rd and 67th percentiles of the ERA5 seasonal-mean
    Niño 3.4 anomaly over the hindcast period. They define the boundaries between
    below-normal, near-normal, and above-normal categories.

    Returns
    -------
    (lower_threshold, upper_threshold)  both in Kelvin (anomaly)
    """
    seasonal_means = []
    for year in range(hindcast_start, hindcast_end + 1):
        seas = era5_anomaly.sel(
            time=(
                (era5_anomaly.time.dt.year  == year) &
                (era5_anomaly.time.dt.month.isin(target_months))
            )
        )
        if len(seas) >= 30:
            seasonal_means.append(float(seas.mean()))

    arr = np.array(seasonal_means)
    t_lower = float(np.percentile(arr, 100 * config.TERCILE_LOWER))
    t_upper = float(np.percentile(arr, 100 * config.TERCILE_UPPER))

    print(f"[BAYES] ERA5 tercile thresholds: BN < {t_lower:.3f} K ≤ NN ≤ "
          f"{t_upper:.3f} K < AN")
    return t_lower, t_upper


def compute_tercile_probs(posterior: GaussianDist,
                           t_lower: float,
                           t_upper: float) -> TercileProbs:
    """
    Compute the probability of each tercile category from the posterior Gaussian.

    P(BN) = Φ( (t_lower − μ_post) / σ_post )
    P(AN) = 1 − Φ( (t_upper − μ_post) / σ_post )
    P(NN) = 1 − P(BN) − P(AN)

    where Φ is the standard normal CDF.
    """
    norm = stats.norm(loc=posterior.mean, scale=posterior.std)
    p_below  = float(norm.cdf(t_lower))
    p_above  = float(1.0 - norm.cdf(t_upper))
    p_normal = 1.0 - p_below - p_above

    return TercileProbs(
        below_normal=p_below,
        near_normal=p_normal,
        above_normal=p_above,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVOLUTION OVER TIME (how probabilities change as more days are observed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_probability_evolution(era5_anomaly: xr.DataArray,
                                   ensemble: xr.DataArray,
                                   init_year: int,
                                   max_cutoff_day: int,
                                   monthly: bool = False) -> pd.DataFrame:
    """
    Repeat the Bayesian update for cutoff days 1, 2, …, max_cutoff_day and
    record how the tercile probabilities evolve as more observations accumulate.

    This produces the data for the 'probability evolution' plot: a time series
    showing how confident we become about the seasonal outcome as the month progresses.

    Parameters
    ----------
    era5_anomaly    : full ERA5 anomaly time series
    ensemble        : grand C3S ensemble DataArray (member × time)
    init_year       : the forecast year
    max_cutoff_day  : last day to include (e.g. 21 if today is the 21st)

    Returns
    -------
    pd.DataFrame with columns:
      cutoff_day, x_obs, mu_prior, mu_posterior, sigma_posterior,
      prob_below, prob_normal, prob_above
    """
    # Compute prior and thresholds once (they don't change with cutoff day)
    prior = compute_prior_monthly(ensemble) if monthly else compute_prior(ensemble, config.TARGET_MONTHS)
    t_lower, t_upper = compute_tercile_thresholds(
        era5_anomaly,
        config.TARGET_MONTHS,
        config.HINDCAST_START_YEAR,
        config.HINDCAST_END_YEAR,
    )

    rows = []
    for cutoff_day in range(1, max_cutoff_day + 1):

        # Fit regression for this specific cutoff day
        try:
            reg = fit_era5_regression(
                era5_anomaly,
                cutoff_day=cutoff_day,
                init_month=config.INIT_MONTH,
                target_months=config.TARGET_MONTHS,
                hindcast_start=config.HINDCAST_START_YEAR,
                hindcast_end=config.HINDCAST_END_YEAR,
            )
        except RuntimeError as e:
            print(f"[BAYES] Skipping day {cutoff_day}: {e}")
            continue

        # Get observed partial-month running mean from ERA5 for the current year
        obs_partial = era5_anomaly.sel(
            time=(
                (era5_anomaly.time.dt.year  == init_year) &
                (era5_anomaly.time.dt.month == config.INIT_MONTH) &
                (era5_anomaly.time.dt.day   <= cutoff_day)
            )
        )
        if len(obs_partial) == 0:
            print(f"[BAYES] No ERA5 observations for {init_year}-{config.INIT_MONTH:02d} "
                  f"day≤{cutoff_day} — skipping.")
            continue
        x_obs = float(obs_partial.mean())

        # Bayesian update and tercile probabilities
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
    print(f"[BAYES] Evolution computed for days 1–{max_cutoff_day}  "
          f"({len(df)} valid points).")
    return df
