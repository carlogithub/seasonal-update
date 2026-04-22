"""
validate.py — Sanity checks and leave-one-out cross-validation.

Basic checks (always run after the pipeline):
  1. Tercile probabilities sum to 1.0
  2. Posterior uncertainty < prior uncertainty
  3. Grand ensemble has at least 20 members
  4. Regression R² flagged if < 0.05 (low predictability)
  5. ERA5 has data for all hindcast years in the init month

LOO cross-validation (--validate flag):
  For each year y in the hindcast period, refit the ERA5 regression
  excluding y, apply the Bayesian update using that year's partial-month
  ERA5 as the observation (fixed grand-ensemble prior), and compare the
  posterior tercile probabilities to the observed ERA5 seasonal outcome.

  Reports:
    - RPSS vs climatology (1/3, 1/3, 1/3) — overall skill
    - RPSS vs prior (C3S ensemble alone) — skill added by the update
    - Hit rate (highest-probability category matches observation)
    - Per-year breakdown saved to a CSV alongside the output figures
"""

import numpy as np
import pandas as pd
import xarray as xr
import config
from variable import Variable
from season   import Season
from bayesian_update import (
    GaussianDist, TercileProbs,
    fit_era5_regression, bayesian_update,
    compute_tercile_probs, _aggregate,
)


# ─────────────────────────────────────────────────────────────────────────────
# BASIC SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def run_basic_checks(prior: GaussianDist,
                     posterior: GaussianDist,
                     grand_ensemble: xr.DataArray,
                     era5_anomaly: xr.DataArray,
                     reg,
                     variable: Variable,
                     season: Season,
                     cutoff_day: int,
                     t_lower: float,
                     t_upper: float,
                     probs: TercileProbs,
                     prior_probs: TercileProbs,
                     hindcast_start: int,
                     hindcast_end: int) -> bool:
    """
    Run sanity checks on pipeline outputs. Prints a summary table.
    Returns True if no FAIL-level issues were found.
    """
    checks = []

    # 1. Tercile probabilities sum to 1
    for label, p in [("Prior probs sum to 1",     prior_probs),
                     ("Posterior probs sum to 1",  probs)]:
        total  = p.below_normal + p.near_normal + p.above_normal
        ok     = abs(total - 1.0) < 0.001
        checks.append((label, "FAIL" if not ok else "OK", f"sum={total:.5f}"))

    # 2. Posterior std < prior std
    ok = posterior.std < prior.std
    checks.append(("Posterior σ < prior σ",
                   "OK" if ok else "WARN",
                   f"σ_post={posterior.std:.3f}  σ_prior={prior.std:.3f}"))

    # 3. Ensemble size
    n  = grand_ensemble.sizes["member"]
    ok = n >= 20
    checks.append(("Grand ensemble ≥ 20 members",
                   "OK" if ok else "WARN",
                   f"n={n}"))

    # 4. Regression R²
    ok = reg.r2 >= 0.05
    checks.append(("Regression R² ≥ 0.05",
                   "OK" if ok else "WARN",
                   f"R²={reg.r2:.3f}"))

    # 5. ERA5 init-month coverage
    missing = []
    for year in range(hindcast_start, hindcast_end + 1):
        partial = era5_anomaly.sel(
            time=(
                (era5_anomaly.time.dt.year  == year) &
                (era5_anomaly.time.dt.month == season.init_month) &
                (era5_anomaly.time.dt.day   <= cutoff_day)
            )
        )
        if len(partial) < max(1, cutoff_day // 2):
            missing.append(year)
    ok  = len(missing) == 0
    msg = "all years present" if ok else f"missing: {missing}"
    checks.append(("ERA5 init-month coverage", "OK" if ok else "WARN", msg))

    print("\n── Basic checks ──────────────────────────────────────────────────────")
    any_fail = False
    for name, status, detail in checks:
        icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}[status]
        print(f"  {icon}  {name:<38}  {status:<5}  {detail}")
        if status == "FAIL":
            any_fail = True
    print()
    return not any_fail


# ─────────────────────────────────────────────────────────────────────────────
# LOO CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _rps(p_bn: float, p_nn: float, p_an: float, obs_cat: int) -> float:
    """
    Ranked Probability Score for a 3-category forecast.

    obs_cat: 0=BN, 1=NN, 2=AN
    RPS = Σ_k (CDF_forecast(k) - CDF_obs(k))²  for k = 0,1
    """
    cdf_f = [p_bn, p_bn + p_nn]
    cdf_o = [1.0 if obs_cat == 0 else 0.0,
             0.0 if obs_cat == 2 else 1.0]
    return sum((f - o) ** 2 for f, o in zip(cdf_f, cdf_o))


def run_loo_validation(era5_anomaly: xr.DataArray,
                        prior: GaussianDist,
                        variable: Variable,
                        season: Season,
                        cutoff_day: int,
                        hindcast_start: int,
                        hindcast_end: int,
                        t_lower: float,
                        t_upper: float,
                        save_path: str | None = None) -> pd.DataFrame:
    """
    Leave-one-out cross-validation of the Bayesian update.

    For each hindcast year y (hindcast_start … hindcast_end):
      1. Refit the ERA5 regression excluding year y.
      2. Apply the Bayesian update using year y's partial-month ERA5 (up to
         cutoff_day) as the observation and the fixed grand-ensemble prior.
      3. Record the posterior tercile probabilities and the observed outcome.

    Prints RPSS vs climatology, RPSS vs prior, and hit rate.
    Optionally saves a per-year CSV to save_path.

    Returns a DataFrame with one row per validated year.
    """
    print(f"\n── LOO cross-validation  "
          f"({hindcast_start}–{hindcast_end}, cutoff day {cutoff_day}) ──────")

    prior_probs = compute_tercile_probs(prior, t_lower, t_upper)
    rows = []

    for year in range(hindcast_start, hindcast_end + 1):

        # ── Partial-month predictor for year y ───────────────────────────────
        partial = era5_anomaly.sel(
            time=(
                (era5_anomaly.time.dt.year  == year) &
                (era5_anomaly.time.dt.month == season.init_month) &
                (era5_anomaly.time.dt.day   <= cutoff_day)
            )
        )
        if len(partial) < max(1, cutoff_day // 2):
            continue
        x_obs = float(variable.apply_transform(_aggregate(partial, variable)))

        # ── LOO regression (exclude year y) ──────────────────────────────────
        try:
            reg_loo = fit_era5_regression(
                era5_anomaly, variable, season, cutoff_day,
                hindcast_start, hindcast_end,
                exclude_years=[year],
                verbose=False,
            )
        except RuntimeError:
            continue

        # ── Posterior and tercile probs ───────────────────────────────────────
        post  = bayesian_update(prior, reg_loo, x_obs)
        probs = compute_tercile_probs(post, t_lower, t_upper)

        # ── Observed seasonal outcome for year y ─────────────────────────────
        seasonal_vals = []
        for cal_month in season.target_months:
            cal_year = season.target_year(year, cal_month)
            m = era5_anomaly.sel(
                time=(
                    (era5_anomaly.time.dt.year  == cal_year) &
                    (era5_anomaly.time.dt.month == cal_month)
                )
            )
            if len(m) > 0:
                seasonal_vals.extend(m.values.tolist())
        if len(seasonal_vals) < 28:
            continue

        if variable.obs_type == "mean":
            obs_agg = float(np.mean(seasonal_vals))
        else:
            obs_agg = float(np.sum(seasonal_vals))
        obs_t = float(variable.apply_transform(obs_agg))

        if obs_t < t_lower:
            obs_cat = 0   # BN
        elif obs_t <= t_upper:
            obs_cat = 1   # NN
        else:
            obs_cat = 2   # AN

        rows.append({
            "year":     year,
            "x_obs":    round(x_obs, 4),
            "obs_cat":  obs_cat,
            "obs_label": ["BN", "NN", "AN"][obs_cat],
            "p_bn":     round(probs.below_normal,  3),
            "p_nn":     round(probs.near_normal,   3),
            "p_an":     round(probs.above_normal,  3),
            "mu_post":  round(post.mean, 4),
        })

    if not rows:
        print("  [VALIDATE] No LOO samples available — check ERA5 coverage.\n")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    n  = len(df)

    # ── Skill scores ─────────────────────────────────────────────────────────
    rps_post  = df.apply(lambda r: _rps(r.p_bn, r.p_nn, r.p_an, r.obs_cat), axis=1)
    rps_clim  = df.apply(lambda r: _rps(1/3, 1/3, 1/3, r.obs_cat), axis=1)
    rps_prior = df.apply(
        lambda r: _rps(prior_probs.below_normal, prior_probs.near_normal,
                        prior_probs.above_normal, r.obs_cat), axis=1
    )

    rpss_clim  = float(1.0 - rps_post.mean() / rps_clim.mean())
    rpss_prior = float(1.0 - rps_post.mean() / rps_prior.mean()) \
                 if rps_prior.mean() > 1e-9 else np.nan

    df["pred_cat"] = df[["p_bn", "p_nn", "p_an"]].values.argmax(axis=1)
    hit_rate = float((df["pred_cat"] == df["obs_cat"]).mean())

    obs_counts = df["obs_label"].value_counts().to_dict()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"  Years validated  : {n}")
    print(f"  Observed freq    : "
          f"BN={obs_counts.get('BN', 0)}  "
          f"NN={obs_counts.get('NN', 0)}  "
          f"AN={obs_counts.get('AN', 0)}")
    print(f"  Mean RPS         : "
          f"posterior={rps_post.mean():.4f}  "
          f"prior={rps_prior.mean():.4f}  "
          f"climatology={rps_clim.mean():.4f}")
    print(f"  RPSS vs clim     : {rpss_clim:+.3f}  "
          f"({'SKILFUL' if rpss_clim > 0 else 'NO SKILL'})")
    print(f"  RPSS vs prior    : {rpss_prior:+.3f}  "
          f"({'UPDATE ADDS SKILL' if rpss_prior > 0 else 'UPDATE REDUCES SKILL'})")
    print(f"  Hit rate         : {hit_rate:.1%}  (random baseline = {1/3:.1%})")

    if save_path:
        df.drop(columns=["pred_cat"]).to_csv(save_path, index=False)
        print(f"  Per-year results → {save_path}")
    print()

    return df
