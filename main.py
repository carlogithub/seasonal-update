"""
main.py — Orchestrate the seasonal forecast Bayesian update pipeline.

Usage
-----
    python main.py --init-year 2026 --cutoff-date 2026-04-21

Arguments
---------
  --init-year    : year of the forecast initialisation (default: current year)
  --cutoff-date  : date up to which ERA5 observations are used, format YYYY-MM-DD
                   (default: today)
  --skip-download: skip CDS downloads if data files already exist (useful for
                   re-running the analysis without re-downloading large files)

Pipeline
--------
  1. Download ERA5 SST + C3S hindcast climatologies + C3S forecast ensembles
  2. Load ERA5, compute Niño 3.4 anomaly time series
  3. Load per-model C3S data, debias, pool into grand ensemble
  4. Fit ERA5-based regression for the observed cutoff day
  5. Compute Gaussian prior from the grand ensemble
  6. Run Bayesian update → posterior distribution → tercile probabilities
  7. Compute probability evolution (day 1 to cutoff day)
  8. Save two figures: updated plume + tercile probability evolution
"""

import argparse
import os
from datetime import date, datetime

import config
import download
import process
import bayesian_update
import visualize


def parse_args():
    today = date.today()
    parser = argparse.ArgumentParser(
        description="Bayesian update of C3S seasonal Niño 3.4 forecast with ERA5"
    )
    parser.add_argument(
        "--init-year",
        type=int,
        default=today.year,
        help="Forecast initialisation year (default: current year)",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=today.strftime("%Y-%m-%d"),
        help="Observation cutoff date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip CDS downloads (use existing files in data/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    init_year   = args.init_year
    cutoff_date = datetime.strptime(args.cutoff_date, "%Y-%m-%d").date()
    cutoff_day  = cutoff_date.day   # day-of-month, e.g. 21

    print("=" * 60)
    print(f"  Seasonal forecast Bayesian update")
    print(f"  Initialisation : {init_year}-{config.INIT_MONTH:02d}-01")
    print(f"  Target season  : {config.SEASON_LABEL}")
    print(f"  Cutoff date    : {cutoff_date}")
    print(f"  Cutoff day     : day {cutoff_day} of {config.INIT_MONTH:02d}")
    print("=" * 60)

    # ── STEP 1: Downloads ────────────────────────────────────────────────────
    if not args.skip_download:
        print("\n[STEP 1] Downloading data from CDS …")
        file_paths = download.download_all(init_year)
    else:
        print("\n[STEP 1] Skipping downloads (--skip-download flag set).")
        # Build the expected file path dict without downloading
        file_paths = {
            "era5": os.path.join(config.ERA5_DIR, "era5_sst_nino34.nc"),
            "hindcast_clim": {
                name: os.path.join(
                    config.HINDCAST_DIR,
                    f"hindcast_clim_{name}_m{config.INIT_MONTH:02d}.grib",
                )
                for name in config.MODELS
            },
            "forecast": {
                name: os.path.join(
                    config.FORECAST_DIR,
                    f"forecast_{name}_{init_year}_m{config.INIT_MONTH:02d}.grib",
                )
                for name in config.MODELS
            },
        }

    # ── STEP 2: ERA5 Niño 3.4 anomaly ────────────────────────────────────────
    print("\n[STEP 2] Computing ERA5 Niño 3.4 anomaly time series …")

    era5_raw     = process.load_era5_nino34(file_paths["era5"])
    era5_anomaly = process.compute_era5_anomaly(
        era5_raw,
        clim_start=config.HINDCAST_START_YEAR,
        clim_end=config.HINDCAST_END_YEAR,
    )

    # ── STEP 3: C3S ensemble ─────────────────────────────────────────────────
    print("\n[STEP 3] Loading C3S forecast ensemble and debiasing per model …")

    model_ensembles = {}
    for model_name in config.MODELS:
        hc_path = file_paths["hindcast_clim"].get(model_name)
        fc_path = file_paths["forecast"].get(model_name)

        if not hc_path or not os.path.exists(hc_path):
            print(f"  [SKIP] {model_name}: hindcast climatology file not found.")
            continue
        if not fc_path or not os.path.exists(fc_path):
            print(f"  [SKIP] {model_name}: forecast file not found.")
            continue

        try:
            hindcast_clim = process.load_hindcast_climatology(hc_path)
            ensemble_da   = process.load_forecast_ensemble(fc_path, hindcast_clim, init_year)
            model_ensembles[model_name] = ensemble_da
        except Exception as exc:
            print(f"  [SKIP] {model_name}: error during processing — {exc}")

    if not model_ensembles:
        raise RuntimeError(
            "No model ensembles could be loaded. "
            "Check that the data downloads succeeded."
        )

    grand_ensemble = process.build_grand_ensemble(model_ensembles)

    # ── STEP 4–6: Bayesian update at the cutoff date ─────────────────────────
    print(f"\n[STEP 4–6] Bayesian update for cutoff day {cutoff_day} …")

    # Regression: how informative is partial April obs about full AMJ mean?
    reg = bayesian_update.fit_era5_regression(
        era5_anomaly,
        cutoff_day=cutoff_day,
        init_month=config.INIT_MONTH,
        target_months=config.TARGET_MONTHS,
        hindcast_start=config.HINDCAST_START_YEAR,
        hindcast_end=config.HINDCAST_END_YEAR,
    )

    # Prior distribution from C3S ensemble
    prior = bayesian_update.compute_prior(grand_ensemble, config.TARGET_MONTHS)

    # Observed partial-month running mean from ERA5
    obs_partial = era5_anomaly.sel(
        time=(
            (era5_anomaly.time.dt.year  == init_year) &
            (era5_anomaly.time.dt.month == config.INIT_MONTH) &
            (era5_anomaly.time.dt.day   <= cutoff_day)
        )
    )
    if len(obs_partial) == 0:
        raise RuntimeError(
            f"No ERA5 data found for {init_year}-{config.INIT_MONTH:02d} "
            f"up to day {cutoff_day}. Check that ERA5 covers this period."
        )
    x_obs = float(obs_partial.mean())
    print(f"  Observed partial-month Niño 3.4 anomaly: {x_obs:.3f} K")

    # Bayesian posterior
    posterior = bayesian_update.bayesian_update(prior, reg, x_obs)

    # Tercile thresholds and final probabilities
    t_lower, t_upper = bayesian_update.compute_tercile_thresholds(
        era5_anomaly,
        config.TARGET_MONTHS,
        config.HINDCAST_START_YEAR,
        config.HINDCAST_END_YEAR,
    )
    probs = bayesian_update.compute_tercile_probs(posterior, t_lower, t_upper)

    print(f"\n  ── Final tercile probabilities for {config.SEASON_LABEL} {init_year} ──")
    print(f"     Prior  (C3S only)  : BN={bayesian_update.compute_tercile_probs(prior, t_lower, t_upper).below_normal*100:.1f}%  "
          f"NN={bayesian_update.compute_tercile_probs(prior, t_lower, t_upper).near_normal*100:.1f}%  "
          f"AN={bayesian_update.compute_tercile_probs(prior, t_lower, t_upper).above_normal*100:.1f}%")
    print(f"     Posterior (updated): BN={probs.below_normal*100:.1f}%  "
          f"NN={probs.near_normal*100:.1f}%  "
          f"AN={probs.above_normal*100:.1f}%")

    # ── STEP 7: Probability evolution ────────────────────────────────────────
    print(f"\n[STEP 7] Computing probability evolution for days 1–{cutoff_day} …")

    evolution_df = bayesian_update.compute_probability_evolution(
        era5_anomaly,
        grand_ensemble,
        init_year=init_year,
        max_cutoff_day=cutoff_day,
    )

    # ── STEP 8: Save figures ─────────────────────────────────────────────────
    print("\n[STEP 8] Saving figures …")

    plume_path = os.path.join(
        config.OUTPUT_DIR,
        f"plume_{config.SEASON_LABEL}_{init_year}_cutoff{cutoff_day:02d}.png",
    )
    evol_path = os.path.join(
        config.OUTPUT_DIR,
        f"tercile_evolution_{config.SEASON_LABEL}_{init_year}_cutoff{cutoff_day:02d}.png",
    )

    visualize.plot_updated_plume(
        ensemble=grand_ensemble,
        era5_obs=era5_anomaly,
        prior=prior,
        posterior=posterior,
        cutoff_date=cutoff_date,
        init_year=init_year,
        save_path=plume_path,
    )

    visualize.plot_tercile_evolution(
        evolution_df=evolution_df,
        init_year=init_year,
        save_path=evol_path,
    )

    print("\nDone.")
    print(f"  Plume figure    → {plume_path}")
    print(f"  Evolution figure → {evol_path}")


if __name__ == "__main__":
    main()
