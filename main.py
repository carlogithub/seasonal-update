"""
main.py — Orchestrate the seasonal forecast Bayesian update pipeline.

Usage
-----
    python main.py --location nino34 --variable nino34 --season amj \\
                   --init-year 2026 --cutoff-date 2026-04-21

    python main.py --location terrassa --variable t2m --season jja \\
                   --init-year 2026 --cutoff-date 2026-04-21

    python main.py --location terrassa --variable tp --season djf \\
                   --init-year 2026 --cutoff-date 2026-10-15 \\
                   --skip-download

Arguments
---------
  --location     : location key from location.LOCATIONS  (e.g. nino34, terrassa)
  --variable     : variable key from variable.VARIABLES  (e.g. nino34, t2m, tp)
  --season       : season key  from season.SEASONS       (e.g. amj, jja, djf)
  --init-year    : forecast initialisation year (default: current year)
  --cutoff-date  : ERA5 observation cutoff, YYYY-MM-DD   (default: today)
  --skip-download: skip CDS downloads if files already exist

Pipeline
--------
  1. Download ERA5 + C3S monthly forecasts
  2. Load ERA5, compute anomaly time series
  3. Load per-model C3S data, debias, pool into grand ensemble
  4. Fit ERA5-based regression for the observation cutoff day
  5. Compute Gaussian prior from the grand ensemble
  6. Run Bayesian update → posterior → tercile probabilities
  7. Compute probability evolution (day 1 to cutoff day)
  8. Save figures
"""

import argparse
import os
from datetime import date, datetime

import config
import download
import process
import bayesian_update
import validate
import visualize

from location import LOCATIONS, Location
from variable import VARIABLES
from season   import SEASONS


def parse_args():
    today = date.today()
    parser = argparse.ArgumentParser(
        description="Bayesian update of C3S seasonal forecast with ERA5 observations"
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        choices=list(LOCATIONS.keys()),
        help=f"Preset location key: {list(LOCATIONS.keys())}",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Custom location latitude °N (use with --lon instead of --location)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Custom location longitude °E (use with --lat instead of --location)",
    )
    parser.add_argument(
        "--location-name",
        type=str,
        default=None,
        help="Name for a custom lat/lon location (optional, used in titles and filenames)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="nino34",
        choices=list(VARIABLES.keys()),
        help=f"Variable key: {list(VARIABLES.keys())}",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="amj",
        choices=list(SEASONS.keys()),
        help=f"Season key: {list(SEASONS.keys())}",
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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run leave-one-out cross-validation after the main pipeline",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.lat is not None and args.lon is not None:
        loc_name = args.location_name or (
            f"{abs(args.lat):.2f}{'N' if args.lat >= 0 else 'S'}_"
            f"{abs(args.lon):.2f}{'E' if args.lon >= 0 else 'W'}"
        )
        location = Location(loc_name, args.lat, args.lon)
    elif args.location is not None:
        location = LOCATIONS[args.location]
    else:
        raise SystemExit("error: provide --location OR both --lat and --lon")

    variable    = VARIABLES[args.variable]
    season      = SEASONS[args.season]
    init_year   = args.init_year
    cutoff_date = datetime.strptime(args.cutoff_date, "%Y-%m-%d").date()
    cutoff_day  = cutoff_date.day

    print("=" * 65)
    print(f"  Seasonal forecast Bayesian update")
    print(f"  Location       : {location.name}")
    print(f"  Variable       : {variable.name}")
    print(f"  Season         : {season.label}  (init month: {season.init_month:02d})")
    print(f"  Initialisation : {init_year}-{season.init_month:02d}-01")
    print(f"  Cutoff date    : {cutoff_date}  (day {cutoff_day})")
    print("=" * 65)

    # ── STEP 1: Downloads ────────────────────────────────────────────────────
    if not args.skip_download:
        print("\n[STEP 1] Downloading data from CDS …")
        file_paths = download.download_all(location, variable, season, init_year)
    else:
        print("\n[STEP 1] Skipping downloads (--skip-download flag set).")
        forecast_paths   = {}
        hindcast_paths   = {}
        for name in config.MODELS:
            postproc = os.path.join(
                config.FORECAST_DIR,
                f"forecast_postproc_{name}_{location.slug}_{variable.slug}"
                f"_{init_year}_m{season.init_month:02d}.grib",
            )
            monthly = os.path.join(
                config.FORECAST_DIR,
                f"forecast_monthly_{name}_{location.slug}_{variable.slug}"
                f"_{init_year}_m{season.init_month:02d}.grib",
            )
            if os.path.exists(postproc):
                forecast_paths[name]  = postproc
                hindcast_paths[name]  = None
            else:
                forecast_paths[name]  = monthly
                hindcast_paths[name]  = os.path.join(
                    config.FORECAST_DIR,
                    f"hindcast_monthly_{name}_{location.slug}_{variable.slug}"
                    f"_m{season.init_month:02d}.grib",
                )
        file_paths = {
            "era5": os.path.join(
                config.ERA5_DIR,
                f"era5_{variable.slug}_{location.slug}.nc",
            ),
            "hindcast_clim": hindcast_paths,
            "forecast":      forecast_paths,
        }

    # ── STEP 2: ERA5 anomaly ─────────────────────────────────────────────────
    print("\n[STEP 2] Computing ERA5 anomaly time series …")

    era5_raw     = process.load_era5(file_paths["era5"], variable)
    era5_anomaly = process.compute_era5_anomaly(
        era5_raw, variable, season,
        clim_start=config.HINDCAST_START_YEAR,
        clim_end=config.HINDCAST_END_YEAR,
    )

    # ── STEP 3: C3S ensemble ─────────────────────────────────────────────────
    print("\n[STEP 3] Loading C3S monthly forecast ensemble …")

    model_ensembles = {}
    for model_name in config.MODELS:
        fc_path = file_paths["forecast"].get(model_name)
        if not fc_path or not os.path.exists(fc_path):
            print(f"  [SKIP] {model_name}: forecast file not found.")
            continue
        try:
            hc_path = file_paths["hindcast_clim"].get(model_name)
            ens_da = process.load_model_forecast(
                fc_path, hc_path, era5_raw, variable, season,
            )
            model_ensembles[model_name] = ens_da
        except Exception as exc:
            print(f"  [SKIP] {model_name}: {exc}")

    if not model_ensembles:
        raise RuntimeError(
            "No model ensembles could be loaded. "
            "Check that the data downloads succeeded."
        )

    grand_ensemble = process.build_grand_ensemble(model_ensembles)

    # ── STEPS 4–6: Bayesian update at the cutoff date ────────────────────────
    print(f"\n[STEP 4–6] Bayesian update for cutoff day {cutoff_day} …")

    reg = bayesian_update.fit_era5_regression(
        era5_anomaly, variable, season,
        cutoff_day=cutoff_day,
        hindcast_start=config.HINDCAST_START_YEAR,
        hindcast_end=config.HINDCAST_END_YEAR,
    )

    prior = bayesian_update.compute_prior_monthly(grand_ensemble, variable)

    # Observed partial-month aggregate, transformed
    obs_partial = era5_anomaly.sel(
        time=(
            (era5_anomaly.time.dt.year  == init_year) &
            (era5_anomaly.time.dt.month == season.init_month) &
            (era5_anomaly.time.dt.day   <= cutoff_day)
        )
    )
    if len(obs_partial) == 0:
        raise RuntimeError(
            f"No ERA5 data found for {init_year}-{season.init_month:02d} "
            f"up to day {cutoff_day}. Check ERA5 coverage."
        )
    x_raw = bayesian_update._aggregate(obs_partial, variable)
    x_obs = float(variable.apply_transform(x_raw))
    print(f"  Observed partial-month {variable.short_name}: {x_raw:.4f} → "
          f"transformed: {x_obs:.4f}")

    posterior = bayesian_update.bayesian_update(prior, reg, x_obs)

    t_lower, t_upper = bayesian_update.compute_tercile_thresholds(
        era5_anomaly, variable, season,
        config.HINDCAST_START_YEAR, config.HINDCAST_END_YEAR,
    )
    probs       = bayesian_update.compute_tercile_probs(posterior, t_lower, t_upper)
    prior_probs = bayesian_update.compute_tercile_probs(prior,    t_lower, t_upper)

    print(f"\n  ── Final tercile probabilities for {season.label} {init_year} "
          f"({location.name} {variable.short_name}) ──")
    print(f"     Prior  (C3S only)  : "
          f"BN={prior_probs.below_normal*100:.1f}%  "
          f"NN={prior_probs.near_normal*100:.1f}%  "
          f"AN={prior_probs.above_normal*100:.1f}%")
    print(f"     Posterior (updated): "
          f"BN={probs.below_normal*100:.1f}%  "
          f"NN={probs.near_normal*100:.1f}%  "
          f"AN={probs.above_normal*100:.1f}%")

    # ── STEP 7: Probability evolution ────────────────────────────────────────
    print(f"\n[STEP 7] Computing probability evolution for days 1–{cutoff_day} …")

    evolution_df = bayesian_update.compute_probability_evolution(
        era5_anomaly, grand_ensemble, variable, season,
        init_year=init_year,
        max_cutoff_day=cutoff_day,
    )

    # ── STEP 8: Save figures ─────────────────────────────────────────────────
    print("\n[STEP 8] Saving figures …")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tag = f"{season.label}_{location.slug}_{variable.slug}_{init_year}_cutoff{cutoff_day:02d}"

    plume_path = os.path.join(config.OUTPUT_DIR, f"plume_{tag}.png")
    evol_path  = os.path.join(config.OUTPUT_DIR, f"tercile_evolution_{tag}.png")

    visualize.plot_updated_plume_monthly(
        ensemble=grand_ensemble,
        era5_obs=era5_anomaly,
        prior=prior,
        posterior=posterior,
        variable=variable,
        season=season,
        location=location,
        cutoff_date=cutoff_date,
        init_year=init_year,
        save_path=plume_path,
    )

    visualize.plot_tercile_evolution(
        evolution_df=evolution_df,
        variable=variable,
        season=season,
        location=location,
        init_year=init_year,
        save_path=evol_path,
    )

    if variable.short_name == "nino34":
        enso_path = os.path.join(config.OUTPUT_DIR, f"enso_prob_{tag}.png")
        visualize.plot_enso_nino_probability(
            ensemble=grand_ensemble,
            prior=prior,
            posterior=posterior,
            variable=variable,
            season=season,
            location=location,
            cutoff_date=cutoff_date,
            init_year=init_year,
            thresholds=[0.5, 1.0, 1.5],
            save_path=enso_path,
        )
        print(f"  El Niño prob figure → {enso_path}")

    # ── STEP 9: Basic sanity checks (always) ────────────────────────────────
    print("\n[STEP 9] Running sanity checks …")
    validate.run_basic_checks(
        prior, posterior, grand_ensemble, era5_anomaly, reg,
        variable, season, cutoff_day,
        t_lower, t_upper, probs, prior_probs,
        config.HINDCAST_START_YEAR, config.HINDCAST_END_YEAR,
    )

    # ── STEP 10: LOO cross-validation (--validate only) ─────────────────────
    if args.validate:
        print("[STEP 10] Leave-one-out cross-validation …")
        loo_path = os.path.join(config.OUTPUT_DIR, f"loo_{tag}.csv")
        validate.run_loo_validation(
            era5_anomaly, prior, variable, season,
            cutoff_day=cutoff_day,
            hindcast_start=config.HINDCAST_START_YEAR,
            hindcast_end=config.HINDCAST_END_YEAR,
            t_lower=t_lower,
            t_upper=t_upper,
            save_path=loo_path,
        )

    print("Done.")
    print(f"  Plume figure     → {plume_path}")
    print(f"  Evolution figure → {evol_path}")


if __name__ == "__main__":
    main()
