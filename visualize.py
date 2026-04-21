"""
visualize.py — Plotting functions for the seasonal forecast Bayesian update.

Two figures are produced:

  Figure 1 — Updated Niño 3.4 plume
    Shows the C3S grand ensemble as a shaded percentile envelope (prior),
    the ERA5 observed trajectory up to the cutoff date, and the Bayesian-
    updated ensemble as a shifted/rescaled envelope (posterior).
    A vertical dashed line marks the cutoff date.

  Figure 2 — Tercile probability evolution
    Shows how the probabilities for below-normal (BN), near-normal (NN), and
    above-normal (AN) change day by day as more observations accumulate.
    A horizontal dashed line at 33% marks climatological probability.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
import config
from bayesian_update import GaussianDist, TercileProbs


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (consistent across both figures)
# ─────────────────────────────────────────────────────────────────────────────
COLOUR_PRIOR     = "#4878CF"   # blue  — C3S prior ensemble
COLOUR_POSTERIOR = "#D65F00"   # burnt orange — Bayesian-updated posterior
COLOUR_OBS       = "#222222"   # near-black — ERA5 observation
COLOUR_BELOW     = "#2166AC"   # blue  — below-normal tercile
COLOUR_NORMAL    = "#4DAC26"   # green — near-normal tercile
COLOUR_ABOVE     = "#D73027"   # red   — above-normal tercile


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — UPDATED PLUME
# ─────────────────────────────────────────────────────────────────────────────

def plot_updated_plume(ensemble: xr.DataArray,
                        era5_obs: xr.DataArray,
                        prior: GaussianDist,
                        posterior: GaussianDist,
                        cutoff_date: date,
                        init_year: int,
                        save_path: str | None = None) -> None:
    """
    Plot the C3S ensemble plume before and after the Bayesian update, together
    with the ERA5 observed trajectory.

    The posterior plume is derived from the prior ensemble by linearly shifting
    all members so that the ensemble mean matches the posterior mean, and
    rescaling the spread to match the posterior standard deviation. This
    preserves the ensemble's internal structure (temporal correlations, non-
    Gaussian tails) while incorporating the observational update.

    Parameters
    ----------
    ensemble     : grand C3S ensemble (member × time) DataArray, daily anomalies
    era5_obs     : ERA5 Niño 3.4 anomaly time series (full resolution)
    prior        : Gaussian prior from bayesian_update.compute_prior()
    posterior    : Gaussian posterior from bayesian_update.bayesian_update()
    cutoff_date  : the date up to which ERA5 observations are used
    init_year    : forecast initialisation year
    save_path    : if given, save the figure to this path; otherwise display it
    """
    # ── Build prior ensemble percentiles ─────────────────────────────────────
    ens_values = ensemble.values     # shape: (member, time)
    times = pd.to_datetime(ensemble.time.values)

    pct_prior = {
        p: np.nanpercentile(ens_values, p, axis=0)
        for p in [10, 25, 50, 75, 90]
    }

    # ── Shift and rescale ensemble to get posterior plume ────────────────────
    # For each member m at each time t:
    #   posterior_member = posterior_mean + (prior_member - prior_mean) * (posterior_std / prior_std)
    # This moves the ensemble centre to the posterior mean and compresses/expands
    # the spread to match the posterior standard deviation.
    scale = posterior.std / prior.std if prior.std > 0 else 1.0
    ens_posterior = posterior.mean + (ens_values - prior.mean) * scale

    pct_post = {
        p: np.nanpercentile(ens_posterior, p, axis=0)
        for p in [10, 25, 50, 75, 90]
    }

    # ── ERA5 observed up to cutoff ────────────────────────────────────────────
    cutoff_ts = pd.Timestamp(cutoff_date)
    obs_mask  = (
        pd.to_datetime(era5_obs.time.values) <= cutoff_ts
    )
    # Only show the target-season months in the observation window
    era5_season_mask = np.isin(era5_obs.time.dt.month.values, config.TARGET_MONTHS)
    obs_time   = pd.to_datetime(era5_obs.time.values[obs_mask & era5_season_mask])
    obs_values = era5_obs.values[obs_mask & era5_season_mask]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    # Shaded prior percentile envelope (10–90 and 25–75)
    ax.fill_between(times, pct_prior[10], pct_prior[90],
                    color=COLOUR_PRIOR, alpha=0.15, label="Prior 10–90th pct")
    ax.fill_between(times, pct_prior[25], pct_prior[75],
                    color=COLOUR_PRIOR, alpha=0.30, label="Prior 25–75th pct")
    ax.plot(times, pct_prior[50],
            color=COLOUR_PRIOR, linewidth=1.5, linestyle="--", label="Prior median")

    # Shaded posterior envelope
    ax.fill_between(times, pct_post[10], pct_post[90],
                    color=COLOUR_POSTERIOR, alpha=0.15, label="Posterior 10–90th pct")
    ax.fill_between(times, pct_post[25], pct_post[75],
                    color=COLOUR_POSTERIOR, alpha=0.30, label="Posterior 25–75th pct")
    ax.plot(times, pct_post[50],
            color=COLOUR_POSTERIOR, linewidth=1.5, linestyle="--", label="Posterior median")

    # ERA5 observed line
    if len(obs_time) > 0:
        ax.plot(obs_time, obs_values,
                color=COLOUR_OBS, linewidth=2.0, label="ERA5 observed")

    # Vertical line at cutoff date
    ax.axvline(cutoff_ts, color="grey", linewidth=1.0, linestyle=":", alpha=0.8)
    ax.text(cutoff_ts, ax.get_ylim()[1] * 0.97,
            f" {cutoff_date.strftime('%d %b')}",
            va="top", ha="left", fontsize=9, color="grey")

    # Zero anomaly line
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # weekly ticks
    plt.xticks(rotation=30, ha="right")

    ax.set_ylabel("Niño 3.4 anomaly (K)", fontsize=11)
    ax.set_title(
        f"C3S multi-model Niño 3.4 forecast — {config.SEASON_LABEL} {init_year}\n"
        f"Bayesian update using ERA5 observed up to {cutoff_date.strftime('%d %b %Y')}",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlim(times[0], times[-1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Plume figure saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — TERCILE PROBABILITY EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_tercile_evolution(evolution_df: pd.DataFrame,
                            init_year: int,
                            save_path: str | None = None) -> None:
    """
    Plot the evolution of tercile probabilities as a function of observation
    cutoff day.

    Each coloured line shows how the probability of one of the three seasonal
    categories (BN / NN / AN) changes as more days of the initialisation month
    are observed. The horizontal dashed line at 33% is the climatological
    baseline — if the updated probability is above it, the forecast provides
    skill beyond climatology.

    Parameters
    ----------
    evolution_df : DataFrame from bayesian_update.compute_probability_evolution()
    init_year    : the forecast year (for the plot title)
    save_path    : if given, save the figure to this path; otherwise display it
    """
    if evolution_df.empty:
        print("[PLOT] No evolution data to plot.")
        return

    days       = evolution_df["cutoff_day"].values
    p_below    = evolution_df["prob_below"].values * 100    # convert to percent
    p_normal   = evolution_df["prob_normal"].values * 100
    p_above    = evolution_df["prob_above"].values * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(days, p_below,  color=COLOUR_BELOW,  linewidth=2.0, marker="o",
            markersize=4, label="Below normal (BN)")
    ax.plot(days, p_normal, color=COLOUR_NORMAL, linewidth=2.0, marker="s",
            markersize=4, label="Near normal (NN)")
    ax.plot(days, p_above,  color=COLOUR_ABOVE,  linewidth=2.0, marker="^",
            markersize=4, label="Above normal (AN)")

    # Climatological baseline (33%)
    ax.axhline(100 / 3, color="grey", linewidth=1.0, linestyle="--", alpha=0.7,
               label="Climatology (33%)")

    ax.set_xlabel(f"Days observed into {_month_name(config.INIT_MONTH)} {init_year}",
                  fontsize=11)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(days[0] - 0.5, days[-1] + 0.5)
    ax.set_title(
        f"Evolution of {config.SEASON_LABEL} Niño 3.4 tercile probabilities — {init_year}\n"
        f"C3S multi-model forecast updated with ERA5 observations",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=10)

    # Show final values as annotations on the right edge
    for p_series, colour, label in [
        (p_below,  COLOUR_BELOW,  "BN"),
        (p_normal, COLOUR_NORMAL, "NN"),
        (p_above,  COLOUR_ABOVE,  "AN"),
    ]:
        ax.annotate(
            f"{p_series[-1]:.0f}%",
            xy=(days[-1], p_series[-1]),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=colour,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Evolution figure saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _month_name(month: int) -> str:
    """Return the full month name for a calendar month number (1–12)."""
    import calendar
    return calendar.month_name[month]
