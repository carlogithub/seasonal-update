"""
visualize.py — Plotting for the seasonal forecast Bayesian update.

Two figures are produced:

  Figure 1 — Updated plume (monthly box-whisker)
    Prior (C3S ensemble) vs posterior (Bayesian-updated) for each target month,
    with ERA5 observed monthly mean overlaid.

  Figure 2 — Tercile probability evolution
    How BN / NN / AN probabilities change as more days of the init month
    are observed.

All functions accept Variable, Season, and Location so axis labels and
titles adapt to the data being plotted.
"""

import os
import calendar
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date
import config
from variable import Variable
from season   import Season
from location import Location
from bayesian_update import GaussianDist, TercileProbs


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
COLOUR_PRIOR     = "#4878CF"   # blue  — C3S prior ensemble
COLOUR_POSTERIOR = "#D65F00"   # burnt orange — Bayesian-updated posterior
COLOUR_OBS       = "#222222"   # near-black — ERA5 observation
COLOUR_BELOW     = "#2166AC"   # blue  — below-normal
COLOUR_NORMAL    = "#4DAC26"   # green — near-normal
COLOUR_ABOVE     = "#D73027"   # red   — above-normal


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — MONTHLY PLUME (box-whisker per target month)
# ─────────────────────────────────────────────────────────────────────────────

def plot_updated_plume_monthly(ensemble: xr.DataArray,
                                era5_obs: xr.DataArray,
                                prior: GaussianDist,
                                posterior: GaussianDist,
                                variable: Variable,
                                season: Season,
                                location: Location,
                                cutoff_date: date,
                                init_year: int,
                                save_path: str | None = None) -> None:
    """
    Box-whisker plume showing prior and posterior for each target month.

    The posterior ensemble is derived from the prior by shifting the ensemble
    mean to the posterior mean and rescaling spread to the posterior std.
    This preserves the ensemble's internal structure while incorporating the
    observational update.

    ERA5 observed monthly mean/total is shown as a diamond marker where
    the target month falls on or before the cutoff date.

    Parameters
    ----------
    ensemble    : grand C3S ensemble (member × month)
    era5_obs    : ERA5 anomaly time series (daily)
    prior       : Gaussian prior
    posterior   : Gaussian posterior
    variable    : Variable object (for labels and aggregation)
    season      : Season object (for month labels and year mapping)
    location    : Location object (for title)
    cutoff_date : the observation cutoff
    init_year   : forecast initialisation year
    save_path   : save figure to this path if given
    """
    target_months = ensemble.month.values
    month_names   = [calendar.month_abbr[m] for m in target_months]
    x             = np.arange(len(target_months))
    width         = 0.35

    ens_values = ensemble.values   # (member, month)

    # Shift/rescale ensemble to produce the posterior plume
    scale    = posterior.std / prior.std if prior.std > 0 else 1.0
    ens_post = posterior.mean + (ens_values - prior.mean) * scale

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (cal_month, mname) in enumerate(zip(target_months, month_names)):
        prior_col = ens_values[:, i]
        post_col  = ens_post[:, i]

        # Prior box (left of centre)
        ax.boxplot(prior_col, positions=[x[i] - width/2], widths=width * 0.8,
                   patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor=COLOUR_PRIOR, alpha=0.5),
                   medianprops=dict(color=COLOUR_PRIOR, linewidth=2),
                   whiskerprops=dict(color=COLOUR_PRIOR),
                   capprops=dict(color=COLOUR_PRIOR),
                   flierprops=dict(marker=".", color=COLOUR_PRIOR, alpha=0.3))

        # Posterior box (right of centre)
        ax.boxplot(post_col, positions=[x[i] + width/2], widths=width * 0.8,
                   patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor=COLOUR_POSTERIOR, alpha=0.5),
                   medianprops=dict(color=COLOUR_POSTERIOR, linewidth=2),
                   whiskerprops=dict(color=COLOUR_POSTERIOR),
                   capprops=dict(color=COLOUR_POSTERIOR),
                   flierprops=dict(marker=".", color=COLOUR_POSTERIOR, alpha=0.3))

        # ERA5 observation for this month if it falls on or before the cutoff
        cal_year  = season.target_year(init_year, cal_month)
        month_ts  = pd.Timestamp(year=cal_year, month=cal_month, day=1)
        cutoff_ts = pd.Timestamp(cutoff_date)
        if month_ts <= cutoff_ts:
            obs_month = era5_obs.sel(
                time=(
                    (era5_obs.time.dt.year  == cal_year) &
                    (era5_obs.time.dt.month == cal_month)
                )
            )
            if len(obs_month) > 0:
                if variable.obs_type == "mean":
                    obs_val = float(obs_month.mean())
                else:
                    obs_val = float(obs_month.sum())
                obs_val = float(variable.apply_transform(obs_val))
                ax.plot(x[i], obs_val, marker="D", color=COLOUR_OBS,
                        markersize=10, zorder=5,
                        label="ERA5 obs" if i == 0 else "")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(month_names, fontsize=11)
    ax.set_ylabel(variable.units, fontsize=11)
    ax.set_title(
        f"C3S multi-model {variable.name} — {season.label} {init_year}\n"
        f"{location.name} | Prior (blue) vs posterior (orange) "
        f"updated to {cutoff_date.strftime('%d %b %Y')}",
        fontsize=11,
    )

    legend_elements = [
        mpatches.Patch(facecolor=COLOUR_PRIOR,     alpha=0.5, label="Prior (C3S ensemble)"),
        mpatches.Patch(facecolor=COLOUR_POSTERIOR, alpha=0.5, label="Posterior (updated)"),
        plt.Line2D([0], [0], marker="D", color=COLOUR_OBS, linestyle="None",
                   markersize=8, label="ERA5 observed"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Monthly plume figure saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — TERCILE PROBABILITY EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_tercile_evolution(evolution_df: pd.DataFrame,
                            variable: Variable,
                            season: Season,
                            location: Location,
                            init_year: int,
                            save_path: str | None = None) -> None:
    """
    Show how BN / NN / AN probabilities evolve as more days are observed.

    Parameters
    ----------
    evolution_df : DataFrame from bayesian_update.compute_probability_evolution()
    variable     : Variable object (for title)
    season       : Season object (for title and x-axis label)
    location     : Location object (for title)
    init_year    : forecast year
    save_path    : save figure to this path if given
    """
    if evolution_df.empty:
        print("[PLOT] No evolution data to plot.")
        return

    days     = evolution_df["cutoff_day"].values
    p_below  = evolution_df["prob_below"].values * 100
    p_normal = evolution_df["prob_normal"].values * 100
    p_above  = evolution_df["prob_above"].values * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(days, p_below,  color=COLOUR_BELOW,  linewidth=2, marker="o",
            markersize=4, label="Below normal (BN)")
    ax.plot(days, p_normal, color=COLOUR_NORMAL, linewidth=2, marker="s",
            markersize=4, label="Near normal (NN)")
    ax.plot(days, p_above,  color=COLOUR_ABOVE,  linewidth=2, marker="^",
            markersize=4, label="Above normal (AN)")

    ax.axhline(100 / 3, color="grey", linewidth=1, linestyle="--", alpha=0.7,
               label="Climatology (33%)")

    init_month_name = calendar.month_name[season.init_month]
    ax.set_xlabel(f"Days observed into {init_month_name} {init_year}", fontsize=11)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(days[0] - 0.5, days[-1] + 0.5)
    ax.set_title(
        f"Evolution of {season.label} {variable.name} tercile probabilities — {init_year}\n"
        f"{location.name} | C3S multi-model forecast updated with ERA5",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=10)

    # Annotate final values on the right edge
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
