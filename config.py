"""
config.py — Paths and C3S model registry.

This file contains only infrastructure settings (where to store files) and the
catalogue of C3S seasonal forecast models.  All scientific configuration
(which variable, which location, which season) is passed at runtime via the
CLI in main.py and held in the location.py / variable.py / season.py dataclasses.

To add a new model, add an entry to MODELS with the correct originating_centre
and system number from the CDS seasonal forecast catalogue.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_DIR, "data")
FORECAST_DIR = os.path.join(DATA_DIR, "forecasts")
ERA5_DIR     = os.path.join(DATA_DIR, "era5")
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "output")

# ─────────────────────────────────────────────────────────────────────────────
# HINDCAST / CLIMATOLOGY PERIOD
# ─────────────────────────────────────────────────────────────────────────────
# Standard C3S hindcast period used for debiasing and regression training.

HINDCAST_START_YEAR = 1993
HINDCAST_END_YEAR   = 2016

# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN UPDATE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

TERCILE_LOWER = 1.0 / 3.0   # 33.3 % — lower boundary of near-normal category
TERCILE_UPPER = 2.0 / 3.0   # 66.7 % — upper boundary of near-normal category

# ─────────────────────────────────────────────────────────────────────────────
# C3S MULTI-SYSTEM SEASONAL FORECAST MODELS
# ─────────────────────────────────────────────────────────────────────────────
# originating_centre : string used in the CDS API request
# system             : model version number (check CDS catalogue if this changes)
# n_members          : number of ensemble members for forecast runs
#
# NOTE: as of April 2026 only the ECMWF SEAS5.1 entry (system=51) reliably
# returns data for recent operational forecasts.  The other models return
# MarsNoDataError — their system numbers may need updating from the CDS catalogue.

MODELS = {
    "ecmwf": {
        "originating_centre": "ecmwf",
        "system": "51",         # SEAS5.1
        "n_members": 51,
    },
    "ukmo": {
        "originating_centre": "ukmo",
        "system": "600",        # GloSea6 — may need updating
        "n_members": 28,
    },
    "meteo_france": {
        "originating_centre": "meteo_france",
        "system": "8",          # System 8 — may need updating
        "n_members": 25,
    },
    "dwd": {
        "originating_centre": "dwd",
        "system": "21",         # GCFS2.1 — may need updating
        "n_members": 30,
    },
    "cmcc": {
        "originating_centre": "cmcc",
        "system": "35",         # SPS3.5 — may need updating
        "n_members": 40,
    },
}
