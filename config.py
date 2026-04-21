"""
config.py — Central configuration for the seasonal forecast Bayesian update system.

All parameters controlling data download, processing, and analysis live here.
Edit this file to change the target season, models, or region without touching
the scientific code.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Root directory of the project (wherever this config file lives)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sub-directories for downloaded data and figures
DATA_DIR      = os.path.join(PROJECT_DIR, "data")
HINDCAST_DIR  = os.path.join(DATA_DIR, "hindcasts")   # C3S hindcast climatology GRIBs
FORECAST_DIR  = os.path.join(DATA_DIR, "forecasts")   # C3S forecast ensemble GRIBs
ERA5_DIR      = os.path.join(DATA_DIR, "era5")        # ERA5 NetCDF files
OUTPUT_DIR    = os.path.join(PROJECT_DIR, "output")   # saved figures

# ─────────────────────────────────────────────────────────────────────────────
# NINO 3.4 REGION
# ─────────────────────────────────────────────────────────────────────────────
# The Niño 3.4 index is the area-averaged SST *anomaly* over 5°S–5°N, 120°W–170°W.
# It is the standard measure of ENSO state used in seasonal forecasting.

NINO34_BOX = {"north": 5, "south": -5, "west": -170, "east": -120}

# CDS area-subsetting uses the order [North, West, South, East]
NINO34_AREA = [
    NINO34_BOX["north"],
    NINO34_BOX["west"],
    NINO34_BOX["south"],
    NINO34_BOX["east"],
]

# ─────────────────────────────────────────────────────────────────────────────
# C3S MULTI-SYSTEM SEASONAL FORECAST MODELS
# ─────────────────────────────────────────────────────────────────────────────
# C3S combines forecasts from several European modelling centres.
# Each model has its own systematic biases, so we debias each model separately
# using its own hindcast climatology before pooling all members into one
# grand ensemble.
#
# originating_centre : string used in the CDS API request
# system             : model version number (check CDS if this changes)
# n_members          : number of ensemble members available for the forecast
# hindcast_years     : years used to build the model's climatology (typically 1993–2016)
#
# NOTE: not all models provide sea_surface_temperature. If a download fails
# the code will issue a warning and skip that model gracefully.

MODELS = {
    "ecmwf": {
        "originating_centre": "ecmwf",
        "system": "51",            # SEAS5.1
        "n_members": 51,           # members 0–50
        "hindcast_years": list(range(1993, 2017)),
    },
    "ukmo": {
        "originating_centre": "ukmo",
        "system": "600",           # GloSea6.1
        "n_members": 28,           # verify on CDS — may vary
        "hindcast_years": list(range(1993, 2017)),
    },
    "meteo_france": {
        "originating_centre": "meteo_france",
        "system": "8",             # MF System 8
        "n_members": 25,
        "hindcast_years": list(range(1993, 2017)),
    },
    "dwd": {
        "originating_centre": "dwd",
        "system": "21",            # GCFS2.1
        "n_members": 30,
        "hindcast_years": list(range(1993, 2017)),
    },
    "cmcc": {
        "originating_centre": "cmcc",
        "system": "35",            # SPS3.5
        "n_members": 40,
        "hindcast_years": list(range(1993, 2017)),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FORECAST / TARGET SEASON
# ─────────────────────────────────────────────────────────────────────────────
# INIT_MONTH    : calendar month on whose 1st the forecast is issued (1=Jan … 12=Dec)
# TARGET_MONTHS : the months whose mean Niño 3.4 we want to forecast
# SEASON_LABEL  : human-readable label for plots and filenames

INIT_MONTH     = 4            # April initialisation
TARGET_MONTHS  = [4, 5, 6]   # April–May–June (AMJ)
SEASON_LABEL   = "AMJ"

# Number of lead days to download (must cover the full target season).
# April(30) + May(31) + June(30) = 91 days; we add a small buffer.
FORECAST_LEADTIME_DAYS = 95

# ─────────────────────────────────────────────────────────────────────────────
# HINDCAST / CLIMATOLOGY PERIOD
# ─────────────────────────────────────────────────────────────────────────────
HINDCAST_START_YEAR = 1993
HINDCAST_END_YEAR   = 2016   # inclusive; standard C3S hindcast period

# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN UPDATE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# Tercile boundaries: the season is divided into three equally likely
# categories — below normal (BN), near normal (NN), above normal (AN).
TERCILE_LOWER = 1.0 / 3.0   # ~33.3 %
TERCILE_UPPER = 2.0 / 3.0   # ~66.7 %
