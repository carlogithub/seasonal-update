"""
variable.py — Defines the meteorological variables supported by the pipeline.

Each Variable knows:
  - which CDS variable name to request for ERA5 and for C3S forecasts
  - how to transform its values before the Bayesian regression (important for
    precipitation, which is bounded at zero and non-Gaussian)
  - how to aggregate it over time (mean for temperature/SST, total for precip)
  - what physical units to display on plots

Transforms
----------
  "none"     : no transform — use values as-is (temperature, SST)
  "cuberoot" : apply x → sign(x)·|x|^(1/3) before regression and plots.
               This stabilises precipitation, which is heavily right-skewed.
               Values must be untransformed (cubed) before physical interpretation.

The cube-root is preferred over log(1+x) because it handles zero naturally,
preserves the sign when used on anomalies, and is invertible everywhere.

Observation type
----------------
  "mean"  : the period aggregate is a time average (temperature, SST)
  "total" : the period aggregate is a sum (precipitation)

Units / offsets
---------------
  kelvin_offset : subtracted when converting raw model SST/T2M from Kelvin to °C
                  (0.0 if the variable is already in °C or in anomaly space)
  mm_scale      : multiply raw TP (m) by this factor to get mm
                  (0.0 for non-precipitation variables)
"""

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Variable:
    """
    Describes one meteorological variable and how to handle it.

    Attributes
    ----------
    name              : human-readable name (used in plot labels)
    short_name        : abbreviated identifier (used in filenames / print messages)
    era5_name         : CDS variable name for ERA5 downloads
    c3s_name          : CDS variable name for C3S seasonal-monthly-single-levels
    c3s_postproc_name : CDS variable name for seasonal-postprocessed-single-levels
                        (already bias-corrected anomalies); empty string if not available
    transform         : "none" | "cuberoot"
    obs_type          : "mean" | "total"
    units             : physical unit string for axis labels
    kelvin_offset     : subtract from raw values to convert K → °C (0 if not needed)
    mm_scale          : multiply raw TP values by this to get mm (0 if not precipitation)
    """
    name:               str
    short_name:         str
    era5_name:          str
    c3s_name:           str
    c3s_postproc_name:  str
    transform:          str    # "none" or "cuberoot"
    obs_type:           str    # "mean" or "total"
    units:              str
    kelvin_offset:      float = 0.0
    mm_scale:           float = 0.0

    @property
    def slug(self) -> str:
        return self.short_name.lower()

    def apply_transform(self, x):
        """
        Apply the variable's transform to a scalar, array, or DataArray.
        For precipitation (cuberoot), this is x → sign(x)·|x|^(1/3).
        For all others, this is a no-op.
        """
        import numpy as np
        if self.transform == "none":
            return x
        elif self.transform == "cuberoot":
            # np.cbrt handles arrays, DataArrays, and scalars
            # Using np.cbrt rather than **(1/3) to avoid issues with negative values
            return np.cbrt(x)
        else:
            raise ValueError(f"Unknown transform '{self.transform}'")

    def invert_transform(self, x):
        """
        Invert the transform (cube-root → cube).
        Used when converting posterior/prior statistics back to physical units.
        """
        import numpy as np
        if self.transform == "none":
            return x
        elif self.transform == "cuberoot":
            return x ** 3
        else:
            raise ValueError(f"Unknown transform '{self.transform}'")


# ─────────────────────────────────────────────────────────────────────────────
# PRESET VARIABLES
# ─────────────────────────────────────────────────────────────────────────────

# Sea-surface temperature Niño 3.4 anomaly
# ERA5 delivers SST in Kelvin; we subtract the climatological mean to get anomalies.
# CDS variable name for seasonal forecasts is "sea_surface_temperature".
NINO34_VAR = Variable(
    name              = "Niño 3.4 SST anomaly",
    short_name        = "nino34",
    era5_name         = "sea_surface_temperature",
    c3s_name          = "sea_surface_temperature",
    c3s_postproc_name = "sea_surface_temperature_anomaly",
    transform         = "none",
    obs_type          = "mean",
    units             = "K (anomaly)",
    kelvin_offset     = 0.0,
    mm_scale          = 0.0,
)

# 2-metre air temperature
# ERA5 and C3S both deliver T2M in Kelvin; we subtract the climatological mean
# for anomalies.  kelvin_offset=273.15 converts absolute values to °C when needed,
# but since the Bayesian regression operates on anomalies it is not applied there.
T2M = Variable(
    name              = "2m temperature",
    short_name        = "t2m",
    era5_name         = "2m_temperature",
    c3s_name          = "2m_temperature",
    c3s_postproc_name = "2m_temperature_anomaly",
    transform         = "none",
    obs_type          = "mean",
    units             = "°C (anomaly)",
    kelvin_offset     = 273.15,
    mm_scale          = 0.0,
)

# Total precipitation
# ERA5 delivers TP in metres (m) per day; C3S monthly forecasts in m/s or m/month
# depending on the model — process.py handles the unit conversion.
# mm_scale converts m → mm (×1000).
# transform="cuberoot" stabilises the right-skewed distribution before regression.
TP = Variable(
    name              = "Total precipitation",
    short_name        = "tp",
    era5_name         = "total_precipitation",
    c3s_name          = "total_precipitation",
    c3s_postproc_name = "total_precipitation_anomalous_rate_of_accumulation",
    transform         = "cuberoot",
    obs_type          = "total",
    units             = "mm (cube-root anomaly)",
    kelvin_offset     = 0.0,
    mm_scale          = 1000.0,
)


# Registry: maps CLI --variable argument to a Variable object
VARIABLES = {
    "nino34": NINO34_VAR,
    "t2m":    T2M,
    "tp":     TP,
}
