"""
season.py — Defines forecast seasons and their relationship to initialisation months.

A Season ties together:
  - an initialisation month (the month on whose 1st the forecast is issued)
  - a list of target months (whose mean/total we want to forecast)
  - a human-readable label (used in filenames and plot titles)

Key subtleties handled here
---------------------------

Lead-time months
    CDS uses 1-based lead months: leadtime_month=1 means the first calendar
    month after initialisation.  For example, if init_month=4 (April), then
    leadtime_month=1 → April, leadtime_month=2 → May, leadtime_month=3 → June.
    We compute this as  ((target_month - init_month) % 12) + 1  to handle
    wrap-around correctly for DJF (where December follows October init).

Year boundary for DJF
    When the target season crosses a year boundary — e.g. DJF initialised in
    October: December is in init_year, January and February are in init_year+1 —
    we need to know which calendar year a given target month belongs to.
    target_year(init_year, target_month) returns the correct year:
      - init_year   if target_month >= init_month
      - init_year+1 if target_month <  init_month  (crossed a year boundary)

Usage
-----
    from season import JJA, DJF, SEASONS
    season = SEASONS["djf"]
    leads  = season.leadtime_months()        # [3, 4, 5]  (Oct init → Dec/Jan/Feb)
    yr     = season.target_year(2026, 1)     # 2027  (January is next year for Oct init)
"""

from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Season:
    """
    Describes a forecast season.

    Attributes
    ----------
    label         : short label used in filenames/titles, e.g. "AMJ", "JJA", "DJF"
    init_month    : calendar month of forecast initialisation (1=Jan … 12=Dec)
    target_months : list of calendar months forming the target season, e.g. [6,7,8]
    """
    label:         str
    init_month:    int
    target_months: List[int]

    def leadtime_months(self) -> List[int]:
        """
        Return the CDS leadtime_month values corresponding to each target month.

        CDS counts lead months from 1 (= first calendar month of the forecast).
        The formula is:  ((target - init) % 12) + 1

        Example: init=10 (Oct), targets=[12, 1, 2]
          Dec: ((12-10) % 12) + 1 = 3
          Jan: ((1 -10) % 12) + 1 = 4
          Feb: ((2 -10) % 12) + 1 = 5
        """
        return [((m - self.init_month) % 12) + 1 for m in self.target_months]

    def target_year(self, init_year: int, target_month: int) -> int:
        """
        Return the calendar year that a given target month falls in.

        If the target month is in the same or a later calendar month than the
        initialisation month, it is in the same year.  If it is earlier (i.e.
        we crossed a year boundary — e.g. January after an October init), it
        is in init_year + 1.

        Parameters
        ----------
        init_year    : the year the forecast was initialised
        target_month : the calendar month we want the year for

        Returns
        -------
        Calendar year (int).
        """
        if target_month >= self.init_month:
            return init_year
        else:
            return init_year + 1

    @property
    def slug(self) -> str:
        return self.label.lower()


# ─────────────────────────────────────────────────────────────────────────────
# PRESET SEASONS
# ─────────────────────────────────────────────────────────────────────────────

# April–May–June, initialised in April
# Used for the Niño 3.4 case: partial April observations → AMJ forecast
AMJ = Season(
    label         = "AMJ",
    init_month    = 4,
    target_months = [4, 5, 6],
)

# June–July–August, initialised in April
# The main boreal summer season for European temperature and precipitation
JJA = Season(
    label         = "JJA",
    init_month    = 4,
    target_months = [6, 7, 8],
)

# December–January–February, initialised in October
# The main boreal winter season.  Note: December is in init_year,
# January and February are in init_year+1.
DJF = Season(
    label         = "DJF",
    init_month    = 10,
    target_months = [12, 1, 2],
)


# Registry: maps CLI --season argument to a Season object
SEASONS = {
    "amj": AMJ,
    "jja": JJA,
    "djf": DJF,
}
