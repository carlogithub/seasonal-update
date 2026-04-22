"""
location.py — Defines geographic locations used in the Bayesian update pipeline.

A Location holds the coordinates of a point of interest and knows how to
produce the CDS area bounding box needed for data downloads.  For point
locations (e.g. a city) we add a small buffer (box_deg) on each side so that
the ERA5 and forecast grids have at least one interior grid point.

The Niño 3.4 case is special: the region is defined by its own fixed lat/lon
box rather than a single point, so we subclass Location and override cds_area.

Usage
-----
    from location import TERRASSA, NINO34_LOC, LOCATIONS
    area = TERRASSA.cds_area     # [42.06, 1.51, 41.06, 2.51]  (N, W, S, E)
    slug = TERRASSA.slug         # "terrassa"
"""

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Location:
    """
    A named geographic location.

    Attributes
    ----------
    name    : human-readable name (used in plot titles)
    lat     : centre latitude  (°N, positive north)
    lon     : centre longitude (°E, positive east)
    box_deg : half-width of the bounding box in degrees (default 0.5°)
              → the CDS area will be [lat+0.5, lon-0.5, lat-0.5, lon+0.5]
    """
    name:    str
    lat:     float
    lon:     float
    box_deg: float = 0.5

    @property
    def cds_area(self) -> list:
        """
        Return the CDS area bounding box as [North, West, South, East].
        This is the format expected by cdsapi for spatial subsetting.
        """
        return [
            self.lat + self.box_deg,   # North
            self.lon - self.box_deg,   # West
            self.lat - self.box_deg,   # South
            self.lon + self.box_deg,   # East
        ]

    @property
    def slug(self) -> str:
        """
        URL/filename-safe lowercase identifier (spaces → underscores).
        Used to build cache file names.
        """
        return self.name.lower().replace(" ", "_")

    @property
    def is_point(self) -> bool:
        """True for single-point locations; False for area-average regions."""
        return True


# ─────────────────────────────────────────────────────────────────────────────
# NINO 3.4 — SPECIAL SUBCLASS WITH FIXED REGION BOX
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Nino34Location(Location):
    """
    The Niño 3.4 region (5°S–5°N, 120°W–170°W).

    Unlike a point location, this region has its own fixed bounding box that
    is not derived from a centre point + buffer.  We store the corners directly
    and override cds_area to return them verbatim.

    CDS convention: area = [North, West, South, East].
    For the Niño 3.4 box: North=5, West=-170, South=-5, East=-120.
    """
    # Override the parent defaults to a sensible centre (equatorial Pacific)
    name:    str   = "Niño 3.4"
    lat:     float = 0.0
    lon:     float = -145.0
    box_deg: float = 0.0   # not used — area is fixed below

    @property
    def cds_area(self) -> list:
        """Fixed Niño 3.4 bounding box: 5°S–5°N, 120°W–170°W."""
        return [5, -170, -5, -120]   # [North, West, South, East]

    @property
    def slug(self) -> str:
        return "nino34"

    @property
    def is_point(self) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PRESET LOCATIONS
# ─────────────────────────────────────────────────────────────────────────────

TERRASSA   = Location("Terrassa", lat=41.56, lon=2.01)
NINO34_LOC = Nino34Location()

# Registry: maps CLI --location argument to a Location object
LOCATIONS = {
    "terrassa": TERRASSA,
    "nino34":   NINO34_LOC,
}
