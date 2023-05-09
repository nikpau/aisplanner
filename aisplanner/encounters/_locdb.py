"""
Database of suitable locations for for COLREGS-relevant encounters.
"""
from pytsa.targetship import BoundingBox
from dataclasses import dataclass, fields

@dataclass
class LocationDatabase:
    # Frederikshavn to Gothenburg
    fre_got: BoundingBox = BoundingBox(
        LATMIN=57.378,
        LATMAX=57.778,
        LONMIN=10.446,
        LONMAX=11.872
    )
    elbe_approach: BoundingBox = BoundingBox(
        LATMIN=53.9807,
        LATMAX = 54.049,
        LONMIN=7.487,
        LONMAX=7.7734
    )
    # Helsingor to Helsingborg
    hel_hel: BoundingBox = BoundingBox(
        LATMIN=55.998,
        LATMAX=56.064,
        LONMIN=12.560,
        LONMAX=12.745
    )
    # Hirtsals to Kristiansand
    hir_krs: BoundingBox = BoundingBox(
        LATMIN=57.393,
        LATMAX=58.240,
        LONMIN=7.280,
        LONMAX=9.995
    )
    # Return all locations as a list
    @classmethod
    def all(cls,utm=False):
        if utm:
            return [getattr(cls,field.name).to_utm() for field in fields(cls)]
        else:
            return [getattr(cls,field.name) for field in fields(cls)]