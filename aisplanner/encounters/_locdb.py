"""
Database of suitable locations for for COLREGS-relevant encounters.
"""
from pytsa.structs import LatLonBoundingBox
from dataclasses import dataclass, fields

@dataclass
class LocationDatabase:
    # Frederikshavn to Gothenburg
    fre_got: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=57.378,
        LATMAX=57.778,
        LONMIN=10.446,
        LONMAX=11.872,
        __name__="fre_got"
    )
    elbe_approach: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=53.9807,
        LATMAX = 54.049,
        LONMIN=7.487,
        LONMAX=7.7734,
        __name__="elbe_approach"
    )
    # Helsingor to Helsingborg
    hel_hel: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=55.998,
        LATMAX=56.064,
        LONMIN=12.560,
        LONMAX=12.745,
        __name__="hel_hel"
    )
    # Hirtsals to Kristiansand
    hir_krs: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=57.393,
        LATMAX=58.240,
        LONMIN=7.280,
        LONMAX=9.995,
        __name__="hir_krs"
    )
    # Return all locations as a list
    @classmethod
    def all(cls,utm=False):
        if utm:
            return [getattr(cls,field.name).to_utm() for field in fields(cls)]
        else:
            return [getattr(cls,field.name) for field in fields(cls)]