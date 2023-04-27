"""
Database suitable locations for for COLREGS-relevant encounters.
"""
from pytsa.targetship import BoundingBox
from dataclasses import dataclass

@dataclass
class LocationDatabase:
    # Frederikshavn to Gothenburg
    fre_got: BoundingBox = BoundingBox(
        LATMIN=57.378,
        LATMAX=57.778,
        LONMIN=10.446,
        LONMAX=11.872
    )