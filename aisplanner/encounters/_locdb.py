"""
Database of suitable locations for for COLREGS-relevant encounters.
"""
from typing import Union
from pytsa.structs import LatLonBoundingBox, UTMBoundingBox
from dataclasses import dataclass, fields

@dataclass
class LocationDatabase:
    # Frederikshavn to Gothenburg
    fre_got: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=57.378,
        LATMAX=57.778,
        LONMIN=10.446,
        LONMAX=11.872,
        name="Frederikshavn_to_Gothenburg"
    )
    elbe_approach: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=53.9807,
        LATMAX = 54.049,
        LONMIN=7.487,
        LONMAX=7.7734,
        name="Elbe_approach"
    )
    german_bight: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=53.7016,
        LATMAX = 54.2652,
        LONMIN=7.2455,
        LONMAX=9.0739,
        name="German_bight"
    )
    # # Helsingor to Helsingborg
    # hel_hel: LatLonBoundingBox = LatLonBoundingBox(
    #     LATMIN=55.998,
    #     LATMAX=56.064,
    #     LONMIN=12.560,
    #     LONMAX=12.745,
    #     name="Helsingor_to_Helsingborg"
    # )
    # Hirtsals to Kristiansand
    hir_krs: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=57.393,
        LATMAX=58.240,
        LONMIN=7.280,
        LONMAX=9.995,
        name="Hirtsals_to_Kristiansand"
    )
    # North north-sea
    nns: LatLonBoundingBox = LatLonBoundingBox(
        LATMIN=57.88,
        LATMAX=58.34,
        LONMIN=9.80,
        LONMAX=11.27,
        name="North_north_sea"
    )

    # Return all locations as a list
    @classmethod
    def all(cls,utm=False):
        if utm:
            return [getattr(cls,field.name).to_utm() for field in fields(cls)]
        else:
            return [getattr(cls,field.name) for field in fields(cls)]


class GriddedNorthSea:
    """
    Class for splitting up the North Sea into a grid of cells.
    """
    LATMIN = 51.85
    LATMAX = 60.49
    LONMIN = 4.85
    LONMAX = 14.3
    def __init__(self,nrows,ncols, utm=True) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.cells: Union[list[LatLonBoundingBox],list[UTMBoundingBox]] = []
        self.utm = utm
        self._setup()
        
    def _breakpoints(self):
        """
        Create breakpoints of latitude and longitude
        coordinates from initial ``frame`` resembling cell borders.
        """
        nrow = self.nrows
        ncol = self.ncols

        # Set limiters for cells 
        lat_extent = self.LATMAX-self.LATMIN
        lon_extent = self.LONMAX-self.LONMIN
        # Split into equal sized cells
        csplit = [self.LONMIN + i*(lon_extent/ncol) for i in range(ncol)]
        rsplit = [self.LATMIN + i*(lat_extent/nrow) for i in range(nrow)]
        csplit.append(self.LONMAX) # Add endpoints
        rsplit.append(self.LATMAX) # Add endpoints
        return rsplit, csplit
    
    def _setup(self) -> None:
        """
        Create individual Cells from
        provided frame and cell number and assigns
        ascending indices. The first cell
        starts in the North-West:
        
        N
        ^
        |
        |----> E
        
        Example of a 5x5 Grid:
        ------------------------------------
        |  0   |  1   |  2   |  3   |  4   |
        |      |      |      |      |      |
        ------------------------------------
        |  5   |  6   |  7   |  8   |  9   |
        |      |      |      |      |      |            
        ------------------------------------       
        | ...  | ...  | ...  | ...  | ...  |             
                
        """
        boxnum = 0
        latbp, lonbp = self._breakpoints()
        latbp= sorted(latbp)[::-1] # Descending Latitudes
        lonbp = sorted(lonbp) # Ascending Longitudes
        for lat_max, lat_min in zip(latbp,latbp[1:]):
            for lon_min, lon_max in zip(lonbp,lonbp[1:]):
                boxnum += 1
                box = LatLonBoundingBox(
                    LATMIN=lat_min,LATMAX=lat_max,
                    LONMIN=lon_min,LONMAX=lon_max,
                )
                box.name = f"{box.LATMIN:.2f}°N-{box.LATMAX:.2f}°N_{box.LONMIN:.2f}°E-{box.LONMAX:.2f}°E"
                box.number = boxnum # Assign ascending indices
                if self.utm:
                    box = box.to_utm()
                self.cells.append(box)