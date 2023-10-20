from aisplanner.encounters import utils
import os
from glob import glob
import multiprocessing as mp
from pytsa import ShipType
from .psd import PSDPointExtractor
from aisplanner.encounters._locdb import GriddedNorthSea, LatLonBoundingBox


# Extract a list of TargetVessel objects from raw AIS messages
# by scanning through 30 minute intervals over all search areas
# in the LocationDatabase. Time frame is 2021-01-01 to 2021-12-31.
areas = GriddedNorthSea(nrows=8, ncols=8, utm=True).cells

def rawPSDextraction(debug=False):

    if debug:
        for loc in areas:
            _do(loc)
    else:
        # Run in parallel
        with mp.Pool(len(areas)) as pool:
            pool.map(_do, areas)

def _do(loc: LatLonBoundingBox):
    s = PSDPointExtractor(
        search_areas=loc,
        msg12318files=glob(f"{os.environ.get('AISDECODED')}/2021_*.csv"),
        msg5files=glob(f"{os.environ.get('MSG5DECODED')}/2021_*.csv"),
        ship_types=[ShipType.CARGO, ShipType.TANKER, ShipType.PASSENGER],
        time_delta=60
    )
    s.search()
    s.save_results(f"{os.environ.get('RESPATH')}")
    
if __name__ == "__main__":
    rawPSDextraction(debug=True)