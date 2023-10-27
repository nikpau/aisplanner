from pathlib import Path
from aisplanner.encounters import utils
import os
from glob import glob
import multiprocessing as mp
from pytsa import ShipType
from psd import PSDPointExtractor
from aisplanner.encounters._locdb import GriddedNorthSea, LatLonBoundingBox


# Extract a list of TargetVessel objects from raw AIS messages
# by scanning through 30 minute intervals over all search areas
# in the LocationDatabase. Time frame is 2021-01-01 to 2021-12-31.
north_sea, = GriddedNorthSea(nrows=1, ncols=1, utm=False).cells

# Files to search through
# msg12318files=glob(f"{os.environ.get('AISDECODED')}/2021*.csv"),
msg12318files=glob("data/aisrecords/*.csv"),
# msg5files=glob(f"{os.environ.get('MSG5DECODED')}/2021*.csv"),
msg5files=glob("data/aisrecords/msgtype5/*.csv"),

def rawPSDextraction(debug=False):

    if debug:
        for dyn, stat in zip(msg12318files,msg5files):
            _do(dyn,stat)
    else:
        # Run in parallel
        #with mp.Pool(len(areas)) as pool:
        with mp.Pool(mp.cpu_count()-1) as pool:
            pool.starmap(_do, zip(msg12318files,msg5files))

def _do(msg12318file: Path, msg5file: Path):
    s = PSDPointExtractor(
        search_areas=north_sea,
        msg12318files=msg12318file,
        msg5files=msg5file,
        ship_types=[
            ShipType.CARGO.value, 
            ShipType.TANKER.value, 
            ShipType.PASSENGER.value
            ]
    )
    s.search()
    # s.save_results(f"{os.environ.get('RESPATH')}",loc.number)
    s.save_results("psdresults")
    
if __name__ == "__main__":
    rawPSDextraction(debug=False)