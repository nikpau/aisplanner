from aisplanner.encounters import utils
import os
from glob import glob
import multiprocessing as mp
from pytsa import ShipType
from aisplanner.encounters.filter import (
    TrajectoryExtractionAgent
)
from aisplanner.encounters._locdb import NorthSea, BoundingBox


# Extract a list of TargetShip objects from raw AIS messages
# by scanning through 30 minute intervals over all search areas
# in the LocationDatabase. Time frame is 2021-01-01 to 2021-12-31.
areas = NorthSea

def search_for(ship_type: ShipType, debug=False):

    if debug:
        for loc in areas:
            _do(loc, ship_type)
    else:
        # Run in parallel
        with mp.Pool(len(areas)) as pool:
            pool.starmap(_do, [(loc, ship_type) for loc in areas])

    # Check raw trajectories for overlapping trajectories
    # and save them in the './results/overlapping' directory
    utils.overlaps_from_raw()

    # Check overlapping trajectories for encounters
    # and save them in the './results/encounters' directory
    utils.encounters_from_overlapping()

def _do(loc: BoundingBox, ship_type: ShipType):
    s = TrajectoryExtractionAgent(
        search_areas=loc,
        msg12318files=glob(f"{os.environ.get('AISDECODED')}/2021_*.csv"),
        msg5files=glob(f"{os.environ.get('MSG5DECODED')}/2021_*.csv"),
        ship_types=ship_type,
        time_delta=30
    )
    s.search()
    # Saved object's type is list[TargetShip]
    s.save_results(f"{os.environ.get('RESPATH')}/2021_{loc.name}.tr")