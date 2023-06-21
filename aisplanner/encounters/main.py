import utils
import os
from glob import glob
import multiprocessing as mp
from getpass import getuser
from aisplanner.encounters.filter import (
    TrajectoryExtractionAgent
)
from aisplanner.encounters._locdb import LocationDatabase, LatLonBoundingBox
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    
    # Extract a list of TargetVessel objects from raw AIS messages
    # by scanning through 30 minute intervals over all search areas
    # in the LocationDatabase. Time frame is 2021-01-01 to 2021-12-31.
    areas = LocationDatabase.all(utm=True)
    def _do(loc: LatLonBoundingBox):
        s = TrajectoryExtractionAgent(
            search_areas=loc,
            msg12318files=glob(f"{os.environ.get('AISDECODED')}/2021_*.tr"),
            msg5files=glob(f"{os.environ.get('MSG5DECODED')}/2021_*.tr"),
        )
        s.search()
        # Saved object's type is list[TargetVessel]
        s.save_results(f"{os.environ.get('ENCOUNTERS_DIR')}/2021_{loc.name}.tr")
    
    # Run in parallel
    with mp.Pool(len(areas)) as pool:
        pool.map(_do, areas)
    
    # Check raw trajectories for overlapping trajectories
    # and save them in the './results/overlapping' directory
    utils.overlaps_from_raw()

    # Check overlapping trajectories for encounters
    # and save them in the './results/encounters' directory
    utils.encounters_from_overlapping()