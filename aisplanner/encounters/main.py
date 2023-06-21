import utils
from getpass import getuser
from aisplanner.encounters.filter import (
    TrajectoryExtractionAgent
)
from aisplanner.encounters._locdb import LocationDatabase

if __name__ == "__main__":
    
    # # Extract a list of TargetVessel objects from raw AIS messages
    # # by scanning through 30 minute intervals over all search areas
    # # in the LocationDatabase. Time frame is 2021-01-01 to 2021-12-31.
    # s = TrajectoryExtractionAgent(
    #     remote_host="",
    #     remote_dir="",
    #     search_areas=LocationDatabase.all(utm=True),
    #     filelist=[f"/home/{getuser()}/Dropbox/TU Dresden/aisplanner/data/aisrecords/2021*.csv"],
    # )
    # s.search()
    # # Saved object's type is list[TargetVessel]
    # s.save_results(f"/home/{getuser()}/Dropbox/TU Dresden/aisplanner/results/2021.tr")
    
    # # Check raw trajectories for overlapping trajectories
    # # and save them in the './results/overlapping' directory
    # utils.overlaps_from_raw()

    # Check overlapping trajectories for encounters
    # and save them in the './results/encounters' directory
    utils.encounters_from_overlapping()