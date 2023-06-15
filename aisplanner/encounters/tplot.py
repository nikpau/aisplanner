"""
Trajectory plotting functions
""""""
Module for plotting colregs encounters found via filter.py
"""
import pickle
from pathlib import Path
from typing import Union
from itertools import permutations

from dotenv import load_dotenv
from pytsa.targetship import TargetVessel, TrajectoryMatcher

from aisplanner.encounters.filter import unique_2_permuts

# Types
MMSI = int

load_dotenv()

RESDIR = Path("results")
REMOTEHOST = "taurus.hrsk.tu-dresden.de"
REMOTEDIR = Path("/warm_archive/ws/s2075466-ais/decoded/jan2020_to_jun2022")

# Load pickled results
def load_results(filename: Union[str, Path]):
    if not isinstance(filename, Path):
        filename = Path(filename)
    with open(RESDIR / filename, "rb") as f:
        results = pickle.load(f)
    return results


if __name__ == "__main__":
    # Load results
    res: list[TargetVessel] = load_results("2021_hel_hel.tr")
    print(len(res))

    # Plot trajectories
    uniques = unique_2_permuts(res[:5000])
    
    def plot_encounter(v1,v2):
        tm = TrajectoryMatcher(v1,v2)
        if not tm.disjoint_trajectories and tm.overlapping_trajectories:
            print(f"Plotting encounter for {v1.mmsi} and {v2.mmsi}")
            tm.observe_interval(10).plot(every=6, path="out/plots/")
            return
        print(f"Disjoint trajectories for {v1.mmsi} and {v2.mmsi}")
        return
        
    for v1,v2 in permutations(res,2):
        #v1.overwrite_rot()
        #v2.overwrite_rot()
        try:
            plot_encounter(v1,v2)
        except Exception as e:
            print(e)
            continue