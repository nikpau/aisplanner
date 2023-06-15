from pytsa import TrajectoryMatcher
from pytsa.targetship import TargetVessel
from itertools import permutations
import pickle
from ..misc import logger
from pathlib import Path
from typing import Union

RESDIR = Path("results")
REMOTEHOST = "taurus.hrsk.tu-dresden.de"
REMOTEDIR = Path("/warm_archive/ws/s2075466-ais/decoded/jan2020_to_jun2022")

def load_results(filename: Union[str, Path]):
    if not isinstance(filename, Path):
        filename = Path(filename)
    with open(RESDIR / filename, "rb") as f:
        results = pickle.load(f)
    return results

def save_overlapping_trajectories(tgts: list[TargetVessel], filename: str):
    out = set()
    found = 0
    for v1, v2 in permutations(tgts, 2):
        tm = TrajectoryMatcher(v1, v2)
        if not tm.disjoint_trajectories and tm.overlapping_trajectories:
            found += 1
            logger.info(
                f"Found overlapping trajectories for {v1.mmsi} "
                f"and {v2.mmsi}. Total: {found}"
            )
            out.add(v1)
            out.add(v2)
    with open(filename, "wb") as f:
        pickle.dump(out, f)
    return

def plot_encounter(v1: TargetVessel,v2: TargetVessel):
        tm = TrajectoryMatcher(v1,v2)
        if not tm.disjoint_trajectories and tm.overlapping_trajectories:
            print(f"Plotting encounter for {v1.mmsi} and {v2.mmsi}")
            tm.observe_interval(10).plot(every=6, path="out/plots/")
            return
        print(f"Disjoint trajectories for {v1.mmsi} and {v2.mmsi}")
        return
    