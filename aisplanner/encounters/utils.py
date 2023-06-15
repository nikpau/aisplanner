from pytsa import TrajectoryMatcher
from pytsa.targetship import TargetVessel
from itertools import permutations
import pickle
from ..misc import logger

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
    