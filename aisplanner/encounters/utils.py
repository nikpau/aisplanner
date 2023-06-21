from pytsa import TrajectoryMatcher
from pytsa.targetship import TargetVessel
from itertools import permutations
import pickle
import os
import dotenv
from aisplanner.misc import logger
from pathlib import Path
from typing import Any, Union
from dataclasses import dataclass
from aisplanner.encounters.filter import (
    ForwardBackwardScan, EncounterSituation,
    Ship, Position,ColregsSituation
)
dotenv.load_dotenv()

RESDIR = Path(os.environ.get("RESPATH"))

# Sampling frequency for the trajectories in seconds
_SAMPLINGFREQ = 10

# Rolling window width for evasive maneuver detection
_WINDOWWIDTH = 30

@dataclass
class OverlappingPair:
    v1: TargetVessel
    v2: TargetVessel

    def __call__(self, *args: Any, **kwds: Any) -> tuple[TargetVessel, TargetVessel]:
        return self.v1, self.v2

    def __hash__(self) -> int:
        return hash((self.v1, self.v2))
    
    def same_mmsi(self) -> bool:
        """Check if both vessels of the pair have the same MMSI."""
        return self.v1.mmsi == self.v2.mmsi


def load_results(filename: Union[str, Path]):
    if not isinstance(filename, Path):
        filename = Path(filename)
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def save_overlapping_trajectories(tgts: list[TargetVessel], filename: Path):
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
            out.add(OverlappingPair(v1, v2))
    with open(filename, "wb") as f:
        pickle.dump(out, f)
    return

def plot_encounter(v1: TargetVessel,v2: TargetVessel):
        tm = TrajectoryMatcher(v1,v2)
        if not tm.disjoint_trajectories and tm.overlapping_trajectories:
            print(f"Plotting encounter for {v1.mmsi} and {v2.mmsi}")
            tm.observe_interval(_SAMPLINGFREQ).plot(every=6, path="out/plots/")
            return
        print(f"Disjoint trajectories for {v1.mmsi} and {v2.mmsi}")
        return

def _overlapping_pipeline(file: str, outfile: str):
    tgts = load_results(file)
    save_overlapping_trajectories(tgts, outfile)
    return

def overlaps_from_raw():
    """
    Extract overlapping trajectories from raw 
    trajectories in the results directory
    and saves them in the overlapping directory
    under the same filename.
    """
    import multiprocessing as mp

    # Get all files from the results directory
    files = list(RESDIR.glob("*.tr"))

    # Create new files for storing the overlapping trajectories
    # Create "overlapping" directory if it doesn't exist
    if not RESDIR.joinpath("overlapping").exists():
        RESDIR.joinpath("overlapping").mkdir()
    outfiles = [
        RESDIR.joinpath("overlapping")/f"{f.stem}_ol.tr" for f in files
    ]

    # Create a pool of workers
    with mp.Pool() as pool:
        pool.starmap(_overlapping_pipeline, zip(files, outfiles))

    return

def has_encounter(v1: TargetVessel, v2: TargetVessel, etype: ColregsSituation) -> bool:
    """
    Check if any message-pair in the trajectories of
    v1 and v2 has a COLREGS relevant encounter.
    """
    encs = []
    tm = TrajectoryMatcher(v1, v2)
    tm.observe_interval(_SAMPLINGFREQ)
    for o1, o2 in zip(tm.obs_vessel1, tm.obs_vessel2):
        s1 = Ship(pos=Position(o1[0],o1[1]), cog=o1[2], sog=o1[3])
        s2 = Ship(pos=Position(o2[0],o2[1]), cog=o2[2], sog=o2[3])
        found = EncounterSituation(s1, s2).analyze()
        if found is not None:
            encs.append(found)
    resset = set(encs)
    if len(resset) == 1:
        return list(resset)[0] == etype and len(encs) > 30
    return False
    # return len(set(encs)) == 1# and len(encs) > 100
    
def _encounter_pipeline(file: str):
    out: set[OverlappingPair] = set()
    overlaps: list[OverlappingPair] = load_results(file)
    for vpair in overlaps:
        if vpair.same_mmsi():
            continue
        if has_encounter(*vpair(),ColregsSituation.CROSSING):
            scanner = ForwardBackwardScan(*vpair(),interval=_SAMPLINGFREQ)
            if scanner(_WINDOWWIDTH):
                out.add(vpair)
    # Create "encounters" directory if it doesn't exist
    if not RESDIR.joinpath("encounters").exists():
        RESDIR.joinpath("encounters").mkdir()
    outpath = RESDIR.joinpath("encounters")/f"{Path(file).stem}_en.tr"
    with open(outpath, "wb") as f:
        pickle.dump(out, f)
    return 

def encounters_from_overlapping() -> None:
    """
    Find encounters from overlapping trajectories
    in the results/overlapping directory.
    
    Results are saved as a pickle file
    containing a set of OverlappingPair objects
    in the './results/encounters' directory.
    """
    import multiprocessing as mp

    # Get all files from the results directory
    files = list(RESDIR.joinpath("overlapping").glob("*.tr"))

    # Create a pool of workers
    with mp.Pool() as pool:
        pool.map(_encounter_pipeline, files)

    return