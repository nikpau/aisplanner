from pytsa import TrajectoryMatcher
from pytsa.targetship import TargetVessel
from itertools import permutations
import pickle
from aisplanner.misc import logger
from pathlib import Path
from typing import Any, Union
from dataclasses import dataclass
from aisplanner.encounters.filter import (
    ForwardBackwardScan, EncounterSituation,
    Ship
)

RESDIR = Path("results")
REMOTEHOST = "taurus.hrsk.tu-dresden.de"
REMOTEDIR = Path("/warm_archive/ws/s2075466-ais/decoded/jan2020_to_jun2022")

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
            tm.observe_interval(10).plot(every=6, path="out/plots/")
            return
        print(f"Disjoint trajectories for {v1.mmsi} and {v2.mmsi}")
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
    outfiles = [
        RESDIR.joinpath("overlapping")/f"{f.stem}_ol.tr" for f in files
    ]

    # Pipeline function for multiprocessing
    def pipeline(file: str, outfile: str):
        tgts = load_results(file)
        save_overlapping_trajectories(tgts, outfile)
        return

    # Create a pool of workers
    with mp.Pool() as pool:
        pool.starmap(pipeline, zip(files, outfiles))

    return 

def has_encounter(v1: TargetVessel, v2: TargetVessel) -> bool:
    """
    Check if any message-pair in the trajectories of
    v1 and v2 has a COLREGS relevant encounter.
    """
    encs = []
    tm = TrajectoryMatcher(v1, v2)
    tm.observe_interval()
    for o1, o2 in zip(tm.obs_vessel1, tm.obs_vessel2):
        s1 = Ship(pos=o1[0:2], cog=o1[2], sog=o1[3])
        s2 = Ship(pos=o2[0:2], cog=o2[2], sog=o2[3])
        enc = EncounterSituation(s1, s2).analyze()
        if enc is not None:
            encs.extend(enc)
    return len(encs) > 0
        

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

    # Pipeline function for multiprocessing
    def pipeline(file: str):
        out: set[OverlappingPair] = set()
        tgts: list[OverlappingPair] = load_results(file)
        for vobj in tgts:
            if vobj.same_mmsi():
                continue
            if has_encounter(*vobj()):
                fb = ForwardBackwardScan(*vobj())
                if fb():
                    out.add(vobj)
        outpath = RESDIR.joinpath("encounters")/f"{Path(file).stem}_en.tr"
        with open(outpath, "wb") as f:
            pickle.dump(out, f)
        return 

    # Create a pool of workers
    with mp.Pool() as pool:
        pool.map(pipeline, files)

    return