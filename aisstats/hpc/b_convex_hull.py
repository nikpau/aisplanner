from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import speed_filter, _cvh_area, plot_convex_hull_area_histogram
from pathlib import Path
from pytsa import SearchAgent
from pytsa.structs import AISMessage
from functools import partial
from pytsa.trajectories.rules import *
from pytsa.trajectories import inspect
import pickle

np.seterr(all='raise')

DEGREES = np.linspace(0,1e5,101)
MINLENS = np.linspace(0,100,101)

def online_average(avg, new, n):
    return avg + (new - avg) / n

def average_change_of_course(ships, split=False):
    """
    Calulates the mean of the cosine of the angles
    enclosed between three consecutive messages 
    for several standard deviations.
    """
    delta = np.full((len(MINLENS),len(DEGREES)),np.nan)
    
    # count for running average
    counts = np.full((len(MINLENS),len(DEGREES)),0)
    for ship in ships.values():
        for track in ship.tracks:
            length = len(track)
            if length < 3:
                print("Track too short")
                continue
            try:
                area = _cvh_area(track)
            except:
                print("Convex hull could not be calculated.")
                continue
            s = inspect.average_smoothness(track)
            d = 180 - 180*s
            
            # Find the index of the minimum length
            minlen_idx = np.argmin(np.abs(MINLENS-length))
            area_idx = np.argmin(np.abs(DEGREES-area))
            
            # If there is no value yet, set it
            if counts[minlen_idx,area_idx] == 0:
                delta[minlen_idx,area_idx] = d
                counts[minlen_idx,area_idx] = 1
                continue
            
            # Update the running average
            counts[minlen_idx,area_idx] += 1
            delta[minlen_idx,area_idx] = online_average(
                delta[minlen_idx,area_idx], 
                d,
                counts[minlen_idx,area_idx]
            )
            
    # Save the results
    split = "_split" if split else "_raw"
    with open(f"/home/s2075466/aisplanner/results/avg_smoothness_cvh{split}.pkl","wb") as f:
        pickle.dump(delta,f)
        
    # Save the counts
    with open(f"/home/s2075466/aisplanner/results/avg_smoothness_counts_cvh{split}.pkl","wb") as f:
        pickle.dump(counts,f)

if __name__ == "__main__":
    for split in [True,False]:
        SEARCHAREA = NorthSea

        DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07*.csv"))
        STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07*.csv"))

        SA = SearchAgent(
                dynamic_paths=DYNAMIC_MESSAGES,
                frame=SEARCHAREA,
                static_paths=STATIC_MESSAGES,
                preprocessor=partial(speed_filter, speeds= (1,30))
            )

        ships = SA.extract_all(njobs=16, skip_tsplit=split)
        average_change_of_course(ships,split=split)

