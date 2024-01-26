"""
Module for plotting the rate of rejection for different 
values for the minimum number of observation per trajectory,
and the spatial standard deviation threshold.
"""
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import speed_filter
from pathlib import Path
from pytsa import SearchAgent, TargetShip
from functools import partial
import pytsa
import pytsa.trajectories.inspect as inspect
from pytsa.trajectories.rules import *
import logging
import pickle
import multiprocessing as mp
import sys

np.seterr(all='raise')

SDS = np.linspace(0,0.1,101)
MINLENS = np.linspace(0,100,101)

def online_average(avg, new, n):
    return avg + (new - avg) / n

def average_complexity(ships: dict[int,TargetShip]):
    """
    Calulates the mean of the cosine of the angles
    enclosed between three consecutive messages 
    for several standard deviations.
    """
    smthness = np.full((len(MINLENS),len(SDS)),np.nan)
    
    # count for running average
    counts = np.full((len(MINLENS),len(SDS)),0)
    for ship in ships.values():
        for track in ship.tracks:
            length = len(track)
            if length < 3:
                print("Track too short")
                continue
            sd = np.std([p.lon for p in track]) + np.std([p.lat for p in track])
            s = inspect.average_smoothness(track)
            
            # Find the index of the minimum length
            minlen_idx = np.argmin(np.abs(MINLENS-length))
            sd_idx = np.argmin(np.abs(SDS-sd))
            
            # If there is no value yet, set it
            if counts[minlen_idx,sd_idx] == 0:
                smthness[minlen_idx,sd_idx] = s
                counts[minlen_idx,sd_idx] = 1
                continue
            
            # Update the running average
            counts[minlen_idx,sd_idx] += 1
            smthness[minlen_idx,sd_idx] = online_average(
                smthness[minlen_idx,sd_idx], 
                s,
                counts[minlen_idx,sd_idx]
            )
            
    # Save the results
    with open(f"/home/s2075466/aisplanner/results/avg_smoothness_zoom.pkl","wb") as f:
        pickle.dump(smthness,f)
        
    # Save the counts
    with open(f"/home/s2075466/aisplanner/results/avg_smoothness_counts_zoom.pkl","wb") as f:
        pickle.dump(counts,f)

if __name__ == "__main__":
    SEARCHAREA = NorthSea

    DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
    STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

    SA = SearchAgent(
            dynamic_paths=DYNAMIC_MESSAGES,
            frame=SEARCHAREA,
            static_paths=STATIC_MESSAGES,
            preprocessor=partial(speed_filter, speeds= (1,30))
        )

    ships = SA.get_all_ships(njobs=16)

    average_complexity(ships)