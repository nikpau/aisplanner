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

sd = float(sys.argv[1])
minlen = int(sys.argv[2])

# Set logging level to warning
SDS = np.array([0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5])
MINLENS = np.array([0,5,10,15,20,30,40,50,60,80,100,200,400,500])

def average_complexity(ships: dict[int,TargetShip]):
    """
    Calulates the mean of the cosine of the angles
    enclosed between three consecutive messages 
    for several standard deviations.
    """
    recipe = Recipe(
            partial(too_few_obs,n=minlen),
            partial(spatial_deviation,sd=sd)
        )
    inpsctr = pytsa.Inspector(
            data=ships,
            recipe=recipe
        )
    print("Inspecting...")
    acc, rej = inpsctr.inspect(njobs=1)
    del acc
    smthness = []
    for ship in rej.values():
        for track in ship.tracks:
            smthness.append(inspect.average_smoothness(track))
            
    # Save the results
    with open(f"/home/s2075466/aisplanner/results/avg_smoothness_{sd}_{minlen}.pkl","wb") as f:
        pickle.dump(np.mean(smthness),f)

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

    ships = SA.get_all_ships(njobs=8)

    average_complexity(ships)