"""
Module for plotting the rate of rejection for different 
values for the minimum number of observation per trajectory,
and the spatial standard deviation threshold.
"""
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import speed_filter
from pathlib import Path
from pytsa import SearchAgent, TimePosition, TargetShip
from functools import partial
import pytsa
import pytsa.tsea.split as split
from pytsa.trajectories.rules import *
import logging
import pickle

# Set logging level to warning
logging.basicConfig(level=logging.WARNING,force = True)

def average_complexity(ships: dict[int,TargetShip]):
    """
    Calulates the mean of the cosine of the angles
    enclosed between three consecutive messages 
    for several standard deviations.
    """
    sds = np.array([0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5])
    minlens = np.array([0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500])
    avg_cosines = np.zeros((len(minlens),len(sds)))
    
    for i, minlen in enumerate(minlens): 
        for j, sd in enumerate(sds):
            recipe = Recipe(
                    partial(too_few_obs,n=minlen),
                    partial(too_small_spatial_deviation,sd=sd)
                )
            inpsctr = pytsa.Inspector(
                    data=ships,
                    recipe=recipe
                )
            acc, rej = inpsctr.inspect(njobs=1)
            del acc
            tcosines = []
            for ship in rej.values():
                for track in ship.tracks:
                    for i in range(1,len(track)-1):
                        a = track[i-1]
                        b = track[i]
                        c = track[i+1]
                        tcosines.append(
                            split.cosine_of_angle_between(a,b,c)
                        )
            avg_cosines[i,j] = np.nanmean(np.abs(tcosines))
            
    # Save the results
    with open("/home/s2075466/aisplanner/results/avg_cosines.pkl","wb") as f:
        pickle.dump(avg_cosines,f)

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )

ships = SA.get_all_ships(njobs=16,skip_filter=True)

average_complexity(ships)