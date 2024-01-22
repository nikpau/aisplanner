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
import pytsa.tsea.split as split
from pytsa.trajectories.rules import *
import logging
import pickle

# Set logging level to warning
logging.basicConfig(level=logging.WARNING,force = True)
SDS = np.array([0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5])
MINLENS = np.array([0,5,10,15,20,30,40,50,60,70,80,90,100,200,300,400,500])

def rej_rate_3d(ships: dict[int,TargetShip]):
    """
    Calulates the mean of the cosine of the angles
    enclosed between three consecutive messages 
    for several standard deviations.
    """
    rej_rate = np.zeros((len(MINLENS),len(SDS)))
    
    for i, minlen in enumerate(MINLENS): 
        for j, sd in enumerate(SDS):
            recipe = Recipe(
                    partial(too_few_obs,n=minlen),
                    partial(spatial_deviation,sd=sd)
                )
            inpsctr = pytsa.Inspector(
                    data=ships,
                    recipe=recipe
                )
            acc, rej = inpsctr.inspect(njobs=1)

            # Calculate rejection rate
            ntracks_rejected = 0
            ntracks_accepted = 0
            for ship in rej.values():
                ntracks_rejected += len(ship.tracks)
            for ship in acc.values():
                ntracks_accepted += len(ship.tracks)
            
            rej_rate[i,j] = ntracks_rejected/(ntracks_rejected+ntracks_accepted)
            
            
    # Save the results
    with open("/home/s2075466/aisplanner/results/3d_rej_rate.pkl","wb") as f:
        pickle.dump(rej_rate,f)

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

    rej_rate_3d(ships)