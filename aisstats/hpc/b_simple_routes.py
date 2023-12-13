"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from matplotlib import pyplot as plt
from pytsa.decode import filedescriptor as fd
import numpy as np
import pandas as pd
import multiprocessing as mp
from pytsa import TargetShip
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import speed_filter
from pathlib import Path
from pytsa import SearchAgent, TimePosition, ShipType
from functools import partial
from pytsa.trajectories.rules import *

from aisstats.errchecker import COLORWHEEL

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07_12.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07_12.csv"))


def plot_simple_route(tv: TargetShip, mode: str) -> None:
    """
    Plot the trajectory of a target vessel
    """
    
    # Positions and speeds
    rlons, rlats = [], []
    for track in tv.tracks:
        for msg in track:
            rlons.append(msg.lon)
            rlats.append(msg.lat)       
        # De-mean
    latmean = sum(lats) / len(lats)
    lonmean = sum(lons) / len(lons)
        # Subtract mean from all positions
    lons, lats = [], []
    for track in tv.tracks:
        for msg in track:
            msg.lat -= latmean
            msg.lon -= lonmean
            lons.append(msg.lon)
            lats.append(msg.lat)
        
        rlons.append(lons)
        rlats.append(lats)

    
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Plot the trajectory
    for i, (lo,la) in enumerate(zip(rlons,rlats)):
        ax.plot(lo,la,color=COLORWHEEL[i%len(COLORWHEEL)],ls="-", alpha = 0.9)
        ax.scatter(lo,la,color=COLORWHEEL[i%len(COLORWHEEL)],s=1)
        
    # Set labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Save figure
    plt.savefig(f"/home/s2075466/aisplanner/results/{tv.mmsi}_{mode}.png",dpi=300)
    plt.close()

if __name__ == "__main__":
    SA = SearchAgent(
        msg12318file=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        msg5file=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )
    
    # Create starting positions for the search.
    # This is just the center of the search area.
    center = SEARCHAREA.center
    tpos = TimePosition(
        timestamp="2021-07-01", # arbitrary date
        lat=center.lat,
        lon=center.lon
    )
    SA.init(tpos)

    ships = SA.get_all_ships(njobs=16)

    ExampleRecipe = Recipe(
        partial(too_few_obs, n=50),
        partial(too_small_spatial_deviation, sd=0.2)
    )
    from pytsa.trajectories import Inspector
    inspctr = Inspector(
        data=ships,
        recipe=ExampleRecipe
    )
    accepted, rejected = inspctr.inspect(njobs=1)
    
    for i in range(60,200):
        tv = rejected[i]
        plot_simple_route(tv, "rejected")