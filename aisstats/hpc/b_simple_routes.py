"""
HPC script for calculating descriptive data
for the AIS data set
"""
from glob import glob
from pathlib import Path
from matplotlib import pyplot as plt
from pytsa.decode import filedescriptor as fd
import numpy as np
import pandas as pd
import multiprocessing as mp
from pytsa import TargetShip, BoundingBox
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import speed_filter
from pathlib import Path
from pytsa import SearchAgent, TimePosition, ShipType
from functools import partial
from pytsa.trajectories.rules import *
from aisplanner.encounters.encmap import plot_coastline
from aisstats.errchecker import COLORWHEEL_MAP
import geopandas as gpd

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07_12.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07_12.csv"))

def plot_simple_route(tv: TargetShip, mode: str) -> None:
    """
    Plot the trajectory of a target vessel
    """
    lons,lats = [],[]
    for track in tv.tracks:
        for msg in track:
            lons.append(msg.lon)
            lats.append(msg.lat)
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    plot_coastline(SEARCHAREA,ax)
    # Plot the trajectory
    ax.plot(lons,lats,color=COLORWHEEL_MAP[0],ls="-", alpha = 0.9)
    ax.scatter(lons,lats,color=COLORWHEEL_MAP[0],s=10)
        
    # Set labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Set y and x limits to 1.5 times the max and min
    # of the trajectory
    ax.set_ylim(min(lats)-0.001,max(lats)+0.001)
    ax.set_xlim(min(lons)-0.001,max(lons)+0.001)
    
    # Save figure
    plt.savefig(f"/home/s2075466/aisplanner/results/{tv.mmsi}_{mode}.png",dpi=300)
    plt.close()

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
    )

    ships = SA.get_all_ships(njobs=16,skip_tsplit=True)
    
    tvs = list(ships.values())
    for i in range(60,200):
        tv = tvs[i]
        plot_simple_route(tv, "rejected")