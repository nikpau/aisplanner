"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from matplotlib import cm, pyplot as plt
from pytsa import TargetShip, BoundingBox
from aisplanner.encounters.main import NorthSea
from pathlib import Path
from pytsa import SearchAgent
from pytsa.trajectories.rules import *
from aisstats.errchecker import COLORWHEEL_MAP, speed_filter,plot_coastline

SEARCHAREA = NorthSea

AMSTERDAM = BoundingBox(
    LATMIN=52.79,
    LATMAX=53.28,
    LONMIN=7.25,
    LONMAX=8.25
)

FISHING_GROUNDS = BoundingBox(
    LATMIN=54.5,
    LATMAX=55,
    LONMIN=7.25,
    LONMAX=8.25
)

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07*.csv"))

def plot_trajectories_on_map(ships: dict[int,TargetShip], 
                             extent: BoundingBox):
    """
    Plot all trajectories on a map.
    """
    # fig, ax = plt.subplots(figsize=(8,4))
    fig, ax = plt.subplots(figsize=(10,12))
    idx = 0
    plot_coastline(extent=extent,ax=ax)
    for ship in ships.values():
        for track in ship.tracks:
            idx += 1
            tlon = [p.lon for p in track]
            tlat = [p.lat for p in track]
            tsog = [p.SOG for p in track]
            ax.plot(
                tlon,
                tlat,
                alpha=0.5, linewidth=1, marker = "x", markersize = 4,
                c = COLORWHEEL_MAP[5]
            )
            
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"/home/s2075466/aisplanner/results/maps/trmap_raw_tshd.png",dpi=600)
    plt.savefig(f"/home/s2075466/aisplanner/results/maps/trmap_raw_tshd.pdf")
    plt.close()

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds = (1,30)),
    )

    ships = SA.get_all_ships(njobs=16)#,skip_tsplit=True)
    
    # Plot the trajectories
    plot_trajectories_on_map(ships,FISHING_GROUNDS)