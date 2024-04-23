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
from aisstats.errchecker import COLORWHEEL,COLORWHEEL_MAP, speed_filter,plot_coastline

SEARCHAREA = NorthSea

AMSTERDAM = BoundingBox(
    LATMIN=52.79,
    LATMAX=53.28,
    LONMIN=5.5,
    LONMAX=6.5
)

FISHING_GROUNDS = BoundingBox(
    LATMIN=54.5,
    LATMAX=55,
    LONMIN=7.25,
    LONMAX=8.25
)

AABENRAA = BoundingBox(
    LATMIN=54.9,
    LATMAX=55.5,
    LONMIN=9.2,
    LONMAX=10
)

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07_1*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07_1*.csv"))

def plot_trajectories_on_map(ships: dict[int,TargetShip], 
                             extent: BoundingBox,
                             savename: str):
    """
    Plot all trajectories on a map.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    # fig, ax = plt.subplots(figsize=(10,12))
    idx = 0
    plot_coastline(
        datapath=Path("/home/s2075466/aisplanner/data/geometry"),
        extent=extent,
        ax=ax)
    for ship in ships.values():
        for track in ship.tracks:
            idx += 1
            tlon = [p.lon for p in track]
            tlat = [p.lat for p in track]
            tsog = [p.SOG for p in track]
            ax.plot(
                tlon,
                tlat,
                alpha=0.5, linewidth=0.8, marker = "x", markersize = 2,
                c = COLORWHEEL_MAP[5]
            )

    # Add Aabenraa to the plot
    ax.plot(9.421,55.044,marker="x",markersize=8,c=COLORWHEEL[6])
    ax.text(9.361,55.054,"Aabenraa",fontsize=14,c="white")
    
    # Add Haderslev to the plot
    ax.plot(9.489,55.251,marker="x",markersize=8,c=COLORWHEEL[6])
    ax.text(9.439,55.261,"Haderslev",fontsize=14,c="white")
    
    # Add Assens to the plot
    ax.plot(9.9,55.271,marker="x",markersize=8,c=COLORWHEEL[6])
    ax.text(9.91,55.271,"Assens",fontsize=14,c="white")
            
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"/home/s2075466/aisplanner/results/maps/{savename}.png",dpi=600)
    plt.savefig(f"/home/s2075466/aisplanner/results/maps/{savename}.pdf")
    plt.close()

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds = (1,30)),
    )

    # ships = SA.extract_all(njobs=16,skip_tsplit=True)
    # plot_trajectories_on_map(ships,AABENRAA,"trmap_raw")
    
    # With filter
    from pytsa.utils import haversine
    ships = SA.extract_all(njobs=16,skip_tsplit=False)
    #plot_trajectories_on_map(ships,AABENRAA,"trmap_raw_tshd")
    lens = []
    for ship in ships.values():
        for track in ship.tracks:
            lens.append(
                sum(
                    haversine(
                        track[i].lat,
                        track[i].lon,
                        track[i+1].lat,
                        track[i+1].lon
                    ) for i in range(len(track)-1)
                )
            )
    print(f"Mean track length: {np.mean(lens)}")