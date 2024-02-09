"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from matplotlib import cm, pyplot as plt
from pytsa import TargetShip, BoundingBox
from aisplanner.encounters.main import NorthSea
from pathlib import Path
from pytsa import SearchAgent, ShipType
from pytsa.trajectories.rules import *
from aisstats.errchecker import COLORWHEEL,COLORWHEEL_MAP, speed_filter,plot_coastline

SEARCHAREA = NorthSea

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
        if ship.ship_type in ShipType.TUGTOW.value:
            for track in ship.tracks:
                idx += 1
                tlon = [p.lon for p in track]
                tlat = [p.lat for p in track]
                ax.plot(
                    tlon,
                    tlat,
                    alpha=0.5, linewidth=0.8, marker = "x", markersize = 2,
                    c = COLORWHEEL_MAP[5]
                )

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
    
    # With filter
    ships = SA.extract_all(njobs=16,skip_tsplit=False)
    plot_trajectories_on_map(ships,NorthSea,"trmap_tugtow")