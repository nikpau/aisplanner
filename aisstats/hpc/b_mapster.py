"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from matplotlib import cm, pyplot as plt
from matplotlib.lines import Line2D
from pytsa import TargetShip, BoundingBox
from aisplanner.encounters.main import NorthSea
from pathlib import Path
from pytsa import SearchAgent
from pytsa.trajectories.rules import *
from pytsa.structs import LENGTH_BINS
from pytsa.tsea.split import get_length_bin
from aisstats.errchecker import COLORWHEEL,COLORWHEEL_MAP, speed_filter,plot_coastline
import matplotlib as mpl

cmap = mpl.colormaps.get_cmap('seismic').resampled(10)
colors = cmap(np.arange(0, cmap.N))

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

COPENHAGEN = BoundingBox(
    LATMIN=55.52,
    LATMAX=56.16,
    LONMIN=12.15,
    LONMAX=13.13
)

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_08_1*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_08_1*.csv"))

def plot_trajectories_on_map(ships: dict[int,TargetShip], 
                             extent: BoundingBox,
                             savename: str):
    """
    Plot all trajectories on a map.
    """
    fig, ax = plt.subplots(figsize=(10,6))
    # fig, ax = plt.subplots(figsize=(10,12))
    idx = 0
    plot_coastline(
        datapath=Path("/home/s2075466/aisplanner/data/geometry"),
        extent=extent,
        ax=ax
    )
    for ship in ships.values():
        if ship.length is None:
            continue
        for k, (l1, l2) in enumerate(zip(LENGTH_BINS,LENGTH_BINS[1:])):
            if l1 <= ship.length < l2:
                break
        for track in ship.tracks:
            idx += 1
            tlon = [p.lon for p in track]
            tlat = [p.lat for p in track]
            ax.plot(
                tlon,
                tlat,
                alpha=0.4, linewidth=0.4, marker = "x", markersize = 1,
                c = colors[k]
            )

    # Add Copenhagen to the plot
    ax.text(12.5,55.68,"Copenhagen",fontsize=14,c="white")
    
    # Add Malmö to the plot
    ax.text(13,55.6,"Malmö",fontsize=14,c="white")
    
    # Add Helsingborg to the plot
    ax.text(12.7,56.05,"Helsingborg",fontsize=14,c="white")
            
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    
    bins_pairs = [(LENGTH_BINS[i], LENGTH_BINS[i+1]) for i in range(len(LENGTH_BINS)-1)]

    # Step 3: Generate a list of custom Line2D objects for the legend
    custom_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(bins_pairs))]
    labels = []
    for pair in bins_pairs:
        if pair[1] == np.inf:
            labels.append(f'{int(pair[0])}-$\infty$')
        else:
            labels.append(f'{int(pair[0])}-{int(pair[1])}')
    ax.legend(custom_lines, labels, title='Ship lengths [m]', loc='upper right', fontsize = 12)
    
    plt.tight_layout()
    plt.savefig(f"/home/s2075466/aisplanner/results/maps/{savename}.png",dpi=300)
    plt.savefig(f"/home/s2075466/aisplanner/results/maps/{savename}.pdf")
    plt.close()

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds = (1,30)),
    )

    ships = SA.extract_all(njobs=30,skip_tsplit=True, alpha=0.03)
    plot_trajectories_on_map(ships,COPENHAGEN,"trmap_copenhagen_raw")
    
    # With filter
    from pytsa.utils import haversine
    ships = SA.extract_all(njobs=30,skip_tsplit=False)
    plot_trajectories_on_map(ships,COPENHAGEN,"trmap_copenhagen_raw_tshd")
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
    print(f"Median track length: {np.median(lens)}")