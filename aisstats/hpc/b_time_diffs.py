"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from aisplanner.encounters.main import NorthSea
from pathlib import Path
from pytsa import SearchAgent
from pytsa.trajectories.rules import *
from aisstats.errchecker import COLORWHEEL, speed_filter

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

def plot_time_diffs(sa:SearchAgent):
    """
    Plot the time difference between two consecutive
    messages.
    """
    f, ax = plt.subplots(1,1,figsize=(6,4))
    ax: plt.Axes
    ships = sa.get_all_ships(njobs=16,skip_tsplit=True)
    time_diffs = []
    it = 0
    maxlen = len(ships)
    for ship in ships.values():
        it +=1
        print(f"Working on ship {it}/{maxlen}")
        for track in ship.tracks:
            for i in range(1,len(track)):
                time_diffs.append(track[i].timestamp - track[i-1].timestamp)
    
    # Quantiles of time diffs
    qs = np.quantile(
        time_diffs,
        [0.99,0.95,0.90]
    )

    # Boxplot of time diffs
    ax.boxplot(
        time_diffs,
        vert=False,
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor=COLORWHEEL[0], color=COLORWHEEL[0]),
        medianprops=dict(color=COLORWHEEL[1]),
        whiskerprops=dict(color=COLORWHEEL[1]),
        capprops=dict(color=COLORWHEEL[1]),
        flierprops=dict(color=COLORWHEEL[1], markeredgecolor=COLORWHEEL[1])
    )
    # Remove y ticks
    ax.set_yticks([])
    
    # Legend with heading
    ax.legend(handles=[
        patches.Patch(color=COLORWHEEL[0],label=f"1% larger than {qs[0]:.2f} s"),
        patches.Patch(color=COLORWHEEL[0],label=f"5% larger than {qs[1]:.2f} s"),
        patches.Patch(color=COLORWHEEL[0],label=f"10% larger than {qs[2]:.2f} s")
    ])
    
    ax.set_xlabel("Time difference [s]")
    
    ax.set_title(
        "Time difference between two consecutive messages",fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig("/home/s2075466/aisplanner/results/time_diffs.pdf")

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds = (1,30)),
    )
    plot_time_diffs(SA)