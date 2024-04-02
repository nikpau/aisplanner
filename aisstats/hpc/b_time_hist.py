"""
HPC script for calculating descriptive data
for the AIS data set
"""
from functools import partial
from pathlib import Path

import numpy as np
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import COLORWHEEL, speed_filter
from pathlib import Path
import ciso8601
import pandas as pd
import multiprocessing as mp
from matplotlib import pyplot as plt
from aisplanner.encounters.main import NorthSea
from pathlib import Path
from pytsa import SearchAgent
from matplotlib import patches

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07_*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07_*.csv"))

SEARCHAREA = NorthSea

def plot_td_histogram(tdiffs: np.ndarray,
                        savename: str) -> None:
    """
    Plots a histogram of the temporal differences 
    between two consecutive messages.
    """
    
    fig, ax = plt.subplots(figsize=(6,4))

    # Quantiles of time diffs
    qs = np.quantile(
        tdiffs,
        [0.99,0.95,0.90]
    )
    tdiffs500 = tdiffs[tdiffs <= 500]
    ax.hist(tdiffs500, bins=100, color=COLORWHEEL[0])


    q_labels = [
        f"1% larger {qs[0]:.2f} s",
        f"5% larger {qs[1]:.2f} s",
        f"10% larger {qs[2]:.2f} s"
    ]

    # Heading Quantiles as vertical lines
    hl11 = ax.axvline(qs[0],color=COLORWHEEL[0],label=q_labels[0],ls="--")
    hl21 = ax.axvline(qs[1],color=COLORWHEEL[0],label=q_labels[1],ls="-.")
    hl31 = ax.axvline(qs[2],color=COLORWHEEL[0],label=q_labels[2],ls=":")
    
    ax.set_xlabel("Time difference [s]")
    ax.set_xlim(0,500)
    ax.set_ylabel("Number of observations")
    ax.set_yscale('log')

    # Legend with heading
    ax.legend(handles=[hl11,hl21,hl31])    # Heading Quantiles as vertical lines

    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(savename,dpi=400)

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds = (1,30))
    )
    
    ships = SA.extract_all(njobs=16,skip_tsplit=True)
    temp_diffs = []
    for ship in ships.values():
        for track in ship.tracks:
            for i in range(1,len(track)):
                diff = track[i].timestamp - track[i-1].timestamp
                temp_diffs.append(diff)
    
    plot_td_histogram(np.array(temp_diffs), savename="/home/s2075466/aisplanner/results/td_hist_21.pdf")