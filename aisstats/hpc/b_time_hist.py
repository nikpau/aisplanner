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

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

SEARCHAREA = NorthSea

def plot_td_histogram(speeds: np.ndarray,
                        savename: str) -> None:
    """
    Plots a histogram of the temporal differences 
    between two consecutive messages.
    """
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(speeds, bins=100, color=COLORWHEEL[0])
    ax.set_xlabel("Time difference [s]")
    ax.set_ylabel("Number of observations")
    ax.set_yscale('log')
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
                if diff > 500:
                    continue
                temp_diffs.append(diff)

    plot_td_histogram(temp_diffs, savename="/home/s2075466/aisplanner/results/td_hist_21.pdf")