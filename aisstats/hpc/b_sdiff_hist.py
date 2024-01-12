"""
HPC script for calculating descriptive data
for the AIS data set
"""
from functools import partial
from pathlib import Path

import numpy as np
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import COLORWHEEL, speed_filter, TEST_FILE_DYN, TEST_FILE_STA
from pathlib import Path
from pytsa.tsea.split import speed_from_position, avg_speed
import pandas as pd
import multiprocessing as mp
from matplotlib import pyplot as plt
from aisplanner.encounters.main import NorthSea
from pathlib import Path
from pytsa import SearchAgent

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

SEARCHAREA = NorthSea


def plot_sdiff_histogram(sa: SearchAgent):
    """
    Plots a histogram of the differences in reported
    speed and calculated speed.
    between two consecutive messages.
    """
    f, ax = plt.subplots(1,1,figsize=(8,5))

    ships = sa.get_all_ships(skip_tsplit=True)
    diffs = []
    it = 0
    maxlen = len(ships)
    for ship in ships.values():
        it +=1
        print(f"Working on ship {it}/{maxlen}")
        for track in ship.tracks:
            for i in range(1,len(track)):
                _chspeed = avg_speed(track[i-1],track[i]) - speed_from_position(track[i-1],track[i])
                diffs.append(_chspeed)

    # Quantiles of heading changes
    h_qs = np.quantile(
        diffs,
        [0.005,0.995,0.025,0.975,0.05,0.95]
    )
    q_labels_h = [
        f"99% within [{h_qs[0]:.2f} kn,{h_qs[1]:.2f} kn]",
        f"95% within [{h_qs[2]:.2f} kn,{h_qs[3]:.2f} kn]",
        f"90% within [{h_qs[4]:.2f} kn,{h_qs[5]:.2f} kn]"
    ]
    # Heading Quantiles as vertical lines
    hl11 = ax.axvline(h_qs[0],color=COLORWHEEL[0],label=q_labels_h[0],ls="--")
    hl12 = ax.axvline(h_qs[1],color=COLORWHEEL[0],label=q_labels_h[0],ls="--")
    hl21 = ax.axvline(h_qs[2],color=COLORWHEEL[0],label=q_labels_h[1],ls="-.")
    hl22 = ax.axvline(h_qs[3],color=COLORWHEEL[0],label=q_labels_h[1],ls="-.")
    hl31 = ax.axvline(h_qs[4],color=COLORWHEEL[0],label=q_labels_h[2],ls=":")
    hl32 = ax.axvline(h_qs[5],color=COLORWHEEL[0],label=q_labels_h[2],ls=":")
    
    # Histogram of heading changes
    ax.hist(
        diffs,
        bins=np.append(-np.inf,np.append(np.linspace(-30,30,100), np.inf)),
        density=True,
        alpha=0.8,
        color=COLORWHEEL[0]
    )
    # ax.set_xlim(-30,30)
    
    # Legend with heading
    ax.legend(handles=[hl11,hl21,hl31])
    ax.set_xlabel(r"$\overline{SOG}_{m_i}^{m_{i+1}} - \widehat{SOG}_{m_i}^{m_{i+1}}$")
    ax.set_ylabel("Density")
    ax.set_title(
        "Difference between reported and calculated speed\n"
    )
    plt.tight_layout()
    plt.savefig(f"/home/s2075466/aisplanner/results/diffs_rep_calc_speed.pdf")
    #plt.savefig(f"results/diffs_rep_calc_speed.pdf")
    plt.close()
    

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths= DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds = (1,30))
    )

    plot_sdiff_histogram(sa=SA)