"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from aisplanner.encounters.main import NorthSea
from pathlib import Path
from pytsa import SearchAgent
from pytsa.trajectories.rules import *
from aisstats.errchecker import COLORWHEEL, speed_filter, _heading_change

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

TEST_FILE_DYN = 'data/aisrecords/2021_07_01.csv'
TEST_FILE_STA = 'data/aisrecords/msgtype5/2021_07_01.csv'


def plot_heading_and_speed_changes(sa: SearchAgent):
    """
    Plot the changes in heading and speed between
    two consecutive messages.
    """
    f, ax = plt.subplots(1,2,figsize=(16,5))
    ax: list[plt.Axes]

    ships = sa.get_all_ships(skip_tsplit=True)
    heading_changes = []
    speed_changes = []
    it = 0
    maxlen = len(ships)
    for ship in ships.values():
        it +=1
        print(f"Working on ship {it}/{maxlen}")
        for track in ship.tracks:
            for i in range(1,len(track)):
                _chheading = _heading_change(
                    track[i-1].COG,
                    track[i].COG
                )
                _chspeed = abs(track[i].SOG - track[i-1].SOG)
                heading_changes.append(_chheading)
                speed_changes.append(_chspeed)
                
    # Quantiles of heading changes
    h_qs = np.quantile(
        heading_changes,
        [0.005,0.995,0.025,0.975,0.05,0.95]
    )
    q_labels_h = [
        f"99% within [{h_qs[0]:.2f}°,{h_qs[1]:.2f}°]",
        f"95% within [{h_qs[2]:.2f}°,{h_qs[3]:.2f}°]",
        f"90% within [{h_qs[4]:.2f}°,{h_qs[5]:.2f}°]"
    ]
    # Heading Quantiles as vertical lines
    hl11 = ax[0].axvline(h_qs[0],color=COLORWHEEL[0],label=q_labels_h[0],ls="--")
    hl12 = ax[0].axvline(h_qs[1],color=COLORWHEEL[0],label=q_labels_h[0],ls="--")
    hl21 = ax[0].axvline(h_qs[2],color=COLORWHEEL[0],label=q_labels_h[1],ls="-.")
    hl22 = ax[0].axvline(h_qs[3],color=COLORWHEEL[0],label=q_labels_h[1],ls="-.")
    hl31 = ax[0].axvline(h_qs[4],color=COLORWHEEL[0],label=q_labels_h[2],ls=":")
    hl32 = ax[0].axvline(h_qs[5],color=COLORWHEEL[0],label=q_labels_h[2],ls=":")
    
    # Histogram of heading changes
    ax[0].hist(
        heading_changes,
        bins=100,
        density=True,
        alpha=0.8,
        color=COLORWHEEL[0]
    )
    
    # Quantiles of speed changes
    s_qs = np.quantile(
        speed_changes,
        [0.99,0.95,0.90]
    )
    
    q_labels_s = [
        f"99% smaller than {s_qs[0]:.2f} kn",
        f"95% smaller than {s_qs[1]:.2f} kn",
        f"90% smaller than {s_qs[2]:.2f} kn"
    ]
    
    # Speed Quantiles as vertical lines
    sl1 = ax[1].axvline(s_qs[0],color=COLORWHEEL[0],label=q_labels_s[0],ls="--")
    sl2 = ax[1].axvline(s_qs[1],color=COLORWHEEL[0],label=q_labels_s[1],ls="-.")
    sl3 = ax[1].axvline(s_qs[2],color=COLORWHEEL[0],label=q_labels_s[2],ls=":")
    
    # Histogram of speed changes
    ax[1].hist(
        speed_changes,
        bins=200,
        density=True,
        alpha=0.8,
        color=COLORWHEEL[0]
    )
    
    # Legend with heading
    ax[0].legend(handles=[hl11,hl21,hl31])
    ax[0].set_xlabel("Change in heading [°]")
    ax[0].set_ylabel("Density")
    ax[0].set_title(
        "Change in heading between two consecutive messages",fontsize=10
    )
    
    ax[1].legend(handles=[sl1,sl2,sl3])
    ax[1].set_xlabel("Absolute change in speed [knots]")
    ax[1].set_ylabel("Density")
    ax[1].set_title(
        "Change in speed between two consecutive messages",fontsize=10
    )
    ax[1].set_xlim(-0.2,4)
    
    plt.tight_layout()
    plt.savefig("aisstats/out/heading_speed_changes_all.pdf")
    plt.savefig(f"/home/s2075466/aisplanner/results/heading_speed_changes_all.pdf")

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds = (1,30)),
    )


    plot_heading_and_speed_changes(SA)