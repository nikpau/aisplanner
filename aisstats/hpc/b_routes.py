"""
HPC script for calculating descriptive data
for the AIS data set
"""
from glob import glob
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from aisplanner.encounters.filter import haversine
from aisplanner.encounters.main import NorthSea
from pathlib import Path
from pytsa import SearchAgent
from pytsa.trajectories.rules import *
from aisstats.errchecker import COLORWHEEL, speed_filter, _heading_change
from pytsa.tsea.split import Splitter, AISMessage

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/data/walrus/ws/s2075466-ais/decoded/jan2020_to_jun2022').glob("2021_07_1*.csv"))
STATIC_MESSAGES = list(Path('/data/walrus/ws/s2075466-ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07_1*.csv"))

TEST_FILE_DYN = 'data/aisrecords/2021_07_01.csv'
TEST_FILE_STA = 'data/aisrecords/msgtype5/2021_07_01.csv'

def plot_simple_route(track: list[AISMessage]) -> None:
    """
    Plot the trajectory of a target vessel
    """
    lons,lats = [],[]
    latmean = np.mean([msg.lat for msg in track])
    lonmean = np.mean([msg.lon for msg in track])
    for msg in track:
        lons.append(msg.lon - lonmean)
        lats.append(msg.lat - latmean)
    
    fig, ax = plt.subplots(1,2,figsize=(8,3))
    ax: list[plt.Axes]
    
    # Plot the trajectory
    ax[0].plot(lons,lats,color=COLORWHEEL[0],ls="-", alpha = 0.9,lw=0.5)
    ax[0].scatter(lons,lats,color=COLORWHEEL[0],s=15,marker="x", alpha = 0.9,linewidths=0.5)
        
    # Set labels
    ax[0].set_xlabel('Longitude [°]')
    ax[0].set_ylabel('Latitude [°]')
    
    ax[0].set_title(f"Trajectory",fontsize=8)
    
    # Plot speed info
    cogchange = [
        abs(_heading_change(m1.COG,m2.COG)) for m1,m2 in zip(track,track[1:])
    ]
    spl = Splitter()
    speedchange = [
        spl.avg_speed(m1,m2) - spl.speed_from_position(m1,m2) for m1,m2 in zip(track,track[1:])
    ]
    
    ax[1].plot(speedchange,color=COLORWHEEL[0],ls="-", alpha = 0.9,lw=0.5)
    ax[1].scatter(range(len(speedchange)),speedchange,color=COLORWHEEL[0],s=15,marker="x", alpha = 0.9,linewidths=0.5)
    ax[1].set_xlabel("Message number")
    #ax[1].set_ylabel("Abs. COG change [°]")
    ax[1].set_ylabel("Difference in speed [kn]")
    
    # Add title
    ax[1].set_title(r"$\overline{SOG}_{m_i}^{m_{i+1}} - \widehat{SOG}_{m_i}^{m_{i+1}}$",fontsize=8)
    # ax[1].set_title("Abs. COG change\nbetween consecutive messages",fontsize=8)
    
    # Save figure
    fname = f"/home/s2075466/aisplanner/results/routes/{track[0].sender}"
    # Check if the file already exists and if so, add a number to the end
    # of the filename
    if Path(fname).exists():
        i = 1
        while Path(fname).exists():
            fname = f"/home/s2075466/aisplanner/results/routes/{track[0].sender}_{i}"
            i += 1
    
    plt.tight_layout()
    plt.savefig(f"{fname}.png",dpi=300)
    plt.savefig(f"{fname}.pdf")
    plt.close()

if __name__ == "__main__":
    SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds = (1,30)),
    )


    ships = SA.extract_all(njobs=20,skip_tsplit=True)
    
    # Sort the trajectories by standard deviation
    # of the calculated speeds
    tracks = []
    for tv in ships.values():
        for track in tv.tracks:
            if len(track) > 1 and len(track) < 200:
                tracks.append(track)
    
    def _sort_by_speed(track: list[AISMessage]) -> float:
        return min(
            Splitter().avg_speed(m1,m2) - Splitter().speed_from_position(m1,m2) for m1,m2 in zip(track,track[1:])
        )
        
    def _sort_by_length(track: list[AISMessage]) -> float:
        return len(track)

    def _sort_by_heading_change(track: list[AISMessage]) -> float:
        return sum(
            abs(_heading_change(m2.COG,m1.COG)) for m1,m2 in zip(track[len(track)//2:],track[(len(track)//2)+1:])
        ) + \
            1/(sum(
                abs(_heading_change(m2.COG,m1.COG)) for m1,m2 in zip(track[:len(track)//2],track[(len(track)//2)-1:]))\
                    +1)
    
    def _sort_by_max_distance(track: list[AISMessage]) -> float:
        return max(
            haversine(m1.lon,m1.lat,m2.lon,m2.lat) for m1,m2 in zip(track,track[1:])
        )
    
    
    tracks = sorted(tracks,key=_sort_by_speed ,reverse=True)
    
    for i in range(1000):
        t = tracks[i]
        plot_simple_route(t)