"""
Module for plotting the rate of rejection for different 
values for the minimum number of observation per trajectory,
and the spatial standard deviation threshold.
"""
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import plot_trlen_vs_nmsg, speed_filter
from pathlib import Path
from pytsa import SearchAgent, TimePosition
from functools import partial
from pytsa.trajectories.rules import *

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07*.csv"))

SA = SearchAgent(
        msg12318file=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        msg5file=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )
    
# Create starting positions for the search.
# This is just the center of the search area.
center = SEARCHAREA.center
tpos = TimePosition(
    timestamp="2021-07-01", # arbitrary date
    lat=center.lat,
    lon=center.lon
)
SA.init(tpos)

ships = SA.extract_all(njobs=16,skip_filter=True)

plot_trlen_vs_nmsg(ships, savename="/home/s2075466/aisplanner/results/trlen_vs_nmsg_07_21.pdf")

SA = SearchAgent(
        msg12318file=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        msg5file=STATIC_MESSAGES,
    )
    
# Create starting positions for the search.
# This is just the center of the search area.
center = SEARCHAREA.center
tpos = TimePosition(
    timestamp="2021-07-01", # arbitrary date
    lat=center.lat,
    lon=center.lon
)
SA.init(tpos)

ships = SA.extract_all(njobs=16,skip_filter=True)

plot_trlen_vs_nmsg(ships, savename="/home/s2075466/aisplanner/results/trlen_vs_nmsg_07_21_raw.pdf")