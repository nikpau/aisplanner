"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from pytsa.decode import filedescriptor as fd
import numpy as np
import pandas as pd
import multiprocessing as mp
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import plot_speed_histogram
from pathlib import Path
from pytsa import SearchAgent, TimePosition, ShipType

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))


SEARCHAREA = NorthSea
SA = SearchAgent(
        msg12318file=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        msg5file=STATIC_MESSAGES
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

plot_speed_histogram(SA, savename="/home/s2075466/aisplanner/results/speed_hist_21.pdf")