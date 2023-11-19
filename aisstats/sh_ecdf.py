"""
Script to generate the ECDF and its quantiles for two variables:

    1. The difference in speed between two consecutive AIS
    messages of the same ship.
    2. The difference in course between two consecutive AIS
    messages of the same ship.

Barnard (HPC) specific script.
"""
from pathlib import Path
from pytsa import SearchAgent
from pytsa.tsea.search_agent import _heading_change 
import numpy as np
from functools import partial
from errchecker import speed_filter
from aisplanner.encounters.main import GriddedNorthSea
import pickle

def quantiles(data, quantiles):
    """
    Calculate the quantiles of a given dataset.

    Parameters
    ----------
    data : array
        The data to calculate the quantiles for.
    quantiles : array
        The quantiles to calculate.

    Returns
    -------
    array
        The quantiles of the data.
    """
    quantiles = np.quantile(data, quantiles)
    return quantiles

# MAIN MATTER ---------------------------------------------------------------    

SEARCHAREA = GriddedNorthSea(nrows=1, ncols=1, utm=False).cells[0]

DYNAMIC_MESSAGES = Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv")
STATIC_MESSAGES = Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv")
print(len(list(DYNAMIC_MESSAGES)))
print(len(list(STATIC_MESSAGES)))

dynamic_chunks = np.array_split(list(DYNAMIC_MESSAGES), 30)
static_chunks = np.array_split(list(STATIC_MESSAGES), 30)

heading_changes = []
speed_changes = []

for dc,sc in zip(dynamic_chunks, static_chunks):
    SA = SearchAgent(
        msg12318file=list(dc),
        frame=SEARCHAREA,
        msg5file=list(sc),
        preprocessor=partial(speed_filter, speeds= (1,30))
    )
    ships = SA.get_all_ships(njobs=24,skip_filter=True)
    for ship in ships.values():
        for track in ship.tracks:
            for i in range(1, len(track)):
                heading_changes.append(
                    _heading_change(
                        track[i-1].COG, track[i].COG
                        )
                )
                speed_changes.append(abs(track[i].SOG - track[i-1].SOG))
                
squants = quantiles(speed_changes, np.linspace(0,1,101))
hquants = quantiles(heading_changes, np.linspace(0,1,101))

# Save the quantiles
with open('/home/s2075466/aisplanner/results/squants.pkl', 'wb') as f:
    pickle.dump(squants, f)
with open('/home/s2075466/aisplanner/results/hquants.pkl', 'wb') as f:    
    pickle.dump(hquants, f)