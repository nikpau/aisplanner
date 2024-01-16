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
from pytsa.utils import heading_change
import pytsa.tsea.split as split
import numpy as np
from functools import partial
from aisplanner.encounters.main import NorthSea
import ciso8601
import pickle
COLORWHEEL = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51","#E45C3A","#732626"]
from aisstats.errchecker import haversine, speed_filter

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

def _date_transformer(datefile: Path) -> float:
    """
    Transform a date string to a float.

    Parameters
    ----------
    date : str
        The date string to transform.

    Returns
    -------
    float
        The date as a float.
    """
    return ciso8601.parse_datetime(datefile.stem.replace("_", "-"))

# MAIN MATTER ---------------------------------------------------------------    

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))

STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

if len(DYNAMIC_MESSAGES) != len(STATIC_MESSAGES):

    print(
        "Number of dynamic and static messages do not match."
        f"Dynamic: {len(DYNAMIC_MESSAGES)}, static: {len(STATIC_MESSAGES)}\n"
        "Processing only common files."
    )
    # Find the difference
    d = set([f.stem for f in DYNAMIC_MESSAGES])
    s = set([f.stem for f in STATIC_MESSAGES])
    
    # Find all files that are in d and s
    common = d.intersection(s)
    common = list(common)
    
    # Remove all files that are not in common
    DYNAMIC_MESSAGES = [f for f in DYNAMIC_MESSAGES if f.stem in common]
    STATIC_MESSAGES = [f for f in STATIC_MESSAGES if f.stem in common]
    
# Sort the files by date
DYNAMIC_MESSAGES = sorted(DYNAMIC_MESSAGES, key=_date_transformer)
STATIC_MESSAGES = sorted(STATIC_MESSAGES, key=_date_transformer)

assert all([d.stem == s.stem for d,s in zip(DYNAMIC_MESSAGES, STATIC_MESSAGES)]),\
    "Dynamic and static messages are not in the same order."

dynamic_chunks = np.array_split(DYNAMIC_MESSAGES, 80)
static_chunks = np.array_split(STATIC_MESSAGES, 80)

heading_changes = []
speed_changes = []
diff_speeds = [] # Difference between reported speed and speed calculated from positions
time_diffs = []
ddiffs = []

for dc,sc in zip(DYNAMIC_MESSAGES, STATIC_MESSAGES):
    SA = SearchAgent(
        msg12318file=dc,
        frame=SEARCHAREA,
        msg5file=sc,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )
    
    ships = SA.get_all_ships(njobs=16,skip_tsplit=True)
    l = len(ships)
    for idx, ship in enumerate(ships.values()):
        print(f"Processing ship {idx+1}/{l}")
        for track in ship.tracks:
            for i in range(1, len(track)):
                heading_changes.append(
                    heading_change(
                        track[i-1].COG, track[i].COG
                        )
                )
                speed_changes.append(abs(track[i].SOG - track[i-1].SOG))
                rspeed = split.avg_speed(track[i-1],track[i])
                cspeed = split.speed_from_position(track[i-1],track[i])
                diff_speeds.append(rspeed - cspeed)
                time_diffs.append(track[i].timestamp - track[i-1].timestamp)
                ddiffs.append(haversine(track[i].lon,track[i].lat,track[i-1].lon,track[i-1].lat))
                
squants = quantiles(speed_changes, np.linspace(0,1,1001))
hquants = quantiles(heading_changes, np.linspace(0,1,1001))
tquants = quantiles(time_diffs, np.linspace(0,1,1001))
diffquants = quantiles(diff_speeds, np.linspace(0,1,1001))
dquants = quantiles(ddiffs, np.linspace(0,1,1001))

# Save the quantiles
with open('/home/s2075466/aisplanner/results/squants.pkl', 'wb') as f:
    pickle.dump(squants, f)
with open('/home/s2075466/aisplanner/results/hquants.pkl', 'wb') as f:    
    pickle.dump(hquants, f)
with open('/home/s2075466/aisplanner/results/tquants.pkl', 'wb') as f:
    pickle.dump(tquants, f)
with open('/home/s2075466/aisplanner/results/diffquants.pkl', 'wb') as f:
    pickle.dump(diffquants, f)
with open('/home/s2075466/aisplanner/results/dquants.pkl', 'wb') as f:
    pickle.dump(dquants, f)