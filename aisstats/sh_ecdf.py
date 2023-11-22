"""
Script to generate the ECDF and its quantiles for two variables:

    1. The difference in speed between two consecutive AIS
    messages of the same ship.
    2. The difference in course between two consecutive AIS
    messages of the same ship.

Barnard (HPC) specific script.
"""
from pathlib import Path
from pytsa import SearchAgent, TimePosition
from pytsa.tsea.search_agent import _heading_change, haversine
import numpy as np
from functools import partial
from errchecker import area_center, speed_filter
from aisplanner.encounters.main import GriddedNorthSea
import ciso8601
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

SEARCHAREA = GriddedNorthSea(nrows=1, ncols=1, utm=False).cells[0]

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
distances = []


for dc,sc in zip(DYNAMIC_MESSAGES, STATIC_MESSAGES):
    SA = SearchAgent(
        msg12318file=dc,
        frame=SEARCHAREA,
        msg5file=sc,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )
    
    # Create starting positions for the search.
    # This is just the center of the search area.
    center = area_center(SEARCHAREA)
    tpos = TimePosition(
        timestamp="2021-07-01", # arbitrary date
        lat=center.lat,
        lon=center.lon
    )
    SA.init(tpos)
    
    ships = SA.get_all_ships(njobs=8,skip_filter=True)
    l = len(ships)
    for idx, ship in enumerate(ships.values()):
        print(f"Processing ship {idx+1}/{l}")
        for track in ship.tracks:
            for i in range(1, len(track)):
                heading_changes.append(
                    _heading_change(
                        track[i-1].COG, track[i].COG
                        )
                )
                speed_changes.append(abs(track[i].SOG - track[i-1].SOG))
                distances.append(haversine(
                    track[i-1].lon, track[i-1].lat,
                    track[i].lon, track[i].lat
                ))
                
squants = quantiles(speed_changes, np.linspace(0,1,101))
hquants = quantiles(heading_changes, np.linspace(0,1,101))
dquants = quantiles(distances, np.linspace(0,1,101))

# Save the quantiles
with open('/home/s2075466/aisplanner/results/squants.pkl', 'wb') as f:
    pickle.dump(squants, f)
with open('/home/s2075466/aisplanner/results/hquants.pkl', 'wb') as f:    
    pickle.dump(hquants, f)
with open('/home/s2075466/aisplanner/results/dquants.pkl', 'wb') as f:
    pickle.dump(dquants, f)