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
from pytsa.structs import ShipType
import pytsa.tsea.split as split
import numpy as np
from functools import partial
from aisplanner.encounters.main import NorthSea
import pickle
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

# MAIN MATTER ---------------------------------------------------------------    

SEARCHAREA = NorthSea


DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob(f"2021_04_01.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob(f"2021_04_01.csv"))

LENGTH_BINS = np.append(np.linspace(0,200,9),np.inf) 

# Values for each length bin
turning_rate = {f"{b0}-{b1}": [] for b0,b1 in zip(LENGTH_BINS,LENGTH_BINS[1:])}
speed_changes = {f"{b0}-{b1}": [] for b0,b1 in zip(LENGTH_BINS,LENGTH_BINS[1:])}
diff_speeds = {f"{b0}-{b1}": [] for b0,b1 in zip(LENGTH_BINS,LENGTH_BINS[1:])} # Difference between reported speed and speed calculated from positions
time_diffs = {f"{b0}-{b1}": [] for b0,b1 in zip(LENGTH_BINS,LENGTH_BINS[1:])}
ddiffs = {f"{b0}-{b1}": [] for b0,b1 in zip(LENGTH_BINS,LENGTH_BINS[1:])}

def get_bin_from_length(length):
    for b0,b1 in zip(LENGTH_BINS,LENGTH_BINS[1:]):
        if b0 <= length < b1:
            return f"{b0}-{b1}"
    return f"{b0}-{b1}"

SA = SearchAgent(
    dynamic_paths=DYNAMIC_MESSAGES,
    frame=SEARCHAREA,
    static_paths=STATIC_MESSAGES,
    preprocessor=partial(speed_filter, speeds= (1,30))
)

ships = SA.extract_all(njobs=48,skip_tsplit=True)
l = len(ships)
for idx, ship in enumerate(ships.values()):
    print(f"Processing ship {idx+1}/{l}")
    if not any(ship.length):
        continue
    else:
        thebin = get_bin_from_length(ship.length[0])
        
    for track in ship.tracks:
        for i in range(1, len(track)):
            td = track[i].timestamp - track[i-1].timestamp
            if td >= 371: # 95th percentile of time differences
                continue
            tr = heading_change(track[i-1].COG, track[i].COG) / (td)
            turning_rate[thebin].append(tr)
            speed_changes[thebin].append(abs(track[i].SOG - track[i-1].SOG))
            rspeed = split.Splitter.avg_speed(track[i-1],track[i])
            cspeed = split.Splitter.speed_from_position(track[i-1],track[i])
            diff_speeds[thebin].append(rspeed - cspeed)
            time_diffs[thebin].append(td)
            ddiffs[thebin].append(haversine(track[i].lon,track[i].lat,track[i-1].lon,track[i-1].lat))

_ls = np.linspace(0,1,1001)        
for L1, L2 in zip(LENGTH_BINS,LENGTH_BINS[1:]):
    with open(f'/home/s2075466/aisplanner/results/trquants_21_{L1}-{L2}.pkl', 'wb') as f:    
        pickle.dump(quantiles(turning_rate[f"{L1}-{L2}"],_ls), f)
    with open(f'/home/s2075466/aisplanner/results/squants_21_{L1}-{L2}.pkl', 'wb') as f:
        pickle.dump(quantiles(speed_changes[f"{L1}-{L2}"],_ls), f)
    with open(f'/home/s2075466/aisplanner/results/tquants_21_{L1}-{L2}.pkl', 'wb') as f:
        pickle.dump(quantiles(time_diffs[f"{L1}-{L2}"],_ls) , f)
    with open(f'/home/s2075466/aisplanner/results/diffquants_21_{L1}-{L2}.pkl', 'wb') as f:
        pickle.dump(quantiles(diff_speeds[f"{L1}-{L2}"],_ls), f)
    with open(f'/home/s2075466/aisplanner/results/dquants_21_{L1}-{L2}.pkl', 'wb') as f:
        pickle.dump(quantiles(ddiffs[f"{L1}-{L2}"],_ls), f)