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


DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob(f"2021_07_01.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob(f"2021_07_01.csv"))


# Turning rate for each ship type
turning_rate = {s: [] for s in ShipType}
speed_changes = {s: [] for s in ShipType}
diff_speeds = {s: [] for s in ShipType} # Difference between reported speed and speed calculated from positions
time_diffs = {s: [] for s in ShipType}
ddiffs = {s: [] for s in ShipType}

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
    _shiptype = ShipType.from_value(ship.ship_type)
    for track in ship.tracks:
        for i in range(1, len(track)):
            td = track[i].timestamp - track[i-1].timestamp
            if td >= 371: # 95th percentile of time differences
                continue
            tr = heading_change(track[i-1].COG, track[i].COG) / (td)
            turning_rate[_shiptype].append(tr)
            speed_changes[_shiptype].append(abs(track[i].SOG - track[i-1].SOG))
            rspeed = split.Splitter.avg_speed(track[i-1],track[i])
            cspeed = split.Splitter.speed_from_position(track[i-1],track[i])
            diff_speeds[_shiptype].append(rspeed - cspeed)
            time_diffs[_shiptype].append(td)
            ddiffs[_shiptype].append(haversine(track[i].lon,track[i].lat,track[i-1].lon,track[i-1].lat))

_ls = np.linspace(0,1,1001)        
for s in ShipType:
    with open(f'/home/s2075466/aisplanner/results/trquants_21_{s.name}.pkl', 'wb') as f:    
        pickle.dump(quantiles(turning_rate[s],_ls), f)
    with open(f'/home/s2075466/aisplanner/results/squants_21_{s.name}.pkl', 'wb') as f:
        pickle.dump(quantiles(diff_speeds,_ls), f)
    with open(f'/home/s2075466/aisplanner/results/tquants_21_{s.name}.pkl', 'wb') as f:
        pickle.dump(quantiles(time_diffs,_ls) , f)
    with open(f'/home/s2075466/aisplanner/results/diffquants_21_{s.name}.pkl', 'wb') as f:
        pickle.dump(quantiles(time_diffs,_ls), f)
    with open(f'/home/s2075466/aisplanner/results/dquants_21_{s.name}.pkl', 'wb') as f:
        pickle.dump(quantiles(ddiffs,_ls), f)