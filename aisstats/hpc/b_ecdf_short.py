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

# MAIN MATTER ---------------------------------------------------------------    

def date_list(days: int = 7) -> list[Path]:
    """
    Select `days` consectutive dates
    in the month of July and return
    them in a list.
    """
    dates = []
    for day in range(1,days+1):
        if day < 10:
            day = f"0{day}"
        dates.append(f"2021_07_{day}")
    return dates

for days, name in zip([1,7,14],["one_day","one_week","two_week"]):
    
    SEARCHAREA = NorthSea

    DYNAMIC_MESSAGES = Path('/home/s2075466/ais/decoded/jan2020_to_jun2022')
    STATIC_MESSAGES = Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5')

    turning_rate = []
    speed_changes = []
    diff_speeds = [] # Difference between reported speed and speed calculated from positions
    time_diffs = []
    ddiffs = []

    dates = date_list(days)
    dyn = [DYNAMIC_MESSAGES / f"{day}.csv" for day in dates]
    sta = [STATIC_MESSAGES / f"{day}.csv" for day in dates]

    SA = SearchAgent(
        dynamic_paths=dyn,
        frame=SEARCHAREA,
        static_paths=sta,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )

    ships = SA.extract_all(njobs=16,skip_tsplit=True)
    l = len(ships)
    for idx, ship in enumerate(ships.values()):
        print(f"Processing ship {idx+1}/{l}")
        for track in ship.tracks:
            for i in range(1, len(track)):
                turning_rate.append(
                    heading_change(
                        track[i-1].COG, track[i].COG
                        ) / (track[i].timestamp - track[i-1].timestamp)
                )
                speed_changes.append(abs(track[i].SOG - track[i-1].SOG))
                rspeed = split.avg_speed(track[i-1],track[i])
                cspeed = split.speed_from_position(track[i-1],track[i])
                diff_speeds.append(rspeed - cspeed)
                time_diffs.append(track[i].timestamp - track[i-1].timestamp)
                ddiffs.append(haversine(track[i].lon,track[i].lat,track[i-1].lon,track[i-1].lat))
                    
    squants = quantiles(speed_changes, np.linspace(0,1,10001))
    trquants = quantiles(turning_rate, np.linspace(0,1,10001))
    tquants = quantiles(time_diffs, np.linspace(0,1,10001))
    diffquants = quantiles(diff_speeds, np.linspace(0,1,10001))
    dquants = quantiles(ddiffs, np.linspace(0,1,10001))

    # Save the quantiles
    suffix = name
    with open(f'/home/s2075466/aisplanner/results/squants_{suffix}.pkl', 'wb') as f:
        pickle.dump(squants, f)
    with open(f'/home/s2075466/aisplanner/results/trquants_{suffix}.pkl', 'wb') as f:    
        pickle.dump(trquants, f)
    with open(f'/home/s2075466/aisplanner/results/tquants_{suffix}.pkl', 'wb') as f:
        pickle.dump(tquants, f)
    with open(f'/home/s2075466/aisplanner/results/diffquants_{suffix}.pkl', 'wb') as f:
        pickle.dump(diffquants, f)
    with open(f'/home/s2075466/aisplanner/results/dquants_{suffix}.pkl', 'wb') as f:
        pickle.dump(dquants, f)