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

SEARCHAREA = NorthSea

intervals = [(1,2),(1,5),(1,9),(1,13)]

OBS = {}

for interval in intervals:
    nobs = 0
    for month in range(*interval):
        try:
            if month < 10:
                month = f"0{month}"
            DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob(f"2021_{month}*.csv"))
            STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob(f"2021_{month}*.csv"))


            turning_rate = []
            speed_changes = []
            diff_speeds = [] # Difference between reported speed and speed calculated from positions
            time_diffs = []
            ddiffs = []

            SA = SearchAgent(
                dynamic_paths=DYNAMIC_MESSAGES,
                frame=SEARCHAREA,
                static_paths=STATIC_MESSAGES,
                preprocessor=partial(speed_filter, speeds= (1,30))
            )

            ships = SA.extract_all(njobs=16,skip_tsplit=True)
            l = len(ships)
            for idx, ship in enumerate(ships.values()):
                # print(f"Processing ship {idx+1}/{l}")
                for track in ship.tracks:
                    nobs += len(track)
        except Exception as e:
            print(e)
            print(f"Failed for month {month}")
            continue
    OBS[interval] = nobs

print(f"Total number of observations for intervals: Obs = {OBS}")
            #         for i in range(1, len(track)):
            #             turning_rate.append(
            #                 heading_change(
            #                     track[i-1].COG, track[i].COG
            #                     ) / (track[i].timestamp - track[i-1].timestamp)
            #             )
            #             speed_changes.append(abs(track[i].SOG - track[i-1].SOG))
            #             rspeed = split.avg_speed(track[i-1],track[i])
            #             cspeed = split.speed_from_position(track[i-1],track[i])
            #             diff_speeds.append(rspeed - cspeed)
            #             time_diffs.append(track[i].timestamp - track[i-1].timestamp)
            #             ddiffs.append(haversine(track[i].lon,track[i].lat,track[i-1].lon,track[i-1].lat))
                            
            # squants = quantiles(speed_changes, np.linspace(0,1,10001))
            # trquants = quantiles(turning_rate, np.linspace(0,1,10001))
            # tquants = quantiles(time_diffs, np.linspace(0,1,10001))
            # diffquants = quantiles(diff_speeds, np.linspace(0,1,10001))
            # dquants = quantiles(ddiffs, np.linspace(0,1,10001))

            # # Save the quantiles
            # with open(f'/home/s2075466/aisplanner/results/squants_{interval[0]}-{interval[1]-1}.pkl', 'wb') as f:
            #     pickle.dump(squants, f)
            # with open(f'/home/s2075466/aisplanner/results/trquants_{interval[0]}-{interval[1]-1}.pkl', 'wb') as f:    
            #     pickle.dump(trquants, f)
            # with open(f'/home/s2075466/aisplanner/results/tquants_{interval[0]}-{interval[1]-1}.pkl', 'wb') as f:
            #     pickle.dump(tquants, f)
            # with open(f'/home/s2075466/aisplanner/results/diffquants_{interval[0]}-{interval[1]-1}.pkl', 'wb') as f:
            #     pickle.dump(diffquants, f)
            # with open(f'/home/s2075466/aisplanner/results/dquants_{interval[0]}-{interval[1]-1}.pkl', 'wb') as f:
            #     pickle.dump(dquants, f)
        # except Exception as e:
        #     print(e)
        #     print(f"Failed for month {month}")
        #     continue