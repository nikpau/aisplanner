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
import random
import pandas as pd

# Beta distribution
from scipy.stats import beta

HD_ESTIMATES = pd.DataFrame(columns=["Month","Variable","Quantile","Estimate"])

COLORWHEEL = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51","#E45C3A","#732626"]
from aisstats.errchecker import speed_filter
from pytsa.utils import haversine

def harrel_davis(x: np.ndarray, q: float, n: int):
    """
    Harrel-Davis quantile estimator.
    """
    # Get the parameters for the beta distribution
    a = (n+1) * q
    b = (n+1) * (1-q)
    
    # Get the beta distribution
    dist = beta(a,b)
    
    # Probability that some value is less or equal to (i-1)/n
    # and less or equal to i/n.
    # Get lower probability
    p0 = np.array([(i-1)/n for i in range(1,n+1)])
    # Get upper probability
    p1 = np.array([i/n for i in range(1,n+1)])
    
    # Get the weights
    w = dist.cdf(p1) - dist.cdf(p0)
    
    # Sort the data
    x = np.sort(x)
    return np.sum(x*w)


# MAIN MATTER ---------------------------------------------------------------    

# Bootstrap repetitions
B = 2000

SEARCHAREA = NorthSea

intervals = [(1,2),(1,5),(1,9),(1,13)]

OBS = {}

for interval in intervals:
    nobs = 0
    turning_rate = []
    speed_changes = []
    diff_speeds = [] # Difference between reported speed and speed calculated from positions
    time_diffs = []
    ddiffs = []
    for month in range(*interval):
        try:
            if month < 10:
                month = f"0{month}"
            DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob(f"2021_{month}*.csv"))
            STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob(f"2021_{month}*.csv"))

            SA = SearchAgent(
                dynamic_paths=DYNAMIC_MESSAGES,
                frame=SEARCHAREA,
                static_paths=STATIC_MESSAGES,
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
                        rspeed = split.Splitter.avg_speed(track[i-1],track[i])
                        cspeed = split.Splitter.speed_from_position(track[i-1],track[i])
                        diff_speeds.append(rspeed - cspeed)
                        time_diffs.append(track[i].timestamp - track[i-1].timestamp)
                        ddiffs.append(haversine(track[i].lon,track[i].lat,track[i-1].lon,track[i-1].lat))
            
        except Exception as e:
            print(e)
            print(f"Failed for month {month}")
            continue

    for q in [0.01,0.05,0.1]:
        
        hd_est_speed = []
        hd_est_turning_upper = []
        hd_est_turning_lower = []
        hd_est_ddiff_upper = []
        hd_est_ddiff_lower = []
        hd_est_time_upper = []
        hd_est_time_lower = []
        hd_est_diff_speed_upper = []
        hd_est_diff_speed_lower = []
        for b in range(B):
            
            print(f"Bootstrap repetition {b+1}/{B}")
            # Resample the data with replacement
            __speed_changes = np.random.choice(speed_changes, len(speed_changes), replace=True)
            __turning_rate = np.random.choice(turning_rate, len(turning_rate), replace=True)
            __ddiffs = np.random.choice(ddiffs, len(ddiffs), replace=True)
            __time_diffs = np.random.choice(time_diffs, len(time_diffs), replace=True)
            __diff_speeds = np.random.choice(diff_speeds, len(diff_speeds), replace=True)
        
            hd_est_speed.append(harrel_davis(__speed_changes, q, len(__speed_changes)))
            
            hd_est_turning_upper.append(harrel_davis(__turning_rate, 1-q/2, len(__turning_rate)))
            hd_est_turning_lower.append(harrel_davis(__turning_rate, q/2, len(__turning_rate)))
            
            hd_est_ddiff_upper.append(harrel_davis(__ddiffs, 1-q/2, len(__ddiffs)))
            hd_est_ddiff_lower.append(harrel_davis(__ddiffs, q/2, len(__ddiffs)))
            
            hd_est_time_upper.append(harrel_davis(__time_diffs, 1-q/2, len(__time_diffs)))
            hd_est_time_lower.append(harrel_davis(__time_diffs, q/2, len(__time_diffs)))
            
            hd_est_diff_speed_upper.append(harrel_davis(__diff_speeds, 1-q/2, len(__diff_speeds)))
            hd_est_diff_speed_lower.append(harrel_davis(__diff_speeds, q/2, len(__diff_speeds)))
            
            
        HD_ESTIMATES = pd.concat([
            HD_ESTIMATES,
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Speed",
                "Quantile": q,
                "Estimate": hd_est_speed
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Turning upper",
                "Quantile": 1-q/2,
                "Estimate": hd_est_turning_upper
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Turning lower",
                "Quantile": q/2,
                "Estimate": hd_est_turning_lower
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Distance upper",
                "Quantile": 1-q/2,
                "Estimate": hd_est_ddiff_upper
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Distance lower",
                "Quantile": q/2,
                "Estimate": hd_est_ddiff_lower
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Time upper",
                "Quantile": 1-q/2,
                "Estimate": hd_est_time_upper
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Time lower",
                "Quantile": q/2,
                "Estimate": hd_est_time_lower
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Diff Speed upper",
                "Quantile": 1-q/2,
                "Estimate": hd_est_diff_speed_upper
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Diff Speed lower",
                "Quantile": q/2,
                "Estimate": hd_est_diff_speed_lower
            })
        ])
        
HD_ESTIMATES.to_csv("/home/s2075466/aisplanner/results/hd_estimates.csv")