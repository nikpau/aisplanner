"""
Script to generate the ECDF and its quantiles for two variables:

    1. The difference in speed between two consecutive AIS
    messages of the same ship.
    2. The difference in course between two consecutive AIS
    messages of the same ship.

Barnard (HPC) specific script.
"""
from multiprocessing import Pool, shared_memory
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

def bootstrap_replication(shm_names, lengths, q):
    # Reconstruct numpy arrays from shared memory
    shm_speed_changes = shared_memory.SharedMemory(name=shm_names['speed_changes'])
    speed_changes = np.ndarray(lengths['speed_changes'], dtype=np.float64, buffer=shm_speed_changes.buf)
    
    shm_turning_rate = shared_memory.SharedMemory(name=shm_names['turning_rate'])
    turning_rate = np.ndarray(lengths['turning_rate'], dtype=np.float64, buffer=shm_turning_rate.buf)
    
    shm_ddiffs = shared_memory.SharedMemory(name=shm_names['ddiffs'])
    ddiffs = np.ndarray(lengths['ddiffs'], dtype=np.float64, buffer=shm_ddiffs.buf)
    
    shm_time_diffs = shared_memory.SharedMemory(name=shm_names['time_diffs'])
    time_diffs = np.ndarray(lengths['time_diffs'], dtype=np.float64, buffer=shm_time_diffs.buf)
    
    shm_diff_speeds = shared_memory.SharedMemory(name=shm_names['diff_speeds'])
    diff_speeds = np.ndarray(lengths['diff_speeds'], dtype=np.float64, buffer=shm_diff_speeds.buf)

    np.random.seed()  # Ensure each process has a different seed
    
    # Resample the data with replacement
    __speed_changes = np.random.choice(speed_changes, len(speed_changes), replace=True)
    __turning_rate = np.random.choice(turning_rate, len(turning_rate), replace=True)
    __ddiffs = np.random.choice(ddiffs, len(ddiffs), replace=True)
    __time_diffs = np.random.choice(time_diffs, len(time_diffs), replace=True)
    __diff_speeds = np.random.choice(diff_speeds, len(diff_speeds), replace=True)

    print("Calculating Harrel-Davis estimator")
    result = {
        'speed': harrel_davis(__speed_changes, q, len(__speed_changes)),
        'turning_upper': harrel_davis(__turning_rate, 1-q/2, len(__turning_rate)),
        'turning_lower': harrel_davis(__turning_rate, q/2, len(__turning_rate)),
        'ddiff_upper': harrel_davis(__ddiffs, 1-q/2, len(__ddiffs)),
        'ddiff_lower': harrel_davis(__ddiffs, q/2, len(__ddiffs)),
        'time_upper': harrel_davis(__time_diffs, 1-q/2, len(__time_diffs)),
        'time_lower': harrel_davis(__time_diffs, q/2, len(__time_diffs)),
        'diff_speed_upper': harrel_davis(__diff_speeds, 1-q/2, len(__diff_speeds)),
        'diff_speed_lower': harrel_davis(__diff_speeds, q/2, len(__diff_speeds))
    }

    # Clean up by closing the shared memory blocks
    shm_speed_changes.close()
    shm_turning_rate.close()
    shm_ddiffs.close()
    shm_time_diffs.close()
    shm_diff_speeds.close()

    return result

def multiprocessing_bootstrap(B, shm_names, q, lengths):
    # Setup multiprocessing pool
    with Pool() as pool:
        results = pool.starmap(
            bootstrap_replication,
            [(shm_names, lengths, q) for b in range(B)]
        )
    
    # Unpack results
    hd_est_speed = [res['speed'] for res in results]
    hd_est_turning_upper = [res['turning_upper'] for res in results]
    hd_est_turning_lower = [res['turning_lower'] for res in results]
    hd_est_ddiff_upper = [res['ddiff_upper'] for res in results]
    hd_est_ddiff_lower = [res['ddiff_lower'] for res in results]
    hd_est_time_upper = [res['time_upper'] for res in results]
    hd_est_time_lower = [res['time_lower'] for res in results]
    hd_est_diff_speed_upper = [res['diff_speed_upper'] for res in results]
    hd_est_diff_speed_lower = [res['diff_speed_lower'] for res in results]

    return {
        'hd_est_speed': hd_est_speed,
        'hd_est_turning_upper': hd_est_turning_upper,
        'hd_est_turning_lower': hd_est_turning_lower,
        'hd_est_ddiff_upper': hd_est_ddiff_upper,
        'hd_est_ddiff_lower': hd_est_ddiff_lower,
        'hd_est_time_upper': hd_est_time_upper,
        'hd_est_time_lower': hd_est_time_lower,
        'hd_est_diff_speed_upper': hd_est_diff_speed_upper,
        'hd_est_diff_speed_lower': hd_est_diff_speed_lower,
    }

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
        
    speed_changes = np.array(speed_changes)
    turning_rate = np.array(turning_rate)
    diff_speeds = np.array(diff_speeds)
    time_diffs = np.array(time_diffs)
    ddiffs = np.array(ddiffs)
    
    shm_speed_changes = shared_memory.SharedMemory(create=True, size=speed_changes.nbytes)
    shm_turning_rate = shared_memory.SharedMemory(create=True, size=turning_rate.nbytes)
    shm_ddiffs = shared_memory.SharedMemory(create=True, size=ddiffs.nbytes)
    shm_time_diffs = shared_memory.SharedMemory(create=True, size=time_diffs.nbytes)
    shm_diff_speeds = shared_memory.SharedMemory(create=True, size=diff_speeds.nbytes)
    
    np_speed_changes = np.ndarray(speed_changes.shape, dtype=speed_changes.dtype, buffer=shm_speed_changes.buf)
    np_turning_rate = np.ndarray(turning_rate.shape, dtype=turning_rate.dtype, buffer=shm_turning_rate.buf)
    np_ddiffs = np.ndarray(ddiffs.shape, dtype=ddiffs.dtype, buffer=shm_ddiffs.buf)
    np_time_diffs = np.ndarray(time_diffs.shape, dtype=time_diffs.dtype, buffer=shm_time_diffs.buf)
    np_diff_speeds = np.ndarray(diff_speeds.shape, dtype=diff_speeds.dtype, buffer=shm_diff_speeds.buf)

    np_speed_changes[:] = speed_changes[:]
    np_turning_rate[:] = turning_rate[:]
    np_ddiffs[:] = ddiffs[:]
    np_time_diffs[:] = time_diffs[:]
    np_diff_speeds[:] = diff_speeds[:]
    
    shm_names = {
    'speed_changes': shm_speed_changes.name,
    'turning_rate': shm_turning_rate.name,
    'ddiffs': shm_ddiffs.name,
    'time_diffs': shm_time_diffs.name,
    'diff_speeds': shm_diff_speeds.name
    }
    
    lengths = {
        'speed_changes': len(speed_changes),
        'turning_rate': len(turning_rate),
        'ddiffs': len(ddiffs),
        'time_diffs': len(time_diffs),
        'diff_speeds': len(diff_speeds)
    }
    

    for q in [0.01,0.05,0.1]:
        
        res = multiprocessing_bootstrap(
            B, shm_names, q,lengths
        )
            
        HD_ESTIMATES = pd.concat([
            HD_ESTIMATES,
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Speed",
                "Quantile": q,
                "Estimate": res["hd_est_speed"]
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Turning upper",
                "Quantile": 1-q/2,
                "Estimate": res["hd_est_turning_upper"]
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Turning lower",
                "Quantile": q/2,
                "Estimate": res["hd_est_turning_lower"]
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Distance upper",
                "Quantile": 1-q/2,
                "Estimate": res["hd_est_ddiff_upper"]
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Distance lower",
                "Quantile": q/2,
                "Estimate": res["hd_est_ddiff_lower"]
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Time upper",
                "Quantile": 1-q/2,
                "Estimate": res["hd_est_time_upper"]
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Time lower",
                "Quantile": q/2,
                "Estimate": res["hd_est_time_lower"]
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Diff Speed upper",
                "Quantile": 1-q/2,
                "Estimate": res["hd_est_diff_speed_upper"]
            }),
            pd.DataFrame({
                "Interval": f"{interval[0]}-{interval[1]}" ,
                "Variable": "Diff Speed lower",
                "Quantile": q/2,
                "Estimate": res["hd_est_diff_speed_lower"]
            })
        ])
        
HD_ESTIMATES.to_csv("/home/s2075466/aisplanner/results/hd_estimates.csv")