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
import uuid

# Beta distribution
from scipy.stats import beta

HD_ESTIMATES = pd.DataFrame(columns=["Month","Variable","Quantile","Estimate"])

COLORWHEEL = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51","#E45C3A","#732626"]
from aisstats.errchecker import speed_filter

resample_size = int(1e6)

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
    
    np.random.seed()  # Ensure each process has a different seed
    
    # Resample the data with replacement
    __speed_changes = np.random.choice(speed_changes, resample_size, replace=True)

    print(f"Calculating Harrel-Davis estimator. Fingerprints: {uuid.uuid4()}")
    result = {
        'speed': harrel_davis(__speed_changes, 1-q, resample_size),
    }

    # Clean up by closing the shared memory blocks
    shm_speed_changes.close()

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
    return {
        'hd_est_speed': hd_est_speed,
    }

# MAIN MATTER ---------------------------------------------------------------    

# Bootstrap repetitions
B = 1000

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
            
        except Exception as e:
            print(e)
            print(f"Failed for month {month}")
            continue
        
    speed_changes = np.array(speed_changes)
    
    shm_speed_changes = shared_memory.SharedMemory(create=True, size=speed_changes.nbytes)
    
    np_speed_changes = np.ndarray(speed_changes.shape, dtype=speed_changes.dtype, buffer=shm_speed_changes.buf)

    np_speed_changes[:] = speed_changes[:]
    
    shm_names = {
    'speed_changes': shm_speed_changes.name,
    }
    
    lengths = {
        'speed_changes': len(speed_changes),
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
                "Quantile": 1-q,
                "Estimate": res["hd_est_speed"]
            }),
        ])
        
HD_ESTIMATES.to_csv("/home/s2075466/aisplanner/results/hd_estimates.csv")