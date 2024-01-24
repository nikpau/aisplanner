"""
Module for plotting the rate of rejection for different 
values for the minimum number of observation per trajectory,
and the spatial standard deviation threshold.
"""
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import speed_filter
from pathlib import Path
from pytsa import SearchAgent, TargetShip
from functools import partial
import pytsa
import pytsa.trajectories.inspect as inspect
from pytsa.trajectories.rules import *
import logging
import pickle
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import ctypes as c

idx = 0

# Set logging level to warning
logging.basicConfig(level=logging.WARNING,force = True)
SDS = [0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5]
MINLENS = [0,5,10,15,20,30,40,50,60,70,80,90,100,200,300,400,500]

def smoothness(lon1,lon2,lon3,lat1,lat2,lat3):
    """
    Calculates the cosine of the angle enclosed between three points.
    """
    v1 = np.array((lon1-lon2,lat1-lat2))
    v2 = np.array((lon3-lon2,lat3-lat2))
    nom = np.dot(v1,v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(nom / den) / np.pi

def average_smoothness(lons: np.ndarray, lats: np.ndarray):
    """
    Calculates the average smoothness of a trajectory.
    All Trajectories will have equal length, however,
    some of the last values may be padding values, as
    shorter trajectories are padded with -1.
    """
    global idx
    smthness = []
    for i in range(1,len(lons)-1):
        # Break if padding values occur
        if lons[i+1] == -1: break 
        smthness.append(
            smoothness(
                lons[i-1],
                lons[i],
                lons[i+1],
                lats[i-1],
                lats[i],
                lats[i+1]
            )
        )
        idx += 1
        print(f"Calculating smoothness for track {idx}")
    return np.mean(smthness)
        

if __name__ == "__main__":
    SEARCHAREA = NorthSea

    DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
    STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

    SA = SearchAgent(
            dynamic_paths=DYNAMIC_MESSAGES,
            frame=SEARCHAREA,
            static_paths=STATIC_MESSAGES,
            preprocessor=partial(speed_filter, speeds= (1,30))
        )

    ships = SA.get_all_ships(njobs=8)
    
    longest_track = max(len(track) for ship in ships.values() for track in ship.tracks)
    ntracks = sum(len(ship.tracks) for ship in ships.values())
    
    # Write all tracks into a numpy array
    tracks = np.zeros((ntracks,longest_track,2))
    idx = 0
    for ship in ships.values():
        for track in ship.tracks:
            # If track is shorter than longest_track, pad with -1
            tracks[idx,:len(track),0] = [p.lon for p in track] + [-1] * (longest_track - len(track))
            tracks[idx,:len(track),1] = [p.lat for p in track] + [-1] * (longest_track - len(track))
            idx += 1
    
    # Create shared array
    shared_arr = mp.Array(c.c_double, ntracks*longest_track*2)
    # Create numpy array from shared array
    tracks = np.frombuffer(shared_arr.get_obj()).reshape(ntracks,longest_track,2)
    
    # Use multiprocessing to calculate the average smoothness
    # of all tracks. All workers will read from the same shared
    # array.
    def calc_smoothness(i):
        return average_smoothness(tracks[i,:,0],tracks[i,:,1])
    
    with mp.Pool(32) as pool:
        OUT = pool.map(calc_smoothness,range(ntracks))
    
    # Save the results
    with open("/home/s2075466/aisplanner/results/avg_smoothness.pkl","wb") as f:
        pickle.dump(OUT,f)