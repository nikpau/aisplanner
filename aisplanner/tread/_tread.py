from __future__ import annotations
import argparse
from glob import glob
import time

import pickle
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

plt.rc('font',**{'family':'sans-serif','sans-serif':['Liberation Sans']})

import numpy as np
import pandas as pd
from misc import logger
from sklearn.neighbors import NearestNeighbors

Latitude = float
Longitude = float
MMSI = int

@dataclass
class BoundingBox:
    MIN_LAT: Latitude # [°N]
    MAX_LAT: Latitude # [°N]
    MIN_LON: Longitude # [°E]
    MAX_LON: Longitude # [°E]


# Global file path for saving objects
GLOBALPATH= Path("/home/s2075466/HHOS/out")

# Print info to stdout
VERBOSE = True

# Frame for Vessel detection
GLOBALFRAME = BoundingBox(
    MIN_LAT = 51.9, # [°N]
    MAX_LAT = 60.5, # [°N]
    MIN_LON = 4,    # [°E]
    MAX_LON = 16    # [°E]
)

# Number of iterations until checking for lost vessels
CHECK_FOR_LOST = 1e4

# Time needed before labeling the vessel as being 'lost'
TAU = 600 # [s]

# Minimum speed for a vessel to be considered 'sailing'
MIN_SPEED = 0.5 # [kn]

# Minimum AIS transmitting frequency
MIN_AIS_FREQ = 2 # [s]

# Parameters for incemental DBSCAN
N_ENTRY   = 4 # [-]
N_EXIT    = 4 # [-]
N_STAT    = 4 # [-]

# Storages for all messages, which would land in the clusters. 
# This is done to decide on an epsilon for the DBSCAN algortithm
# before it is instantiated
@dataclass
class Storage:
    entry: List[AISMessage] = field(default_factory=lambda: [])
    exit: List[AISMessage] = field(default_factory=lambda: [])
    stationary: List[AISMessage] = field(default_factory=lambda: [])
    not_assigned: List[AISMessage] = field(default_factory=lambda: [])

    def as_list(self):
        return [
            self.entry,  
            self.exit,
            self.stationary,
            self.not_assigned
        ]

@dataclass
class CLUSTERS:
    entry:      DBSCAN
    exit:       DBSCAN
    stationary: DBSCAN

@dataclass
class Location:
    lat: float
    lon: float

    def __hash__(self) -> int:
        return hash((self.lat,self.lon))

# Cluster Objects for Entry, Exit and Stationary Points
def _init_clusters(eps_entry,eps_exit,eps_stat):
    """
    Set up DBSCAN instances with provided epsilon
    parameters. 
    """
    _entry_cluster = DBSCAN(
        eps=eps_entry,min_samples=N_ENTRY, n_jobs=-3,
        metric="haversine")
    _exit_cluster = DBSCAN(
        eps=eps_exit,min_samples=N_EXIT, n_jobs=-3,
        metric="haversine")
    
    # The stationary cluster is used for ports, oil rigs etc.
    # The cluster type "PO" refers to this cluster. 
    _stationary_cluster = DBSCAN(
        eps=eps_stat,min_samples=N_STAT, n_jobs=-3,
        metric="haversine")
    
    return CLUSTERS(
        _entry_cluster,_exit_cluster,_stationary_cluster)

@dataclass(frozen=True)
class VesselStatus:
    sailing: int = 1
    stationary: int = 2
    lost: int = -1

@dataclass
class AISMessage:
    """
    AIS Message object
    """
    sender: int
    timestamp: datetime
    lat: Latitude
    lon: Longitude
    COG: float # Course over ground
    SOG: float # Speed over ground
    cluster_type: str =  "" # ["EN","EX","PO"]
    label_group: Union[int,None] = None


@dataclass
class Vessel:
    mmsi: int
    status: int
    last_update: datetime
    avg_speed: float # Will be set to -1 if unknown
    track: List[AISMessage] = field(default_factory = lambda: list())

@dataclass
class ReducedVessel:
    """Reduced vessel object
    for usage in the epsilon 
    determination prior to clustering"""
    mmsi: int
    status: int
    avg_speed: float
    last_update: datetime
    last_2_messages: deque


@dataclass(frozen=True)
class ClusterType:
    """
    Types of waypoints which 
    are eligible for clustering
    """
    stationary: str = "PO"
    entry: str = "EN"
    exit: str = "EX"
    not_assigned: str = ""

    @classmethod
    def as_list(cls) -> list:
        return [cls.entry,cls.exit,cls.stationary]

@dataclass
class Route:
    avg_start: Tuple[Latitude,Longitude] = -1, -1 # Average lat lon of route begin
    avg_end: Tuple[Latitude,Longitude] = -1, -1 # Average lat lon of route end
    elements: List[AISMessage] = field(default_factory = lambda: list())
    
def _load_data(filepath: PathLike, **csv_options) -> pd.DataFrame:
    return pd.read_csv(filepath,**csv_options)

def _add_to_cluster(
        container: List[AISMessage], cluster: DBSCAN) -> List[AISMessage]:
    """
    Waypoint clustering with DBSCAN. 

    Args:
        container (list): storage container containing (time sorted)
        AIS Messages. If this function is used together with the TRE
        algorithm, order is ensured.
        cluster (DBSCAN): DBSCAN instance
        
    First, extract the observations for clustering from the messages.
    After extraction, the data is clustered and the resulting labels
    are re-assigned to the messages.
    """
    data = np.array([[m.lat,m.lon] for m in container])
    logger.info(f"Cluster shape: {data.shape}")
    cluster.fit(data)
    return _update_labels(cluster.labels_,container)

def _clear_track(vessel: Vessel) -> None:
    """
    Clear a vessel's track to refill it with messages 
    that gave gone through clustering.
    """
    vessel.track = []

def _update_labels(labels: List[int], 
                   container: List[AISMessage]) -> List[AISMessage]:
    """Assign labels to each message in the repective 
    container. Order must be preserved
    """
    for label, message in zip(labels,container):
        message.label_group = label
    return container

def _reassign(container: List[AISMessage], 
              mmsi_dict: Dict[MMSI,Vessel]) -> Dict[MMSI,Vessel]:
    """Reassign the labeled AIS messages to the 
    track of the dictionary mapping MMSIs to Vessel
    objects.
    Note: The AIS messages in the ``container`` must
    be sorted by thier `timestamp` property. 
    """
    for message in container:
        mmsi_dict[message.sender].track.append(message)

def _sort_track(vessel: Vessel) -> None:
    vessel.track = sorted(vessel.track,key=lambda x: x.timestamp)


def TRE(data: pd.DataFrame, clusters: CLUSTERS):
    """
    Traffic Route Extraction without Anomaly Detection
    adapted from Pallotta et al. (2013).

    This TRE[AD] implementation does not have an 
    incremental route extraction process. It is 
    run once at the end of the function, and
    needs to be rerun everytime the input data changes. 

    Args:
        datapath (PathLike): _description_
    """
    mmsi_dict: Dict[MMSI,Vessel] = {}

    iteration = 0
    datasize = len(data.index)
    date = data["timestamp"].iloc[0]
    storage = Storage()
    
    for mmsi,ts,lat,lon,sog,cog in zip(
        data["MMSI"],data["timestamp"],data["lat"],
        data["lon"],data["speed"],data["course"]):

        iteration += 1
        if VERBOSE:
            print(f"Assigning to cluster storages: {(iteration/datasize)*100:.2f}% done",end="\r")

        if not _is_inside_bounds(lat,lon):
            continue
        
        # The vessel object identified by MMSI does not exist yet
        if mmsi not in mmsi_dict:
            message = AISMessage(
                sender=mmsi,timestamp=ts,
                lat=lat,lon=lon,
                COG=cog,SOG=sog,
                cluster_type = ClusterType.entry,
                label_group=None
            )
            vessel = Vessel(
                mmsi = mmsi,status = VesselStatus.sailing,
                last_update= ts, avg_speed= -1,
                track=[]
            )
            # Add message to vessel's track
            vessel.track.append(message)

            # Also store a copy in the entry message storage
            storage.entry.append(message)
            mmsi_dict[mmsi] = vessel

        # The vessel exists: its parameters are updated and tested
        else:
            vessel = mmsi_dict[mmsi]
            message = AISMessage(
                    sender=mmsi,
                    timestamp=ts,
                    lat=lat,lon=lon,
                    COG=cog,SOG=sog,
                    cluster_type = ClusterType.not_assigned,
                    label_group=None
                )
            
            # Add message to vessel's track
            vessel.track.append(message)
            vessel.last_update = ts

            # Calculate average speed
            vessel_pos = _get_last_position(vessel)
            prev_pos = _get_last_position(vessel, n=2)

            dist = np.linalg.norm(vessel_pos - prev_pos)
            delta_time = (vessel.track[-1].timestamp - vessel.track[-2].timestamp).total_seconds()
            if delta_time < MIN_AIS_FREQ: # Sort out transmissions received by two stations
                vessel.avg_speed = -1.
            else:
                vessel.avg_speed = dist/delta_time
                if vessel.avg_speed < MIN_SPEED and vessel.status== VesselStatus.sailing:
                    # The vessel has stopped. A stationary 
                    # event is generated and clustered into POs 
                    # (ports and offshore platforms) object clustering
                    vessel.status = VesselStatus.stationary
                    vessel.track[-1].cluster_type = ClusterType.stationary
                    storage.stationary.append(vessel.track[-1])
                elif vessel.status == VesselStatus.lost:
                    # Vessel reappears after being lost
                    vessel.status = VesselStatus.sailing
                    vessel.track[-1].cluster_type = ClusterType.entry
                    storage.stationary.append(vessel.track[-1])
                
                # None of the special events happend. Thus, we
                # just store the message without any cluster
                # attached to it. 
                else: storage.not_assigned.append(message)
        
        if iteration % CHECK_FOR_LOST == 0:
            if VERBOSE:
                print("Checking for lost vessels...")
            _check_lost(ts,mmsi_dict,storage)

    # Fill every message in each storage container with
    # thier respective labels 

    logger.info("Clustering entry points")
    storage.entry = _add_to_cluster(storage.entry,clusters.entry)

    logger.info("Clustering stationary points")
    storage.stationary = _add_to_cluster(storage.stationary,clusters.stationary)

    logger.info("Clustering exit points")
    storage.exit = _add_to_cluster(storage.exit,clusters.exit)

    # Currently we have two copies of all AIS Messages: 
    # One being the storage container, the other being
    # the time-sorted vessel tracks. 
    # The messages the the vessel's tracks are not assigned
    # to a label_group as we just saved them without clustering.
    # However, we needed the time-sorted messages to check, if succeeding
    # messages had been apart for more than a certain threshold (see the 
    # `delta_time` variable).
    # After clustering, we are now replacing the unclustered messages in
    # the vessel's tracks with the clustered ones.
    for vessel in mmsi_dict.values():
        _clear_track(vessel)

    for s in storage.as_list():
        _reassign(s,mmsi_dict)    
        
    for vessel in mmsi_dict.values():
        _sort_track(vessel)

    routes = route_manager(mmsi_dict)

    # Save the dictonary containing the routes
    # to the data folder  
    #with open(f'{GLOBALPATH}/routes/routes_{date!r}.pyo', 'wb') as fp:
    with open(f'data/routes/10days/routes_{date.date()}.pyo', 'wb') as fp:
        pickle.dump(routes,fp)

    return

def _reduced_TRE_no_clustering(data: pd.DataFrame) -> Storage:
    
    r_dict: Dict[MMSI,ReducedVessel] = {}
    size = len(data)
    iteration = 0
    storage = Storage()
    
    for mmsi,ts,lat,lon in zip(
        data["MMSI"],data["timestamp"],data["lat"],
        data["lon"]):

        iteration += 1
        if VERBOSE:
            print(f"Sorting messages: {(iteration/size)*100:.2f}% done.",end="\r")

        if not _is_inside_bounds(lat,lon):
            continue
        
        # The vessel object identified by MMSI does not exist yet
        if mmsi not in r_dict:
            storage.entry.append([lat,lon])
            dq = deque(maxlen=2)
            dq.append([lat,lon,ts])
            r_dict[mmsi] = ReducedVessel(
                mmsi=mmsi,status=VesselStatus.sailing,
                avg_speed= -1,last_update=ts,last_2_messages=dq)
        # The vessel exists: its parameters are updated and tested
        else:
            vessel = r_dict[mmsi]
            vessel.last_2_messages.append([lat,lon,ts])
            vessel.last_update = ts

            # Calculate average speed
            vessel_pos = np.array(vessel.last_2_messages[1][0:2]) # ["lat","lon"]
            prev_pos = np.array(vessel.last_2_messages[0][0:2]) # ["lat","lon"]

            dist = np.linalg.norm(vessel_pos - prev_pos)
            delta_time = (vessel.last_2_messages[-1][2] - vessel.last_2_messages[-2][2]).total_seconds()
            if delta_time >= MIN_AIS_FREQ: # Sort out transmissions received by two stations
                vessel.avg_speed = dist/delta_time
                if vessel.avg_speed < MIN_SPEED and vessel.status== VesselStatus.sailing:
                    vessel.status = VesselStatus.stationary
                    storage.stationary.append([lat,lon])
                if vessel.status == VesselStatus.lost:
                    vessel.status = VesselStatus.sailing
                    storage.entry.append([lat,lon])
            else: vessel.avg_speed = -1
        
        if iteration % CHECK_FOR_LOST == 0:
            for vessel in r_dict.values():
                if vessel.status is VesselStatus.lost:
                    continue
                if (ts - vessel.last_update).total_seconds() > TAU and len(vessel.last_2_messages) > 1:
                    vessel.status = VesselStatus.lost
                    storage.exit.append(vessel.last_2_messages[1][0:2])

    return storage

def identify_epsilons(data: pd.DataFrame,eps_cutoff: float = 0) -> Tuple[float,float,float]:
    """
    Find the maximum distance between two samples 
    for one to be considered as in the neighborhood of the other.

    Procedure is done via visual inspection, as proposed in Ester et al. (1998)
    """

    _s = _reduced_TRE_no_clustering(data)
    _names = ["Entry Cluster","Stationary Cluster","Exit Cluster"]
    _colors = ["#264653","#2a9d8f","#e9c46a"]

    f, ax1 = plt.subplots(figsize = (8,4))
    #ax1.set_yscale("log")

    for i, c in enumerate([_s.entry,_s.stationary,_s.exit]):
        print(f"Generating k-dist-graph for {_names[i]}")
        dist = sorted_kdist_graph(c,eps_cutoff)
        ax1.plot(dist, label=_names[i],color=_colors[i])

    ax1.set_xlim(-20,15000)
    ax1.set_ylabel("$\epsilon$",fontsize=14)
    ax1.set_xlabel("$n$ points",fontsize=14)
    ax1.grid(which="both",ls=":")
    plt.suptitle("Sorted 4-dist-graph for sample AIS data")
    ax1.legend()
    plt.tight_layout()
    savepath = Path("out/4-dist_graph.pdf").absolute()
    plt.savefig(savepath)
    return savepath

def _get_last_position(vessel: Vessel, n:int = 1) -> Tuple[Latitude,Longitude]:
    """
    Retrieve the vessels (lat,lon) position
    in its recorded track `n` observations ago.
    """
    return np.array([vessel.track[-n].lat, vessel.track[-n].lon])

def _check_lost(ts: datetime, mmsi_dict: Dict[int,Vessel],storage: Storage) -> None:
    """
    Check if the vessel has not been transmitting 
    since `TAU` seconds.
    """
    for vessel in mmsi_dict.values():
        if vessel.status is VesselStatus.lost:
            continue
        if (ts - vessel.last_update).total_seconds() > TAU:
            # Delcare lost
            vessel.status = VesselStatus.lost
            # Add last point to exit storage and update label
            storage.exit.append(vessel.track[-1])
            vessel.track[-1].cluster_type = ClusterType.exit

def _is_inside_bounds(lat,lon):
    gf = GLOBALFRAME
    return (lat >= gf.MIN_LAT and lat <= gf.MAX_LAT and 
            lon >= gf.MIN_LON and lon <= gf.MAX_LON)

def route_manager(mmsi_dict: Dict[int,Vessel]) -> Dict[str,Route]:
    
    # Temporary routes dict
    routes: Dict[str,Route] = {}
    found = 0

    for vessel in mmsi_dict.values():
        cluster_and_labels = extract_cluster_and_labels(vessel.track)
        # Vessel did not cross more than one waypoint
        if len(set([(c,l) for c,l,_ in cluster_and_labels])) <= 1:
            continue
        _subroutes = subroute_ranges([idx for _,_,idx in cluster_and_labels])
        _names = _generate_route_names(cluster_and_labels)        
        _subroute_idx = 0
        for idx, message in enumerate(vessel.track):
            if _subroute_idx == len(_subroutes):
                break
            start, end = _subroutes[_subroute_idx]
            if start <= idx < end:
                if _names[_subroute_idx] not in routes: # Sub route not in dict
                    routes[_names[_subroute_idx]] = Route()
                    found += 1
                    if VERBOSE:
                        print(f"{found} routes found.", end="\r")
                route = routes[_names[_subroute_idx]]
                if idx == end - 1: # Add Endpoint
                    route.elements.extend([message,vessel.track[idx+1]])
                    _subroute_idx += 1
                else: route.elements.append(message)

    routes = _filter_routes(routes)

    logger.info(f"{len(routes)} routes have been extracted")
    return routes

def _filter_routes(routes: Dict[str,Route]) -> Dict[str,Route]:
    """Subset routes with two filters:

    1. If two routes are the same but from a differnt 
    direction, merge them. Example: One route start at `EN3`
    and ends in `EX8` while another route starts at `EX8` 
    and ends in `EN3`, they will be combined.    

    2. All routes containing the not-clustered placeholder
    `-1` will also be removed
    """
    for name, route in list(routes.items()):
        if ("-1" in name or 
        (name.startswith("EX") and _swap_route_name(name).startswith("EX"))):
            del routes[name]
            continue
        swapped = _swap_route_name(name)
        if swapped in routes:
            route.elements.extend(routes[swapped].elements)
            del routes[swapped]
    return routes

def _swap_route_name(name: str) -> str:
    parts = re.split('(\d+)',name)
    return parts[2] + parts[3] + parts[0] + parts[1]


def extract_cluster_and_labels(track: List[AISMessage]) -> List[Tuple[str,float,int]]:
    """
    Extract all cluster types, label groups  
    and indices for a given vessel track which are not ``None``
    and therefore resemble starting or ending points of 
    a route.
    """
    cluster_and_labels = []
    for idx, message in enumerate(track):
            if message.label_group is not None:
                cluster_and_labels.append(
                    (
                        message.cluster_type,
                        message.label_group,
                        idx
                    )
                )
    return cluster_and_labels

def subroute_ranges(indices: List[int]) ->List[Tuple[int,int]]:
    return [(start,end) for start,end in pairwise(indices)]

def _generate_route_names(cluster_and_label: List[Tuple[str,float,int]])-> List[str]:
    """
    Generate temporary names based on 
    cluster and label for two adjacent routes
    Example: 
        Cluster 1 & 2: "EN", "PO"
        Labels 1 & 2:   3,    5
        Result: "EN3PO5"
    """
    names = []
    for cl0, cl1 in pairwise(cluster_and_label):
        (c0,l0,_), (c1,l1,_) = cl0, cl1 
        names.append(f"{c0}{int(l0)}{c1}{int(l1)}")
    return names

def sorted_kdist_graph(data: list, eps_cutoff: float = 1e-2) -> np.ndarray:

    neigh = NearestNeighbors(n_neighbors=2, metric="haversine")
    fitted = neigh.fit(data)
    dist, _ = fitted.kneighbors(data)
    dist = np.sort(dist,axis=0)[::-1]
    dist = dist[:,1]
    return dist[dist > eps_cutoff]

def preprocessor(datapath: Union[Path,List[Path]] = None) -> pd.DataFrame:
    """
    Data pre-processing for decoded AIS message file(s) 
    """

    if isinstance(datapath,list):
        logger.info(f"Preprocessing data file list {datapath}")
        data = _load_and_concat(datapath)
    else:
        logger.info(f"Preprocessing data file {datapath}")
        data = _load_data(datapath,sep=",")
    
    # Convert timestamp to time-object
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Sort by ascending date
    data = data.sort_values(by="timestamp")

    # Drop raw messages and inferred lat and lon
    logger.info("Dropping unused columns")
    data: pd.DataFrame = data.drop(
        ["raw_message","latitude","longitude","repeat",
         "turn","accuracy","radio","heading"],
        axis=1)

    # Remove duplicates based on Lateral position to account 
    # for multiple AIS receivers reporting the same vessel
    logger.info("Dropping duplicates")
    data = data.drop_duplicates(subset=["timestamp","MMSI"], keep="first")

    # Drop all values with invalid, or default values.
    # Taken from 
    # https://gpsd.gitlab.io/gpsd/AIVDM.html#_type_5_static_and_voyage_related_data
    
    logger.info("Removing NA observations")
    data = data.drop(data[data["course"] == 360].index)
    data = data.drop(data[data.speed == 0.0].index)
    data = data.drop(data[
         (data.status == "NavigationStatus.Moored") & 
         (data.status == "NavigationStatus.EngagedInFishing")].index)

    return data

def _prompt_for_eps(savepath: Path[str]) -> Tuple[float,float,float]:

    while True:
        inp = input(
            "Please provide comma separated epsilon values "
            "for the entry, exit and stationary cluster.\n"
            "The k-dist graph for visual inspection has been "
            f"saved at '{savepath}'.\nInput: "
        )
        try:
            en, ex, st = tuple(map(float,inp.split(",")))
            break
        except Exception:
            print(
                "Input could not be parsed. "
                "Please try again."
                )
    return en, ex, st

def _load_and_concat(files: List[Path[str]]) -> pd.DataFrame:
    """Concatenate an arbitrary amount of csv files
    into one large data frame"""
    dfs = []
    for file in files:
        df = pd.read_csv(file,sep=",",index_col=0)
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def run_eps_ident(datapath: Union[Path,List[Path]]) -> None:
    
    messages = preprocessor(datapath)
    
    # Find the maximum distance between
    # two samples for one to be considered
    # as in the neighborhood of the other.
    return identify_epsilons(messages)

def run_tre(datapath: Union[Path,List[Path]], eps: List[float]):

    messages = preprocessor(datapath)

    # Parameters for incemental DBSCAN
    cl = _init_clusters(*eps)

    # Extract routes from data
    TRE(messages,cl)
    
if __name__ == "__main__":

    #SOURCE = glob("data/oneweek/*.csv")
    SOURCE = [
        "data/10days/2021_06_20.csv",
        "data/10days/2021_06_21.csv",
        # "data/10days/2021_06_22.csv",
        # "data/10days/2021_06_23.csv",
        # "data/10days/2021_06_24.csv",
        # "data/10days/2021_06_25.csv",
        # "data/10days/2021_06_26.csv",
    ]

    # SOURCE = ["data/decoded/2021_07_03-calm.csv"]

    parser = argparse.ArgumentParser()

    parser.add_argument("-i","--identify",default=False,action='store_true')
    parser.add_argument(
        "-f","--full", default=False,
        help="Provide three comma separated eps vals for the DBSCAN instances"
    )

    args = parser.parse_args()

    if args.identify:
        logger.info("Epsilon Identification process started.")
        run_eps_ident(SOURCE)

    if args.full:
        logger.info("TRE starting...")
        eps = [float(item) for item in args.full.split(',')]
        run_tre(SOURCE,eps)