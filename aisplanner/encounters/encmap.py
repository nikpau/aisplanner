"""
Module for plotting encounters on a map.
"""

import geopandas as gpd
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from pytsa.targetship import TargetVessel
from aisplanner.encounters.utils import OverlappingPair, load_results
import pickle

# Types
Latitue = float
Longitude = float

GEODATA = Path("data/geometry")
ENCOUNTERS = Path("results/encounters")

def plot_coastline(ax: plt.Axes = None, save_plot: bool = False) -> None:
    """
    Plots the coastline of the North-Sea area.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,15))
    coasts = glob(f"{GEODATA}/*.json")
    for coast in coasts:
        gdf = gpd.read_file(coast)
        gdf.plot(ax=ax, color="#003049", alpha=0.8)
    
    if save_plot:
        plt.savefig("aisplanner/encounters/coastline.png", dpi=300)
    return None

def closest_point(pair: OverlappingPair) -> tuple[Latitue,Longitude]:
    """
    Returns the closest point between two vessels.
    """
    v1, v2 = pair()
    # Get the closest point between the two trajectories
    mindist = np.inf
    for v1msg, v2msg in zip(v1.track, v2.track):
        dist = np.linalg.norm(
            np.array([v1msg.lat,v1msg.lon]) - 
            np.array([v2msg.lat,v2msg.lon])
        )
        if dist < mindist:
            mindist = dist
            closest = (v1msg.lon,v1msg.lat)
    return closest

def map_region(
    single_region_path: Path, 
    ax: plt.Axes,
    seen: set[OverlappingPair] = set()
) -> set[OverlappingPair]:
    """
    Adds encounters extracted from a single region to the plot.
    """
    olpairs: list[OverlappingPair] = load_results(single_region_path)
    for pair in olpairs:
        if pair in seen:
            continue
        v1, v2 = pair()
        print(f"Plotting encounter for {v1.mmsi} and {v2.mmsi}")
        seen.add(((v1.mmsi,v2.mmsi),(v2.mmsi,v1.mmsi)))
        # Plot the closest point
        closest = closest_point(pair)
        ax.scatter(*closest, color="#e76f51", s=10)
    return seen
        
if __name__ == "__main__":
    # Create a map of the North Sea
    fig, ax = plt.subplots(figsize=(15,15))
    plot_coastline(ax=ax)
    # Plot encounters
    seen = set()
    for file in ENCOUNTERS.glob("*.tr"):
        seen = map_region(file, ax)
    plt.tight_layout()
    plt.savefig("aisplanner/encounters/map.png", dpi=300)
    plt.close()