"""
Module for plotting encounters on a map.
"""

import geopandas as gpd
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from pytsa import TargetShip
from aisplanner.encounters.utils import OverlappingPair, load_results
import pickle
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from pytsa import BoundingBox
import requests
from osm2geojson import json2geojson

# Types
Latitude = float
Longitude = float

GEODATA = Path("data/geometry")
ENCOUNTERS = Path("results/encounters")

def plot_coastline(datapath: Path,
                    extent: BoundingBox , ax: plt.Axes = None,
                   save_plot: bool = False,
                   return_figure: bool = False,
                   query: str = None) -> plt.Figure | None:
    """
    Plots the coastline of the North-Sea area.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,15))
    coasts = glob(f"{datapath}/*.json")
    for coast in coasts:
        gdf = gpd.read_file(coast)
        gdf.crs = 'epsg:3395' # Mercator projection
        gdf.plot(ax=ax, color="#00657d", alpha=0.8,linewidth=2)
        
    # Additional query for overpass API
    if query is not None:
        url = f"https://overpass-api.de/api/interpreter?data={query}"
        r = requests.get(url)
        data = r.json()
        data = json2geojson(data)
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        gdf.plot(ax=ax, color='#66a2b1', linewidth=.3)
        
    # Crop the plot to the extent
    ax.set_xlim(extent.LONMIN, extent.LONMAX)
    ax.set_ylim(extent.LATMIN, extent.LATMAX)
    
    if save_plot:
        plt.savefig("aisplanner/encounters/coastline.png", dpi=300)
    return None if not return_figure else fig

def closest_point(pair: OverlappingPair) -> tuple[Latitude,Longitude]:
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
    seen: set[OverlappingPair] = set()) -> set[OverlappingPair]:
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

def plot_scatter():
    fig, ax = plt.subplots(figsize=(15,15))
    plot_coastline(ax=ax)
    # Plot encounters
    seen = set()
    for region in ENCOUNTERS.glob("*.tr"):
        seen = map_region(region, ax)
    plt.tight_layout()
    plt.savefig("aisplanner/encounters/map.png", dpi=300)
    plt.close()

def extract_region(
    single_region_path: Path,
    seen: set[OverlappingPair] = set()) -> tuple[list[Longitude],list[Latitude]]:
    """
    Export the closest points between two vessels in a single region
    as a list of longitude and latitude coordinates.
    """
    olpairs: list[OverlappingPair] = load_results(single_region_path)
    lats, lons = np.empty_like(list(olpairs),dtype=np.float32), np.empty_like(list(olpairs),dtype=np.float32)
    for i, pair in enumerate(olpairs):
        if pair in seen:
            continue
        v1, v2 = pair()
        seen.add(((v1.mmsi,v2.mmsi),(v2.mmsi,v1.mmsi)))
        # Plot the closest point
        closest = closest_point(pair)
        lons[i], lats[i] = closest
    return lons, lats, seen

def plot_heat_map():
    fig, ax = plt.subplots(figsize=(15,15))
    # Plot encounters
    seen = set()
    lons, lats = [], []
    for region in ENCOUNTERS.glob("*.tr"):
        rlons, rlats, seen = extract_region(region, seen)
        lons.append(rlons) 
        lats.append(rlats)
    # Flatten
    lons = np.concatenate(lons)
    lats = np.concatenate(lats)
    x,y = np.arange(min(lons),max(lons)), np.arange(min(lats),max(lats))
    xx,yy = np.mgrid[min(lons):max(lons):1000j, min(lats):max(lats):1000j]
    vals = np.vstack([lons,lats])
    kernel = gaussian_kde(vals)
    z = kernel([xx.ravel(), yy.ravel()]).T.reshape(xx.shape)
    ax.contourf(xx,yy,z, cmap=cm.inferno,levels=100)
    cset = ax.contour(xx,yy,z, colors="k", levels=10)
    ax.clabel(cset, inline=True, fontsize=10)
    plot_coastline(ax=ax)
    plt.savefig("aisplanner/encounters/heatmap.png", dpi=300)
    
if __name__ == "__main__":
    plot_heat_map()
    #plot_scatter()