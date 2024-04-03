"""
Module for visually inspecting how far
the interpolated speed of a target vessel
deviates from the the speed calulated from
the distance between two consecutive
positions and the time between them.
"""
import colorsys
import io
from datetime import datetime
from pathlib import Path
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.collections import LineCollection
import pytsa
from PIL import Image
from pytsa import SearchAgent, TargetShip, TimePosition, BoundingBox
from pytsa.structs import Position
import pytsa.tsea.split as split 
from aisplanner.encounters.main import NorthSea
from aisplanner.encounters.filter import haversine
from matplotlib import patches, pyplot as plt
from matplotlib import cm
from matplotlib import dates as mdates
import numpy as np
from aisplanner.encounters.encmap import plot_coastline
from scipy.stats import gaussian_kde
import multiprocessing as mp
from aisplanner.misc import MemoryLoader
from pytsa.trajectories.rules import Recipe
from KDEpy.bw_selection import improved_sheather_jones
from pytsa.utils import mi2nm
import matplotlib.lines as lines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Rules for inspection of trajectories
from pytsa.trajectories.rules import too_few_obs,spatial_deviation
from functools import partial

TEST_FILE_DYN = 'data/aisrecords/2021_08_02.csv'
TEST_FILE_STA = 'data/aisrecords/msgtype5/2021_08_02.csv'
# TEST_FILE_DYN = '/warm_archive/ws/s2075466-ais/decoded/jan2020_to_jun2022/2021_07_01.csv'
# TEST_FILE_STA = '/warm_archive/ws/s2075466-ais/decoded/msgtype5/2021_07_01.csv'
SEARCHAREA = NorthSea
SAMPLINGRATE = 30 # seconds
# Sampling rate in hours
SAMPLINGRATE_H = SAMPLINGRATE / 60 / 60
import pickle
import pandas as pd
from functools import partial
from aisplanner.dataprep import _file_descriptors as fd

# Color converter
cc = matplotlib.colors.ColorConverter.to_rgb
def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


# Use matplotlib style `bmh`
# plt.style.use('bmh')
plt.style.use('default')
plt.rcParams["font.family"] = "monospace"
COLORWHEEL = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51","#E45C3A","#732626"]
COLORWHEEL3 = ["#335c67","#fff3b0","#e09f3e","#9e2a2b","#540b0e"]
COLORWHEEL_DARK = [scale_lightness(cc(c), 0.6) for c in COLORWHEEL]
COLORWHEEL2 = ["#386641", "#6a994e", "#a7c957", "#f2e8cf", "#bc4749"]
COLORWHEEL2_DARK = [scale_lightness(cc(c), 0.6) for c in COLORWHEEL2]
COLORWHEEL_MAP = ["#0466c8","#0353a4","#023e7d","#002855","#001845","#001233","#33415c","#5c677d","#7d8597","#979dac"]

def _bw_sel(kde: gaussian_kde) -> float:
    data = kde.dataset.reshape(-1,1)
    return improved_sheather_jones(data)

def align_yaxis(ax1: plt.Axes, v1, ax2: plt.Axes, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

def flatten(l: list[list]) -> list:
    """
    Flatten a list of lists.
    """
    return [item for sublist in l for item in sublist]  

def speed_filter(df: pd.DataFrame, speeds: tuple[float,float]) -> pd.DataFrame:
    """
    Filter out all rows in a dataframe
    where the speed is not within a given
    interval.
    """
    return df[
        (df[fd.Fields12318.speed.name] >= speeds[0]) &
        (df[fd.Fields12318.speed.name] <= speeds[1])
    ]

def area_center(area: pytsa.BoundingBox) -> Position:
    """
    Get the center of the search area in UTM coordinates.
    """
    return Position(
        (area.LATMIN + area.LATMAX)/2,
        (area.LONMIN + area.LONMAX)/2
    )

def plot_speeds_and_route(tv: TargetShip, mode: str) -> None:
    """
    Plot the trajectory of a target vessel
    as well as the speeds calculated from
    individual positions and the interpolated
    speeds.
    """
    
    tss = []
    # Real positions and speeds
    rlons, rlats, rspeeds, rcog = [], [], [], []
    for track in tv.tracks:
        lons, lats, speeds, cog, ts = [], [], [], [], []
        for msg in track:
            lons.append(msg.lon)
            lats.append(msg.lat)
            speeds.append(msg.SOG)
            cog.append(msg.COG)
            ts.append(msg.timestamp)
        rlons.append(lons)
        rlats.append(lats)
        rspeeds.append(speeds)
        rcog.append(cog)
        tss.append(ts)
    
    fig, axd = plt.subplot_mosaic(
        [
            ['left', 'upper right'],
            ['left', 'lower right']
        ], layout="constrained", figsize=(10,6)
    )
    
    # Plot the trajectory
    for i, (lo,la, sp) in enumerate(zip(rlons,rlats,rspeeds)):
        f = axd["left"].scatter(lo,la,c=sp,s=8,cmap='inferno')
        axd["left"].plot(lo,la,color=COLORWHEEL[i%len(COLORWHEEL)],ls="--", alpha = 0.9)
    
    # Colorbar
    fig.colorbar(f, ax=axd["left"], label="Speed [knots]")
    
    # Plot the real speeds
    axd["lower right"].xaxis.set_major_locator(mdates.AutoDateLocator())
    for nr, (time,speed) in enumerate(zip(tss,rspeeds)):
        axd["lower right"].plot(
            [datetime.fromtimestamp(t) for t in time],
            speed,
            color=COLORWHEEL[nr%len(COLORWHEEL)])
        
    # Plot course over ground
    axd["upper right"].xaxis.set_major_locator(mdates.AutoDateLocator())
    for nr, (time,cog) in enumerate(zip(tss,rcog)):
        axd["upper right"].plot(
            [datetime.fromtimestamp(t) for t in time],
            cog,
            color=COLORWHEEL[nr%len(COLORWHEEL)])
    
    # Set labels
    axd["left"].set_xlabel('Longitude')
    axd["left"].set_ylabel('Latitude')
    
    
    axd["upper right"].set_title(f"Course over ground [°]")
    axd["upper right"].set_xlabel('Time')
    axd["upper right"].set_ylabel('Course over ground [°]')
    
    
    axd["lower right"].set_title(f"Speed [knots]")
    axd["lower right"].set_xlabel('Time')
    axd["lower right"].set_ylabel('Speed [knots]')
    
    # Set y lim
    axd["lower right"].set_ylim(-1,20)
    
    
    # Set legend
    axd["left"].legend(handles=[
        patches.Patch(color='k',label=f"{len(rlons)} trajectories"),
    ]
    )
    axd["upper right"].legend()
    axd["lower right"].legend()
    
    # Save figure
    fig.autofmt_xdate()
    plt.savefig(f"aisstats/out/errchecker/{tv.mmsi}_{mode}.png",dpi=300)
    plt.close()
    
def plot_speed_scatter(sa: SearchAgent,savename: str) -> None:
    """
    Plot the speeds calculated from
    individual positions against the
    calculated speeds.
    """
    rspeeds = []
    cspeeds = []
    ships = sa.extract_all(skip_tsplit=True)
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(6,6))
    ax1: plt.Axes; ax2: plt.Axes
    for ship in ships.values():
        for track in ship.tracks:
            for i in range(1,len(track)):
                rspeed = split.avg_speed(track[i-1],track[i])
                cspeed = split.speed_from_position(track[i-1],track[i])
                rspeeds.append(rspeed)
                cspeeds.append(cspeed)
    ax1.scatter(
        rspeeds,
        cspeeds,
        color=COLORWHEEL[0],
        alpha=0.5,
        s=1
    )

    # Plot line y = x
    ax2.plot(
        np.linspace(1,30,100),
        np.linspace(1,30,100),
        color=COLORWHEEL3[3],
        lw=0.8, label = r"$\overline{SOG}_{m_i}^{m_{i+1}}=\widehat{SOG}_{m_i}^{m_{i+1}}$"
    )
    
    
    ax2.scatter(
        rspeeds,
        cspeeds,
        color=COLORWHEEL[0],
        alpha=0.5,
        s=1
    )

    ax2.set_ylim(0,60)
    ax1.set_ylim(100,1.05*max(cspeeds))
    ax1.set_yscale('log')
    
    # hide the spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(top='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
        
    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    ax2.set_xlabel("Average speed as reported [kn]")
    fig.supylabel("Speed calculated from positions [kn]")
    ax1.set_axisbelow(True)
    ax2.legend(loc = "upper right",fontsize=8)
    
    
    plt.tight_layout()
    plt.savefig(savename,dpi=300)
    plt.close()

def plot_trajectory_jitter(ships: dict[int,TargetShip]) -> None:
    
    np.random.seed(424)
    
    sds = np.array([0,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3])
    
    for ship in ships.values():
        # Center trajectory
        # Get mean of lat and lon
        for track in ship.tracks:
            latmean = sum(p.lat for p in track) / len(track)
            lonmean = sum(p.lon for p in track) / len(track)
            # Subtract mean from all positions
            for msg in track:
                msg.lat -= latmean
                msg.lon -= lonmean
    
    ships: list[TargetShip] = list(ships.values())
            
    # Plot trajectories for different sds
    ncols = 4
    div,mod = divmod(len(sds)-1,ncols)
    nrows = div + 1 if mod else div
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(4*ncols,8))
    subpos = [0.55,0.55,0.4,0.4]
    axs: dict[int,plt.Axes] # type hint
    niter = 0
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            
            trnr = 0
            # Find trajectories whose standard deviation
            # is within the current range
            lons, lats = [], []
            
            np.random.shuffle(ships)
            for ship in ships:
                # Get standard deviation of lat and lon
                for track in ship.tracks:
                    lo = [p.lon for p in track]
                    la = [p.lat for p in track]
                    latstd = np.std(la)
                    lonstd = np.std(lo)
                    # Check if within range
                    if sds[idx] <= (latstd + lonstd) <= sds[idx+1]:
                        if trnr == 50:
                            break
                        trnr += 1
                        lons.append(lo)
                        lats.append(la)
                        
            if row != 1:
                inset = axs[row,col].inset_axes(subpos)
                
                # Add lines for x and y axes
                inset.axhline(0, color='k', linewidth=0.5)
                inset.axvline(0, color='k', linewidth=0.5)
            
            axs[row,col].axhline(0, color='k', linewidth=0.5)
            axs[row,col].axvline(0, color='k', linewidth=0.5)

            for la, lo in zip(lats,lons):
                axs[row,col].plot(
                    lo,
                    la,
                    alpha=0.5, 
                    marker = "x", 
                    markersize = 0.65, 
                    color = COLORWHEEL_MAP[niter % len(COLORWHEEL_MAP)],
                    linewidth = 0.6
                )
            
                if row != 1:
                    # Plot center region in inset
                    inset.plot(    
                        lo,la,
                        color = COLORWHEEL_MAP[niter % len(COLORWHEEL_MAP)],
                        linewidth = 1,
                        marker = "x",
                        markersize = 1
                    )
                    
                        
                    # inset.set_axes_locator(ip)
                    inset.set_xlim(-0.02,0.02)
                    inset.set_ylim(-0.02,0.02)
                    
                niter += 1
                
            axs[row,col].set_xlabel("Longitude")
            axs[row,col].set_ylabel("Latitude")
            axs[row,col].set_title(
                "$\sigma_{ssd}\in$"
                f"[{sds[row*ncols+col]:.2f},{sds[row*ncols+col+1]:.2f}]",
                fontsize=16
            )

            # Set limits
            axs[row,col].set_xlim(-0.25,0.65)
            axs[row,col].set_ylim(-0.25,0.65)
            
            idx += 1
            
    plt.tight_layout()
    plt.savefig(f"aisstats/out/trjitter.pdf")#, dpi=500)
    plt.close()
    
def plot_simple_route(tv: TargetShip, mode: str) -> None:
    """
    Plot the trajectory of a target vessel
    """
    
    # Positions and speeds
    rlons, rlats = [], []
    for track in tv.tracks:
        for msg in track:
            rlons.append(msg.lon)
            rlats.append(msg.lat)       
        # De-mean
    latmean = sum(lats) / len(lats)
    lonmean = sum(lons) / len(lons)
        # Subtract mean from all positions
    lons, lats = [], []
    for track in tv.tracks:
        for msg in track:
            msg.lat -= latmean
            msg.lon -= lonmean
            lons.append(msg.lon)
            lats.append(msg.lat)
        
        rlons.append(lons)
        rlats.append(lats)

    
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Plot the trajectory
    for i, (lo,la) in enumerate(zip(rlons,rlats)):
        ax.plot(lo,la,color=COLORWHEEL[i%len(COLORWHEEL)],ls="-", alpha = 0.9)
        ax.scatter(lo,la,color=COLORWHEEL[i%len(COLORWHEEL)],s=1)
        
    # Set labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Save figure
    plt.savefig(f"aisstats/out/errchecker/{tv.mmsi}_{mode}.png",dpi=300)
    plt.close()
    
def plot_trajectories_on_map(ships: dict[int,TargetShip], 
                             mode: str,
                             specs: dict,
                             extent: BoundingBox):
    """
    Plot all trajectories on a map.
    """
    assert mode in ["all","accepted","rejected"]
    fig, ax = plt.subplots(figsize=(10,10))
    idx = 0
    plot_coastline(
        datapath=Path("data/geometry"),
        extent=extent,
        ax=ax
    )
    for ship in ships.values():
        for track in ship.tracks:
            idx += 1
            ax.plot(
                [p.lon for p in track],
                [p.lat for p in track],
                alpha=0.5, linewidth=0.3, marker = "x", markersize = 0.5,
                color = COLORWHEEL_MAP[idx % len(COLORWHEEL_MAP)]
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title(f"Trajectories for {mode} vessels: {specs}", size = 8)
    plt.tight_layout()
    if not specs:
        plt.savefig(f"aisstats/out/trajmap_{mode}.png",dpi=800)
    else:
        plt.savefig(f"aisstats/out/trajmap_{mode}_{specs['Min traj length [msg]']}.png",dpi=800)
    plt.close()
    
def _maxmin_norm(arr:np.ndarray) -> np.ndarray:
    """
    Normalize an array by subtracting the minimum
    and dividing by the maximum.
    """
    return (arr - arr.min()) / (arr.max() - arr.min())
    
def plot_latlon_kde(ships: dict[int,TargetShip], mode: str):
    """
    Plot the kernel density estimate of
    the latitude and longitude of all
    trajectories.
    """
    assert mode in ["all","accepted","rejected"]
    
    # Get all latitudes and longitudes
    lats = []
    lons = []
    for ship in ships.values():
        for track in ship.tracks:
            latmin, latmax = min([p.lat for p in track]), max([p.lat for p in track])
            lonmin, lonmax = min([p.lon for p in track]), max([p.lon for p in track])
            # Subtract mean from all positions
            for msg in track:
                msg.lat = (msg.lat - latmin) / (latmax - latmin)
                msg.lon = (msg.lon - lonmin) / (lonmax - lonmin)
            lats += [p.lat for p in track]
            lons += [p.lon for p in track]
    
    # Create KDEs
    lat_kde = gaussian_kde(lats)
    lon_kde = gaussian_kde(lons)
    
    # Plot KDEs
    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(16,10))
    ax1.plot(
        np.linspace(min(lats),max(lats),1000),
        lat_kde(np.linspace(min(lats),max(lats),1000))
    )
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Kernel density estimate of latitudes for {mode} vessels")
    
    ax2.plot(
        np.linspace(min(lons),max(lons),1000),
        lon_kde(np.linspace(min(lons),max(lons),1000))
    )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Kernel density estimate of longitudes for {mode} vessels")
    
    plt.savefig(f"aisstats/out/latlon_kde_{mode}.png",dpi=300)
    plt.close()
    
def plot_latlon_shapes(good: list[TargetShip],bad: list[TargetShip]):
    """
    Plots the standardized distribution of latitudes and longitudes 
    for good and bad trajectories.
    """
    
    # Get all latitudes and longitudes
    glats = []
    glons = []
    for ship in good:
        for track in ship.tracks:
            glats += [p.lat for p in track]
            glons += [p.lon for p in track]

    # Normalize
    glats, glons = np.array(glats), np.array(glons)
    glats = _maxmin_norm(glats)
    glons = _maxmin_norm(glons)
    
     
    blats = []
    blons = []
    for ship in bad:
        for track in ship.tracks:
            blats += [p.lat for p in track]
            blons += [p.lon for p in track]

    # Normalize
    blats, blons = np.array(blats), np.array(blons)
    blats = _maxmin_norm(blats)
    blons = _maxmin_norm(blats)
    
    
    # Create KDEs
    glat_kde = gaussian_kde(glats)
    glon_kde = gaussian_kde(glons)
    blat_kde = gaussian_kde(blats)
    blon_kde = gaussian_kde(blons)
    
    # Plot KDEs
    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(16,10))
    ax1.plot(
        np.linspace(min(glats),max(glats),1000),
        glat_kde(np.linspace(min(glats),max(glats),1000)),
        label="Good trajectories"
    )
    ax1.plot(
        np.linspace(min(blats),max(blats),1000),
        blat_kde(np.linspace(min(blats),max(blats),1000)),
        label="Bad trajectories"
    )
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Kernel density estimate of latitudes")
    ax1.legend()
    
    ax2.plot(
        np.linspace(min(glons),max(glons),1000),
        glon_kde(np.linspace(min(glons),max(glons),1000)),
        label="Good trajectories"
    )
    ax2.plot(
        np.linspace(min(blons),max(blons),1000),
        blon_kde(np.linspace(min(blons),max(blons),1000)),
        label="Bad trajectories"
    )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Kernel density estimate of longitudes")
    ax2.legend()
    
    plt.savefig(f"aisstats/out/latlon_kde.png",dpi=300)
    plt.close()

def plot_speed_histogram(speeds: np.ndarray,
                         savename: str) -> None:
    """
    Plots a histogram of the speeds of all
    vessels in the data set.
    """
    print(f"Number of observations: {len(speeds)}")
    
    # Print how many vessels have a speed of smaller than
    # 1 knot and larger than 30 knots
    print(f"Number of vessels with speed < 1 knot: {sum(speeds < 1)}")
    print(f"Number of vessels with speed > 30 knots: {sum(speeds > 30)}")
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(speeds, bins=100, color=COLORWHEEL[0])
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Number of observations")
    ax.set_yscale('log')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(savename,dpi=400)
    
def lat_lon_outofbounds(lats,lons)->None:
    """
    Print out how many latitudes and longitudes
    are out of bounds as a percentage of the total
    number of observations.
    """
    print(
        f"Number of latitudes out of bounds: "
        f"{sum(lats < 0) + sum(lats > 90)}"
    )
    print(
        f"Number of longitudes out of bounds: "
        f"{sum(lons < 0) + sum(lons > 180)}"
    )
    print(
        f"Percentage of latitudes out of bounds: "
        f"{(sum(lats < 0) + sum(lats > 90)) / len(lats) * 100:.2f}%"
    )
    print(
        f"Percentage of longitudes out of bounds: "
        f"{(sum(lons < 0) + sum(lons > 180)) / len(lons) * 100:.2f}%"
    )
    
def plot_sd_vs_rejection_rate(ships: dict[int,TargetShip],
                              savename: str):
    """
    Plot the standard deviation of the trajectory jitter
    against the rejection rate.
    """
    cw = COLORWHEEL + COLORWHEEL2
    sds = np.array([0,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.1,0.2,0.3])
    minlens = np.array([0,2,5,10,20,30,40,50,60,70])
    fig, ax = plt.subplots(figsize=(8,5))
    for idx, minlen in enumerate(minlens):
        rejected = []
        accepted = []
        for sd in sds:
            recipe = Recipe(
                partial(too_few_obs,n=minlen),
                partial(spatial_deviation,sd=sd)
            )
            inpsctr = pytsa.Inspector(
                data=ships,
                recipe=recipe
            )
            acc, rej = inpsctr.inspect(njobs=1)
            rejected.append(sum([len(r.tracks) for r in rej.values()]))
            accepted.append(sum([len(a.tracks) for a in acc.values()]))
        rejected = np.array(rejected)
        accepted = np.array(accepted)
        total = rejected + accepted
        rejection_rate = rejected / total
        
        ax.plot(sds,
                rejection_rate,
                label=f"{minlen} obs",
                color=cw[idx]
            )
        
    ax.set_xlabel("Standard deviation")
    ax.set_ylabel("Rejection rate")
    ax.legend(loc="lower right" ,title = "Min. trajectory\nlength", fontsize=9, fancybox=False)
    # ax.set_title("Rejection rate vs. standard deviation of trajectory")
    plt.savefig(savename)
    plt.close()
    
def _heading_change(h1,h2):
    """
    calculate the change between two headings
    such that the smallest angle is returned.
    """

    diff = abs(h1-h2)
    if diff > 180:
        diff = 360 - diff
    if (h1 + diff) % 360 == h2:
        return diff
    else:
        return -diff
    
def plot_time_diffs(sa:SearchAgent):
    """
    Plot the time difference between two consecutive
    messages.
    """
    f, ax = plt.subplots(1,1,figsize=(6,4))
    ax: plt.Axes
    ships = sa.extract_all(skip_filter=True)
    time_diffs = []
    it = 0
    maxlen = len(ships)
    for ship in ships.values():
        it +=1
        print(f"Working on ship {it}/{maxlen}")
        for track in ship.tracks:
            for i in range(1,len(track)):
                time_diffs.append(track[i].timestamp - track[i-1].timestamp)
    
    # Quantiles of time diffs
    qs = np.quantile(
        time_diffs,
        [0.99,0.95,0.90]
    )

    # Boxplot of time diffs
    ax.boxplot(
        time_diffs,
        vert=False,
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor=COLORWHEEL[0], color=COLORWHEEL[0]),
        medianprops=dict(color=COLORWHEEL[1]),
        whiskerprops=dict(color=COLORWHEEL[1]),
        capprops=dict(color=COLORWHEEL[1]),
        flierprops=dict(color=COLORWHEEL[1], markeredgecolor=COLORWHEEL[1])
    )
    # Remove y ticks
    ax.set_yticks([])
    
    # Legend with heading
    ax.legend(handles=[
        patches.Patch(color=COLORWHEEL[0],label=f"1% larger than {qs[0]:.2f} s"),
        patches.Patch(color=COLORWHEEL[0],label=f"5% larger than {qs[1]:.2f} s"),
        patches.Patch(color=COLORWHEEL[0],label=f"10% larger than {qs[2]:.2f} s")
    ])
    
    ax.set_xlabel("Time difference [s]")
    
    ax.set_title(
        "Time difference between two consecutive messages",fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig("aisstats/out/time_diffs.pdf")

def plot_reported_vs_calculated_speed(sa:SearchAgent):
    """
    Plot the reported speed against the
    calculated speed.
    """
    f, ax = plt.subplots(1,1,figsize=(6,4))
    ax: plt.Axes
    ships = sa.extract_all(skip_filter=True)
    speeds = []
    it = 0
    maxlen = len(ships)
    for ship in ships.values():
        it +=1
        print(f"Working on ship {it}/{maxlen}")
        for track in ship.tracks:
            for i in range(1,len(track)):
                rspeed = split.avg_speed(track[i-1],track[i])
                cspeed = split.speed_from_position(track[i-1],track[i])
                speeds.append(rspeed - cspeed)
    
    # Boxplot of speeds
    ax.boxplot(
        speeds,
        vert=False,
        labels=["Speed"],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor=COLORWHEEL[0], color=COLORWHEEL[0]),
        medianprops=dict(color=COLORWHEEL[1]),
        whiskerprops=dict(color=COLORWHEEL[1],width=1.5),
        capprops=dict(color=COLORWHEEL[1]),
        flierprops=dict(color=COLORWHEEL[1], markeredgecolor=COLORWHEEL[1])
    )
    
    # Quantiles of speeds
    s_qs = np.quantile(
        speeds,
        [0.005,0.995,0.025,0.975,0.05,0.95]
    )
    q_labels_h = [
        f"99% within [{s_qs[0]:.2f} kn,{s_qs[1]:.2f} kn]",
        f"95% within [{s_qs[2]:.2f} kn,{s_qs[3]:.2f} kn]",
        f"90% within [{s_qs[4]:.2f} kn,{s_qs[5]:.2f} kn]"
    ]

    # Legend with heading
    ax.legend(handles=[
        patches.Patch(color=COLORWHEEL[0],label=q_labels_h[0]),
        patches.Patch(color=COLORWHEEL[0],label=q_labels_h[1]),
        patches.Patch(color=COLORWHEEL[0],label=q_labels_h[2])
    ])
    
    ax.set_xlabel("Difference [kn]")
    
    ax.set_title(
        "Difference between reported and calculated speed [kn]",fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig("aisstats/out/speed_reported.pdf")
    
def plot_distance_between_messages(sa:SearchAgent):
    """
    Plot the distance between two consecutive messages.
    """
    f, ax = plt.subplots(1,1,figsize=(8,8))
    ax: plt.Axes
    ships = sa.extract_all(skip_filter=True)
    distances = []
    it = 0
    maxlen = len(ships)
    for ship in ships.values():
        it +=1
        print(f"Working on ship {it}/{maxlen}")
        for track in ship.tracks:
            for i in range(1,len(track)):
                d = haversine(
                    track[i-1].lon,
                    track[i-1].lat,
                    track[i].lon,
                    track[i].lat
                )
                distances.append(d)
                
    # KDE of distances
    kde = gaussian_kde(
        distances,
        bw_method=_bw_sel
    )
    xs = np.linspace(0,max(distances),1000)
    ax.plot(
        xs,
        kde(xs),
        label=f"Kernel density estimate",
        color=COLORWHEEL[0]
    )
    
    # Quantiles of distances
    qs = np.quantile(
        distances,
        [0.99,0.95,0.90]
    )

    # Histogram of distances
    ax.hist(
        distances,
        bins=300,
        density=True,
        alpha=0.6,
        color=COLORWHEEL[0]
    )
    
    # Legend with heading
    ax.legend(handles=[
        patches.Patch(color=COLORWHEEL[0],label=f"1% larger than {qs[0]:.2f} miles"),
        patches.Patch(color=COLORWHEEL[0],label=f"5% larger than {qs[1]:.2f} miles"),
        patches.Patch(color=COLORWHEEL[0],label=f"10% larger than {qs[2]:.2f} miles")
    ])
    
    ax.set_xlabel("Distance between two consecutive messages [miles]")
    ax.set_ylabel("Density")
    
    # Log scale
    ax.set_xscale('log')
    
    ax.set_title(
        "Distance between two consecutive messages",fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig("aisstats/out/distance_between_messages.pdf")
    
def plot_heading_and_speed_changes(sa:SearchAgent):
    """
    Plot the changes in heading and speed between
    two consecutive messages.
    """
    f, ax = plt.subplots(2,1,figsize=(8,10))
    ax: list[plt.Axes]

    ships = sa.extract_all(skip_filter=True)
    heading_changes = []
    speed_changes = []
    it = 0
    maxlen = len(ships)
    for ship in ships.values():
        it +=1
        print(f"Working on ship {it}/{maxlen}")
        for track in ship.tracks:
            for i in range(1,len(track)):
                _chheading = _heading_change(
                    track[i-1].COG,
                    track[i].COG
                )
                _chspeed = abs(track[i].SOG - track[i-1].SOG)
                heading_changes.append(_chheading)
                speed_changes.append(_chspeed)
                
    # Quantiles of heading changes
    h_qs = np.quantile(
        heading_changes,
        [0.005,0.995,0.025,0.975,0.05,0.95]
    )
    q_labels_h = [
        f"99% within [{h_qs[0]:.2f}°,{h_qs[1]:.2f}°]",
        f"95% within [{h_qs[2]:.2f}°,{h_qs[3]:.2f}°]",
        f"90% within [{h_qs[4]:.2f}°,{h_qs[5]:.2f}°]"
    ]
    # Heading Quantiles as vertical lines
    hl11 = ax[0].axvline(h_qs[0],color=COLORWHEEL[0],label=q_labels_h[0],ls="--")
    hl12 = ax[0].axvline(h_qs[1],color=COLORWHEEL[0],label=q_labels_h[0],ls="--")
    hl21 = ax[0].axvline(h_qs[2],color=COLORWHEEL[0],label=q_labels_h[1],ls="-.")
    hl22 = ax[0].axvline(h_qs[3],color=COLORWHEEL[0],label=q_labels_h[1],ls="-.")
    hl31 = ax[0].axvline(h_qs[4],color=COLORWHEEL[0],label=q_labels_h[2],ls=":")
    hl32 = ax[0].axvline(h_qs[5],color=COLORWHEEL[0],label=q_labels_h[2],ls=":")
    
    # Histogram of heading changes
    ax[0].hist(
        heading_changes,
        bins=100,
        density=True,
        alpha=0.8,
        color=COLORWHEEL[0]
    )
    
    # Quantiles of speed changes
    s_qs = np.quantile(
        speed_changes,
        [0.99,0.95,0.90]
    )
    
    q_labels_s = [
        f"99% smaller than {s_qs[0]:.2f} kn",
        f"95% smaller than {s_qs[1]:.2f} kn",
        f"90% smaller than {s_qs[2]:.2f} kn"
    ]
    
    # Speed Quantiles as vertical lines
    sl1 = ax[1].axvline(s_qs[0],color=COLORWHEEL[0],label=q_labels_s[0],ls="--")
    sl2 = ax[1].axvline(s_qs[1],color=COLORWHEEL[0],label=q_labels_s[1],ls="-.")
    sl3 = ax[1].axvline(s_qs[2],color=COLORWHEEL[0],label=q_labels_s[2],ls=":")
    
    # Histogram of speed changes
    ax[1].hist(
        speed_changes,
        bins=200,
        density=True,
        alpha=0.8,
        color=COLORWHEEL[0]
    )
    
    # Legend with heading
    ax[0].legend(handles=[hl11,hl21,hl31])
    ax[0].set_xlabel("Change in heading [°]")
    ax[0].set_ylabel("Density")
    ax[0].set_title(
        "Change in heading between two consecutive messages",fontsize=10
    )
    
    ax[1].legend(handles=[sl1,sl2,sl3])
    ax[1].set_xlabel("Absolute change in speed [knots]")
    ax[1].set_ylabel("Density")
    ax[1].set_title(
        "Change in speed between two consecutive messages",fontsize=10
    )
    ax[1].set_xlim(-0.2,4)
    
    plt.tight_layout()
    plt.savefig("aisstats/out/heading_speed_changes_all.pdf")
    
    
def inspect_trajectory_splits(sa: SearchAgent):
    """
    The data-driven splitting of trajectories left some questions
    open. A visual inspection of the trajectories that were acccepted
    showed that some of them seemed to be split although they belonged
    to one trajectory. To inspect this, we plot the changes in heading
    and speed between two consecutive messages, which have been determined
    to split a trajectory.
    """
    f, ax = plt.subplots(2,1,figsize=(8,10))
    sax1 = ax[0].twinx()
    sax2 = ax[1].twinx()
    ax: list[plt.Axes]
    tgaps = np.linspace(180,180*5,5)

    labels = [f"tgap = {tgap} s" for tgap in tgaps]
    hc_hist = []
    sc_hist = [] 
    for col,tgap in enumerate(tgaps):
        ships = sa.extract_all(max_tgap=tgap,max_dgap=np.inf)
        heading_changes = []
        speed_changes = []
        it = 0
        maxlen = len(ships)
        for ship in ships.values():
            it +=1
            print(f"Working on ship {it}/{maxlen}")
            if len(ship.tracks) > 1:
                for i in range(1,len(ship.tracks)):
                    _chheading = _heading_change(
                        ship.tracks[i-1][-1].COG,
                        ship.tracks[i][0].COG
                    )
                    _chspeed = abs(ship.tracks[i][0].SOG - ship.tracks[i-1][-1].SOG)
                    heading_changes.append(_chheading)
                    speed_changes.append(_chspeed)
                    
        # KDE of heading changes
        h_kde = gaussian_kde(heading_changes,bw_method=_bw_sel)
        h_xs = np.linspace(-180,180,1000)
        ax[0].plot(
            h_xs,
            h_kde(h_xs),
            label=f"tgap = {tgap} s",
            color=COLORWHEEL[col]
        )
        
        # KDE of speed changes
        s_kde = gaussian_kde(speed_changes,bw_method="silverman")
        s_xs = np.linspace(0,10,1000)
        ax[1].plot(
            s_xs,
            s_kde(s_xs),
            label=f"tgap = {tgap} s",
            color=COLORWHEEL[col]
        )
        
        # Add to histogram
        hc_hist.append(heading_changes)
        sc_hist.append(speed_changes)
        
    # Histogram of speed changes
    sax2.hist(
        sc_hist,
        bins=100,
        density=True,
        alpha=0.6,
        color=COLORWHEEL
    )
    ax[1].set_xlim(-0.2,10)

    # Histogram of heading changes
    sax1.hist(
        hc_hist,
        bins=100,
        density=True,
        alpha=0.6,
        color=COLORWHEEL
    )
        
    # Legend with heading
    ax[0].legend(title = "Max. time gap between messages")
    ax[0].set_xlabel("Change in heading [°]")
    ax[0].set_ylabel("Density")
    ax[0].set_title(
        "Change in heading between two consecutive messages\n"
        "that split a trajectory",fontsize=10
    )
    align_yaxis(ax[0], 0, sax1, 0)
    
    ax[1].legend(title = "Max. time gap between messages")
    ax[1].set_xlabel("Absolute change in speed [knots]")
    ax[1].set_ylabel("Density")
    ax[1].set_title(
        "Change in speed between two consecutive messages\n"
        "that split a trajectory",fontsize=10
    )
    align_yaxis(ax[1], 0, sax2, 0)
    
    plt.tight_layout()
    plt.savefig("aisstats/out/heading_speed_changes.pdf")
    
def plot_average_complexity(ships: dict[int,TargetShip],
                            savename: str):
    """
    Calulates the mean of the cosine of the angles
    enclosed between three consecutive messages 
    for several standard deviations.
    """
    sds = np.array([0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.1,0.2,0.3])
    avg_cosines = []
    for sd in sds:
        recipe = Recipe(
                # partial(too_few_obs,n=50),
                partial(spatial_deviation,sd=sd)
            )
        inpsctr = pytsa.Inspector(
                data=ships,
                recipe=recipe
            )
        acc, rej = inpsctr.inspect(njobs=1)
        tcosines = []
        for ship in rej.values():
            for track in ship.tracks:
                for i in range(1,len(track)-1):
                    a = track[i-1]
                    b = track[i]
                    c = track[i+1]
                    tcosines.append(
                        split.cosine_of_angle_between(a,b,c)
                    )
        avg_cosines.append(np.nanmean(np.abs(tcosines)))
        
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(sds,avg_cosines,color=COLORWHEEL[0])
    ax.set_xlabel("Standard deviation")
    ax.set_ylabel(r"Average complexity $\bar{\cos(\theta)}$")
    ax.set_title("Average cosine of the angle between three consecutive messages")
    plt.savefig(savename)

def binned_heatmap(targets: dict[int,TargetShip], 
                   bb: BoundingBox,
                   savename: str) -> None:
    """
    Split the bounding box into the closest
    amount of square pixels fitting in the 
    resolution. Then count the number of 
    messages in each pixel and plot a heatmap.
    """
    
    lonpx = 500
    latpx = lonpx
        
    # Create a grid of pixels
    x = np.linspace(bb.LONMIN,bb.LONMAX,lonpx)
    y = np.linspace(bb.LATMIN,bb.LATMAX,latpx)
    xx, yy = np.meshgrid(x,y)
    
    # Count the number of messages in each pixel
    counts = np.zeros((latpx,lonpx))
    for ship in targets.values():
        for track in ship.tracks:
            for msg in track:
                # Find the closest pixel
                i = np.argmin(np.abs(x - msg.lon))
                j = np.argmin(np.abs(y - msg.lat))
                counts[j,i] += 1
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12,10))
    ax: plt.Axes

    # Add coastline redered to an image
    # and plot it on top of the heatmap
    plot_coastline("/home/s2075466/aisplanner/data/geometry/",bb,ax=ax)
    
    # Mask the pixels with no messages
    counts = np.ma.masked_where(counts == 0,counts)

    # Log transform of counts to avoid
    # spots with many messages to dominate
    # the plot
    loglogcounts = np.vectorize(lambda x: np.log(np.log(x+1)+1))(counts)
    
    cmap = matplotlib.colormaps["gist_stern_r"]
    cmap.set_bad(alpha=0)
    ax.grid(False)
    pcm = ax.pcolormesh(xx,yy,loglogcounts,cmap=cmap)#,shading="gouraud")
    #plt.tight_layout()
    
    # Small  inset Colorbar
    cbaxes = inset_axes(ax, width="40%", height="2%", loc=4, borderpad = 2)
    cbaxes.grid(False)
    cbar = fig.colorbar(pcm,cax=cbaxes, orientation="horizontal")
    cbar.set_label(r"Route density ($n_{msg}$)",color="white")

    newticks = np.linspace(1,loglogcounts.max(),3)
    cbar.set_ticks(
        ticks = newticks,
        labels = [f"{(np.exp(np.exp(t))-np.exp(1))/np.exp(1):.0f}" for t in newticks]
    )
    
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color="white")
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    #ax.set_title("Heatmap of messages")
    
    
    plt.savefig(savename,dpi=300)
    
def plot_trlen_vs_nmsg(ships: dict[int,TargetShip],
                       savename: str):
    """
    Plot the length of a trajectory against the number of
    messages in it.
    """
    f,ax = plt.subplots(1,1,figsize=(6,5))
    ll = len(ships)
    nmsg = []
    tlens = []
    for idx,ship in enumerate(ships.values()):
        for track in ship.tracks:
            print(f"Working on ship {idx+1}/{ll}")
            d = 0
            for i in range(1,len(track)):
                d += haversine(
                    track[i-1].lon,
                    track[i-1].lat,
                    track[i].lon,
                    track[i].lat
                )
            tlens.append(mi2nm(d))
            nmsg.append(len(track))
    ax.scatter(tlens,nmsg,color=COLORWHEEL[0],alpha=0.5,s=0.5)        
    ax.set_ylabel("Number of messages per trajectory")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Trajectory length [nm]")
    plt.savefig(savename)
    plt.close()
    
def plot_histograms(ships: dict[int,TargetShip],title: str, specs: dict):
    
    # Temporal gaps between messages
    # and lengths of trajectories
    MAXLEN = 1000 # trajectory length [miles]
    tgaps = []
    dgaps = []
    tlens = [] # Filled with tuples of (tlen, n_messages)
    for ship in ships.values():
        for track in ship.tracks:
            tlen = 0
            for i in range(1,len(track)):
                tgap = track[i].timestamp - track[i-1].timestamp
                d = haversine(
                    track[i-1].lon,
                    track[i-1].lat,
                    track[i].lon,
                    track[i].lat
                )
                tlen += d
                
                dgaps.append(d)
                tgaps.append(tgap)
            tlens.append((tlen,len(track)))
                
    # Vessel speeds for histogram
    speeds = SA.dynamic_msgs[fd.Fields12318.speed.name].values
    
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(32,20))
    
    # Font size
    fs = 12
    
    ax1: plt.Axes
    ax1.hist(tgaps,bins=100,density=False)
    ax1.set_xlabel("Time [s]",size = fs)
    ax1.set_ylabel("Count",size = fs)
    ax1.set_yscale('log')
    ax1.set_title(f"Histogram of Time Gaps\nbetween Messages [s]",size = fs)
    
    ax2: plt.Axes
    ax2.hist(speeds,bins=100,density=False)
    ax2.set_xlabel("Speed [knots]",size = fs)
    ax2.set_yscale('log')
    ax2.set_ylabel("Count",size = fs)
    ax2.set_title("Histogram of Speeds [knots]",size = fs)
    
    ax3: plt.Axes
    ax3.hist([len for len,nobs in tlens],bins=100,density=False)
    ax3.set_xlabel("Trajectory length [miles]",size = fs)
    ax3.set_ylabel("Count",size = fs)
    ax3.set_yscale('log')
    ax3.set_title(f"Histogram of Trajectory Lengths [miles]\ncut at {MAXLEN} miles",size = fs)
    
    ax4: plt.Axes
    ax4.scatter(
        [lens for lens,nobs in tlens],
        [nobs for lens,nobs in tlens],
        s=1.5
    )
    ax4.set_xlabel("Trajectory length [miles]",size = fs)
    ax4.set_ylabel("Number of Messages per Trajectory",size = fs)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_title("Trajectory Length vs. Number of Messages",size = fs)
    
    ax5: plt.Axes
    ax5.hist(dgaps,bins=100,density=False)
    ax5.set_xlabel("Distance between messages [miles]",size = fs)
    ax5.set_ylabel("Count",size = fs)
    ax5.set_yscale('log')
    ax5.set_title(f"Histogram of Distances between Messages [miles]\ncut at {MAXLEN} ",size = fs)

    # Turn off last subplot
    ax6: plt.Axes
    ax6.axis('off')
    
    # Annotate specs to plot
    printed_specs = "\n".join([f"{k}: {v}" for k,v in specs.items()])
    printed_specs = "Specifications:\n" + printed_specs
    plt.annotate(
        printed_specs,
        xy=(0.75, 0.25),
        xycoords='figure fraction',
        fontsize=12,
    )
    
    # Title
    plt.suptitle(f"{title}",size=12)
    
    # Save figure with ascending number
    plt.tight_layout()
    plt.savefig(f"aisstats/out/{title}.png",dpi=300)
    plt.close()
if __name__ == "__main__":
    # Create a SearchAgent object --------------------------------------------------
    SPEEDRANGE = (1,30) # knots
    MAX_TGAP = 400 # seconds
    MAX_DGAP = 4 # miles
    SD = 0.2 # standard deviation for trajectory jitter
    MINLEN = 50 # minimum length of trajectory
    
    specs = {
        "Speed range [kn]": SPEEDRANGE,
        "Max time gap [s]": MAX_TGAP,
        "Max distance gap [mi]": MAX_DGAP,
        "SD [°]": SD,
        "Min traj length [msg]": MINLEN
    }
    
    
    SA = SearchAgent(
        dynamic_paths=TEST_FILE_DYN,
        static_paths=TEST_FILE_STA,
        frame=SEARCHAREA,
        preprocessor=partial(speed_filter, speeds=SPEEDRANGE)
    )

    # Heading changes and speed changes --------------------------------------------
    # inspect_trajectory_splits(SA)
    # plot_heading_and_speed_changes(SA)
    # plot_distance_between_messages(SA)
    # plot_reported_vs_calculated_speed(SA)
    # plot_time_diffs(SA)
    
    # Speed histogram --------------------------------------------------------------
    # plot_speed_histogram(SA,"aisstats/out/speed_histogram.pdf")
    
    
    # f = pd.read_csv(TEST_FILE_DYN)
    # la, lo = f[fd.Fields12318.lat.name].values, f[fd.Fields12318.lon.name].values
    # lat_lon_outofbounds(la,lo)
    
    # ships = SA.extract_all(njobs=2)#,skip_tsplit=True)
    
    # Plot average complexity ------------------------------------------------------
    # plot_average_complexity(ships)
    
    # Plot calculated vs reported speed --------------------------------------------
    plot_speed_scatter(sa=SA,savename="aisstats/out/speed_scatter.png") 

    # Plot trajectory jitter --------------------------------------------------------
    # with MemoryLoader():
        #plot_sd_vs_rejection_rate(ships,"aisstats/out/sd_vs_rejection_rate_08_02_21.pdf.pdf")
        # plot_trajectory_jitter(ships)
    
    # Plot trajectory length by number of observations
    # plot_trlen_vs_nmsg(ships,"aisstats/out/trlen_vs_nmsg_1-30.pdf")
    #
    # Plot histograms of raw data ---------------------------------------------------
    # plot_histograms(
    #     ships,
    #     title=f"Histograms of raw AIS Data for {SPEEDRANGE[0]}-{SPEEDRANGE[1]} knots",
    #     specs={}
    # )
    # Split the ships into accepted and rejected ------------------------------------
    # from pytsa.trajectories.rules import *
    # ExampleRecipe = Recipe(
    #     partial(too_few_obs, n=MINLEN),
    #     partial(spatial_deviation, sd=SD)
    # )
    # from pytsa.trajectories import Inspector
    # inspctr = Inspector(
    #     data=ships,
    #     recipe=ExampleRecipe
    # )
    # accepted, rejected = inspctr.inspect(njobs=1)
    
    # Plot heatmap -----------------------------------------------------------------
    # binned_heatmap(ships,SEARCHAREA,"aisstats/out/heatmap_no_rejoin.png")
    
    # Plot routes and speeds for accepted and rejected ------------------------------
    # vals = list(accepted.values())
    # np.random.shuffle(vals)
    # for i in range(60,100):
    #     plot_simple_route(vals[i],"rejected")
    #     plot_speeds_and_route(list(rejected.values())[i],"rejected")
    #     plot_speeds_and_route(list(accepted.values())[i],"accepted")
    
    # Latlon shape plots -----------------------------------------------------------
    # Get 100 random ships from each
    # accepted and rejected
    # good = [accepted[k] for k in np.random.choice(list(accepted.keys()),500)]
    # bad = [rejected[k] for k in np.random.choice(list(rejected.keys()),500)]
    
    # plot_latlon_shapes(good,bad)
    # plot_latlon_kde(accepted,"accepted")
    # plot_latlon_kde(rejected,"rejected")
    
    # # Construct splines ------------------------------------------------------------
    # rejected = SA._construct_splines(rejected,"linear")
    # accepted = SA._construct_splines(accepted,"linear")


    # Plot trajectories on map ------------------------------------------------------
    # AMSTERDAM = BoundingBox(
    # LATMIN=52.79,
    # LATMAX=53.28,
    # LONMIN=5.5,
    # LONMAX=6.5)
    # plot_trajectories_on_map(ships, "all",{},SEARCHAREA)
    # plot_trajectories_on_map(accepted,"accepted",specs,SEARCHAREA)
    # plot_trajectories_on_map(rejected,"rejected",specs,SEARCHAREA)

    # # Plot histograms of accepted and rejected --------------------------------------
    # plot_histograms(
    #     accepted,
    #     title=f"Histograms of accepted AIS Data for {SPEEDRANGE[0]}-{SPEEDRANGE[1]} knots",
    #     specs=specs
    # )
    # plot_histograms(
    #     rejected,
    #     title=f"Histograms of rejected AIS Data for {SPEEDRANGE[0]}-{SPEEDRANGE[1]} knots",
    #     specs=specs
    # )   