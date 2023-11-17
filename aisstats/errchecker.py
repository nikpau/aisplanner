"""
Module for visually inspecting how far
the interpolated speed of a target vessel
deviates from the the speed calulated from
the distance between two consecutive
positions and the time between them.
"""
import pytsa
from pytsa import SearchAgent, TargetVessel, TimePosition
from pytsa.structs import Position
from aisplanner.encounters.main import GriddedNorthSea
from aisplanner.encounters.filter import haversine
from matplotlib import pyplot as plt
from aisstats.psd import time_range
import numpy as np
from aisplanner.encounters.encmap import plot_coastline
from scipy.stats import gaussian_kde
import multiprocessing as mp
from aisplanner.misc import MemoryLoader
from KDEpy.bw_selection import improved_sheather_jones

TEST_FILE_DYN = 'data/aisrecords/2021_07_01.csv'
TEST_FILE_STA = 'data/aisrecords/msgtype5/2021_07_01.csv'
# TEST_FILE_DYN = '/warm_archive/ws/s2075466-ais/decoded/jan2020_to_jun2022/2021_07_01.csv'
# TEST_FILE_STA = '/warm_archive/ws/s2075466-ais/decoded/msgtype5/2021_07_01.csv'
SEARCHAREA = GriddedNorthSea(nrows=1, ncols=1, utm=False).cells[0]
SAMPLINGRATE = 30 # seconds
# Sampling rate in hours
SAMPLINGRATE_H = SAMPLINGRATE / 60 / 60
import pickle
import pandas as pd
from functools import partial
from aisplanner.dataprep import _file_descriptors as fd

# Use matplotlib style `bmh`
#plt.style.use('bmh')
plt.rcParams["font.family"] = "monospace"

def _bw_sel(kde: gaussian_kde) -> float:
    data = kde.dataset.reshape(-1,1)
    return improved_sheather_jones(data)

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

def area_center(area: pytsa.LatLonBoundingBox) -> Position:
    """
    Get the center of the search area in UTM coordinates.
    """
    return Position(
        (area.LATMIN + area.LATMAX)/2,
        (area.LONMIN + area.LONMAX)/2
    )

def plot_speeds_and_route(tv: TargetVessel, mode: str) -> None:
    """
    Plot the trajectory of a target vessel
    as well as the speeds calculated from
    individual positions and the interpolated
    speeds.
    """
    # Distance from first position to all others
    # as histogram with kernel density estimate
    distances = []
    for track in tv.tracks:
        for i in range(1,len(track)):
            d = haversine(
                track[0].lon,
                track[0].lat,
                track[i].lon,
                track[i].lat
            )
            distances.append(d)
    if not distances:
        return None

    # KDE of distances
    d_kde = gaussian_kde(distances)
    d_xs = np.linspace(0,max(distances),1000)
    d_ys = d_kde(d_xs)
    
    # Real positions and speeds
    tss = []
    rlons, rlats, rspeeds = [], [], []
    for track in tv.tracks:
        lons, lats, speeds, ts = [], [], [], []
        for msg in track:
            lons.append(msg.lon)
            lats.append(msg.lat)
            speeds.append(msg.SOG)
            ts.append(msg.timestamp)
        rlons.append(lons)
        rlats.append(lats)
        rspeeds.append(speeds)
        tss.append(ts)
    
    fig, axd = plt.subplot_mosaic(
        [
            ['left', 'upper right'],
            ['left', 'lower right']
        ], layout="constrained", figsize=(16,10)
    )
    
    # Plot the trajectory
    for lo,la, sp in zip(rlons,rlats,rspeeds):
        f = axd["left"].scatter(lo,la,c=sp,s=18,label='Trajectory',cmap='inferno')
        axd["left"].plot(lo,la,"k--",label='Trajectory', alpha = 0.5)
    
    # Colorbar
    fig.colorbar(f, ax=axd["left"], label="Speed [knots]")
    
    # Plot the distances from first position
    axd["upper right"].hist(distances,bins=20,density=True)
    
    # Plot the KDE
    axd["upper right"].plot(d_xs,d_ys,'r',label='KDE')
    
    # Plot the real speeds
    axd["lower right"].plot(flatten(rspeeds),'r',label='Real speed')
    
    # Set labels
    axd["left"].set_xlabel('Longitude')
    axd["left"].set_ylabel('Latitude')
    axd["upper right"].set_xlabel('Distance to first message [miles]')
    axd["upper right"].set_ylabel('Speed [knots]')
    axd["lower right"].set_xlabel('Time')
    axd["lower right"].set_ylabel('Speed [knots]')
    
    # Set y lim
    axd["lower right"].set_ylim(-1,20)
    
    
    # Set legend
    axd["left"].legend()
    axd["upper right"].legend()
    axd["lower right"].legend()
    
    # Save figure
    plt.savefig(f"aisstats/out/errchecker/{tv.mmsi}_{mode}.png",dpi=300)
    plt.close()

def plot_trajectory_jitter(sa: SearchAgent,
                           tpos: TimePosition,
                           ships: dict[int,TargetVessel],
                           sds: list | np.ndarray,
                           mode: str) -> None:
    
    assert mode in ["accepted","rejected"]
    np.random.seed(424)
    
    for ship in ships.values():
        # Center trajectory
        # Get mean of lat and lon
        for track in ship.tracks:
            latmean = sum([p.lat for p in track]) / len(track)
            lonmean = sum([p.lon for p in track]) / len(track)
            # Subtract mean from all positions
            for msg in track:
                msg.lat -= latmean
                msg.lon -= lonmean
            
    # Plot trajectories for different sds
    ncols = 3
    div,mod = divmod(len(sds),ncols)
    nrows = div + 1 if mod else div
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5*nrows))
    for row in range(nrows):
        for col in range(ncols):
            try:
                acc,rej = sa.split(
                    ships,
                    tpos,
                    overlap_tpos=False,
                    sd=sds[row*ncols+col],
                    minlen=50,
                    njobs=1
                )
            except IndexError:
                break
            to_process = acc if mode == "accepted" else rej
            sign = "<" if mode == "rejected" else ">"
    
            for ship in to_process.values():
                for track in ship.tracks:
                    axs[row,col].plot(
                        [p.lon for p in track],
                        [p.lat for p in track],
                        alpha=0.2, marker = "x", markersize = 0.15, color = "k", linewidth = 0.1
                    )
            axs[row,col].set_xlabel("Longitude")
            axs[row,col].set_ylabel("Latitude")
            axs[row,col].set_title(f"Trajectories with sd {sign} {sds[row*ncols+col]:.2f}")
    
    
    plt.title(f"Trajectory jitter for {mode} vessels.")
    plt.savefig(f"aisstats/out/trjitter_{mode}.png", dpi=500)
    plt.close()
    
def plot_trajectories_on_map(ships: dict[int,TargetVessel], mode: str,specs: dict):
    """
    Plot all trajectories on a map.
    """
    assert mode in ["all","accepted","rejected"]
    fig, ax = plt.subplots(figsize=(10,16))
    plot_coastline(ax=ax)
    for ship in ships.values():
        for track in ship.tracks:
            ax.plot(
                [p.lon for p in track],
                [p.lat for p in track],
                alpha=0.5, linewidth=0.3, marker = "x", markersize = 0.5
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title(f"Trajectories for {mode} vessels: {specs}", size = 8)
    plt.tight_layout()
    plt.savefig(f"aisstats/out/trajmap_{mode}_{specs['Min traj length [msg]']}.png",dpi=800)
    plt.close()
    
def _maxmin_norm(arr:np.ndarray) -> np.ndarray:
    """
    Normalize an array by subtracting the minimum
    and dividing by the maximum.
    """
    return (arr - arr.min()) / (arr.max() - arr.min())
    
def plot_latlon_kde(ships: dict[int,TargetVessel], mode: str):
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

def run_filter(ships: dict[int,TargetVessel], tpos: TimePosition):
    
    #for sd in np.arange(0.01,2,0.1):
    sd = 0.05
    accepted, rejected = SA.split(
        targets=ships,
        tpos=tpos,
        overlap_tpos=False,
        sd=sd,
        minlen=20
    )
    rejected = SA._construct_splines(rejected,"linear")
    accepted = SA._construct_splines(accepted,"linear")
    plot_trajectory_jitter(rejected,sd,"rejected")
    plot_trajectory_jitter(accepted,sd,"accepted")

    for i in range(80,100):
        plot_speeds_and_route(list(rejected.values())[i],"rejected")

    for i in range(80,100):
        plot_speeds_and_route(list(accepted.values())[i],"accepted")
        
def plot_trajectory_length_by_obscount(ships: dict[int,TargetVessel]):
    
    # n_obs bins
    n_obs_bins = np.arange(5,75,5)
    tlens = [[] for _ in range(len(n_obs_bins))]
    
    # Walk through all ships' tracks
    # and append trajectory lengths to
    # the correct bin
    for ship in ships.values():
        for track in ship.tracks:
            n_obs = len(track)
            for i in range(len(n_obs_bins)):
                if n_obs <= n_obs_bins[i]:
                    tlens[i].append(len(track))
                    break
                
    # Set up KDEs for each bin
    kdes = []
    for tlen in tlens:
        kdes.append(gaussian_kde(tlen))
        
    # Plot KDEs and names of bins
    fig, ax = plt.subplots(figsize=(16,10))
    max_tlen = max([max(tlen) for tlen in tlens])
    for i in range(len(n_obs_bins)):
        ax.plot(
            np.linspace(0,max_tlen+50,1000),
            kdes[i](np.linspace(0,max_tlen+50,1000)),
            label=f"n_obs <= {n_obs_bins[i]}"
        )
    ax.set_xlabel("Trajectory length [miles]")
    ax.set_ylabel("Density")
    ax.set_title("Trajectory length by number of observations_1-30")
    ax.legend()
    plt.savefig("aisstats/out/trajlen_by_nobs_1-30.png",dpi=300)
    plt.close()
    
def plot_latlon_shapes(good: list[TargetVessel],bad: list[TargetVessel]):
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
    
def plot_sd_vs_rejection_rate(ships: dict[int,TargetVessel], tpos: TimePosition):
    """
    Plot the standard deviation of the trajectory jitter
    against the rejection rate.
    """
    sds = np.array([0,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.1,0.2,0.3])
    minlens = np.array([0,2,5,10,20,30,40,50,60,70])
    fig, ax = plt.subplots(figsize=(8,5))
    for minlen in minlens:
        rejected = []
        accepted = []
        for sd in sds:
            acc,rej = SA.split(
                targets=ships,
                tpos=tpos,
                overlap_tpos=False,
                sd=sd,
                minlen=minlen
            )
            rejected.append(sum([len(r.tracks) for r in rej.values()]))
            accepted.append(sum([len(a.tracks) for a in acc.values()]))
        rejected = np.array(rejected)
        accepted = np.array(accepted)
        total = rejected + accepted
        rejection_rate = rejected / total
        
        ax.plot(sds,rejection_rate,label=f"tr. len > {minlen} obs")
    ax.set_xlabel("Standard deviation")
    ax.set_ylabel("Rejection rate")
    ax.legend()
    ax.set_title("Rejection rate vs. standard deviation of trajectory")
    plt.savefig("aisstats/out/sd_vs_rejection_rate.pdf")
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
    
def inspect_trajectory_splits(sa: SearchAgent):
    """
    The data-driven splitting of trajectories left some questions
    open. A visual inspection of the trajectories that were acccepted
    showed that some of them seemed to be split although they belonged
    to one trajectory. To inspect this, we plot the changes in heading
    and speed between two consecutive messages, which have been determined
    to split a trajectory.
    """
    colorwheel = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]
    f, ax = plt.subplots(2,1,figsize=(8,10))
    sax1 = ax[0].twinx()
    sax2 = ax[1].twinx()
    ax: list[plt.Axes]
    tgaps = np.linspace(180,180*5,5)

    labels = [f"tgap = {tgap} s" for tgap in tgaps]
    hc_hist = []
    sc_hist = [] 
    for col,tgap in enumerate(tgaps):
        ships = sa.get_all_ships(max_tgap=tgap,max_dgap=np.inf)
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
            color=colorwheel[col]
        )
        
        # KDE of speed changes
        s_kde = gaussian_kde(speed_changes,bw_method=_bw_sel)
        s_xs = np.linspace(0,10,1000)
        ax[1].plot(
            s_xs,
            s_kde(s_xs),
            label=f"tgap = {tgap} s",
            color=colorwheel[col]
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
        color=colorwheel
    )
    ax[1].set_xlim(0,10)

    # Histogram of heading changes
    sax1.hist(
        hc_hist,
        bins=100,
        density=True,
        alpha=0.6,
        color=colorwheel
    )
        
    # Legend with heading
    ax[0].legend(title = "Max. time gap between messages")
    ax[0].set_xlabel("Change in heading [°]")
    ax[0].set_ylabel("Density")
    ax[0].set_title(
        "Change in heading between two consecutive messages\n"
        "that split a trajectory",fontsize=10
    )
    
    ax[1].legend(title = "Max. time gap between messages")
    ax[1].set_xlabel("Absolute change in speed [knots]")
    ax[1].set_ylabel("Density")
    ax[1].set_title(
        "Change in speed between two consecutive messages\n"
        "that split a trajectory",fontsize=10
    )
    plt.tight_layout()
    plt.savefig("aisstats/out/heading_speed_changes.pdf")
            
    
def plot_histograms(ships: dict[int,TargetVessel],title: str, specs: dict):
    
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
        msg12318file=TEST_FILE_DYN,
        msg5file=TEST_FILE_STA,
        frame=SEARCHAREA,
        preprocessor=partial(speed_filter, speeds=SPEEDRANGE)
    )

    # Create starting positions for the search.
    # This is just the center of the search area.
    center = area_center(SEARCHAREA)
    tpos = TimePosition(
        timestamp="2021-07-01",
        lat=center.lat,
        lon=center.lon
    )

    # Initialize the SearchAgent
    SA.init(tpos)

    # Heading changes and speed changes --------------------------------------------
    inspect_trajectory_splits(SA)
    
    # ships = SA.get_all_ships(max_tgap=MAX_TGAP,max_dgap=MAX_DGAP)

    # Plot trajectory jitter --------------------------------------------------------
    # with MemoryLoader():
    #     plot_sd_vs_rejection_rate(ships,tpos)
    #     plot_trajectory_jitter(SA,tpos,ships,np.arange(0.05,0.55,0.05),"rejected")
    
    # Plot trajectory length by number of observations
    # plot_trajectory_length_by_obscount(ships)
    # Plot histograms of raw data ---------------------------------------------------
    # plot_histograms(
    #     ships,
    #     title=f"Histograms of raw AIS Data for {SPEEDRANGE[0]}-{SPEEDRANGE[1]} knots",
    #     specs={}
    # )
    # Split the ships into accepted and rejected ------------------------------------
#     from pytsa.trajectories.rules import *
#     ExampleRecipe = Recipe(
#     partial(too_few_obs, n=100),
#     partial(too_small_spatial_deviation, sd=0.1)
# )
#     from pytsa.trajectories import TrajectorySplitter
#     trsp = TrajectorySplitter(
#         data=ships,
#         recipe=ExampleRecipe
#     )
#     accepted, rejected = trsp.split()
    
    # Plot routes and speeds for accepted and rejected ------------------------------
    # for i in range(80,100):
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


    # # Plot trajectories on map ------------------------------------------------------
    #plot_trajectories_on_map(ships, "all",{})
    # plot_trajectories_on_map(accepted,"accepted",specs)
    # plot_trajectories_on_map(rejected,"rejected",specs)

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