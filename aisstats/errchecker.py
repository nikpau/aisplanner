"""
Module for visually inspecting how far
the interpolated speed of a target vessel
deviates from the the speed calulated from
the distance between two consecutive
positions and the time between them.
"""
import pytsa
from pytsa import SearchAgent, TargetVessel, TimePosition
from aisplanner.encounters.main import GriddedNorthSea
from aisplanner.encounters.filter import haversine
from matplotlib import pyplot as plt
from aisstats.psd import time_range
import numpy as np
from aisplanner.encounters.encmap import plot_coastline

TEST_FILE_DYN = 'data/aisrecords/2021_07_01.csv'
TEST_FILE_STA = 'data/aisrecords/msgtype5/2021_07_01.csv'
SEARCHAREA, = GriddedNorthSea(nrows=1, ncols=1, utm=False).cells
SAMPLINGRATE = 30 # seconds
# Sampling rate in hours
SAMPLINGRATE_H = SAMPLINGRATE / 60 / 60
import pickle
import pandas as pd
from functools import partial
from aisplanner.dataprep import _file_descriptors as fd

# Use matplotlib style `bmh`
plt.style.use('bmh')

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

def area_center(area: pytsa.LatLonBoundingBox) -> pytsa.structs.Position:
        """
        Get the center of the search area in UTM coordinates.
        """
        return pytsa.structs.Position(
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
    # Get start and end time of the trajectory
    t0 = tv.track[0].timestamp
    t1 = tv.track[-1].timestamp
    
    # Lon, lat, calculated speeds, interpolated speeds
    lons, lats, i_speeds,c_speeds = [], [], [], [0]
    
    # Interpolate positions and speeds
    for moment in time_range(int(t0),int(t1),SAMPLINGRATE):
        # Get positions
        lons.append(tv.interpolation.lon(moment))
        lats.append(tv.interpolation.lat(moment))
        i_speeds.append(tv.interpolation.SOG(moment))
        if len(lons) > 1:
            # Calculate speed
            c_speeds.append(
                haversine(
                    lons[-2], lats[-2], lons[-1], lats[-1]
                ) / (SAMPLINGRATE_H)
            )
    
    # Real positions and speeds
    ts = []
    rlons, rlats, rspeeds, rcspeeds = [], [], [], [0]
    for msg in tv.track:
        rlons.append(msg.lon)
        rlats.append(msg.lat)
        rspeeds.append(msg.SOG)
        ts.append(msg.timestamp)
        if len(rlons) > 1:
            rcspeeds.append(
                haversine(
                    rlons[-2], rlats[-2], rlons[-1], rlats[-1]
                ) / ((ts[-1] - ts[-2]) / 60 / 60)
            )
    
    fig, axd = plt.subplot_mosaic(
        [
            ['left', 'upper right'],
            ['left', 'lower right']
        ], layout="constrained", figsize=(16,10)
    )
    
    # Plot the trajectory
    f = axd["left"].scatter(rlons,rlats,c=rspeeds,s=18,label='Trajectory',cmap='inferno')
    axd["left"].plot(rlons,rlats,"k--",label='Trajectory', alpha = 0.5)
    
    # Colorbar
    fig.colorbar(f, ax=axd["left"], label="Speed [knots]")
    
    # Plot the calculated speeds
    axd["upper right"].plot(c_speeds,'r',label='Calculated speed')
    
    # Plot the interpolated speeds
    axd["upper right"].plot(i_speeds,'b',label='Interpolated speed')
    
    # Calulate error
    e = [c-i for c,i in zip(c_speeds,i_speeds)]
    
    # Plot error
    axd["upper right"].plot(e,'g',label='Error (Calc. - Interp.)')
    
    # Plot the real speeds
    axd["lower right"].plot(rspeeds,'r',label='Real speed')
    
    # Plot the calculated speeds
    axd["lower right"].plot(rcspeeds,'b',label='Calculated speed', alpha = 0.5)
    
    # Plot the error
    e = [r-c for r,c in zip(rspeeds,rcspeeds)]
    axd["lower right"].plot(e,'g',label='Error (Real - Calc.)', alpha = 0.5)
    
    # Set labels
    axd["left"].set_xlabel('Longitude')
    axd["left"].set_ylabel('Latitude')
    axd["upper right"].set_xlabel('Time')
    axd["upper right"].set_ylabel('Speed [knots]')
    axd["lower right"].set_xlabel('Time')
    axd["lower right"].set_ylabel('Speed [knots]')
    
    # Set y lim
    axd["lower right"].set_ylim(-30,30)
    axd["upper right"].set_ylim(-30,30)
    
    
    
    # Set legend
    axd["left"].legend()
    axd["upper right"].legend()
    axd["lower right"].legend()
    
    # Save figure
    plt.savefig(f"aisstats/out/errchecker/{tv.mmsi}_{mode}.png",dpi=300)
    plt.close()

def plot_trajectory_jitter(ships: dict[int,TargetVessel],
                           sd: float,
                           mode: str) -> None:
    
    assert mode in ["accepted","rejected"]
    
    for ship in ships.values():
        # Center trajectory
        # Get mean of lat and lon
        latmean = sum([p.lat for p in ship.track]) / len(ship.track)
        lonmean = sum([p.lon for p in ship.track]) / len(ship.track)
        # Subtract mean from all positions
        for p in ship.track:
            p.lat -= latmean
            p.lon -= lonmean
            
    # Plot all trajectories
    fig, ax = plt.subplots(figsize=(16,10))
    for ship in ships.values():
        ax.scatter(
            [p.lon for p in ship.track],
            [p.lat for p in ship.track],
            alpha=0.5,s=0.5
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    sign = "<" if mode == "rejected" else ">"
    
    plt.title(f"Trajectory jitter for {mode} vessels with sd{sign}{sd:.2f}")
    plt.savefig(f"aisstats/out/trjitter_{sd:.2f}_{mode}.png", dpi=300)
    plt.close()
    
def plot_trajectories_on_map(ships: dict[int,TargetVessel], mode: str):
    """
    Plot all trajectories on a map.
    """
    fig, ax = plt.subplots(figsize=(10,16))
    plot_coastline(ax=ax)
    for ship in ships.values():
        ax.plot(
            [p.lon for p in ship.track],
            [p.lat for p in ship.track],
            alpha=0.5, linewidth=0.3
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(f"aisstats/out/trajmap_{mode}.pdf")
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

    
def plot_histograms(ships: dict[int,TargetVessel],title: str):
    
    # Temporal gaps between messages
    # and lengths of trajectories
    MAXGAP = 900 # [s]
    MAXLEN = 1000 # trajectory length [miles]
    MAXDIST = 10 # dist between messages [miles]
    tgaps = []
    dgaps = []
    tlens = [] # FIlled with tuples of (tlen, n_messages)
    for ship in ships.values():
        tlen = 0
        for i in range(1,len(ship.track)):
            tgap = ship.track[i].timestamp - ship.track[i-1].timestamp
            d = haversine(
                ship.track[i-1].lon,
                ship.track[i-1].lat,
                ship.track[i].lon,
                ship.track[i].lat
            )
            tlen += d
            if d < MAXDIST:
                dgaps.append(d)
            if tgap < MAXGAP:
                tgaps.append(tgap)
        tlens.append((tlen,len(ship.track)))
                
    # Vessel speeds for histogram
    speeds = SA.cell_data[fd.Fields12318.speed.name].values
    
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(16,10))
    
    ax1: plt.Axes
    ax1.hist(tgaps,bins=100,density=False)
    ax1.set_xlabel("Time [s]",size = 9)
    ax1.set_ylabel("Count",size = 9)
    ax1.set_title(f"Histogram of Time Gaps\nbetween Messages [s]. Cut at {MAXGAP}s",size = 9)
    
    ax2: plt.Axes
    ax2.hist(speeds,bins=100,density=False)
    ax2.set_xlabel("Speed [knots]",size = 9)
    ax2.set_ylabel("Count",size = 9)
    ax2.set_title("Histogram of Speeds [knots]",size = 9)
    
    ax3: plt.Axes
    ax3.hist([len for len,nobs in tlens if len < 400],bins=100,density=False)
    ax3.set_xlabel("Trajectory length [miles]",size = 9)
    ax3.set_ylabel("Count",size = 9)
    ax3.set_yscale('log')
    ax3.set_title(f"Histogram of Trajectory Lengths [miles]\ncut at {MAXLEN} miles",size = 9)
    
    ax4: plt.Axes
    ax4.scatter(
        [lens for lens,nobs in tlens],
        [nobs for lens,nobs in tlens],
        s=1
    )
    ax4.set_xlabel("Trajectory length [miles]",size = 9)
    ax4.set_ylabel("Number of Messages per Trajectory",size = 9)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_title("Trajectory Length vs. Number of Messages")
    
    ax5: plt.Axes
    ax5.hist(dgaps,bins=100,density=False)
    ax5.set_xlabel("Distance between messages [miles]",size = 9)
    ax5.set_ylabel("Count",size = 9)
    ax5.set_yscale('log')
    ax5.set_title(f"Histogram of Distances between Messages [miles]\ncut at {MAXLEN} ",size = 9)
    
    # Title
    plt.suptitle(f"{title}",size=12)
    
    # Save figure with ascending number
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"aisstats/out/{title}.png",dpi=300)
    plt.close()
if __name__ == "__main__":
    # Create a SearchAgent object --------------------------------------------------
    SPEEDRANGE = (1,30) # knots
    SA = SearchAgent(
        msg12318file=TEST_FILE_DYN,
        msg5file=TEST_FILE_STA,
        frame=SEARCHAREA,
        time_delta=60*24, # 24 hours
        n_cells=1,
        # filter=partial(speed_filter, speeds=SPEEDRANGE)
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
    
    ships = SA.get_raw_ships(tpos,True)

    accepted, rejected = SA.split(
        targets=ships,
        tpos=tpos,
        overlap_tpos=False,
        sd=0.1,
        minlen=100
    )
    rejected = SA._construct_splines(rejected,"linear")
    accepted = SA._construct_splines(accepted,"linear")

    plot_trajectories_on_map(ships, "all")
    plot_trajectories_on_map(accepted,"accepted")
    plot_trajectories_on_map(rejected,"rejected")

    plot_histograms(
        ships,
        title=f"Histograms of raw AIS Data for {SPEEDRANGE[0]}-{SPEEDRANGE[1]} knots"
    )
    plot_histograms(
        accepted,
        title=f"Histograms of accepted AIS Data for {SPEEDRANGE[0]}-{SPEEDRANGE[1]} knots"
    )
    plot_histograms(
        rejected,
        title=f"Histograms of rejected AIS Data for {SPEEDRANGE[0]}-{SPEEDRANGE[1]} knots"
    )   