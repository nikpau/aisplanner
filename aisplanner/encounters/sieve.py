"""
Module for plotting colregs encounters found via filter.py
"""
from aisplanner.encounters.filter import (
    EncounterResult, ColregsSituation, 
    FileStream, EncounterSituations
)
from pytsa.targetship import BoundingBox, TargetVessel
import pytsa
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from dotenv import load_dotenv
import pandas as pd
from aisplanner.dataprep._file_descriptors import DecodedReport
import numpy as np
from datetime import datetime, timedelta
import warnings
from rdp import rdp
import multiprocessing as mp

# Types
MMSI = int

load_dotenv()

RESDIR = Path("results")
REMOTEHOST = "taurus.hrsk.tu-dresden.de"
REMOTEDIR = Path("/warm_archive/ws/s2075466-ais/decoded/jan2020_to_jun2022")

# Load pickled results
def load_results(filename: str | Path):
    if not isinstance(filename, Path):
        filename = Path(filename)
    with open(RESDIR / filename, "rb") as f:
        results = pickle.load(f)
    return results


class COLREGSSieve:

    def __init__(self, encounter: EncounterResult, tlen: int = 10):
        self.encres = encounter
        self.tlen = tlen # Length of recorded trajectories in minutes
        self.fs = FileStream(
            host=REMOTEHOST,
            path=REMOTEDIR
        )
        self.file = self.get_file()

    def get_file(self) -> Path:
        """
        Download the file containing the encounter 
        and return its filepath
        """
        # Add .csv extension
        f = Path(self.encres.file + ".csv")
        # Check if file is already downloaded
        if not (self.fs._DOWNLOADPATH / f).exists():
            return self.fs.get_single_file(self.encres.file + ".csv")
        else:
            return self.fs._DOWNLOADPATH / f
    
    def find_os_tpos(self, celldata: pd.DataFrame) -> pytsa.TimePosition:
        """
        Find the own ship's time position in the given
        cell data
        """
        dt = timedelta(minutes=5)
        date = self.encres.timestamp
        mask = (
            (celldata[DecodedReport.MMSI.name]==self.encres.mmsi[0]) &
            (celldata[DecodedReport.timestamp.name] > (date-dt)) & 
            (celldata[DecodedReport.timestamp.name] < (date+dt))
        )
        ownship: pd.DataFrame = celldata.loc[mask]

        # Check if os is more than one row
        if len(ownship) > 1:
            warnings.warn(
                f"More than one row found for own ship. Result is ambiguous.\n"
                f"Result: {ownship}\n Using observation closest to encounter timestamp."
            )
            # Use observation closest to encounter timestamp
            ownship = ownship.iloc[ownship[DecodedReport.timestamp.name].sub(date).abs().argsort()[:1]]
            
        elif len(ownship) == 0:
            raise ValueError(
                f"No row found for own ship."
            )
        return pytsa.TimePosition(
            ownship[DecodedReport.timestamp.name].iloc[0],
            ownship[DecodedReport.lat.name].iloc[0],
            ownship[DecodedReport.lon.name].iloc[0]
        )
        
    def search_area_center(self, area: pytsa.UTMBoundingBox) -> pytsa.structs.UTMPosition:
        """
        Get the center of the search area.
        """
        return pytsa.TimePosition(
            timestamp=datetime.now(), #Irrelevant
            northing=(area.min_northing + area.max_northing)/2,
            easting=(area.min_easting + area.max_easting)/2
        )
    
    def message_filter(self, records: pd.DataFrame) -> pd.DataFrame:
        """
        Return only records with valid course i.e. not 360
        and with status "UnderWayUsingEngine"
        """
        return records[
            (records[DecodedReport.course.name] != 360) &
            (records[DecodedReport.status.name] == "NavigationStatus.UnderWayUsingEngine") &
            (records[DecodedReport.MMSI.name].isin(self.encres.mmsi))
        ]
    
    def record_trajectories(self) -> dict[MMSI,list[np.ndarray]]:
        """
        Record the positions, course and speed of the
        vessels involved in the encounter
        """
        # We can not init the search agent to the
        # time position of the own ship, because
        # we do not know it yet. Therefore, we
        # initialize it to the center of the cell,
        # which loads the cell data. From the loaded
        # cell data, we can then find the own ship's
        # time position. 
        # NOTE: This only works because we use a single
        # cell. If we specify more than one cell,
        # the cell data of the center and that of the
        # own ship might differ.
        sa = pytsa.SearchAgent(
            datapath=self.file,
            frame=self.encres.area,
            search_radius=20, # nm
            n_cells=1,
            filter=self.message_filter,
        )
        sa.init(self.search_area_center(self.encres.area))
        tpos = self.find_os_tpos(sa.cell_data)
        tpos._is_utm = True

        # Record trajectories in 10 second increments
        # for the given time length
        trajs: dict[MMSI,list[np.ndarray]] = {}

        # Record the encounter from 10 minutes before
        # to 10 minutes after the encounter
        start = self.encres.timestamp - timedelta(minutes=self.tlen)
        end = self.encres.timestamp + timedelta(minutes=self.tlen)

        # Set the time position to the start time
        tpos.timestamp: datetime = start

        # Extract the day from the encounter timestamp
        enc_day = self.encres.timestamp.date().day

        while start <= tpos.timestamp < end:
            # Check if the day has changed
            if tpos.timestamp.day != enc_day or \
                  (tpos.timestamp - timedelta(minutes=self.tlen)).day != enc_day:
                tpos.timestamp += timedelta(seconds=10)
                continue
            ships: list[TargetVessel] = sa.get_ships(tpos)
            for ship in ships:
                if not ship.mmsi in trajs:
                    trajs[ship.mmsi] = []
                trajs[ship.mmsi].append(ship.observe())
            tpos.timestamp += timedelta(seconds=10)

        return trajs
    
    def compress_trajectories(self, trajs: dict[MMSI,list[np.ndarray]]) -> dict[MMSI,list[np.ndarray]]:
        """
        Compress the trajectories to the given length
        """
        for mmsi, traj in trajs.items():
            # Make sure the trajectory is a numpy array
            traj = np.array(traj)
            # Perform the compression using the Douglas-Peucker algorithm
            mask = rdp(traj[:,0:2],epsilon=0.2,return_mask=True)
            # Apply the mask to the trajectory
            trajs[mmsi] = traj[mask]
        return trajs

    def plot(self, trajs: dict[MMSI,list[np.ndarray]]):
        """
        Plot the trajectories of the vessels involved.
        Different colors are for different vessels.
        Points are equidistant in time.
        """
        colors = ["#c1121f","#003049"]
        f, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))
        #trajs = self.compress_trajectories(trajs)
        for i, (mmsi, traj) in enumerate(trajs.items()):
            # Plot every n-th point as an arrow to indicate
            # the course of the vessel
            n = 3
            traj = np.array(traj)
            northing, easting, course, speed, rot, drot =\
                  traj[:,0], traj[:,1], traj[:,2], traj[:,3], traj[:,4], traj[:,5]
            # Scatter plot for each vessel
            ax1.scatter(easting[::n],northing[::n],label=mmsi,c=colors[i])
            # Line plot for each vessel
            ax1.plot(easting,northing,c=colors[i])
            # Transform course to radians such that
            # 0 is north and pi/2 is east
            course = np.pi/2 - np.deg2rad(course)
            ax1.quiver(
                easting[::n],northing[::n],
                np.cos(course[::n]),np.sin(course[::n]),
                color="black",scale=50
            )
            # Draw safety circle from EncounterSituations object
            [ax1.add_patch(
                plt.Circle(
                    (lo,la),
                    EncounterSituations.D1safe*1852,
                    color="r",fill=False
                )
            ) for lo,la in zip(easting[::n*8],northing[::n*8])]

            # Plot the rot and drot as a function of time
            ax2.plot(np.arange(len(rot)),rot,label="ROT[deg/min]",c=colors[i])
            ax2.plot(np.arange(len(rot)),drot,label="DROT[deg/min²]",c=colors[i])

        ax1.legend(title="MMSI")
        ax1.set_xlabel("Easting [m]")
        ax1.set_ylabel("Northing [m]")
        ax1.set_aspect("equal")
        ax1.set_title("Trajectories of vessels")

        ax2.legend()
        ax2.set_xlabel("Time [min]")
        ax2.set_title("Rates of turn")

        plt.suptitle(f"Encounter between {self.encres.mmsi[0]} and {self.encres.mmsi[1]}")
        plt.savefig(f"out/plots/{self.encres.mmsi[0]}_encounter.png",dpi = 300)

if __name__ == "__main__":
    # Load results
    res = load_results("newresults.pkl")
    print(len(res))
    # Plot encounter
    for i in range(100):
        plotter = COLREGSSieve(res[i*10])
        trajs = plotter.record_trajectories()
        plotter.plot(trajs)