"""
This module will filter sets of AIS records by COLREGS-relevant situations.

AIS records must be provided as time-sorted .csv files.

Procedure:
    1. The AIS records are divided into one-minute-long chunks 
    2. For each chunk, we check whether a COLREGS-relevant situation is present
    3. If so:
        1. Record the trajectory for the time of the situation (+- buffer)
        2. Set the first waypoint of the agent to be the start of the situation
        3. Set the last waypoint of the agent to be the goal
        4. Assume, that a linear path connecting start and end is the unplanned route
"""

from typing import IO, Iterator
import pandas as pd
import numpy as np
import pytsa
from pytsa.targetship import TargetVessel
from dataclasses import dataclass
from enum import Enum
import math
import matplotlib.pyplot as plt
import paramiko
import os
from contextlib import closing
import dotenv
from pathlib import Path
from aisplanner.dataprep._file_descriptors import DecodedReport
import ciso8601
from datetime import datetime, timedelta
from itertools import permutations
from aisplanner.misc import logger

# Exceptions
class EndOfFileError(Exception):
    pass

# Load environment variables
dotenv.load_dotenv()

# Types
latitude = float
longitude = float

# Constants
PI = np.pi
TWOPI = 2 * PI
 
@dataclass
class Position:
    lat: latitude
    lon: longitude

    @property
    def x(self) -> longitude:
        return self.lon
    @property
    def y(self) -> latitude:
        return self.lat

@dataclass
class Ship:
    pos: Position
    sog: float
    cog: float

    def __post_init__(self):
        if not (0 <= self.cog <= 2*np.pi):
            self.cog = angle_to_pi(self.cog)

@dataclass
class ColregsSituation(Enum):
    """
    Dataclass to map relative bearings to COLREGS situations.
    """
    # No situation
    NONE = 0
    # Crossing situation
    CROSSING = 1
    # Overtaking situation
    OVERTAKING = 2
    # Head-on situation
    HEADON = 3

def dtr(deg: float) -> float:
    """Convert degrees to radians"""
    return deg * PI / 180

def rtd(rad: float) -> float:
    """Convert radians to degrees"""
    return rad * 180 / PI

def angle_to_pi(angle: float) -> float:
    """Convert an angle to the range [-pi,pi]"""
    return (angle + PI) % (2 * PI) - PI

def angle_to_2pi(angle: float) -> float:
    """Convert an angle to the range [0,2pi]"""
    return angle % (2 * PI)

def relative_velocity(
    o_crs: float, o_spd: float, t_crs: float, t_spd: float) -> tuple[float,float]:
    """
    Calculate the relative velocity between two ships
    and split it into x and y components.
    o_crs: own course in radians
    o_spd: own speed in knots
    t_crs: target course in radians
    t_spd: target speed in knots
    """
    vx_rel = t_spd * np.sin(t_crs) - o_spd * np.sin(o_crs)
    vy_rel = t_spd * np.cos(t_crs) - o_spd * np.cos(o_crs)
    if np.isclose(vx_rel,0):
        vx_rel = 0
    if np.isclose(vy_rel,0):
        vy_rel = 0
    return vx_rel,vy_rel

def vel_from_xy(x: float, y: float) -> float:
    """Calculate the velocity from x and y components"""
    return np.sqrt(x**2 + y**2)

def crv(vx_rel: float, vy_rel: float) -> float:
    """Calculate the course of a relative velocity
    x: x-component of the relative velocity
    y: y-component of the relative velocity
    """
    def alpha(x: float, y: float) -> float:
        """Calculate the angle of a vector in radians"""
        if x >= 0 and y >= 0:
            return 0
        if (x < 0 and y < 0) or (x >= 0 and y < 0):
            return PI
        if x < 0 and y >= 0:
            return TWOPI
    return np.arctan(vx_rel/vy_rel) + alpha(vx_rel,vy_rel)

def rel_dist(own: Position, tgt: Position) -> float:
    """Calculate the relative distance between two ships"""
    own = np.array([own.x,own.y])
    tgt = np.array([tgt.x,tgt.y])
    return np.linalg.norm(own - tgt)

def true_bearing(own: Position, tgt: Position) -> float:
    """Calculate the true bearing between two ships"""
    def alpha(own: Position, tgt: Position) -> float:
        """Calculate the angle of a vector in radians"""
        if tgt.x >= own.x and tgt.y >= own.y:
            return 0
        if (tgt.x < own.x and tgt.y < own.y) or\
              (tgt.x >= own.x and tgt.y < own.y):
            return PI
        if tgt.x < own.x and tgt.y >= own.y:
            return TWOPI
    return np.arctan((tgt.x-own.x)/(tgt.y-own.x)) + alpha(own,tgt)

def relative_bearing(own: Ship, tgt: Ship) -> float:
    """Calculate the relative bearing between two ships"""
    return angle_to_2pi(true_bearing(own.pos,tgt.pos) - own.cog)

def DCPA(d_rel: float, crv: float, true_bearing: float) -> float:
    """Calculate the distance of closest point of approach.
    crv: course of relative velocity"""
    return d_rel * np.sin(crv-true_bearing-PI)

def TCPA(d_rel: float, crv: float, true_bearing: float, v_rel:float) -> float:
    """Calculate the time of closest point of approach.
    crv: course of relative velocity"""
    return d_rel * np.cos(crv-true_bearing-PI)/v_rel

class EncounterSituations:
    """
    Classify COLREGS encounter situatio TWO
    based on Zhang et al. (2021), Table 6.

    10.1109/ACCESS.2021.3060150
    """
    # Sensor ranges [nm] for the ship to detect specific encounters
    D1 = 6 # Head-on
    D2 = 3 # Overtaking
    D3 = 6 # Crossing
    D1safe, D2safe, D3safe = 1, 1, 1 # Safety margins [nm]
    MAX_RANGE = max(D1,D2,D3) # Maximum sensor range [nm]

    def __init__(self, own: Ship, tgt: Ship) -> None:
        self.own = own
        self.tgt = tgt

        self.rel_bearing = relative_bearing(own,tgt)
        self.rel_dist = rel_dist(own.pos,tgt.pos)
        self.rel_vel = relative_velocity(own.cog,own.sog,tgt.cog,tgt.sog)
        self.crv = crv(*self.rel_vel) # Course of relative velocity
        self.true_bearing = true_bearing(own.pos,tgt.pos)
        self.v_rel = vel_from_xy(*self.rel_vel)
        self.dcpa = DCPA(self.rel_dist,self.crv,self.true_bearing)
        self.tcpa = TCPA(self.rel_dist,self.crv,self.true_bearing,self.v_rel)
    
    def __repr__(self) -> str:
        return f"EncounterSituations({self.own}, {self.tgt})"

    def analyze(self) -> list[ColregsSituation]:
        s = []
        # Early stop if the encounter is not active
        if self.tcpa <= 0:
            return s
        if self._is_headon():
            s.append(ColregsSituation.HEADON)
        if self._is_overtaking():
            s.append(ColregsSituation.OVERTAKING)
        if self._is_crossing():
            s.append(ColregsSituation.CROSSING)
        return s
        
    def _is_headon(self) -> bool:
        in_radius = self.rel_dist < self.D1 # Is the target in the sensor range?
        in_bearing = abs(self.rel_bearing) < dtr(5) # Is the target in front of the ship?
        in_safety = abs(self.dcpa) < self.D1safe # Is the target within the safety margin?
        return in_radius and in_bearing and in_safety
    
    def _is_overtaking(self) -> bool:
        in_radius = self.rel_dist < self.D2
        in_bearing = dtr(112.5) < abs(self.rel_bearing) < dtr(247.5)
        in_safety = abs(self.dcpa) < self.D2safe
        return in_radius and in_bearing and in_safety
    
    def _is_crossing(self) -> bool:
        in_radius = self.rel_dist < self.D3
        in_bearing = abs(self.rel_bearing) >= dtr(5)
        in_safety = abs(self.dcpa) < self.D3safe
        return in_radius and in_bearing and in_safety
    
@dataclass
class EncounterResult:
    """
    Store the file, timestamp and MMSI of the encounter
    """
    encounter: list[EncounterSituations]
    file: str
    timestamp: str | datetime
    mmsi: str

    def __post_init__(self) -> None:
        self.encounter_names = [e.name for e in self.encounter]
        if isinstance(self.timestamp, str):
            ciso8601.parse_datetime(self.timestamp)

class FileStream:
    """
    Yield files from a remote folder over ssh.
    """
    _DOWNLOADPATH = Path("./data/aisrecords")
    def __init__(self, host: str, path: str) -> None:
        self.host = host
        self.path = Path(path)
        self.downloadpath = self._set_downloadpath()

        self._check_environment_variables()
        
    def _check_environment_variables(self) -> None:
        """
        Check if required environment variables are set.
        Raise ValueError if any of them is missing.
        """
        required_vars = ["TUUSER", "PRIVKEY"]
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            raise ValueError(f"Environment variables {', '.join(missing_vars)} are not set")

        if not Path(os.environ["PRIVKEY"]).exists():
            raise ValueError("Private key does not exist")   

    def _set_downloadpath(self) -> Path:
        """
        Set the download path to be ./data/aisrecords
        Folder is created if it does not exist.
        """
        downloadpath = self._DOWNLOADPATH
        downloadpath.mkdir(parents=True, exist_ok=True)
        return downloadpath

    def __iter__(self) -> Iterator[IO]:
        with closing(paramiko.SSHClient()) as ssh:
            ssh.load_system_host_keys()
            ssh.connect(
                self.host, 
                username=os.environ["TUUSER"],
                key_filename=os.environ["PRIVKEY"])
            sftp = ssh.open_sftp()
            for file in sftp.listdir(self.path.as_posix()):
                sftp.get(
                    (self.path/file).as_posix(), 
                    (self.downloadpath/file).as_posix()
                )
                yield self.downloadpath/file
            sftp.close()

    def get_single_file(self, filename: str) -> IO:
        with closing(paramiko.SSHClient()) as ssh:
            ssh.load_system_host_keys()
            ssh.connect(
                self.host, 
                username=os.environ["TUUSER"],
                key_filename=os.environ["PRIVKEY"])
            sftp = ssh.open_sftp()
            sftp.get(
                (self.path/filename).as_posix(), 
                (self.downloadpath/filename).as_posix()
            )
            sftp.close()
            return self.downloadpath/filename

class ENCSearchAgent:
    """
    Search agent for COLREGS encounter situations in a
    stream of AIS messages.

    Files are streamed from a remote server over ssh and
    will be deleted after processing.
    
    Situations are saved in a list of EncounterSituations 
    together with the file and timestamp of the encounter.
    
    """
    def __init__(self, 
            remote_host: str | Path, 
            remote_dir: str | Path,
            search_area: pytsa.BoundingBox) -> None:
        
        self.remote_host = remote_host
        self.remote_dir = remote_dir
        self.search_area = search_area

        self.encounters: list[EncounterResult] = []
        
        self.filestream = FileStream(self.remote_host, self.remote_dir)

    def load_next_file(self) -> pd.DataFrame:
        """
        Download AIS messages from remote server and
        open it as a pandas dataframe.
        """
        try:
            self.current_file: Path = next(self.filestream)
        except StopIteration:
            logger.info("No more files to process")
            exit(0)
        df = pd.read_csv(self.current_file, sep=",")
        return self.select_columns(df)

    def delete_current_file(self) -> None:
        """
        Delete file after processing.
        """
        self.current_file.unlink()

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select columns that are needed for the search.
        """
        return df[[
            DecodedReport.lat.name,
            DecodedReport.lon.name,
            DecodedReport.course.name,
            DecodedReport.speed.name,
            DecodedReport.status.name,
        ]]
    
    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter rows that are needed for the search.

        Since we are only considering ships that are moving, 
        therefore only status == 'UnderWayUsingEngine' is
        considered.

        additionally, we are only considering ships that
        transmitted valid course information and a speed
        greater than 0.
        """
        return df[
            (df[DecodedReport.status.name] == "NavigationStatus.UnderWayUsingEngine") &
            (df[DecodedReport.course.name] != 360) &
            (df[DecodedReport.speed.name] > 0)
        ]
    
    def filter_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter rows that are within the search area.
        """
        return df[
            (df[DecodedReport.lat.name] >= self.search_area.LATMIN) &
            (df[DecodedReport.lat.name] <= self.search_area.LATMAX) &
            (df[DecodedReport.lon.name] >= self.search_area.LONMIN) &
            (df[DecodedReport.lon.name] <= self.search_area.LONMAX)
        ]
    
    def prepare_ais_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare AIS messages for search.
        """
        df = self.filter_rows(df)
        return self.filter_location(df)
    
    def search_area_center(self) -> pytsa.targetship.Position:
        """
        Get the center of the search area.
        """
        b = self.search_area
        return pytsa.targetship.Position(
            (b.LATMIN + b.LATMAX)/2,
            (b.LONMIN + b.LONMAX)/2
        )
    
    def get_search_radius(self) -> float:
        """
        Get the radius of the circle around the bounding box
        """
        b = self.search_area
        center = self.search_area_center()
        lat_extent = abs(b.LATMAX - b.LATMIN)
        lon_extent = abs(b.LONMAX - b.LONMIN)
        r_along_lon = np.sqrt(
            (center.lon-b.LONMIN)**2 + (lat_extent/2)**2
        )
        r_along_lat = np.sqrt(
            (center.lat-b.LATMIN)**2 + (lon_extent/2)**2
        )
        return max(r_along_lon, r_along_lat)
    
    def _increment(self, date: datetime, by: int = 3) -> datetime:
        """
        Increment date by .
        """
        return date + timedelta(minutes=by)

    def search(self) -> list[EncounterSituations]:
        """
        
        """
        self.load_next_file()
        # Datetime of current file, starting at midnight
        # NOTE: This only works if the file name is in the format
        #       YYYY-MM-DD.csv
        #       Since the AIS messages we use are in the format
        #       YYYY_MM_DD.csv, we need to replace the _ with -.
        corrected_filename = self.current_file.stem.replace("_", "-")
        start_date = ciso8601.parse_datetime(corrected_filename)

        # Initialize search agent for nearest neighbors
        # and trajectory interpolation
        search_agent = pytsa.SearchAgent(
            datapath=self.current_file,
            frame=self.search_area,
            search_radius=self.get_search_radius(),
            n_cells=1,
            filter=self.prepare_ais_messages
        )

        # Use the center of the search area as the starting position
        center = self.search_area_center()
        tpos = pytsa.TimePosition(
            timestamp=start_date,
            lat=center.lat,
            lon=center.lon
        )

        # Initialize search agent to that starting position
        search_agent.init(tpos.position)

        # Scan the area every 3 minutes. 
        # During every scan, every ship is checked for a possible
        # encounter situation.
        search_date = start_date

        # Increment the search date until the date of search is
        # greater than the date of the current file.
        while start_date.day <= search_date.day:

            ships: list[TargetVessel] = search_agent.get_ships()
            
            # Skip if there are not enough ships in the area
            if len(ships) < 2:
                search_date = self._increment(search_date)
                continue
            
            # Check every ship combination for a possible encounter
            for os, ts in permutations(ships,2):
                os_obs = os.observe() # Get [lat,lon,course,speed]
                ts_obs = ts.observe() # Get [lat,lon,course,speed]

                # Construct ship objects
                OS = Ship(
                    pos=Position(os_obs[0],os_obs[1]),
                    sog=os_obs[3],
                    cog=dtr(os_obs[2])
                )
                TS = Ship(
                    pos=Position(ts_obs[0],ts_obs[1]),
                    sog=ts_obs[3],
                    cog=dtr(ts_obs[2])
                )
                found = EncounterSituations(OS,TS).analyze()
                if found:
                    for encounter in found:
                        self.encounters.append(
                            EncounterResult(
                                encounter=encounter,
                                file=self.current_file.stem,
                                timestamp=search_date,
                                mmsi=os.mmsi,
                            )
                        )
            # End of search for this time step
            search_date = self._increment(search_date)

        return self.encounters
                    
            
# ---------------------------------------------
# Test land
# ---------------------------------------------

# Set search area
# Fredrikshavn - Gothenburg
b = pytsa.targetship.BoundingBox(
    LATMIN=57.378,
    LATMAX=57.778,
    LONMIN=10.446,
    LONMAX=11.872
)

# Initialize search agent
s = ENCSearchAgent(
    remote_host="taurus.hrsk.tu-dresden.de",
    remote_dir=os.environ["AISDECODED"],
    search_area=b
)
res = s.search()


OS = Ship(Position(0,0),1,dtr(45))
TS = Ship(Position(2,2),1,dtr(210))
ES = EncounterSituations(OS,TS).analyze()
print(ES)
v_rel = relative_velocity(OS.cog,OS.sog,TS.cog,TS.sog)
print("DCPA: ",
    DCPA(
        rel_dist(OS.pos,TS.pos), 
        crv(*v_rel), 
        true_bearing(OS.pos,TS.pos),
    )
)
print("TCPA: ",
    TCPA(
        rel_dist(OS.pos,TS.pos),
        crv(*v_rel),
        true_bearing(OS.pos,TS.pos),
        vel_from_xy(*v_rel)
    )
)

f,ax = plt.subplots()

# Draw circles of sensor range
ax.add_artist(plt.Circle((OS.pos.x,OS.pos.y),EncounterSituations.D1,color='r',fill=False))
ax.add_artist(plt.Circle((OS.pos.x,OS.pos.y),EncounterSituations.D2,color='g',fill=False))
ax.add_artist(plt.Circle((OS.pos.x,OS.pos.y),EncounterSituations.D3,color='y',fill=False))

ax.plot([OS.pos.x,TS.pos.x],[OS.pos.y,TS.pos.y])
# Plot arrows for own and target ship
ax.arrow(
    OS.pos.x,OS.pos.y,
    np.sin(OS.cog),np.cos(OS.cog),
    width=0.01,
    color='r'
)
ax.arrow(
    TS.pos.x,TS.pos.y,
    np.sin(TS.cog),np.cos(TS.cog),
    width=0.01,
    color='b'
)
# Plot relative velocity
plt.arrow(
    OS.pos.x,OS.pos.y,
    v_rel[0],v_rel[1],
    width=0.01,
    color='g'
)
plt.xlim(-0.5,6)
plt.ylim(-0.5,6)
plt.show()

