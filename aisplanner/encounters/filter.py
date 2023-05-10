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

from typing import IO, Callable, Iterator, List, Tuple, Union
import pandas as pd
import numpy as np
import pytsa
from pytsa.targetship import TargetVessel, FileLoadingError
from dataclasses import dataclass
from enum import Enum
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
import pickle
import multiprocessing as mp

# Identity function
def _id(x):
    return x

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
    o_crs: float, o_spd: float, t_crs: float, t_spd: float) -> Tuple[float,float]:
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

def nm2m(nm: float) -> float:
    """Convert nautical miles to meters"""
    return nm * 1852

def m2nm(m: float) -> float:
    """Convert meters to nautical miles"""
    return m / 1852

class EncounterSituations:
    """
    Classify COLREGS encounter situatios
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

    def analyze(self) -> List[ColregsSituation]:
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
    encounter: ColregsSituation
    file: str
    timestamp: Union[str, datetime]
    mmsi: str
    area: pytsa.LatLonBoundingBox

    def __post_init__(self) -> None:
        self.encounter_names = self.encounter.name
        if isinstance(self.timestamp, str):
            ciso8601.parse_datetime(self.timestamp)

class FileStream:
    """
    Yield files from a remote folder over ssh.
    """
    _DOWNLOADPATH = Path("./data/aisrecords")
    def __init__(self, host: str, path: str, filter: Callable = _id) -> None:
        self.host = host
        self.path = Path(path) if not isinstance(path, Path) else path
        self.downloadpath = self._set_downloadpath()
        self.username = os.environ["ZIH"]
        self.pkey = os.environ["PKEYLOC"]

        self.filter = filter

        self._check_environment_variables()
        
    def _check_environment_variables(self) -> None:
        """
        Check if required environment variables are set.
        Raise ValueError if any of them is missing.
        """
        required_vars = ["ZIH", "PKEYLOC"]
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            raise ValueError(f"Environment variables {', '.join(missing_vars)} are not set")

        if not Path(os.environ["PKEYLOC"]).exists():
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
                username=self.username,
                key_filename=self.pkey)
            sftp = ssh.open_sftp()
            filelist = sftp.listdir(self.path.as_posix())
            for file in self.filter(filelist):
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
                username=self.username,
                key_filename=self.pkey)
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
            remote_host: Union[str, Path], 
            remote_dir: Union[str, Path],
            search_areas: List[pytsa.UTMBoundingBox],
            filelist: List[Union[Path, str]] = None,
            parallel: bool = False,
            **fsargs) -> None:
        
        self.remote_host = remote_host
        self.remote_dir = remote_dir
        self.search_areas = search_areas
        self.parallel = parallel

        self.encounters_total: List[EncounterResult] = []

        # Check if filelist is given or if files should be streamed
        self._using_remote = False
        
        if filelist is None:
            self.filestream = FileStream(
                self.remote_host, self.remote_dir, **fsargs
            )
            self._using_remote = True
        else: 
            self.filestream  = [Path(file) for file in filelist]
        
        self.filestream = iter(self.filestream)

    def load_next_file(self) -> pd.DataFrame:
        """
        Download AIS messages from remote server and
        open it as a pandas dataframe.
        """
        try:
            self.current_file: Path = next(self.filestream)
        except StopIteration:
            logger.info("No more files to process")
            raise

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
    
    def only_underway(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter rows for ships that are underway.

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
        
    def search_area_center(self, area: pytsa.UTMBoundingBox) -> pytsa.structs.UTMPosition:
        """
        Get the center of the search area.
        """
        return pytsa.structs.UTMPosition(
            (area.min_northing + area.max_northing)/2,
            (area.min_easting + area.max_easting)/2
        )
    
    def get_search_radius(self,area: pytsa.UTMBoundingBox) -> float:
        """
        Get the radius of the circle around the bounding box
        in nautical miles.
        Due to the earth being a sphere, the radius is not
        constant, therefore the error of this approximation
        increases with the distance from the equator.
        """
        center = self.search_area_center(area)
        north_extent = abs(area.max_northing - area.min_northing)
        r = np.sqrt(
            (area.max_easting-center.easting)**2 + (north_extent/2)**2
        )
        return m2nm(r) # convert to nautical miles
    
    def save_results(self,dest: Union[str, Path]) -> None:
        """
        Saves the results as a python object.
        """
        with open(dest, "wb") as f:
            pickle.dump(self.encounters_total, f)
    
    def search(self) -> None:
        """
        Search all areas for encounters.
        """
        while True:
            try:
                self.load_next_file()
                if self.parallel:
                    with mp.Pool(len(self.search_areas)) as pool:
                        encs = pool.map(self._search, self.search_areas)
                    # Flatten list of lists
                    self.encounters_total.extend([enc for sublist in encs for enc in sublist])
                else:
                    for area in self.search_areas:
                        enc = self._search(area)
                        self.encounters_total.extend(enc)
                if self._using_remote:
                    self.delete_current_file()
            except StopIteration:
                return

    def _search(
        self,
        area: pytsa.UTMBoundingBox,
        override_file: Path = None) -> List[EncounterResult]:
        """
        
        """
        if override_file is not None:
            self.current_file = Path(override_file)
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
            datapath=self.current_file if not override_file else override_file,
            frame=area,
            search_radius=self.get_search_radius(area),
            n_cells=1,
            filter=self.only_underway
        )

        # Use the center of the search area as the starting position
        center = self.search_area_center(area)
        tpos = pytsa.TimePosition(
            timestamp=start_date,
            easting=center.easting,
            northing=center.northing
        )

        # Initialize search agent to that starting position
        try:
            search_agent.init(tpos)
        except FileLoadingError as e:
            logger.warning(f"File {self.current_file} could not be loaded:\n{e}")
            return []

        # Scan the area every 3 minutes. 
        # During every scan, every ship is checked for a possible
        # encounter situation.
        search_date = start_date

        # Initialize the list of encounters
        encounters: List[EncounterSituations] = []

        # Increment the search date until the date of search is
        # greater than the date of the current file.
        while start_date.day == search_date.day:

            # print(f"Searching {search_date}")
            ships: List[TargetVessel] = search_agent.get_ships(tpos)
            
            # Skip if there are not enough ships in the area
            if len(ships) < 2:
                tpos = self._update_timeposition(tpos, 3)
                search_date = tpos.timestamp
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
                        encounters.append(
                            EncounterResult(
                                encounter=encounter,
                                file=self.current_file.stem,
                                timestamp=search_date,
                                mmsi=os.mmsi,
                                area=area
                            )
                        )
            # End of search for this time step
            tpos = self._update_timeposition(tpos, 3)
            search_date = tpos.timestamp
        
        return encounters

    def _update_timeposition(self,
            tpos: pytsa.TimePosition, 
            byminutes: int) -> pytsa.TimePosition:
        """
        Update the time of a time position.
        """
        return pytsa.TimePosition(
            timestamp=tpos.timestamp + timedelta(minutes=byminutes),
            easting=tpos.easting,
            northing=tpos.northing,
            as_utm=True
        )
