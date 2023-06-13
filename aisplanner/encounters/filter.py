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

import multiprocessing as mp
import os
import pickle
import psutil
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from itertools import permutations
from pathlib import Path
from typing import IO, Any, Callable, Generator, Iterator, List, Tuple, Union

import ciso8601
import dotenv
import numpy as np
import pandas as pd
import paramiko
import pytsa
from pytsa.targetship import (
    TargetVessel
)
from pytsa.search_agent import FileLoadingError

from aisplanner.dataprep._file_descriptors import DecodedReport
from aisplanner.misc import logger


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


class TargetShipIntpFields(Enum):
    """
    Dataclass to store the indices of the interpolated fields
    of the target ship.
    """
    northing = 0
    easting = 1
    COG = 2
    SOG = 3
    ROT = 4
    dROT = 5

 
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
        if self.rel_dist < nm2m(self.D1safe):
            logger.info(f"Close encounter: {self.rel_dist:.2f} m")
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
        in_radius = self.rel_dist < nm2m(self.D1) # Is the target in the sensor range?
        in_bearing = abs(self.rel_bearing) < dtr(5) # Is the target in front of the ship?
        in_safety = abs(self.dcpa) < nm2m(self.D1safe) # Is the target within the safety margin?
        return in_radius and in_bearing and in_safety
    
    def _is_overtaking(self) -> bool:
        in_radius = self.rel_dist < nm2m(self.D2)
        in_bearing = dtr(112.5) < abs(self.rel_bearing) < dtr(247.5)
        in_safety = abs(self.dcpa) < nm2m(self.D2safe)
        return in_radius and in_bearing and in_safety
    
    def _is_crossing(self) -> bool:
        in_radius = self.rel_dist < nm2m(self.D3)
        in_bearing = abs(self.rel_bearing) >= dtr(5)
        in_safety = abs(self.dcpa) < nm2m(self.D3safe)
        return in_radius and in_bearing and in_safety
    
@dataclass
class EncounterResult:
    """
    Store the file, timestamp and MMSI of the encounter
    """
    encounter: ColregsSituation
    file: str
    timestamp: Union[str, datetime]
    mmsi: List[int] # MMSIs of the ships involved in the encounter
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

class TrajectoryExtractionAgent:
    """
    Search agent for Trajectories in a
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

        # List of Trajectories will be a list of TargetVessels
        # which are dataclasses containing the trajectory of raw 
        # AIS messages as well as their interpolated metrics.
        self.trajectories: List[TargetVessel] = []

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
            pickle.dump(self.trajectories, f)
    
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
                    self.trajectories.extend([enc for sublist in encs for enc in sublist])
                else:
                    for area in self.search_areas:
                        enc = self._search(area)
                        self.trajectories.extend(enc)
                if self._using_remote:
                    self.delete_current_file()
            except StopIteration:
                return
            
    def _search(self,
        area: pytsa.UTMBoundingBox,
        override_file: Path = None) -> List[TargetVessel]:
        """
        Search for valid trajectories in a given area.
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
            filter=MessageFilter.only_german
        )
        
        # Set the maximum temporal deviation of target
        # ships from provided time in `init()`
        search_agent.time_delta = 30 # in minutes

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

        # Scan the area every 30 minutes. 
        # During every scan, every ship is checked for a possible
        # encounter situation.
        search_date = start_date

        # Initialize the list of encounters
        found: List[TargetVessel] = []

        # Increment the search date until the date of search is
        # greater than the date of the current file.
        logger.info(f"Searching {self.current_file}")
        
        # Print memory usage of the process
        logger.info(
            f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB"
            )
        
        
        while start_date.day == search_date.day:

            ships: List[TargetVessel] = search_agent.get_ships(tpos)
            
            # Skip if there are not enough ships in the area
            if len(ships) < 2:
                tpos = self._update_timeposition(tpos, 30)
                search_date = tpos.timestamp
                continue
            
            else:
                found.extend(ships)
            
            # End of search for this time step
            tpos = self._update_timeposition(tpos, 30)
            search_date = tpos.timestamp
        
        return found

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
    
def unique_2_permuts(ships: List[TargetVessel]) -> List[Tuple[TargetVessel,TargetVessel]]:
    """
    Get all unique permutations of the ships.
    """
    uniques = []
    permutes = list(permutations(ships,2))
    for a,b in permutes:
        if not (b,a) in uniques:
            uniques.append((a,b))
            yield (a,b)

@dataclass
class MessageFilter:
    """
    Collection of filters to apply to AIS messages.
    """
    @classmethod
    def only_german(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for all messages that have been recorded by 
        German base stations.
        """
        return df[df[DecodedReport.originator.name].isin(["DEU"])]

class ForwardBackwardScan:
    """
    Scan for evasive maneuvers using a forward-backward scan 
    proposed by Rong et al. (2022).
    
    https://doi.org/10.1016/j.oceaneng.2021.110479

    Input for this class are two target vessels who are 
    involved in an encounter situation as decided by the
    `EncounterSituation` class.

    According to the original paper, the forward-backward scan
    is performed as follows:
        1.  The time-sorted AIS messages of the two vessels are
            scanned forward in time and the distance between
            the two vessels is calculated. 
        2.  The time of minimum distance is identified.
        3.  The AIS messages are scanned backward in time, 
            starting from the time of minimum distance, in 
            order to check if there exists a time interval
            that the sign of every ROT value within the time 
            interval is the same with the rudder direction 
            sign(δ) = sign(ψ''), and the sign of the derivative of ROT at 
            the first message in the interval is the same with sign(δ)
    """

    def __init__(self, t1: TargetVessel, t2: TargetVessel) -> None:

        # Set the interval for the scan
        self.interval = 10 # in seconds

        matcher = pytsa.TrajectoryMatcher(t1, t2)
        if matcher.disjoint_trajectories:
            raise ValueError("Trajectories are disjoint.")
        
        self.start = matcher.start
        self.end = matcher.end
        matcher.observe_interval(interval=self.interval)
        self.matcher = matcher

        # Array of times at which the track is observed
        self.times = np.arange(self.start, self.end, self.interval)

    def __call__(self, interval_length) -> bool:

        min_index = self.forward_scan()
        if min_index is None or min_index < interval_length:
            return False
        success = []
        for obs in [self.matcher.obs_vessel1,self.matcher.obs_vessel2]:
            success.append(
                self.backward_scan(obs[:min_index],interval_length)
            )
        return any(success)
        

    def forward_scan(self) -> float:
        """
        Perform the forward scan.

        Returns the index of minimum distance 
        in the 'self.times' attribute.
        """

        # Array of distances between the two vessels
        distances = np.zeros_like(self.times)
        for i, (obs_v1, obs_v2) in enumerate(
            zip(self.matcher.obs_vessel1, self.matcher.obs_vessel2)):
            dist  = np.linalg.norm(obs_v1[0:2] - obs_v2[0:2])
            distances[i] = dist

        # Find the time of minimum distance
        min_dist = np.min(distances)
        min_dist_idx = np.where(distances == min_dist)[0][0]

        return min_dist_idx
    
    def sliding_window(self, input: list, interval_length: int = 3) -> Generator:
        """
        Generator yielding a sliding window of length `interval_length`
        over the input list.
        """
        for i in range(len(input)-interval_length):
            yield input[i:i+interval_length]
    
    def backward_scan(
            self, 
            obs: np.ndarray,
            interval_length: int = 3) -> bool:
        """
        Perform the backward scan.

        Returns True if the backward scan was successful,
        False otherwise.

        A backwards scan is successful if the sign of
        every ROT value within the time interval is the
        same with the rudder direction sign(δ).

        Args:
            obs (np.ndarray): Array of observations
            idx (int): Index of minimum distance
            interval_length (int, optional): Length of the
                interval to scan. Defaults to 3.
        """
        # Walk backwards in time using a sliding window
        # of length `interval_length`
        for window in zip(self.sliding_window(obs, interval_length)):
            # Get the ROT values
            rot = window[:, TargetShipIntpFields.ROT.value]
            drot = window[:, TargetShipIntpFields.dROT.value]

            # Check if the signs of the ROT values are the same
            # as the rudder direction
            if all(np.sign(rot) == 1) or all(np.sign(rot) == -1):
                rots_sign = np.sign(rot[0])
            else:
                continue
            
            # Check if the sign of the dROT value at the first
            # message in the interval is the same as the rudder
            # direction
            if np.sign(drot[0]) == rots_sign:
                return True
            else:
                continue
        return False
            
