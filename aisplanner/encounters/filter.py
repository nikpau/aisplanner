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
from pytsa.search_agent import FileLoadingError
from pytsa import ShipType, TargetVessel

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
Northing = float
Easting = float

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
    northing: Northing
    easting: Easting

    @property
    def x(self) -> Easting:
        return self.easting
    @property
    def y(self) -> Northing:
        return self.northing

@dataclass
class Ship:
    pos: Position
    sog: float
    cog: float

    def __post_init__(self):
        self.cog = dtr(self.cog)

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

    def __hash__(self) -> int:
        return super().__hash__()

def dtr(deg: float) -> float:
    """Convert degrees to radians"""
    return deg * PI / 180

def rtd(rad: float) -> float:
    """Convert radians to degrees"""
    return rad * 180 / PI

def angle_to_pi(angle: float) -> float:
    """Convert an angle to the range [0,pi]"""
    return angle % PI

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
    o_vx, o_vy = velocity_components(o_crs,o_spd)
    t_vx, t_vy = velocity_components(t_crs,t_spd)
    return t_vx - o_vx, t_vy - o_vy

def velocity_components(crs: float, spd: float) -> Tuple[float,float]:
    """Calculate the x and y components of a velocity vector"""
    return spd * np.sin(crs), spd * np.cos(crs)

def vel_from_xy(x: float, y: float) -> float:
    """Calculate the velocity from x and y components"""
    return np.sqrt(x**2 + y**2)

def crv(vx_rel: float, vy_rel: float) -> float:
    """Calculate the course of a relative velocity
    x: x-component of the relative velocity
    y: y-component of the relative velocity
    """
    def alpha(x: float, y: float) -> float:
        if x >= 0 and y >= 0:
            return 0
        if (x < 0 and y < 0) or (x >= 0 and y < 0):
            return PI
        if x < 0 and y >= 0:
            return TWOPI
    return np.arctan(vx_rel/vy_rel) + alpha(vx_rel,vy_rel)

def relative_distance(own: Position, tgt: Position) -> float:
    """Calculate the relative distance between two ships"""
    own = np.array([own.x,own.y])
    tgt = np.array([tgt.x,tgt.y])
    return np.linalg.norm(own - tgt)

def true_bearing(own: Position, tgt: Position) -> float:
    """Calculate the true bearing between two ships"""
    def alpha(own: Position, tgt: Position) -> float:
        if tgt.x >= own.x and tgt.y >= own.y:
            return 0
        if (tgt.x < own.x and tgt.y < own.y) or\
              (tgt.x >= own.x and tgt.y < own.y):
            return PI
        if tgt.x < own.x and tgt.y >= own.y:
            return TWOPI
    return np.arctan2((tgt.x-own.x),(tgt.y-own.y)) + alpha(own,tgt)

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

class EncounterSituation:
    """
    Classify COLREGS encounter situatios
    based on Zhang et al. (2021), Table 6.

    10.1109/ACCESS.2021.3060150
    """
    # Sensor ranges [nm] for the ship to detect specific encounters
    # TODO: Check if these values are justified
    D1 = .5 # 6 # Head-on
    D2 = .5 # 3 # Overtaking
    D3 = .5 # 6 # Crossing
    D1safe, D2safe, D3safe = 1, 1, 1 # Safety margins [nm]
    MAX_RANGE = max(D1,D2,D3) # Maximum sensor range [nm]

    def __init__(self, own: Ship, tgt: Ship) -> None:
        self.own = own
        self.tgt = tgt

        self.rel_bearing = relative_bearing(own,tgt)
        self.rel_dist = relative_distance(own.pos,tgt.pos)
        self.rel_vel = relative_velocity(own.cog,own.sog,tgt.cog,tgt.sog)
        self.crv = crv(*self.rel_vel) # Course of relative velocity
        self.true_bearing = true_bearing(own.pos,tgt.pos)
        self.v_rel = vel_from_xy(*self.rel_vel)
        self.dcpa = DCPA(self.rel_dist,self.crv,self.true_bearing)
        self.tcpa = TCPA(self.rel_dist,self.crv,self.true_bearing,self.v_rel)
        
    def __repr__(self) -> str:
        return f"EncounterSituations({self.own}, {self.tgt})"

    def analyze(self) -> Union[ColregsSituation,None]:
        # Early stop if the encounter is not active
        if self.tcpa <= 0:
            return 
        if self._is_headon():
            return ColregsSituation.HEADON
        if self._is_overtaking():
            return ColregsSituation.OVERTAKING
        if self._is_crossing():
            return ColregsSituation.CROSSING
        
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
            search_areas: List[pytsa.UTMBoundingBox],
            ship_types: list[ShipType],
            msg12318files: List[Union[Path, str]] = None,
            msg5files: List[Union[Path, str]] = None,
            parallel: bool = False) -> None:
        
        # Check if search areas are given
        if not isinstance(search_areas, list):
            search_areas = [search_areas]
        if len(search_areas) == 1 and parallel:
            raise ValueError(
                "Only one search area given. "
                "Parallel processing is not possible."
                )
        self.search_areas = search_areas
        self.parallel = parallel

        # List of ShipTypes to be searched for
        self.ship_types = ship_types


        # List of Trajectories will be a list of TargetVessels
        # which are dataclasses containing the trajectory of raw 
        # AIS messages as well as their interpolated metrics.
        self.trajectories: List[TargetVessel] = []

        # Check if filelist is given or if files should be streamed
        self._using_remote = False
        
        if msg12318files is None:
            raise ValueError("No filelist for dynamic messages given")
        self.dynamic_filestream  = [Path(file) for file in msg12318files]
        self.dynamic_filestream = iter(self.dynamic_filestream)

        if msg5files is None:
            logger.info(
                "No filelist for static messages given. "
                "Static messages will not be processed."
                )
            self.static_filestream = None
        else:
            self.static_filestream = [Path(file) for file in msg5files]
            self.static_filestream = iter(self.static_filestream)

    def load_next_file(self) -> None:
        """
        Download AIS messages from remote server and
        open it as a pandas dataframe.
        """
        try:
            self.current_file: Path = next(self.dynamic_filestream)
            self.current_static_file: Path = next(self.static_filestream)
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
            msg12318file=self.current_file if not override_file else override_file,
            msg5file=self.current_static_file,
            frame=area,
            search_radius=self.get_search_radius(area),
            n_cells=1,
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

            for i, ship in enumerate(ships):
                if ship.ship_type not in self.ship_types:
                    del ships[i]
            
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

    def __init__(self, t1: TargetVessel, t2: TargetVessel, interval: int = 10) -> None:


        matcher = pytsa.TrajectoryMatcher(t1, t2)
        if matcher.disjoint_trajectories or not matcher.overlapping_trajectories:
            raise ValueError("Trajectories are disjoint or do not overlap.")
        
        self.start = matcher.start
        self.end = matcher.end
        #                  -------v Set the interval for the scan [s]
        matcher.observe_interval(interval=interval)
        self.matcher = matcher

        # Array of times at which the track is observed
        self.times = np.arange(self.start, self.end, interval)

    def __call__(self, window_width) -> bool:

        min_index = self.forward_scan()
        if min_index is None or min_index < window_width:
            return False
        success = []
        for obs in [self.matcher.obs_vessel1,self.matcher.obs_vessel2]:
            success.append(
                self.backward_scan(obs[:min_index],window_width)
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

        # Return the time of minimum distance
        return np.where(distances == np.min(distances))[0][0]
    
    def sliding_window(self, input: list, interval_length: int = 3) -> Generator:
        """
        Generator yielding a sliding window of length `interval_length`
        over the input list.
        """
        for i in range(len(input)-interval_length):
            yield input[i:i+interval_length]
            
    def reverse_sliding_window(self, input: list, interval_length: int = 3) -> Generator:
        """
        Generator yielding a sliding window of length `interval_length`
        over the reversed input list.
        """
        for i in range(len(input)-interval_length,0,-1):
            yield input[i:i+interval_length]
        
    
    def backward_scan(
            self, 
            obs: np.ndarray,
            window_width: int = 3) -> bool:
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
            window_width (int, optional): Number of observations
                                        to scan per window. Defaults to 3.
        """
        # Walk backwards in time using a sliding window
        # of length `interval_length`
        for window in self.reverse_sliding_window(obs, window_width):
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
            # We take the last value of the dROT array because
            # the array is reversed
            if np.sign(drot[-1]) == rots_sign:
                logger.info(
                    f"Evasive maneuver detected between "
                    f"{self.matcher.vessel1.mmsi} and {self.matcher.vessel2.mmsi}"
                )
                return True
            else:
                continue
        return False