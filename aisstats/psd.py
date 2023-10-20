"""
Implementation of the Probabilistic ship domain (PSD) after
Zhang and Meng (2019)
https://doi.org/10.1016/j.oceaneng.2019.106130
"""
from pathlib import Path
import pickle
import numpy as np
from pytsa.structs import ShipType
import pytsa
from aisplanner.dataprep import _file_descriptors as fd
from more_itertools import pairwise
from aisplanner.encounters import (
    TrajectoryExtractionAgent, 
    true_bearing, TargetVessel, 
    ALL_SHIPS, FileLoadingError,
    Position
)
import multiprocessing as mp
from aisplanner.misc import logger
import ciso8601
from datetime import timedelta
from typing import Generator, List, Union
import psutil
import os

Latitude = float
Longitude = float

# Value proxy for degernate cases
DEGEN = "degen"

# Bins for the lengths of the ships.
# The PSD will be calculated for each bin.
LBINS = np.arange(0, 250 + 25, 25) # 0-250m in 25m bins

# Bins for the speeds of the ships.
# We assume that the PSD looks different for different speeds.
# The PSD will be calculated for each bin.
SBINS = np.arange(0, 30 + 2.5, 2.5) # 0-30kts in 2.5kts bins

def get_slot_name(length: int, speed: int) -> str:
    """
    Returns the slot name for a given length and speed.
    """
    for llower, lupper in pairwise(LBINS):
        for slower, supper in pairwise(SBINS):
            if llower <= length <= lupper and slower <= speed <= supper:
                return f"{llower}-{lupper}_{slower}kn-{supper}kn"
    return DEGEN # Degenerate case

def slot_gen() -> list[str]:
    """
    Generator for the slots of the rawPSD class.
    The slots will be the speed and length bins.
    """
    slots = []
    for llower, lupper in pairwise(LBINS):
        for slower, supper in pairwise(SBINS):
            slots.append(f"{llower}-{lupper}_{slower}kn-{supper}kn") 
    return slots

class rawPSD:
    """
    This class holds all extraced points around an
    abstract ship type comprised of a length and speed bin.
    
    The points are distance and bearing pairs for each combination
    of length and speed bin.
    
    These points will be used to calculate the PSD.
    """
    
    def __init__(self, ship_type: ShipType) -> None:
        self.ship_type = ship_type
        self.data = {s: [] for s in slot_gen()}
        # Degenerate cases where the speed
        # and length is outside the range of the bins
        self.data.update({DEGEN: []})

class PSDPointExtractor(TrajectoryExtractionAgent):
    
    def __init__(self,*,
            search_areas: List[pytsa.UTMBoundingBox],
            ship_types: list[ShipType],
            time_delta: int = 30,
            msg12318files: List[Union[Path, str]] = None,
            msg5files: List[Union[Path, str]] = None) -> None:
        
        # Check if search areas are given
        if not isinstance(search_areas, list):
            search_areas = [search_areas]

        self.search_areas = search_areas

        # List of ShipTypes to be searched for
        if not isinstance(ship_types, list):
            ship_types = [ship_types]
        if not ship_types:
            ship_types = ALL_SHIPS
        self.ship_types = ship_types
        
        # Set the maximum temporal deviation of target
        # ships from provided time in `init()`
        self.time_delta = time_delta # in minutes

        # Initialize the list of raw PSD points 
        self.cargo_raw_psd = rawPSD(ShipType.CARGO)
        self.passenger_raw_psd = rawPSD(ShipType.PASSENGER)
        self.tanker_raw_psd = rawPSD(ShipType.TANKER)
        

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
            
    def save_results(self, directory: Union[str, Path],boxnumber: int) -> None:
        """
        Saves the raw PSD points as pickled objects.
        """
        # Check if destination is a Path
        if not isinstance(directory, Path):
            directory = Path(directory)
        # Check if destination is a directory
        # and create it if it does not exist
        if not directory.is_dir():
            directory.mkdir(parents=True)
            
        # Save the raw PSD points
        with open(directory / f"{boxnumber}_cargo_raw.psd", "wb") as f:
            pickle.dump(self.cargo_raw_psd, f)
        with open(directory / f"{boxnumber}_passenger_raw.psd", "wb") as f:
            pickle.dump(self.passenger_raw_psd, f)
        with open(directory / f"{boxnumber}_tanker_raw.psd", "wb") as f:
            pickle.dump(self.tanker_raw_psd, f)

    def _search(
        self,
        area: pytsa.UTMBoundingBox) -> List[TargetVessel]:
        """
        Search for valid trajectories in a given area.
        """
        # Datetime of current file, starting at midnight
        # NOTE: This only works if the file name is in the format
        #       YYYY-MM-DD.csv
        #       Since the AIS messages we use are in the format
        #       YYYY_MM_DD.csv, we need to replace the _ with -.
        corrected_filename = self.current_file.stem.replace("_", "-")
        start_date = ciso8601.parse_datetime(corrected_filename)
        
        logger.info(f"Begin searching {area.name} at {start_date}")

        # Initialize search agent for nearest neighbors
        # and trajectory interpolation
        search_agent = pytsa.SearchAgent(
            msg12318file=self.current_file,
            msg5file=self.current_static_file,
            frame=area,
            search_radius=self.get_search_radius(area),
            n_cells=1,
        )
        
        # Set the maximum temporal deviation of target
        # ships from provided time in `init()`
        search_agent.time_delta = self.time_delta # in minutes

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
        # We need to increment the search date by 30 minutes
        # to avoid overlapping the search times.
        # as the search agent will search around
        # search_date +- time_delta
        search_date = start_date + timedelta(minutes=self.time_delta)

        # Increment the search date until the date of search is
        # greater than the date of the current file.
        logger.info(f"Searching {self.current_file}")
        
        # Print memory usage of the process
        logger.info(
            f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB"
            )
        
        
        while start_date.day == search_date.day:

            ships: List[TargetVessel] = search_agent.get_ships(tpos)
            if not ships:
                tpos = self._update_timeposition(tpos, self.time_delta)
                search_date = tpos.timestamp
                logger.info(f"Skipping {search_date}. No ships found.")
                continue
            
            
            # For all ship types we are interested in...
            for ship_type in self.ship_types:
                # ...find the ships of that type...
                seen = set()
                for own_vessel, index in find_by_type(ships, ship_type):
                    own_vessel: TargetVessel # make the type checker happy
                    other_vessels: list[TargetVessel] = ships.remove(ships[index])
                    # ...and get the minimum distance and bearing
                    # to all other ships of all other types
                    for other_vessel in other_vessels:
                        
                        # Check if we have already seen this pair
                        if (own_vessel.mmsi,other_vessel.mmsi)in seen or\
                            (other_vessel.mmsi,own_vessel.mmsi) in seen:
                            continue
                        
                        logger.info(
                            f"Checking {own_vessel.mmsi} and {other_vessel.mmsi}"
                            )
                        seen.add(((own_vessel.mmsi,other_vessel.mmsi),
                                (other_vessel.mmsi,own_vessel.mmsi)))
                        
                        # Find the closest point between the two trajectories
                        cp = closest_point(own_vessel,other_vessel)
                        mindist, own_pos, other_pos, speed = cp
                        own_pos = Position(*own_pos)
                        other_pos = Position(*other_pos)
                        
                        # Find the true bearing of the other vessel
                        # relative to the own vessel
                        bearing = true_bearing(own_pos, other_pos)
                        
                        # Find the slot name for the current ship
                        # and save the distance and bearing
                        # to the correstponding rawPSD object
                        res = (mindist, bearing)
                        own_length = own_vessel.length
                        own_speed = speed
                        slotname = get_slot_name(own_length, own_speed)
                        self.raw_psd_selector(ship_type).data[slotname].append(res)
            
            # End of search for this time step
            # We need to move on in time twice the time delta
            # as the search agent will search 
            # search_date +- time_delta
            tpos = self._update_timeposition(tpos, 2*self.time_delta)
            search_date = tpos.timestamp
            
        logger.info(f"Finished searching {self.current_file}")
            
        return None
    
    def search(self) -> None:
        """
        Search all areas for raw PSD points.
        """
        while True:
            try:
                self.load_next_file()
                for area in self.search_areas:
                    self._search(area)
                if self._using_remote:
                    self.delete_current_file()
            except StopIteration:
                return

    def raw_psd_selector(self,ship_type: ShipType) -> rawPSD:
        """
        Returns the rawPSD object for a given ship type.
        """
        if ship_type == ShipType.CARGO:
            return self.cargo_raw_psd
        elif ship_type == ShipType.PASSENGER:
            return self.passenger_raw_psd
        elif ship_type == ShipType.TANKER:
            return self.tanker_raw_psd
        else:
            raise ValueError(f"Invalid ship type: {ship_type}")
    
def closest_point(own: TargetVessel,tgt: TargetVessel
                  ) -> tuple[float,tuple[Latitude,Longitude],tuple[Latitude,Longitude],float]:
    """
    Returns the distance between the closest point of two vessels'
    trajectories, the coordinates of that point,
    and the speed of the own vessel at that point.
    """
    # Get the closest point between the two trajectories
    mindist = np.inf
    for own_msg, tgt_msg in zip(own.track, tgt.track):
        dist = np.linalg.norm(
            np.array([own_msg.lat,own_msg.lon]) - 
            np.array([tgt_msg.lat,tgt_msg.lon])
        )
        if dist < mindist:
            mindist = dist
            speed = own_msg.SOG
            own_pos = (own_msg.lon,own_msg.lat)
            tgt_pos = (tgt_msg.lon,tgt_msg.lat)
    return mindist, own_pos, tgt_pos, speed

def find_by_type(
    vessels: list[TargetVessel],
    ship_type: ShipType) -> Generator[tuple[TargetVessel,int],None,None]:
    """
    Given a list of TargetVessel objects, returns a generator
    of tuples containing the TargetVessel object of the requested
    ship type and the index of the vessel in the list.
    """
    for i, vessel in enumerate(vessels):
        if vessel.ship_type == ship_type:
            yield vessel, i