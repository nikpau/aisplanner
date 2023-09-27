"""
This module extracts descriptive statistics on the distribution of ship sizes
and speeds in the ais data set.
"""

from typing import Sequence
from pytsa.structs import ShipType
from aisplanner.dataprep import _file_descriptors as fd
from attr import define
import os
from glob import glob
from pathlib import Path
import pandas as pd
import pickle

def _check_env_init() -> None:
    """
    Checks whether the needed env variables are set.
    """
    for var in ["AISDECODED","MSG5DECODED"]:
        if var not in os.environ:
            raise EnvironmentError(
                f"Environment variable {var} is not set."
            )
    return None

@define
class ShipDims:
    """ 
    Data structure for storing dimensions of ship classes.
    """
    Length = int
    Width = int
    MMSI = int
    
    name: str
    ship_types: Sequence[int]
    dims: set[tuple[MMSI,Length,Width]]
    dests: set[str]
    def __str__(self):
        return self.name

    def clear(self):
        self.dims.clear()
    
# Define ship categories
CargoDims = ShipDims(
    name = "Cargo Ships",
    ship_types = ShipType.CARGO.value,
    dims = set(),
    dests = set()
)

TankerDims = ShipDims(
    name = "Tankers",
    ship_types = ShipType.TANKER.value,
    dims = set(),
    dests = set()
)
    
PassengerDims = ShipDims(
    name = "Passenger Ships",
    ship_types = ShipType.PASSENGER.value,
    dims = set(),
    dests = set()
)
    
    
def load_file(file: Path,fields: list[str] = []) -> pd.DataFrame:
    """
    Loads a single data file into a pandas DataFrame.
    If fields is empty, all columns are loaded.
    """
    return pd.read_csv(file,sep=",",usecols=fields)

def get_dynamic_filenames() -> list[Path]:
    """
    Returns a list of all ais data files.
    """
    _f = glob(f"{os.environ['AISDECODED']}/*.csv")
    _f.sort()
    return [Path(f) for f in _f]

def get_static_filenames() -> list[Path]:
    """
    Returns a list of all ais data files.
    """
    _f = glob(f"{os.environ['MSG5DECODED']}/*.csv")
    _f.sort()
    return [Path(f) for f in _f]
    
def static_extraction(statfile: Path, 
                      cats: list[ShipDims]) -> list[ShipDims]:
    """
    Main search loop, that iterates over the ais data set and extracts
    the length, width and speed of each ship, and stores them in their
    respective data structures.
    """
    
    staticdata = load_file(
        Path(statfile),
        fields=[
            fd.Fields5.MMSI.name,
            fd.Fields5.ship_type.name,
            fd.Fields5.to_bow.name,
            fd.Fields5.to_stern.name,
            fd.Fields5.to_port.name,
            fd.Fields5.to_starboard.name,
            fd.Fields5.destination.name
        ]
    )
    # Keep only unique MMSIs
    staticdata = staticdata.drop_duplicates(subset=[fd.Fields5.MMSI.name])
    
    # Add length and width columns
    staticdata["length"] =\
        staticdata[fd.Fields5.to_bow.name] + staticdata[fd.Fields5.to_stern.name]
    staticdata["width"] =\
        staticdata[fd.Fields5.to_port.name] + staticdata[fd.Fields5.to_starboard.name]

    # Initialize ship categories on first iteration
    if not cats:
        cats: list[ShipDims] = [
            CargoDims,
            TankerDims,
            PassengerDims,
        ]

    for cat in cats:
        cat_filter = staticdata[fd.Fields5.ship_type.name].isin(cat.ship_types)
        filtered = staticdata[cat_filter]
        # Add all dimensions to the set (and MMSI)
        cat.dims.update(
            zip(
                filtered[fd.Fields5.MMSI.name],
                filtered["length"],
                filtered["width"]
            )
        )
        # Add all destinations to the set
        cat.dests.update(
            filtered[fd.Fields5.destination.name].to_list()
        )

    return cats

def save_static_results(cats: list[ShipDims]) -> None:
    """
    Saves the results of the static extraction.
    """
    with open(f"aisstats/extracted/dims.pickle","wb") as f:
        pickle.dump(cats,f)
    return None

def static_run() -> None:
    """
    Runs the static extraction.
    """
    _check_env_init()
    sf = get_static_filenames()
    
    cats = []
    for f in sf:
        print(f"Processing {f.name}...")
        cats = static_extraction(f,cats)
        print(
            f"Done. Found {sum(len(c.dims) for c in cats)} ships in total "
            f"and {sum(len(c.dests) for c in cats)} destinations."
        )
    # Save results
    save_static_results(cats)
    print("Done.")
    return None
        
def single_file_static_run(filename: str) -> None:
    """
    Runs the static extraction on a single file.
    """
    f = Path(filename)
    cats = static_extraction(f)
    print(f"Processed {f.name}:\nFound {sum(len(c.dims) for c in cats)} ships in total.")
    # Save results
    with open(f"aisstats/extracted/static_{f.name}.pickle","wb") as f:
        pickle.dump(cats,f)
        
if __name__ == "__main__":
    static_run()
    #single_file_static_run("data/aisrecords/msgtype5/2021_04_01.csv")
