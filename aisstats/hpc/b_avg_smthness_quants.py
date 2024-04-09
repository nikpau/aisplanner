from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import speed_filter, average_smoothness_quantiles
from pathlib import Path
from pytsa import SearchAgent
from pytsa.structs import AISMessage
from functools import partial
from pytsa.trajectories.rules import *
from pytsa.trajectories import inspect
import pickle

if __name__ == "__main__":
    SEARCHAREA = NorthSea

    DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07*.csv"))
    STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07*.csv"))

    SA = SearchAgent(
            dynamic_paths=DYNAMIC_MESSAGES,
            frame=SEARCHAREA,
            static_paths=STATIC_MESSAGES,
            preprocessor=partial(speed_filter, speeds= (1,30))
        )

    ships = SA.extract_all(njobs=16)
    average_smoothness_quantiles(ships)

