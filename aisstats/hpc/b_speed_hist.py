"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path

import numpy as np
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import plot_speed_histogram
from pathlib import Path
from pytsa import SearchAgent, TimePosition
import ciso8601
import pandas as pd
import multiprocessing as mp


def _date_transformer(datefile: Path) -> float:
    return ciso8601.parse_datetime(datefile.stem.replace("_", "-"))

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))

def load_and_count(path: Path) -> int:
    print(f"Processing {path}")
    df = pd.read_csv(path,usecols=['speed'])
    out = df['speed'].values
    return out

if __name__ == "__main__":
    with mp.Pool(20) as pool:
        results = pool.map(load_and_count, DYNAMIC_MESSAGES)
    
    sp = np.hstack(results)

    plot_speed_histogram(sp, savename="/home/s2075466/aisplanner/results/speed_hist_21.pdf")