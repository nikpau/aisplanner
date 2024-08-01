"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path

import numpy as np
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import plot_speed_histogram, COLORWHEEL
from pathlib import Path
from pytsa.structs import ShipType
import ciso8601
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt


def _date_transformer(datefile: Path) -> float:
    return ciso8601.parse_datetime(datefile.stem.replace("_", "-"))

STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07_1*.csv"))

def load_and_count(path: Path) -> int:
    stdict = {s:[] for s in ShipType}
    print(f"Processing {path}")
    df = pd.read_csv(path,usecols=['to_bow',"to_stern","ship_type"])
    df["length"] = df["to_bow"] + df["to_stern"]
    
    for row in df.itertuples():
        stdict[ShipType.from_value(row.ship_type)].append(row.length)
    
    return stdict

if __name__ == "__main__":
    with mp.Pool(23) as pool:
        results = pool.map(load_and_count, STATIC_MESSAGES)
    
    # Join all dicts into one
    stdict = {s:[] for s in ShipType}
    for res in results:
        for s in ShipType:
            stdict[s].extend(res[s])
            
    fig, axs = plt.subplots(4,3,figsize=(12,8))
    
    for i,s in enumerate(ShipType):
        row, col = divmod(i,3)
        axs[row,col].hist(stdict[s],bins=50,color=COLORWHEEL[i%len(COLORWHEEL)])
        axs[row,col].set_title(f"Ship Type {s.name}")
        axs[row,col].set_xlabel("Length [m]")
        axs[row,col].set_ylabel("Number of observations")
        axs[row,col].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("/home/s2075466/aisplanner/results/length_hist_21.pdf")