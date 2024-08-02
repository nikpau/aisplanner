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

STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_1*.csv"))

def load_and_count(path: Path, mmsis: set[int]) -> int:
    seen_mmsis = set()
    stdict = {s:[] for s in ShipType}
    print(f"Processing {path}")
    df = pd.read_csv(path,usecols=['MMSI','to_bow',"to_stern","ship_type"])
    # Only keep 
    
    df["length"] = df["to_bow"] + df["to_stern"]
    
    for row in df.itertuples():
        if row.MMSI not in mmsis:
            seen_mmsis.add(row.MMSI) 
            stdict[ShipType.from_value(row.ship_type)].append((row.MMSI,row.length))
        else:
            continue
    
    return stdict

if __name__ == "__main__":
    with mp.Pool(23) as pool:
        results = pool.map(load_and_count, STATIC_MESSAGES)
    
    # Join all dicts into one
    stdict = {s:[] for s in ShipType}
    for res in results:
        for s in ShipType:
            stdict[s].extend(res[s])
            
    # Last filter for unique MMSIs
    seen_mmsis = set()
    for s in ShipType:
        for mmsi, length in stdict[s]:
            if mmsi not in seen_mmsis:
                seen_mmsis.add(mmsi)
            else:
                stdict[s].remove((mmsi,length))
            
    fig, axs = plt.subplots(4,3,figsize=(12,8))
    
    for i,s in enumerate(ShipType):
        row, col = divmod(i,3)
        bins = np.linspace(0,300,50)
        # Add inf 
        bins = np.append(bins, np.inf)
        axs[row,col].hist(stdict[s],bins=bins,color=COLORWHEEL[i%len(COLORWHEEL)])
        axs[row,col].set_title(f"Ship Type {s.name}")
        axs[row,col].set_xlabel("Length [m]")
        axs[row,col].set_ylabel("Number of observations")
        # axs[row,col].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("/home/s2075466/aisplanner/results/length_hist_21.pdf")