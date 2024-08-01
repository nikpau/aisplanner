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

STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07_01.csv"))

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
            
    for i,s in enumerate(ShipType):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(stdict[s], bins=np.linspace(0,300,100), color=COLORWHEEL[0])
        plt.savefig(f"/home/s2075466/aisplanner/results/length_hist_{s}.pdf",dpi=300)
        plt.close()