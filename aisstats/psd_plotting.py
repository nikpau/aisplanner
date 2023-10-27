"""
Plotting module for the Probabalistic Ship Domain (PSD).
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from psd import rawPSD
from pathlib import Path
from glob import glob
from pytsa.structs import ShipType

def load_psd_file(path: Path) -> rawPSD:
    with open(path, "rb") as f:
        psd = pickle.load(f)
    return psd

def merge_rawPSD(psds: list[rawPSD], ship_type: ShipType) -> rawPSD:
    """
    Merge the `data` dictionaries of rawPSD objects.
    """
    if not all(isinstance(psd, rawPSD) for psd in psds):
        raise TypeError("All objects must be of type rawPSD")
    for psd in psds:
        if psd.ship_type != ship_type:
            psds.remove(psd)
    new = rawPSD(ship_type=ship_type)
    for d in tuple(p.data for p in psds):
        for key, value in d.items():
            new.data[key].extend(value)
    return new

def plot_polar_psd(psd: rawPSD) -> None:
    
    f, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    for name, data in psd.data.items():
        thetas = [t[1] for t in data] # bearing to target vessel
        rs = [t[0] for t in data] # distance to target vessel in miles
        ax.scatter(thetas,rs, c = "k",s=1)
        ax.set_rmax(2)
    plt.title(f"PSD for {psd.ship_type.name}")
    plt.savefig(f"psdresults/plots/{psd.ship_type.name}.png", dpi=300)
    plt.close()
            
if __name__ == "__main__":
    files = glob("psdresults/*.rpsd")
    
    files = [load_psd_file(f) for f in files]
    
    passenger = merge_rawPSD(files, ship_type=ShipType.PASSENGER)
    tanker = merge_rawPSD(files, ship_type=ShipType.TANKER)
    cargo = merge_rawPSD(files, ship_type=ShipType.CARGO)
    
    for st in (passenger, tanker, cargo):
        plot_polar_psd(st)