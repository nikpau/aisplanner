"""
Trajectory plotting functions
Module for plotting colregs encounters found via filter.py
"""
from itertools import permutations

from aisplanner.encounters.utils import load_results, plot_encounter, OverlappingPair

if __name__ == "__main__":
    # Load results
    res: list[OverlappingPair] = load_results("results/overlapping/2021_fre_got_overlapping.tr")
    print(len(res))
    for vobj in res:
        if vobj.v1.mmsi == vobj.v2.mmsi:
            continue
        try:
            plot_encounter(*vobj())
        except Exception as e:
            print(e)
            continue