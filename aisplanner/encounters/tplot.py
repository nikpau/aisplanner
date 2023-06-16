"""
Trajectory plotting functions
Module for plotting colregs encounters found via filter.py
"""
from aisplanner.encounters.utils import (
    load_results, plot_encounter, OverlappingPair
)

if __name__ == "__main__":
    # Load results
    res: list[OverlappingPair] = load_results("results/overlapping/2021_fre_got_overlapping.tr")
    print(len(res))
    for vobj in res:
        if vobj.same_mmsi():
            continue
        try:
            plot_encounter(*vobj())
        except Exception as e:
            print(e)
            continue