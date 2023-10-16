"""
Trajectory plotting functions
Module for plotting colregs encounters found via filter.py
"""
from glob import glob
from aisplanner.encounters.utils import (
    load_results, plot_encounter, OverlappingPair
)

if __name__ == "__main__":
    # Load results
    encounter_files = glob("results_single_sequence/encounters/*.tr")
    for file in encounter_files:
        res: list[OverlappingPair] = load_results(file)
        print(f"Found {len(res)} evasive maneuvers.")
        for vobj in res:
            if vobj.same_mmsi():
                continue
            try:
                plot_encounter(*vobj())
            except Exception as e:
                print(e)
                continue