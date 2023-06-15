"""
Trajectory plotting functions
Module for plotting colregs encounters found via filter.py
"""
from itertools import permutations

from pytsa.targetship import TargetVessel
from .utils import load_results, plot_encounter

if __name__ == "__main__":
    # Load results
    res: list[TargetVessel] = load_results("2021_hel_hel.tr")
    print(len(res))
    for v1,v2 in permutations(res,2):
        try:
            plot_encounter(v1,v2)
        except Exception as e:
            print(e)
            continue