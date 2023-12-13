"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from pytsa.decode import filedescriptor as fd
import numpy as np
import pandas as pd
import multiprocessing as mp

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/curated/jan2020_to_jun2022').glob("*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/curated/jan2020_to_jun2022/msgtype5').glob("*.csv"))

def routine(file: Path) -> np.ndarray:
    df = pd.read_csv(file)
    nobs = len(df) # number of observations in the entire file
    print(f"Processing {file} with {nobs:_} observations.")
    unique_originators = df["originator"].unique()
    return unique_originators

if __name__ == "__main__":
    with mp.Pool(20) as pool:
        results = pool.map(routine, DYNAMIC_MESSAGES + STATIC_MESSAGES)

    # Extract the results from the list of lists
    # and put them in a numpy array.
    results = np.hstack(results)
    uniques = np.unique(results)
    print(uniques)