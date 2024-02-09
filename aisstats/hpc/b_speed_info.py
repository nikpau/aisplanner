"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from pytsa.decode import filedescriptor as fd
import numpy as np
import pandas as pd
import multiprocessing as mp

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))

def routine(file: Path) -> np.ndarray:
    df = pd.read_csv(file)
    nobs = len(df) # number of observations in the entire file
    print(f"Processing {file} with {nobs:_} observations.")
    speeds = df[fd.Msg12318Columns.SPEED.value].values
    
    return len(speeds), sum(speeds > 1), sum(speeds < 30)

if __name__ == "__main__":
    with mp.Pool(20) as pool:
        results = pool.map(routine, DYNAMIC_MESSAGES)

    # Extract the results from the list of tuples
    # and put them in a numpy array.
    nobs, nobs1, nobs30 = zip(*results)
    
    totnobs = sum(nobs)
    totnobs1 = sum(nobs1)
    totnobs30 = sum(nobs30)
    frac1 = totnobs1 / totnobs
    frac30 = totnobs30 / totnobs
    
    print(
        f"Total number of observations: {totnobs:_}\n"
        f"Fraction of observations with speed > 1: {frac1:.2%}\n"
        f"Fraction of observations with speed < 30: {frac30:.2%}\n"
    )