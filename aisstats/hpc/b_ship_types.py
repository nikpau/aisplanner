"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from pytsa.decode import filedescriptor as fd
import numpy as np
import pandas as pd
import multiprocessing as mp
import pickle

STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

def routine(file: Path) -> np.ndarray:
    df = pd.read_csv(file)
    nobs = len(df) # number of observations in the entire file
    print(f"Processing {file} with {nobs:_} observations.")
    df = df.drop_duplicates(subset=fd.Msg5Columns.MMSI.value)
    
    return df

if __name__ == "__main__":
    with mp.Pool(20) as pool:
        results = pool.map(routine, STATIC_MESSAGES)

    # Concat the results
    df: pd.DataFrame = pd.concat(results)
    df = df.drop_duplicates(subset=fd.Msg5Columns.MMSI.value)
    
    shiptypes = df[fd.Msg5Columns.SHIPTYPE.value].value_counts()
    
    types = shiptypes.index
    counts = shiptypes.values
    
    with open("/home/s2075466/aisplanner/results/ship_types.pkl", "wb") as f:
        pickle.dump([(t,v) for t,v in zip(types,counts)], f)