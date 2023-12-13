"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from pytsa.decode import filedescriptor as fd
import numpy as np
import pandas as pd
import multiprocessing as mp

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("*.csv"))

def routine(file: Path) -> np.ndarray:
    df = pd.read_csv(file)
    nobs = len(df) # number of observations in the entire file
    print(f"Processing {file} with {nobs:_} observations.")
    unique_mmsi = df[fd.Msg12318Columns.MMSI.value].unique()
    df[fd.BaseColumns.TIMESTAMP.value] = pd.to_datetime(
        df[fd.BaseColumns.TIMESTAMP.value]
    )
    
    # Make a bin for every hour in the day
    # and count the number of observations in each bin.
    # First, make a column with the hour of the day.
    df["hour"] = df[fd.BaseColumns.TIMESTAMP.value].dt.hour
    # Then, group by hour and count the number of observations in each bin.
    hcounts = df.groupby("hour")[fd.BaseColumns.TIMESTAMP.value].count()
    
    # Generate a list of hour counts with 0 for hours with no observations.
    hcounts = hcounts.reindex(range(24), fill_value=0).values
    
    # Make a bin for every month in the year
    # and count the number of observations in each bin.
    # First, make a column with the month of the year.
    df["month"] = df[fd.BaseColumns.TIMESTAMP.value].dt.month
    # Then, group by month and count the number of observations in each bin.
    mcounts = df.groupby("month")[fd.BaseColumns.TIMESTAMP.value].count()
    
    # Generate a list of month counts with 0 for months with no observations.
    mcounts = mcounts.reindex(range(1,13), fill_value=0).values
    
    # Make a bin for each of the 27 message types
    # and count the number of observations in each bin.
    # First, make a column with the message type.
    df["msgtype"] = df[fd.BaseColumns.MESSAGE_ID.value]
    # Then, group by message type and count the number of observations in each bin.
    msgcounts = df.groupby("msgtype")[fd.BaseColumns.MESSAGE_ID.value].count()
    
    # Generate a list of message type counts with 0 for message types with no observations.
    msgcounts = msgcounts.reindex(range(1,28), fill_value=0).values
    
    return np.hstack([nobs, hcounts, mcounts, msgcounts]), unique_mmsi

if __name__ == "__main__":
    with mp.Pool(20) as pool:
        results = pool.map(routine, DYNAMIC_MESSAGES + STATIC_MESSAGES)

    # Extract the results from the list of tuples
    # and put them in a numpy array.
    descr = np.array([r[0] for r in results])
    # Extract the unique MMSI numbers from the list of tuples.
    unique_mmsi = np.concatenate([r[1] for r in results])
    # Get the unique MMSI numbers in the entire data set.
    unique_mmsi = np.unique(unique_mmsi)
    
    # Sum over the results to get the total number of observations
    # and bin counts.
    out = descr.sum(axis=0)
    
    # Append the number of unique MMSI numbers to the results.
    out = np.append(out, len(unique_mmsi))
    
    # Make a single-line dataframe with the results.
    df = pd.DataFrame(
        data=out.reshape(1,-1),
        columns=[
            "nobs",
            *[f"hour_{i}" for i in range(24)],
            *[f"month_{i}" for i in range(1,13)],
            *[f"msgtype_{i}" for i in range(1,28)],
            "unique_mmsis"
        ]
    )

    # Save the dataframe to a csv file.    
    df.to_csv("/home/s2075466/aisplanner/results/descriptives.csv", index=False)