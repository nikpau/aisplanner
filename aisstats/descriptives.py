"""
This module extracts descriptive statistics on the distribution of ship sizes
and speeds in the ais data set.
"""
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.stats import gaussian_kde
from aisstats.errchecker import (
    COLORWHEEL, COLORWHEEL2,
    COLORWHEEL_DARK, COLORWHEEL2_DARK,
    COLORWHEEL3
)
import numpy as np

import colorsys



RESFILE = Path("results/descriptives.csv")
FONTSIZE = 12

# Load pandas dataframe and return the only row in it
# as a numpy array.
def _load_df(file: Path) -> pd.DataFrame:
    return pd.read_csv(file).values[0]

descr = _load_df(RESFILE)

# Descr is a numpy array with the following entries:
# 0: total number of observations in the data set
# 1-24: number of observations per hour of the day
# 25-36: number of observations per month of the year
# 37-63: number of observations per message type
# 64: number of unique MMSI numbers in the data set

# Plot the number of observations per hour of the day
# as a bar plot with a kernel density estimate.
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.bar(range(24), descr[1:25], color=COLORWHEEL[0])
ax.set_xlabel("Hour of the day",fontsize=FONTSIZE)
ax.set_ylabel("Number of observations",fontsize=FONTSIZE)
ax.set_xticks(range(0,24,3))
ax.set_xticklabels(range(0,24,3))
ax.set_axisbelow(True)

ax.set_ylim(0.95*min(descr[1:25]), 1.05*max(descr[1:25]))

plt.savefig("results/hourly_obs.pdf")
plt.close()

# Plot the number of observations per month of the year
# as a bar plot with a kernel density estimate.
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.bar(range(1,13), descr[25:37], color=COLORWHEEL[0])
ax.set_xlabel("Month of the year",fontsize=FONTSIZE)
ax.set_ylabel("Number of observations",fontsize=FONTSIZE)
ax.set_xticks(range(1,13))
ax.set_xticklabels(range(1,13))
ax.set_axisbelow(True)
plt.savefig("results/monthly_obs.pdf")
plt.close()

# Plot bar chart
fig, ax = plt.subplots(1,1, figsize=(6,4))

# Remove message types with no observations
# sort by number of observations and save indices
msgtypecounts = descr[37:64]
msgnumber = np.arange(1,28)
idx = np.where(msgtypecounts > 0)[0]
msgtypecounts = msgtypecounts[idx]
msgnumber = msgnumber[idx]

sidx = np.argsort(msgtypecounts)[::-1]
msgtypecounts = msgtypecounts[sidx]
msgnumber = msgnumber[sidx]

barh = ax.barh(np.arange(len(msgnumber)), 
       msgtypecounts, 
       color=COLORWHEEL3
    )

ax.set_yticks(np.arange(len(msgnumber)),labels = [f"Type {i}" for i in msgnumber])
ax.invert_yaxis()
ax.set_axisbelow(True)

# Add legend with counts
ax.legend(
    barh,
    [f"{c:,.0f}" for c in msgtypecounts],
    title="Number of observations",
    loc="lower right"
)

ax.set_xlabel("Number of observations",fontsize=FONTSIZE)

ax.set_title("Message types",fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig("results/msgtypes.pdf")

print(f"Total number of observations: {descr[0]:,.0f}")
print(f"Number of unique MMSI numbers: {descr[64]:,.0f}")