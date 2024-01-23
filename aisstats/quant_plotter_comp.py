import pickle
from matplotlib import pyplot as plt
import numpy as np
from aisstats.errchecker import COLORWHEEL_MAP

names = ["squants","hquants","tquants","diffquants","dquants"]
description = [
    "Speed changes",
    "Heading changes",
    "Time difference between messages",
    "Difference between reported\nand calculated speed",
    "Distance between\nconsecutive messages"
]
ylabel = [
    "Speed [kn]",
    "Heading [deg]",
    "Time [s]",
    "Speed [kn]",
    "Distance [mi]"
]

months = [f"0{i}" for i in range(1,10)] + [str(i) for i in range(10,13)]

intervals = [(1,1),(1,4),(1,8),(1,12)]

int2name = ["1 Month","4 Months","8 Months","12 Months"]

fig, ax = plt.subplots(1,5, figsize=(15,3))
for k, (lower, upper) in enumerate(intervals):
    for i in range(5):
        with open(f"results/{names[i]}_{lower}-{upper}.pkl","rb") as f:
            quantiles = pickle.load(f)
        ax[i].plot(
            np.linspace(0,1,1001), 
            quantiles, 
            label=int2name[k], 
            c=COLORWHEEL_MAP[k%len(COLORWHEEL_MAP)]
        )
        ax[i].set_title(description[i],fontsize=8)
        ax[i].set_xlabel("Quantile")
        ax[i].set_ylabel(ylabel[i])
        if i in [2,3,4]:
            # Log scale
            ax[i].set_yscale("log")
        ax[i].legend(fontsize=6,ncol=2)
plt.tight_layout()
plt.savefig("results/quantiles-comparison.pdf",dpi=300)