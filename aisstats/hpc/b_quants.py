import pickle
from matplotlib import pyplot as plt
import numpy as np

# Load the quantiles
with open('results/squants.pkl', 'rb') as f:
    squants = pickle.load(f)
with open('results/hquants.pkl', 'rb') as f:    
    hquants = pickle.load(f)
with open('results/tquants.pkl', 'rb') as f:
    tquants = pickle.load(f)
with open('results/diffquants.pkl', 'rb') as f:
    diffquants = pickle.load(f)
    
# Plot the quantiles
fig, ax = plt.subplots(1,4, figsize=(12,4))
ax: list[plt.Axes]
ax[0].plot(np.linspace(0,1,1001), squants)
ax[0].set_title("Speed changes")
ax[0].set_xlabel("Quantile")
ax[0].set_ylabel("Speed (knots)")
ax[1].plot(np.linspace(0,1,1001), hquants)
ax[1].set_title("Heading changes")
ax[1].set_xlabel("Quantile")
ax[1].set_ylabel("Heading (degrees)")
ax[2].plot(np.linspace(0,1,1001), tquants)
ax[2].set_title("Time difference\nbetween messages")
ax[2].set_xlabel("Quantile")
ax[2].set_ylabel("Diff [s]")
ax[2].set_ylim(0, 4)
ax[3].plot(np.linspace(0,1,1001), diffquants)
ax[3].set_title("Difference between\nreported and\ncalculated speed")
ax[3].set_xlabel("Quantile")
ax[3].set_ylabel("Diff [kn]")
fig.tight_layout()
plt.show()
# fig.savefig('/home/s2075466/aisplanner/results/quantiles.png', dpi=300)