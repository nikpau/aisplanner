import matplotlib.pyplot as plt
import pickle
import numpy as np 
from pathlib import Path
from pytsa import ShipType
from aisstats.errchecker import COLORWHEEL, COLORWHEEL_MAP

def ship_type_to_color(t) -> str:
    """
    Convert a ship type to a color
    """
    for i, st in enumerate(ShipType):
        if isinstance(st.value, range):
            if t in st.value:
                return COLORWHEEL_MAP[i]
        else:         
            if t == st.value:
                return COLORWHEEL_MAP[i]

def group_ship_types(types: np.ndarray[int], 
                     counts: np.ndarray[int]) -> tuple[list[int], list[int]]:
    """
    Group the ship types by type in the 
    order of the ShipType enum.
    """
    newtypes = []
    newcounts = []
    for st in ShipType:
        if isinstance(st.value, range):
            newtypes.append(st.name)
            newcounts.append(sum(counts[np.isin(types, st.value)]))
        else:
            newtypes.append(st.name)
            newcounts.append(sum(counts[types == st.value]))
    return newtypes, newcounts
    

# Load the ship types
with open("results/ship_types.pkl", "rb") as f:
    types, counts = zip(*pickle.load(f))
    
# Group the ship types by type
types, counts = group_ship_types(np.array(types), np.array(counts))
    
# Sort the ship types by type
sorter = np.argsort(counts)
types = np.array(types)[sorter][::-1]
counts = np.array(counts)[sorter][::-1]

nobs = sum(counts)

# Plot the ship types as a bar plot
fig, ax = plt.subplots(1,1, figsize=(6,4))

# Bar plot with colors depending on ship type
# from the pytsa module
for i, t in enumerate(types):
    ax.bar(i, counts[i]/nobs, color=(COLORWHEEL_MAP+COLORWHEEL)[i], label=t)

ax.set_axisbelow(True)

# Log scale on the y-axis
# ax.set_yscale("log")

ax.set_xticks(range(len(types)))
ax.set_xticklabels(types, rotation=45)
ax.set_xlabel("Ship type", fontsize=12)
ax.set_ylabel("Fraction of observations", fontsize=12)

plt.show()