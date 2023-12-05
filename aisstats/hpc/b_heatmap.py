from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import binned_heatmap, speed_filter
from pathlib import Path
from pytsa import SearchAgent, TimePosition, ShipType
from functools import partial
from pytsa.trajectories.rules import *

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07*.csv"))

SA = SearchAgent(
        msg12318file=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        msg5file=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )
    
# Create starting positions for the search.
# This is just the center of the search area.
center = SEARCHAREA.center
tpos = TimePosition(
    timestamp="2021-07-01", # arbitrary date
    lat=center.lat,
    lon=center.lon
)
SA.init(tpos)

ships = SA.get_all_ships(njobs=16)

ExampleRecipe = Recipe(
    partial(too_few_obs, n=50),
    partial(too_small_spatial_deviation, sd=0.2)
)
from pytsa.trajectories import Inspector
inspctr = Inspector(
    data=ships,
    recipe=ExampleRecipe
)
accepted, rejected = inspctr.inspect(njobs=1)
binned_heatmap(accepted, SEARCHAREA, savename=f"/home/s2075466/aisplanner/results/accepted_all_07_21.png")
binned_heatmap(rejected, SEARCHAREA, savename=f"/home/s2075466/aisplanner/results/rejected_all_07_21.png")
# Split up accepted and rejected trajectories
# to only contain ships of type CARGO, TANKER,
# PASSENGER and FISHING
types = [ShipType.CARGO, ShipType.TANKER, ShipType.PASSENGER, ShipType.FISHING]
names = [t.name for t in types]
expanded = []
for t in types:
    if isinstance(t.value, int):
        expanded.append([t.value])
    else:
        expanded.append(list(t.value))
for i,t in enumerate(expanded):
    a = {mmsi:s for mmsi,s in accepted.items() if any(st in t for st in s.ship_type)}
    r = {mmsi:s for mmsi,s in rejected.items() if any(st in t for st in s.ship_type)}
    binned_heatmap(a, SEARCHAREA, savename=f"/home/s2075466/aisplanner/results/accepted_{names[i]}_07_21.png")
    binned_heatmap(r, SEARCHAREA, savename=f"/home/s2075466/aisplanner/results/rejected_{names[i]}_07_21.png")