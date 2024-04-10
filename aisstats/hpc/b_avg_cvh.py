from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import _cvh_area, speed_filter
from pathlib import Path
from pytsa import SearchAgent, TimePosition, ShipType
from pytsa.structs import AISMessage
from functools import partial
from pytsa.trajectories.rules import *

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )

ships = SA.extract_all(njobs=16)

ExampleRecipe = Recipe(
    partial(too_few_obs, n=50),
)
from pytsa.trajectories import Inspector
inspctr = Inspector(
    data=ships,
    recipe=ExampleRecipe
)
accepted, _ = inspctr.inspect(njobs=1)
# Split up accepted and rejected trajectories
# to only contain ships of type CARGO, TANKER,
# PASSENGER and FISHING
types = [t for t in ShipType]
names = [t.name for t in types]
expanded = []
for t in types:
    if isinstance(t.value, int):
        expanded.append([t.value])
    else:
        expanded.append(list(t.value))

with open("/home/s2075466/aisplanner/results/avg_cvh_areas.csv", "w") as f:
    # Header
    f.write("ship_type,avg_cvh_area\n")
    for i,t in enumerate(expanded):
        a = {mmsi:s for mmsi,s in accepted.items() if any(st in t for st in s.ship_type)}
        # Calculate average spatial deviation per ship type
        # and save the results to a file
        cvh_areas = []
        for mmsi, ship in a.items():
            for track in ship.tracks:
                try:
                    cvh_areas.append(_cvh_area(track))
                except:
                    print(f"Could not calculate convex hull for {mmsi}")
        f.write(f"{names[i]},{np.mean(cvh_areas)}\n")
        