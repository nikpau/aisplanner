from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import binned_heatmap, speed_filter
from pathlib import Path
from pytsa import SearchAgent, TimePosition, ShipType
from functools import partial
from pytsa.trajectories.rules import *

def spatial_dev(track: list[AISMessage]) -> float:
    """
    Calculate the spatial deviation of a track.
    """
    lons = [p.lon for p in track]
    lats = [p.lat for p in track]
    return np.std(lons) + np.std(lats)

SEARCHAREA = NorthSea

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021_07_01*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021_07_01*.csv"))

SA = SearchAgent(
        dynamic_paths=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        static_paths=STATIC_MESSAGES,
        preprocessor=partial(speed_filter, speeds= (1,30))
    )

ships = SA.get_all_ships(njobs=16)

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
types = [ShipType.CARGO, ShipType.TANKER, ShipType.PASSENGER, ShipType.FISHING]
names = [t.name for t in types]
expanded = []
for t in types:
    if isinstance(t.value, int):
        expanded.append([t.value])
    else:
        expanded.append(list(t.value))

with open("/home/s2075466/aisplanner/results/avg_spatial_deviation.csv", "w") as f:
    # Header
    f.write("ship_type,avg_spatial_deviation\n")
    for i,t in enumerate(expanded):
        a = {mmsi:s for mmsi,s in accepted.items() if any(st in t for st in s.ship_type)}
        # Calculate average spatial deviation per ship type
        # and save the results to a file
        deviations = []
        for mmsi, ship in a.items():
            for track in ship.tracks:
                deviations.append(spatial_dev(track))
        f.write(f"{names[i]},{np.mean(deviations)}\n")
        