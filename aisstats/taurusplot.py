from aisplanner.encounters.main import GriddedNorthSea
from errchecker import binned_heatmap, speed_filter
from pathlib import Path
from pytsa import SearchAgent, TimePosition
from functools import partial
from pytsa.trajectories.rules import *

SEARCHAREA = GriddedNorthSea(nrows=1, ncols=1, utm=False).cells[0]

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

ships = SA.get_all_ships(njobs=16,skip_filter=True)

ExampleRecipe = Recipe(
    partial(too_few_obs, n=50),
    partial(too_small_spatial_deviation, sd=0.2)
)
from pytsa.trajectories import Inspector
inspctr = Inspector(
    data=ships,
    recipe=ExampleRecipe
)
accepted, rejected = inspctr.inspect(njobs=8)

binned_heatmap(accepted, SEARCHAREA, savename="/home/s2075466/aisplanner/results/acceped.png")
binned_heatmap(rejected, SEARCHAREA, savename="/home/s2075466/aisplanner/results/rejected.png")