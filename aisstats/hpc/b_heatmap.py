from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import binned_heatmap, speed_filter
from pathlib import Path
from pytsa import SearchAgent, ShipType
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

ships = SA.extract_all(njobs=32)

R = Recipe(
    partial(too_few_obs, n=50),
    partial(convex_hull_area, area=5e3)
)
from pytsa.trajectories import Inspector
inspctr = Inspector(
    data=ships,
    recipe=R
)
accepted, rejected = inspctr.inspect(njobs=1)
print(
    f"No. of accepted trajectories: {sum([len(t.tracks) for t in accepted.values()])}\n"
    f"No. of rejected trajectories: {sum([len(t.tracks) for t in rejected.values()])}\n"
    f"Total no. of accepted observations: {sum([sum(len(tr) for tr in t.tracks) for t in accepted.values()])}\n"
    f"Total no. of rejected observations: {sum([sum(len(tr) for tr in t.tracks) for t in rejected.values()])}\n"
)
binned_heatmap(accepted, SEARCHAREA, savename=f"/home/s2075466/aisplanner/results/maps/hm_accepted_all_21.png")
binned_heatmap(rejected, SEARCHAREA, savename=f"/home/s2075466/aisplanner/results/maps/hm_rejected_all_21.png")
# Split up accepted and rejected trajectories

for st in ShipType:
    a = {mmsi:s for mmsi,s in accepted.items() if s.ship_type == st}
    r = {mmsi:s for mmsi,s in rejected.items() if s.ship_type == st}
    binned_heatmap(a, SEARCHAREA, savename=f"/home/s2075466/aisplanner/results/maps/hm_accepted_{st.name}_21.png")
    binned_heatmap(r, SEARCHAREA, savename=f"/home/s2075466/aisplanner/results/maps/hm_rejected_{st.name}_21.png")