"""
HPC script for calculating descriptive data
for the AIS data set
"""
from pathlib import Path
from aisplanner.encounters.main import NorthSea
from aisstats.errchecker import plot_speed_histogram
from pathlib import Path
from pytsa import SearchAgent, TimePosition
import ciso8601


def _date_transformer(datefile: Path) -> float:
    return ciso8601.parse_datetime(datefile.stem.replace("_", "-"))

DYNAMIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022').glob("2021*.csv"))
STATIC_MESSAGES = list(Path('/home/s2075466/ais/decoded/jan2020_to_jun2022/msgtype5').glob("2021*.csv"))

if len(DYNAMIC_MESSAGES) != len(STATIC_MESSAGES):

    print(
        "Number of dynamic and static messages do not match."
        f"Dynamic: {len(DYNAMIC_MESSAGES)}, static: {len(STATIC_MESSAGES)}\n"
        "Processing only common files."
    )
    # Find the difference
    d = set([f.stem for f in DYNAMIC_MESSAGES])
    s = set([f.stem for f in STATIC_MESSAGES])
    
    # Find all files that are in d and s
    common = d.intersection(s)
    common = list(common)
    
    # Remove all files that are not in common
    DYNAMIC_MESSAGES = [f for f in DYNAMIC_MESSAGES if f.stem in common]
    STATIC_MESSAGES = [f for f in STATIC_MESSAGES if f.stem in common]
    
# Sort the files by date
DYNAMIC_MESSAGES = sorted(DYNAMIC_MESSAGES, key=_date_transformer)
STATIC_MESSAGES = sorted(STATIC_MESSAGES, key=_date_transformer)

SEARCHAREA = NorthSea
SA = SearchAgent(
        msg12318file=DYNAMIC_MESSAGES,
        frame=SEARCHAREA,
        msg5file=STATIC_MESSAGES
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

plot_speed_histogram(SA, savename="/home/s2075466/aisplanner/results/speed_hist_21.pdf")