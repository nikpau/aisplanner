
import os
from aisplanner.encounters import (
    EncounterSituations, Ship, Position, ENCSearchAgent,
    dtr, crv, true_bearing, rel_dist, relative_velocity, vel_from_xy,
    DCPA, TCPA
)
from aisplanner.encounters._locdb import LocationDatabase
import matplotlib.pyplot as plt
import numpy as np
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Initialize search agent
s = ENCSearchAgent(
    remote_host="taurus.hrsk.tu-dresden.de",
    remote_dir=os.environ["AISDECODED"],
    search_areas=LocationDatabase.all(),
)
s.search()
s.save_results("~/TUD/aisplanner/results/results.pkl")

exit()


OS = Ship(Position(0,0),1,dtr(45))
TS = Ship(Position(2,2),1,dtr(210))
ES = EncounterSituations(OS,TS).analyze()
print(ES)
v_rel = relative_velocity(OS.cog,OS.sog,TS.cog,TS.sog)
print("DCPA: ",
    DCPA(
        rel_dist(OS.pos,TS.pos), 
        crv(*v_rel), 
        true_bearing(OS.pos,TS.pos),
    )
)
print("TCPA: ",
    TCPA(
        rel_dist(OS.pos,TS.pos),
        crv(*v_rel),
        true_bearing(OS.pos,TS.pos),
        vel_from_xy(*v_rel)
    )
)

f,ax = plt.subplots()

# Draw circles of sensor range
ax.add_artist(plt.Circle((OS.pos.x,OS.pos.y),EncounterSituations.D1,color='r',fill=False))
ax.add_artist(plt.Circle((OS.pos.x,OS.pos.y),EncounterSituations.D2,color='g',fill=False))
ax.add_artist(plt.Circle((OS.pos.x,OS.pos.y),EncounterSituations.D3,color='y',fill=False))

ax.plot([OS.pos.x,TS.pos.x],[OS.pos.y,TS.pos.y])
# Plot arrows for own and target ship
ax.arrow(
    OS.pos.x,OS.pos.y,
    np.sin(OS.cog),np.cos(OS.cog),
    width=0.01,
    color='r'
)
ax.arrow(
    TS.pos.x,TS.pos.y,
    np.sin(TS.cog),np.cos(TS.cog),
    width=0.01,
    color='b'
)
# Plot relative velocity
plt.arrow(
    OS.pos.x,OS.pos.y,
    v_rel[0],v_rel[1],
    width=0.01,
    color='g'
)
plt.xlim(-0.5,6)
plt.ylim(-0.5,6)
plt.show()
