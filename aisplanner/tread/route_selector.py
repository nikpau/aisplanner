from itertools import cycle
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
import geopandas as gpd
from pathlib import Path
from aisplanner.tread._tread import Route, AISMessage, ClusterType, Location

# Path to the directory containing the geometry files
GEOMPATH = Path("data/geometry")

# Locations to search for
LOCATIONS = {
    "Hamburg": Location(53.5417,9.9024),
    "Kiel": Location(54.364428,10.171100)
}

def load_routes(path: Path) -> dict[str,Route]:
    """Load routes from a file"""
    with open(path, "rb") as p_obj:
        routes: dict[str,Route] = pickle.load(p_obj)
    return routes

def decompose(route: Route) -> dict[ClusterType,list[AISMessage]]:
    """
    Sort the messages of a route by 
    their cluster type
    """
    out = {
        ClusterType.entry: [],
        ClusterType.exit: [],
        ClusterType.stationary: [],
        ClusterType.not_assigned: []
    }
    for message in route.elements:
        out[message.cluster_type].append(message)
    return out


def set_endpoints(decomp: dict[ClusterType,list[AISMessage]]) -> Route:
    """Set the coorinates for route start and route end.
    Repack back to a route object"""
    new = Route()

    # Check if the route either starts at a 
    # stationary or an entry point.
    if not decomp[ClusterType.entry]:
        startpoint = ClusterType.stationary
    else: startpoint = ClusterType.entry

    en_latlon = np.array([[m.lat,m.lon] for m in decomp[startpoint]])
    new.avg_start = Location(
        lat=np.mean(en_latlon.T[0]),
        lon=np.mean(en_latlon.T[1])
    )

    # Check if the route either ends at a 
    # stationary or an exit point.
    if not decomp[ClusterType.stationary]:
        endpoint = ClusterType.exit
    else: endpoint = ClusterType.stationary

    ex_latlon = np.array([[m.lat,m.lon] for m in decomp[endpoint]])
    new.avg_end = Location(
        lat=np.mean(ex_latlon.T[0]),
        lon=np.mean(ex_latlon.T[1])
        )

    # Add all messages back to the route
    for v in decomp.values():
        new.elements.extend(v)
    return new

def locate_routes(
        routes: dict[str,Route], 
        location: str, eps: float = 0.2) -> list[Route]:
    """
    Find route(s) starting or ending not more than `eps` degrees
    away from provided location.
    """
    assert location in LOCATIONS.keys(), f"Location {location} not found"
    l = LOCATIONS[location]
    found = []
    for route in list(routes.values()):
        r = set_endpoints(decompose(route))
        found.append(r)

    return found

def located_to_file(file: Path[str],location: str, destination: Path[str], plot: bool = False):
    """Saves located routes starting or ending at `location`
    to files"""

    assert location in LOCATIONS.keys(), f"Location {location} not found"
    l = LOCATIONS[location]

    routes = load_routes(file)
    found = locate_routes(routes,l)

    # Sort the routes by length of its elements
    found = sorted(found,key=lambda x: len(x.elements))[::-1]

    for route in found:
        route.elements = sorted(route.elements, key=lambda x: x.timestamp)
        mmsis = np.unique([m.sender for m in route.elements])
        for mmsi in mmsis:
            coords = np.array(
                [[m.lat,m.lon] for m in route.elements if m.sender==mmsi]
            )
            l1la,l1lo = coords[0]
            l2la,l2lo = coords[-1]

            fname = destination.joinpath(
                f"route-[{mmsi}]-[{l1la:.2f}-{l1lo:.2f}]-[{l2la:.2f}-{l2lo:.2f}]"
                )
            print(f"Saving {fname}")
            np.savetxt(
                fname,coords,fmt="%8f",
                delimiter=",",header="lat,lon",comments=""
            )
    if plot:
        plot_found(found,location)
    return

def plot_found(routes: dict[str,Route], location: str):
    # Plot coastline geometry
    f, ax = plt.subplots()

    def lonlat_from_route(route: Route) -> tuple[list[float],list[float]]:
        lat, lon = [], []
        for message in route.elements:
            lat.append(message.lat)
            lon.append(message.lon)
        return lon, lat

    for route in routes.values():
        ax.scatter(*lonlat_from_route(route), marker = ".", s=1)

    files = GEOMPATH.glob("*.json")
    for file in files:
        f = gpd.read_file(file)
        f.plot(ax=ax,color="#283618",marker = ".")

    ax.set_xlim(7.482,7.75)
    ax.set_ylim(53.95,54.05)
    plt.savefig(f"out/routes_from_{location}.png",dpi = 300)


def inspect_longest(route_folder: Path[str], destination: Path[str], _length: int = 64):
    """
    Inspect the 64 longest routes in the folder as individual
    2x2 plots.
    """
    fig, axs = plt.subplots(ncols=4,nrows=1,figsize=(32,18))
    geofiles = GEOMPATH.glob("*.json")
    gs = [gpd.read_file(gf) for gf in geofiles]
    files = route_folder.glob("*.csv")
    files = [np.loadtxt(file,delimiter=",",skiprows=1) for file in files]
    loaded = sorted(files, key=lambda x: len(x))[::-1][:_length]
    i = 0
    for x in cycle(range(4)):
        if i%4 == 0 and i != 0:
            plt.tight_layout()
            plt.savefig(f"{destination}/img/routes-{i}", dpi = 300)
            [axs[n].clear() for n in range(4)]
        if i == _length - 1:
            return
        for g in gs:
            g.plot(ax=axs[x],color="#283618",markersize=0.05,marker = ".")
        axs[x].scatter(loaded[i].T[1],loaded[i].T[0],s=0.2,c="#c1121f")
        axs[x].set_title(f"Route {i}")
        i += 1
        print(i)
    return

plot_found(load_routes(Path("routes/routes_elbeapproach.pyo")), "elbeapproach")
exit()

parser = argparse.ArgumentParser()
parser.add_argument("-f","--file", type=str, help="Path to routes file", default=None)
parser.add_argument("-l","--location", type=str, help="Location to search for.", default=None)
parser.add_argument("-d","--destination", type=str, help="Path to save extracted routes.", default=None)
parser.add_argument("-p", "--plot", action="store_true", help="Plot found routes.")

args = parser.parse_args()

if any([args.file is None, args.location is None, args.destination is None]):
    parser.print_help()
    raise SystemExit(1)

if args.plot:
    plot = True
    located_to_file(Path(args.file),args.location,Path(args.destination),plot)
else:
    located_to_file(Path(args.file),args.location,Path(args.destination))