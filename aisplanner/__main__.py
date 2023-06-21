"""
Argument parser for the command line interface.
"""
import argparse
import sys
from aisplanner import __version__
from aisplanner import __author__
from aisplanner.encounters.main import search_for
from pytsa import ShipType

def parse_args():
    parser = argparse.ArgumentParser(
        description="AISplanner: A tool for analyzing AIS data.",)
    parser.add_argument("-t", "--type,", type=str, help="Ship type to search for.")
    parser.add_argument("-v", "--version", action="version", version=f"{__version__}")
    parser.add_argument("-a", "--author", action="version", version=f"{__author__}")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.type:
        try:
            ship_type = ShipType[f"{args.type}"]
        except ValueError:
            print(f"Invalid ship type: {args.type}")
            sys.exit(1)
        search_for(ship_type)
    else:
        print("No ship type given. Use -h for help.")
        sys.exit(1)

if __name__ == "__main__":
    main()
