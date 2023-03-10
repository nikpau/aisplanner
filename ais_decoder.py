"""
This module defines some structures and functions 
to decode, sort, and export AIS Messages.

The structures are specific to the EMSA dataset
being analyzed with it. 
"""
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from os import PathLike
import os
from typing import Callable
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import pyais as ais

load_dotenv()

class StructuralError(Exception):
    pass

# Message type tuples
_STATIC_TYPES = (5,)
_DYNAMIC_TYPES = (1,2,3,18,)

_TIMEFORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Delcare a float replacing any 
# `not available` values.
_FLOAT_NA = -999.

# Sentinel type for parameters that have not been
# transmitted or are out out of bounds.
# Some values have "NOT_AVAILABLE" as their default.
# In this module we do not distinguish between defaults
# and not available types.
class _NOT_AVAILABLE_TYPE:
    pass
NOT_AVAILABLE = _NOT_AVAILABLE_TYPE()

# File column descriptors --------------------------------------------
class DynamicReport(int, Enum):
    """
    Column descriptor class 
    for assigning column indices of
    position report messages (types 1,2,3,18)
    to human-understandable names
    """
    timestamp   = 0
    message_id  = 1
    latitude    = 2
    longitude   = 3
    raw_message = 4
    MMSI        = 5
    originator  = 6

class StaticReport(int, Enum):
    """
    Column descriptor class 
    for assigning column indices of
    static/voyage related 
    messages (type 5) to 
    human-understandable names
    """    
    timestamp    = 0
    message_id   = 1
    latitude     = 2
    longitude    = 3
    raw_message1 = 4
    raw_message2 = 5
    MMSI         = 6
    originator   = 7

class TurningRate(int):
    """
    Turning rate decoder from AIS value

    Output: Turning Rate [deg/s]

    Turn rate is encoded as follows:
        0 = not turning
        1…126 = turning right at up to 708 degrees per minute or higher
        1…126 = turning left at up to 708 degrees per minute or higher
        127 = turning right at more than 5deg/30s (No TI available)
        -127 = turning left at more than 5deg/30s (No TI available)
        128 (80 hex) indicates no turn information available (default)

    Values between 0 and 708 degrees/min coded by ROTAIS=4.733 * SQRT(ROTsensor) 
    degrees/min where ROTsensor is the Rate of Turn as input by an external 
    Rate of Turn Indicator. ROTAIS is rounded to the nearest integer value. 
    Thus, to decode the field value, divide by 4.733 and then square it. 
    The resulting value is divided by 60 to return a rotation rate per second
    Sign of the field value should be preserved when squaring it, otherwise 
    the left/right indication will be lost.

    """
    def __new__(cls,__x):
        _ROTCONST = 4.733
        try:
            __x = int(__x)
            sign = -1 if __x < 0 else 1
            if -128 < __x < 128:
                return float(sign * (__x/_ROTCONST)**2 / 60)
            else: return float(_FLOAT_NA)
        except Exception as e:
            raise e

# Tuple of attrs we want to extract from the decoded ais message
# Order is the same as the bits appear in the original message.
_FIELDS_MSG12318 = (
    "repeat",
    "status", 
    "turn", 
    "speed",
    "accuracy",
    "lon",
    "lat",
    "course", 
    "heading",
    "second",
    "maneuver",
    "radio",
    )

_FIELDS_MSG5 = (
    "repeat",
    "ais_version",# 0=[ITU1371], 1-3 = future editions
    "imo",
    "callsign",
    "shipname",
    "ship_type",
    "to_bow",
    "to_stern",
    "to_port",
    "to_starboard",
    "epfd", # Electronic Position Fixing Device (GPS,GLOSSNASS,...)
    "month", # ETA month (UTC)
    "day", # ETA day (UTC)
    "hour", # ETA hour (UTC)
    "minute", # ETA minute (UTC)
    "draught", # Draft
    "destination",
    "dte", # 0=Data terminal ready, 1=Not ready (default).
)

# Decoding plan
def _decode_dynamic_messages(df: pd.DataFrame) -> list[ais.ANY_MESSAGE]:
    """
    Decode AIS messages of types 1,2,3,18 
    supplied as a pandas Series object.
    """
    messages = df[DynamicReport.raw_message.name]
    # Split at exclamation mark and take the last part
    raw = messages.str.split("!",expand=True).iloc[:,-1:]
    # Since we split on the exclamation mark we need to
    # re-add it at the front of the message
    raw = "!" + raw
    return [ais.decode(val) for val in raw.values.ravel()]

def _decode_static_messages(df: pd.DataFrame):
    """
    Decode AIS messages of type 5 
    supplied as a pandas DataFrame.
    """
    msg1 = df[StaticReport.raw_message1.name]
    msg2 = df[StaticReport.raw_message2.name]
    raw1 = msg1.str.split("!",expand=True).iloc[:,-1:]
    raw2 = msg2.str.split("!",expand=True).iloc[:,-1:]
    raw1, raw2 = "!" + raw1, "!" + raw2
    return [ais.decode(*vals) for vals in 
            zip(raw1.values.ravel(),raw2.values.ravel())]


def _extract_fields(messages: list[ais.ANY_MESSAGE],
                    fields: tuple) -> dict[str,np.ndarray]:
    out = np.empty((len(messages),len(fields)))
    for i, msg in enumerate(messages):
        out[i] = [getattr(msg,field,_FLOAT_NA) for field in fields]
    return dict(zip(fields,out.T))

def _get_decoder(
        dataframe: pd.DataFrame
    ) -> tuple[Callable[[pd.DataFrame],list[ais.ANY_MESSAGE]],dict]:
    """
    Returns a message-specific decoding function
    based on message types present in the dataframe.

    Note: Since input dataframes will be processed 
    at once, they can only contain either position 
    record messages (types 1,2,3,18), or static messages
    (type 5) but not both.
    """
    # The "message_id" field is the same across all files,
    # therefore I randomly chose to extract it via the 
    # PositionReport Enum
    types = dataframe[DynamicReport.message_id.name]
    if all(k in dataframe for k in (StaticReport.raw_message1.name, 
        StaticReport.raw_message2.name)):
        # Maybe type 5 
        if all(b in _STATIC_TYPES for b in types.unique()):
            return _decode_static_messages, _FIELDS_MSG5
        else: raise StructuralError(
                "Assumed type-5-only dataframe, but found "
                f"messages of types {types.unique()}"
        )
    elif all(b in _DYNAMIC_TYPES for b in types.unique()):
        return _decode_dynamic_messages, _FIELDS_MSG12318
    else: raise StructuralError(
            "Found not processable combination "
            "of message types. Need either type-5-only dataframe "
            "or type-1-2-3-18-only dataframe. Found messages "
            f"of types {types.unique()}"
    )
    
# Pandas file operations 
#
# The pipeline will first read-in the csv-file
# as a pandas dataframe, decode the raw AIS
# message, and save the extracted information 
# as new columns in the existing file. 
def process(datafile: str | PathLike[str]) -> None:  
    df  = pd.read_csv(datafile,sep=",",quotechar='"',
                      names=[e.name for e in DynamicReport],
                      encoding="utf-8",header=1)
    decoder, fields = _get_decoder(df)
    decoded = decoder(df)
    df = df.assign(**_extract_fields(decoded,fields))
    df.to_csv(datafile)
    return

# MAIN --------------------------------------------------------------------

from pathlib import Path

SOURCE = Path(os.environ("AISSOURCE"))
DEST = Path(os.environ("DECODEDDEST"))

for file in SOURCE.rglob("*.csv"):
    process(
        file,
        f"{DEST.as_posix()}/{'/'.join(file.parts[len(SOURCE.parts):])}"
    )
