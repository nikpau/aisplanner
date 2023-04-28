"""
This module defines some structures and functions 
to decode, sort, and export AIS Messages.

The structures are specific to the EMSA dataset
being analyzed with it. 
"""
from os import PathLike
import os
from typing import Callable, List, Tuple, Dict
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import pyais as ais
from ._file_descriptors import (
    StaticReport, DynamicReport, 
    AISFile, _FIELDS_MSG12318, _FIELDS_MSG5
)

load_dotenv()

class StructuralError(Exception):
    pass

# Message type tuples
_STATIC_TYPES = (5,)
_DYNAMIC_TYPES = (1,2,3,18,)

# Default value for missing data
_NA = "NA"

# Decoding plan
def _decode_dynamic_messages(df: pd.DataFrame) -> List[ais.ANY_MESSAGE]:
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


def _extract_fields(messages: List[ais.ANY_MESSAGE],
                    fields: tuple) -> Dict[str,np.ndarray]:
    out = np.empty((len(messages),len(fields)),dtype=object)
    for i, msg in enumerate(messages):
        out[i] = [getattr(msg,field,_NA) for field in fields]
    return dict(zip(fields,out.T))

def _get_decoder(
        dataframe: pd.DataFrame
    ) -> Tuple[Callable[[pd.DataFrame],List[ais.ANY_MESSAGE]],dict]:
    """
    Returns a message-specific decoding function
    based on message types present in the dataframe.

    Note: Since input dataframes will be processed 
    at once, they can only contain either position 
    record messages (types 1,2,3,18), or static messages
    (type 5) but not both.
    """
    types = dataframe[AISFile.message_id.name]
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
def decode_from_file(source: str | PathLike[str],
            dest: str | PathLike[str]) -> None:  
    df  = pd.read_csv(source,sep=",",quotechar='"',
                      encoding="utf-8",index_col=False)
    df = df.drop(df[df[DynamicReport.raw_message.name].str.contains(r"\n")].index)
    decoder, fields = _get_decoder(df)
    decoded = decoder(df)
    df["DECODE_START"] = "||"
    df = df.assign(**_extract_fields(decoded,fields))
    df.to_csv(dest)
    return

# MAIN --------------------------------------------------------------------

from pathlib import Path

SOURCE = Path(os.environ["AISSOURCE"])
DEST = Path(os.environ["DECODEDDEST"])

for file in SOURCE.rglob("*.csv"):
    decode_from_file(
        file,
        f"{DEST.as_posix()}/{'/'.join(file.parts[len(SOURCE.parts):])}"
    )
