from enum import Enum

class AISFile(int,Enum):
    """
    Column descriptor class
    for fields present in all
    AIS files.
    """
    timestamp = 0
    message_id = 1

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

class DecodedReport(int, Enum):
    """
    Column descriptor class for assigning column indices of
    decoded position report messages (types 1,2,3,18) to
    human-understandable names.
    """
    timestamp = 0 # TODO: this may be wrong
    MMSI      = 5
    status    = 10
    turn      = 11
    speed     = 12
    lon       = 14
    lat       = 15
    course    = 16
    originator= 6

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

class Fields12318(Enum):
    """
    Column descriptor class for assigning column indices of
    decoded position report messages (types 1,2,3,18) to
    human-understandable names.
    """
    repeat      = 0
    status      = 1
    turn        = 2
    speed       = 3
    accuracy    = 4
    lon         = 5
    lat         = 6
    course      = 7
    heading     = 8
    second      = 9
    maneuver    = 10
    radio       = 11
    MMSI        = 12
    
class Fields5(Enum):
    """
    Column descriptor class for assigning column indices of
    decoded static/voyage related messages (type 5) to
    human-understandable names.
    """
    repeat      = 0
    ais_version = 1
    imo         = 2
    callsign    = 3
    shipname    = 4
    ship_type   = 5
    to_bow      = 6
    to_stern    = 7
    to_port     = 8
    to_starboard= 9
    epfd        = 10
    month       = 11
    day         = 12
    hour        = 13
    minute      = 14
    draught     = 15
    destination = 16
    dte         = 17
    MMSI        = 18