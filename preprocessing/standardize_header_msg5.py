from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

HEADER = "timestamp,message_id,latitude,longitude,raw_message1,raw_message2,MMSI,originator"

# Hard coded file paths
SOURCE = Path(os.environ["MSG5SOURCE"])
DEST = Path(os.environ["TEMPDEST"])

def _replace_header(source: Path[str], dest: Path[str]) -> None:

    with open(source, "r") as src:
        s = src.read().splitlines()
    del s[0] # remove header
    
    s.insert(0,HEADER) # Replace header
    
    with open(dest,"w") as dest:
        for line in s:
            dest.write(f"{line}\n")

files = SOURCE.rglob("*.csv")
for file in files:
    _replace_header(
        file,
        f"{DEST.as_posix()}/{'/'.join(file.parts[len(SOURCE.parts):])}"
    )