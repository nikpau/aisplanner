from pathlib import Path

HEADER = "timestamp,message_id,latitude,longitude,raw_message1,raw_message2,MMSI,originator"

# Hard coded file paths
SOURCE = Path("/warm_archive/ws/s2075466-ais/curated/jan2020_to_jun2022/msgtype5")
DEST = Path("/lustre/ssd/ws/s2075466-ais-temp/msgtype5")

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