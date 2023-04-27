"""
Remove all null bytes contained in
files found at "SOURCE" and save the files
at "DEST". Folder strucutre is preserved. 
"""
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

SOURCE = Path(os.environ["AISRAW"])
DEST = Path(os.environ["TEMDEST"])

def remove_null_bytes(
        old: str | Path[str], new: str | Path[str]) -> str:
    """
    
    """
    with open(old,"rb") as file:
        data = file.read()
    with open(new,"wb") as newfile:
        newfile.write(data.replace(b"\x00",b""))

for file in SOURCE.rglob("*.csv"):
    remove_null_bytes(
        file,
        f"{DEST.as_posix()}/{'/'.join(file.parts[len(SOURCE.parts):])}"
    )