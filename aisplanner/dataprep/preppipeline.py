from pathlib import Path
import multiprocessing as mp

import numpy as np

from aisplanner.dataprep.ais_decoder import decode_from_file

HEADER = "timestamp,message_id,latitude,longitude,raw_message1,raw_message2,MMSI,originator"

def remove_null_bytes(old: Path, new: Path) -> str:
    """
    Remove all null bytes contained in
    files found at "SOURCE" and save the files
    at "DEST". Folder strucutre is preserved. 
    """
    print(f"Removing null bytes...")
    with open(old,"rb") as file:
        data = file.read()
    with open(new,"wb") as newfile:
        newfile.write(data.replace(b"\x00",b""))

def prepare_msg5(source: str, dest: str) -> None:
    """
    1. Step:
        - Raw messages in every odd row start
          with a double quote that is not terminated.
        - On every even row, the messages do not start
          with a double quote but are terminated with one.
    2. Step: 
        AIS messages of type 5 are split into two
        separate lines, because type 5 messages are built from 
        two different fragments. This step appends row 2n to 
        row (2n-1), for n∈𝕫.
    """
    with open(source,"rb") as file:
        
        s = file.read().splitlines()
        del s[0] # remove header

        # 1. Step
        print("Merging raw messages...")
        # Odd lines are the 1st, 3rd, 5th...
        # Due to zero indexing, they become even, however.
        odd_lines = [s[i].decode() for i in range(len(s)) if i%2 == 0 ]
        odd_lines = [line + '"' for line in odd_lines]

        even_lines = [s[i].decode() for i in range(len(s)) if i%2 != 0]
        even_lines = ['"' + line for line in even_lines]

        # 2. Step
        final = [f"{odd},{even}" for odd,even in zip(odd_lines,even_lines)]
        final.insert(0,HEADER)
    
    print("Writing to destination...")
    with open(dest,"w") as dest:
        for line in final:
            dest.write(f"{line}\n")
    

if __name__ == "__main__":
    # SOURCE = Path("/data/walrus/ws/s2075466-aisdata/raw/")
    # DEST = Path("/data/horse/ws/s2075466-aistemp/curated/")
    
    # # Remove null bytes from all files
    # files = list(SOURCE.rglob("*.csv"))
    # for file in files:
    #     print(f"Processing {file}")
    #     remove_null_bytes(
    #         file,
    #         f"{DEST.as_posix()}/{'/'.join(file.parts[len(SOURCE.parts):])}"
    #     )
    
    # Prepare msgtype5 files  
    SOURCE = Path("/data/horse/ws/s2075466-aistemp/curated/msgtype5")
    DEST = Path("/data/horse/ws/s2075466-aistemp/curated/msgtype5")
    
    files = list(SOURCE.rglob("*.csv")) 
    for file in files:
        print(f"Processing {file}")
        prepare_msg5(
            file,
            f"{DEST.as_posix()}/{'/'.join(file.parts[len(SOURCE.parts):])}"
        )
    
    # Decode all files 
    SOURCE = Path("/data/horse/ws/s2075466-aistemp/curated/")
    DEST = Path("/data/horse/ws/s2075466-aistemp/decoded/")
    
    files = list(SOURCE.rglob("*.csv"))
    # Split files into 16 batches
    # to be processed in parallel
    batches = np.array_split(files,16)

    with mp.Pool(processes=16) as pool:
        pool.starmap(
            decode_from_file, 
            [(file, f"{DEST.as_posix()}/{'/'.join(file.parts[len(SOURCE.parts):])}")
                    for file in files])