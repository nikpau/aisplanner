"""
AIS messages of type 5 are split into two
separate lines in our data set. 
This is because type 5 messages are built from 
two different fragments which need to be combined 
in a later stage of processing.

Therefore, this script appends every 2n-th row (n∈𝕫)to the
(2n-1)-th row. 
Put differently, the second row gets appended to the first, 
the 4th to the 3rd, 6th to the 5th and so on...

Headers are ignored and later redefined.

There are also some special properties to this data set
which also need to be addressed:
    - Raw messages in every (2n-1)-th line start
      with a single quote that is not terminated
    - On every 2n-th line, the messages are terminated 
      with a single quote but the starting quote is missing.
      
They are therefore added after reading the input file
"""
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

HEADER = "timestamp,message_id,latitude,longitude,raw_message1,raw_message2,MMSI,originator"

# Hard coded file paths
SOURCE = Path(os.environ["MSG5SOURCE"])
DEST = Path(os.environ["TEMDEST"])

def merge_mm5(source: Path[str], dest: Path[str]) -> None:
    
    with open(source, "r") as src:
        s = src.read().splitlines()
    del s[0] # remove header
    
    # Odd lines are the 1st, 3rd, 5th...
    # Due to zero indexing, they become even, however.
    odd_lines = [s[i] for i in range(len(s)) if i%2 == 0 ]
    odd_lines = [line + '"' for line in odd_lines]
    
    even_lines = [s[i] for i in range(len(s)) if i%2 != 0]
    even_lines = ['"' + line for line in even_lines]
    
    final = [f"{odd},{even}" for odd,even in zip(odd_lines,even_lines)]
    final.insert(0,HEADER)
    
    with open(dest,"w") as dest:
        for line in final:
            dest.write(f"{line}\n")

files = SOURCE.rglob("*.csv")
for file in files:
    merge_mm5(
        file,
        f"{DEST.as_posix()}/{'/'.join(file.parts[len(SOURCE.parts)-1:])}"
    )