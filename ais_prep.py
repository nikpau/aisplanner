import pandas as pd

raw_dynamic = pd.read_csv(
    "data/STIRES archive till 30.06.2020/ais-200101_Dynamic data.csv",
    sep=",",
    usecols=list(range(10))
    )

# Filter AIS messages to only use position report types
raw_dynamic = raw_dynamic[raw_dynamic.AISType.isin([1,2,3,18,19])]

# Filter out all rows wih no available SOG and COG
raw_dynamic = raw_dynamic[raw_dynamic[["SOG","COG"]].notnull().all(1)]

# Remove duplicates based on Lateral position to account 
# for multiple AIS receivers reporting the same vessel
raw_dynamic = raw_dynamic.drop_duplicates(subset=["Timestamp","MMSI"], keep="first")

# Convert timestamp to date objects
raw_dynamic["Timestamp"] = pd.to_datetime(raw_dynamic["Timestamp"])

# Sort by ascending date
raw_dynamic = raw_dynamic.sort_values(by="Timestamp")

# Save as curated copy
raw_dynamic.to_csv("data/curated/dynamic_time_sorted.csv")