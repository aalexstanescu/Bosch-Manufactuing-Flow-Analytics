import os

import pandas as pd

INPUT_FILE = "data/processed/full/bosch_line3_processed.csv"
OUTPUT_FILE = "data/processed/full/bosch_line3_flow_metrics.csv"

print("Loading processed Line 3 events...")

df = pd.read_csv(INPUT_FILE)

# Put every part's station events in chronological order.
df = df.sort_values(
    ["Id", "Timestamp", "Station"]
).reset_index(drop=True)

# Create the next event within each part's route.
df["NextStation"] = (
    df.groupby("Id")["Station"]
    .shift(-1)
)

df["NextTimestamp"] = (
    df.groupby("Id")["Timestamp"]
    .shift(-1)
)

# Difference between the current event and the next event.
df["TransitTime"] = (
    df["NextTimestamp"] - df["Timestamp"]
)

# Position of the station within each part's observed route.
df["RouteOrder"] = (
    df.groupby("Id")
    .cumcount()
    .add(1)
)

# Total number of observed Line 3 stations visited by each part.
df["StationsVisited"] = (
    df.groupby("Id")["Station"]
    .transform("count")
)

# First and last timestamps for each part on Line 3.
df["Line3StartTime"] = (
    df.groupby("Id")["Timestamp"]
    .transform("min")
)

df["Line3EndTime"] = (
    df.groupby("Id")["Timestamp"]
    .transform("max")
)

df["TotalLine3FlowTime"] = (
    df["Line3EndTime"] - df["Line3StartTime"]
)

# The final station in a route has no next station or transit time.
transition_df = df.dropna(
    subset=["NextStation", "TransitTime"]
).copy()

transition_df["NextStation"] = (
    transition_df["NextStation"]
    .astype("int16")
)

# Basic validation.
negative_transit = (
    transition_df["TransitTime"] < 0
).sum()

zero_transit = (
    transition_df["TransitTime"] == 0
).sum()

print("\nValidation summary")
print("------------------")
print(f"Event rows loaded: {len(df):,}")
print(f"Transition rows created: {len(transition_df):,}")
print(f"Negative transit times: {negative_transit:,}")
print(f"Zero transit times: {zero_transit:,}")
print(
    "Positive transit times: "
    f"{(transition_df['TransitTime'] > 0).sum():,}"
)

print("\nFirst 15 transitions:")
print(
    transition_df[
        [
            "Id",
            "Station",
            "NextStation",
            "Timestamp",
            "NextTimestamp",
            "TransitTime",
            "RouteOrder",
            "StationsVisited",
            "TotalLine3FlowTime",
            "Response",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

os.makedirs("data/processed", exist_ok=True)

transition_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved flow metrics to:")
print(OUTPUT_FILE)
