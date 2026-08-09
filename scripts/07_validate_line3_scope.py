import os
import re

import pandas as pd

DATE_FILE = "data/raw/train_date.csv"
LINE3_FILE = "data/processed/bosch_line3_processed.csv"
OUTPUT_FILE = "data/processed/line3_scope_validation.csv"

# Keep this aligned with the current development sample.
SAMPLE_SIZE = 10_000

print("Loading processed Line 3 events...")

line3 = pd.read_csv(LINE3_FILE)

# Find each part's final observed Line 3 event.
last_line3 = (
    line3.sort_values(["Id", "Timestamp", "Station"])
    .groupby("Id")
    .tail(1)
    [["Id", "Station", "Timestamp"]]
    .rename(
        columns={
            "Station": "LastLine3Station",
            "Timestamp": "LastLine3Timestamp",
        }
    )
    .reset_index(drop=True)
)

print(f"Parts with Line 3 activity: {len(last_line3):,}")

# Read only the timestamp column names first.
all_columns = pd.read_csv(
    DATE_FILE,
    nrows=0
).columns.tolist()

other_line_columns = [
    column
    for column in all_columns
    if column == "Id"
    or re.match(r"L[012]_S\d+_D\d+", column)
]

print(
    f"Reading Id plus {len(other_line_columns) - 1:,} "
    "timestamp columns from Lines 0, 1, and 2..."
)

# Read the same 10,000-row development sample used by the Line 3 pipeline.
other_lines = pd.read_csv(
    DATE_FILE,
    usecols=other_line_columns,
    nrows=SAMPLE_SIZE
)

# Calculate earliest and latest observed timestamp outside Line 3.
timestamp_columns = [
    column
    for column in other_lines.columns
    if column != "Id"
]

other_lines["EarliestOtherLineTimestamp"] = (
    other_lines[timestamp_columns].min(axis=1, skipna=True)
)

other_lines["LatestOtherLineTimestamp"] = (
    other_lines[timestamp_columns].max(axis=1, skipna=True)
)

other_lines["HasOtherLineActivity"] = (
    other_lines[timestamp_columns].notna().any(axis=1)
)

other_summary = other_lines[
    [
        "Id",
        "HasOtherLineActivity",
        "EarliestOtherLineTimestamp",
        "LatestOtherLineTimestamp",
    ]
]

validation = last_line3.merge(
    other_summary,
    on="Id",
    how="left",
    validate="one_to_one"
)

# Compare temporal ordering.
validation["OtherLineActivityAfterLine3"] = (
    validation["HasOtherLineActivity"]
    & (
        validation["LatestOtherLineTimestamp"]
        > validation["LastLine3Timestamp"]
    )
)

validation["OtherLineActivityOnlyBeforeOrAtLine3"] = (
    validation["HasOtherLineActivity"]
    & ~validation["OtherLineActivityAfterLine3"]
)

validation["NoOtherLineActivity"] = (
    ~validation["HasOtherLineActivity"]
)

# Useful time difference when activity continues later.
validation["TimeFromLastLine3ToLatestOtherLine"] = (
    validation["LatestOtherLineTimestamp"]
    - validation["LastLine3Timestamp"]
)

# Label each part clearly.
validation["ScopeCategory"] = "No activity outside Line 3"

validation.loc[
    validation["OtherLineActivityOnlyBeforeOrAtLine3"],
    "ScopeCategory",
] = "Other-line activity only before or at final Line 3 event"

validation.loc[
    validation["OtherLineActivityAfterLine3"],
    "ScopeCategory",
] = "Other-line activity after final Line 3 event"

os.makedirs("data/processed", exist_ok=True)

validation.to_csv(
    OUTPUT_FILE,
    index=False
)

# Overall summary.
summary = (
    validation["ScopeCategory"]
    .value_counts()
    .rename_axis("ScopeCategory")
    .reset_index(name="Parts")
)

summary["Percent"] = (
    summary["Parts"]
    / len(validation)
    * 100
)

print("\nLine 3 scope validation")
print("-----------------------")
print(summary.to_string(index=False))

# Specific check for parts ending at Station 37 within Line 3.
station37 = validation[
    validation["LastLine3Station"] == 37
].copy()

station37_summary = (
    station37["ScopeCategory"]
    .value_counts()
    .rename_axis("ScopeCategory")
    .reset_index(name="Parts")
)

station37_summary["Percent"] = (
    station37_summary["Parts"]
    / len(station37)
    * 100
)

print("\nParts whose last observed Line 3 station is 37")
print("------------------------------------------------")
print(f"Parts analyzed: {len(station37):,}")
print(station37_summary.to_string(index=False))

print("\nExamples with activity after Line 3:")
examples = validation[
    validation["OtherLineActivityAfterLine3"]
][
    [
        "Id",
        "LastLine3Station",
        "LastLine3Timestamp",
        "LatestOtherLineTimestamp",
        "TimeFromLastLine3ToLatestOtherLine",
    ]
].head(15)

if examples.empty:
    print("None found.")
else:
    print(examples.to_string(index=False))

print(f"\nSaved validation file to:")
print(OUTPUT_FILE)