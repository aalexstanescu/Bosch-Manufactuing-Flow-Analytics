import os
import re

import pandas as pd

print("Starting Line 3 preprocessing...")

# -------------------------
# SETTINGS
# -------------------------

SAMPLE_SIZE = 10_000

DATE_FILE = "data/raw/train_date.csv"
NUMERIC_FILE = "data/raw/train_numeric.csv"
OUTPUT_FILE = "data/processed/bosch_line3_processed.csv"

# -------------------------
# FIND LINE 3 COLUMNS
# -------------------------

all_date_columns = pd.read_csv(
    DATE_FILE,
    nrows=0
).columns.tolist()

line3_columns = [
    column
    for column in all_date_columns
    if column == "Id" or column.startswith("L3_")
]

print(f"Keeping {len(line3_columns):,} date columns")

# -------------------------
# LOAD DATE DATA
# -------------------------

date_df = pd.read_csv(
    DATE_FILE,
    usecols=line3_columns,
    nrows=SAMPLE_SIZE
)

print(f"Loaded {len(date_df):,} date rows")

# -------------------------
# LOAD QUALITY RESPONSE
# -------------------------

response_df = pd.read_csv(
    NUMERIC_FILE,
    usecols=["Id", "Response"],
    nrows=SAMPLE_SIZE
)

print(f"Loaded {len(response_df):,} response rows")

# -------------------------
# JOIN BY PART ID
# -------------------------

merged_df = date_df.merge(
    response_df,
    on="Id",
    how="left",
    validate="one_to_one"
)

print(f"Joined dataset: {merged_df.shape}")

# -------------------------
# WIDE TO LONG FORMAT
# -------------------------

long_df = merged_df.melt(
    id_vars=["Id", "Response"],
    var_name="TimestampColumn",
    value_name="Timestamp"
)

# Remove missing station events
long_df = long_df.dropna(subset=["Timestamp"]).copy()

# Extract station number from names such as L3_S29_D3316
long_df["Station"] = (
    long_df["TimestampColumn"]
    .str.extract(r"L3_S(\d+)_D\d+", expand=False)
    .astype("int16")
)

# Keep the requested final columns
long_df = long_df[
    ["Id", "Station", "Timestamp", "Response"]
]

# One part may have many date features at the same station with the same
# timestamp. Collapse those duplicates into one part-station event.
long_df = (
    long_df.drop_duplicates(
        subset=["Id", "Station", "Timestamp", "Response"]
    )
    .sort_values(["Id", "Timestamp", "Station"])
    .reset_index(drop=True)
)

# -------------------------
# VALIDATION
# -------------------------

print(f"Long-format rows: {len(long_df):,}")
print(f"Unique parts: {long_df['Id'].nunique():,}")
print(f"Unique stations: {long_df['Station'].nunique():,}")
print(f"Missing responses: {long_df['Response'].isna().sum():,}")

print("\nFirst 10 processed rows:")
print(long_df.head(10).to_string(index=False))

# -------------------------
# SAVE OUTPUT
# -------------------------

os.makedirs("data/processed", exist_ok=True)

long_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved successfully to:")
print(OUTPUT_FILE)