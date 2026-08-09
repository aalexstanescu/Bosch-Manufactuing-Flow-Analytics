"""
Bosch Line 3 Chunked Preprocessing
==================================

Reads train_date.csv and train_numeric.csv in matching chunks so the
full Bosch dataset can be processed without loading everything into RAM.

Output:
    data/processed/sample/bosch_line3_processed.csv
or:
    data/processed/full/bosch_line3_processed.csv
"""

from pathlib import Path
import gc
import re
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = None
# After the sample test works, change the line above to:
# SAMPLE_SIZE = None

CHUNK_SIZE = 5_000
TARGET_LINE = 3

DATE_FILE = Path("data/raw/train_date.csv")
NUMERIC_FILE = Path("data/raw/train_numeric.csv")

RUN_NAME = "sample" if SAMPLE_SIZE is not None else "full"
OUTPUT_DIR = Path("data/processed") / RUN_NAME
OUTPUT_FILE = OUTPUT_DIR / "bosch_line3_processed.csv"


# ============================================================
# FIND LINE 3 COLUMNS
# ============================================================

def get_line_column_map(
    date_file: Path,
    target_line: int,
) -> dict[str, int]:
    """
    Read only the CSV header and return:

        timestamp column name -> station number
    """

    header = pd.read_csv(
        date_file,
        nrows=0,
    ).columns.tolist()

    pattern = re.compile(
        rf"^L{target_line}_S(\d+)_D\d+$"
    )

    column_map: dict[str, int] = {}

    for column in header:
        match = pattern.match(column)

        if match:
            column_map[column] = int(match.group(1))

    if not column_map:
        raise ValueError(
            f"No timestamp columns found for Line {target_line}."
        )

    return column_map


# ============================================================
# PROCESS ONE CHUNK
# ============================================================

def process_chunk(
    date_chunk: pd.DataFrame,
    numeric_chunk: pd.DataFrame,
    line_column_map: dict[str, int],
) -> pd.DataFrame:
    """
    Convert one pair of matching date/numeric chunks into one row
    per part-station event.
    """

    if len(date_chunk) != len(numeric_chunk):
        raise ValueError(
            "Date and numeric chunks have different row counts: "
            f"{len(date_chunk):,} vs. {len(numeric_chunk):,}"
        )

    if not np.array_equal(
        date_chunk["Id"].to_numpy(),
        numeric_chunk["Id"].to_numpy(),
    ):
        raise ValueError(
            "Id values do not match between date and numeric chunks."
        )

    merged_chunk = date_chunk.merge(
        numeric_chunk,
        on="Id",
        how="left",
        validate="one_to_one",
    )

    missing_responses = merged_chunk["Response"].isna().sum()

    if missing_responses:
        raise ValueError(
            f"Found {missing_responses:,} missing Response values."
        )

    long_chunk = merged_chunk.melt(
        id_vars=["Id", "Response"],
        value_vars=list(line_column_map.keys()),
        var_name="TimestampColumn",
        value_name="Timestamp",
    )

    long_chunk = long_chunk.dropna(
        subset=["Timestamp"]
    ).copy()

    long_chunk["Station"] = (
        long_chunk["TimestampColumn"]
        .map(line_column_map)
        .astype("int16")
    )

    station_events = (
        long_chunk.groupby(
            ["Id", "Station"],
            as_index=False,
            sort=False,
        )
        .agg(
            Timestamp=("Timestamp", "min"),
            Response=("Response", "first"),
        )
    )

    station_events = station_events[
        ["Id", "Station", "Timestamp", "Response"]
    ]

    station_events["Id"] = station_events["Id"].astype(
        "int32"
    )
    station_events["Station"] = station_events[
        "Station"
    ].astype("int16")
    station_events["Timestamp"] = station_events[
        "Timestamp"
    ].astype("float32")
    station_events["Response"] = station_events[
        "Response"
    ].astype("int8")

    station_events = station_events.sort_values(
        ["Id", "Timestamp", "Station"]
    ).reset_index(drop=True)

    return station_events


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    start_time = time.time()

    print("Bosch Line 3 chunked preprocessing")
    print("----------------------------------")
    print(f"Mode: {RUN_NAME}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if not DATE_FILE.exists():
        raise FileNotFoundError(
            f"Date file not found: {DATE_FILE}"
        )

    if not NUMERIC_FILE.exists():
        raise FileNotFoundError(
            f"Numeric file not found: {NUMERIC_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(f"Deleted old output: {OUTPUT_FILE}")

    line_column_map = get_line_column_map(
        DATE_FILE,
        TARGET_LINE,
    )

    line_columns = list(line_column_map.keys())

    print(
        f"Line {TARGET_LINE} timestamp columns: "
        f"{len(line_columns):,}"
    )

    date_dtypes = {
        "Id": "int32",
        **{
            column: "float32"
            for column in line_columns
        },
    }

    numeric_dtypes = {
        "Id": "int32",
        "Response": "int8",
    }

    date_reader = pd.read_csv(
        DATE_FILE,
        usecols=["Id"] + line_columns,
        dtype=date_dtypes,
        chunksize=CHUNK_SIZE,
    )

    numeric_reader = pd.read_csv(
        NUMERIC_FILE,
        usecols=["Id", "Response"],
        dtype=numeric_dtypes,
        chunksize=CHUNK_SIZE,
    )

    first_output_chunk = True
    total_source_rows = 0
    total_output_rows = 0
    chunk_number = 0

    for date_chunk, numeric_chunk in zip(
        date_reader,
        numeric_reader,
    ):
        chunk_number += 1

        if SAMPLE_SIZE is not None:
            rows_remaining = SAMPLE_SIZE - total_source_rows

            if rows_remaining <= 0:
                break

            if len(date_chunk) > rows_remaining:
                date_chunk = date_chunk.iloc[
                    :rows_remaining
                ].copy()

                numeric_chunk = numeric_chunk.iloc[
                    :rows_remaining
                ].copy()

        source_rows = len(date_chunk)

        if source_rows == 0:
            break

        station_events = process_chunk(
            date_chunk=date_chunk,
            numeric_chunk=numeric_chunk,
            line_column_map=line_column_map,
        )

        station_events.to_csv(
            OUTPUT_FILE,
            mode="w" if first_output_chunk else "a",
            header=first_output_chunk,
            index=False,
        )

        first_output_chunk = False

        total_source_rows += source_rows
        total_output_rows += len(station_events)

        print(
            f"Chunk {chunk_number:>3}: "
            f"{source_rows:>6,} source rows -> "
            f"{len(station_events):>7,} station events | "
            f"total source rows: {total_source_rows:>9,}"
        )

        del date_chunk
        del numeric_chunk
        del station_events

        gc.collect()

        if (
            SAMPLE_SIZE is not None
            and total_source_rows >= SAMPLE_SIZE
        ):
            break

    if first_output_chunk:
        raise RuntimeError(
            "No output was created. Check the input files."
        )

    elapsed_seconds = time.time() - start_time

    print("\nProcessing complete")
    print("-------------------")
    print(f"Source rows processed: {total_source_rows:,}")
    print(f"Station events written: {total_output_rows:,}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Elapsed time: {elapsed_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    main()