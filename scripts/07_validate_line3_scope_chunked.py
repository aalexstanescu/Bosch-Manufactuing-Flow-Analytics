"""
Bosch Line 3 Scope Validation — Chunked
=======================================

Validates whether parts continue to Lines 0, 1, or 2 after their final
observed Line 3 event.

Inputs:
    data/raw/train_date.csv
    data/processed/full/bosch_line3_processed.csv

Output:
    data/processed/full/line3_scope_validation.csv

This script is chunked so it can safely process the full Bosch dataset.
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

CHUNK_SIZE = 5_000
TARGET_LINE = 3

DATE_FILE = Path("data/raw/train_date.csv")
LINE3_FILE = Path(
    "data/processed/full/bosch_line3_processed.csv"
)
OUTPUT_FILE = Path(
    "data/processed/full/line3_scope_validation.csv"
)


# ============================================================
# FIND OTHER-LINE TIMESTAMP COLUMNS
# ============================================================

def get_other_line_columns(
    date_file: Path,
    target_line: int,
) -> list[str]:
    """
    Return all timestamp columns not belonging to the target line.
    """

    header = pd.read_csv(
        date_file,
        nrows=0,
    ).columns.tolist()

    pattern = re.compile(
        r"^L(\d+)_S\d+_D\d+$"
    )

    other_columns: list[str] = []

    for column in header:
        match = pattern.match(column)

        if match:
            line_number = int(match.group(1))

            if line_number != target_line:
                other_columns.append(column)

    if not other_columns:
        raise ValueError(
            "No other-line timestamp columns were found."
        )

    return other_columns


# ============================================================
# BUILD LAST LINE 3 EVENT TABLE
# ============================================================

def build_last_line3_events(
    line3_file: Path,
) -> pd.DataFrame:
    """
    Create one row per part containing its final observed Line 3 event.
    """

    print("Loading full processed Line 3 events...")

    line3 = pd.read_csv(
        line3_file,
        dtype={
            "Id": "int32",
            "Station": "int16",
            "Timestamp": "float32",
            "Response": "int8",
        },
    )

    last_line3 = (
        line3.sort_values(
            ["Id", "Timestamp", "Station"]
        )
        .groupby("Id", sort=False)
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

    last_line3["Id"] = last_line3["Id"].astype(
        "int32"
    )

    last_line3["LastLine3Station"] = last_line3[
        "LastLine3Station"
    ].astype("int16")

    last_line3["LastLine3Timestamp"] = last_line3[
        "LastLine3Timestamp"
    ].astype("float32")

    del line3
    gc.collect()

    print(
        f"Parts with Line 3 activity: "
        f"{len(last_line3):,}"
    )

    return last_line3


# ============================================================
# PROCESS ONE RAW DATE CHUNK
# ============================================================

def process_date_chunk(
    chunk: pd.DataFrame,
    other_line_columns: list[str],
) -> pd.DataFrame:
    """
    Calculate earliest and latest observed timestamps outside Line 3
    for each part in one chunk.
    """

    timestamp_data = chunk[
        other_line_columns
    ]

    earliest_other = timestamp_data.min(
        axis=1,
        skipna=True,
    )

    latest_other = timestamp_data.max(
        axis=1,
        skipna=True,
    )

    has_other = timestamp_data.notna().any(
        axis=1
    )

    summary = pd.DataFrame(
        {
            "Id": chunk["Id"].to_numpy(),
            "HasOtherLineActivity": (
                has_other.to_numpy()
            ),
            "EarliestOtherLineTimestamp": (
                earliest_other.to_numpy()
            ),
            "LatestOtherLineTimestamp": (
                latest_other.to_numpy()
            ),
        }
    )

    summary["Id"] = summary["Id"].astype(
        "int32"
    )

    return summary


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    start_time = time.time()

    print("Line 3 scope validation — chunked")
    print("---------------------------------")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if not DATE_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {DATE_FILE}"
        )

    if not LINE3_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {LINE3_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(
            f"Deleted old output: {OUTPUT_FILE}"
        )

    last_line3 = build_last_line3_events(
        LINE3_FILE
    )

    line3_ids = set(
        last_line3["Id"].tolist()
    )

    other_line_columns = get_other_line_columns(
        DATE_FILE,
        TARGET_LINE,
    )

    print(
        f"Other-line timestamp columns: "
        f"{len(other_line_columns):,}"
    )

    dtype_map = {
        "Id": "int32",
        **{
            column: "float32"
            for column in other_line_columns
        },
    }

    reader = pd.read_csv(
        DATE_FILE,
        usecols=["Id"] + other_line_columns,
        dtype=dtype_map,
        chunksize=CHUNK_SIZE,
    )

    first_chunk = True
    total_raw_rows = 0
    total_matched_parts = 0
    chunk_number = 0

    for chunk in reader:
        chunk_number += 1
        total_raw_rows += len(chunk)

        chunk = chunk[
            chunk["Id"].isin(line3_ids)
        ].copy()

        if chunk.empty:
            print(
                f"Chunk {chunk_number:>3}: "
                "no Line 3 parts"
            )
            continue

        other_summary = process_date_chunk(
            chunk,
            other_line_columns,
        )

        validation_chunk = other_summary.merge(
            last_line3,
            on="Id",
            how="inner",
            validate="one_to_one",
        )

        validation_chunk[
            "OtherLineActivityAfterLine3"
        ] = (
            validation_chunk[
                "HasOtherLineActivity"
            ]
            & (
                validation_chunk[
                    "LatestOtherLineTimestamp"
                ]
                >
                validation_chunk[
                    "LastLine3Timestamp"
                ]
            )
        )

        validation_chunk[
            "OtherLineActivityOnlyBeforeOrAtLine3"
        ] = (
            validation_chunk[
                "HasOtherLineActivity"
            ]
            & ~validation_chunk[
                "OtherLineActivityAfterLine3"
            ]
        )

        validation_chunk[
            "NoOtherLineActivity"
        ] = ~validation_chunk[
            "HasOtherLineActivity"
        ]

        validation_chunk[
            "TimeFromLastLine3ToLatestOtherLine"
        ] = (
            validation_chunk[
                "LatestOtherLineTimestamp"
            ]
            -
            validation_chunk[
                "LastLine3Timestamp"
            ]
        )

        validation_chunk[
            "ScopeCategory"
        ] = "No activity outside Line 3"

        validation_chunk.loc[
            validation_chunk[
                "OtherLineActivityOnlyBeforeOrAtLine3"
            ],
            "ScopeCategory",
        ] = (
            "Other-line activity only before "
            "or at final Line 3 event"
        )

        validation_chunk.loc[
            validation_chunk[
                "OtherLineActivityAfterLine3"
            ],
            "ScopeCategory",
        ] = (
            "Other-line activity after "
            "final Line 3 event"
        )

        validation_chunk = validation_chunk[
            [
                "Id",
                "LastLine3Station",
                "LastLine3Timestamp",
                "HasOtherLineActivity",
                "EarliestOtherLineTimestamp",
                "LatestOtherLineTimestamp",
                "OtherLineActivityAfterLine3",
                "OtherLineActivityOnlyBeforeOrAtLine3",
                "NoOtherLineActivity",
                "TimeFromLastLine3ToLatestOtherLine",
                "ScopeCategory",
            ]
        ]

        validation_chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False
        total_matched_parts += len(
            validation_chunk
        )

        print(
            f"Chunk {chunk_number:>3}: "
            f"{len(chunk):>5,} matched parts | "
            f"total validated: "
            f"{total_matched_parts:>9,}"
        )

        del chunk
        del other_summary
        del validation_chunk

        gc.collect()

    if first_chunk:
        raise RuntimeError(
            "No validation output was created."
        )

    print("\nReading final validation output...")

    validation = pd.read_csv(
        OUTPUT_FILE,
        dtype={
            "Id": "int32",
            "LastLine3Station": "int16",
            "HasOtherLineActivity": "bool",
            "OtherLineActivityAfterLine3": "bool",
            "OtherLineActivityOnlyBeforeOrAtLine3": "bool",
            "NoOtherLineActivity": "bool",
        },
    )

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
    print(
        summary.to_string(index=False)
    )

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

    print(
        "\nParts whose last observed "
        "Line 3 station is 37"
    )
    print(
        "----------------------------------------"
    )
    print(
        f"Parts analyzed: {len(station37):,}"
    )
    print(
        station37_summary.to_string(
            index=False
        )
    )

    examples = validation[
        validation[
            "OtherLineActivityAfterLine3"
        ]
    ][
        [
            "Id",
            "LastLine3Station",
            "LastLine3Timestamp",
            "LatestOtherLineTimestamp",
            "TimeFromLastLine3ToLatestOtherLine",
        ]
    ].head(15)

    print(
        "\nExamples with activity "
        "after Line 3:"
    )

    if examples.empty:
        print("None found.")
    else:
        print(
            examples.to_string(index=False)
        )

    elapsed_seconds = (
        time.time() - start_time
    )

    print("\nValidation complete")
    print("-------------------")
    print(
        f"Raw rows scanned: "
        f"{total_raw_rows:,}"
    )
    print(
        f"Parts validated: "
        f"{len(validation):,}"
    )
    print(
        f"Output file: {OUTPUT_FILE}"
    )
    print(
        f"Elapsed time: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()