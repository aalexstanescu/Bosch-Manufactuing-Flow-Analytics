import os
import numpy as np
import pandas as pd

EVENT_FILE = "data/processed/full/bosch_line3_processed.csv"

PART_ROUTE_OUTPUT = "data/processed/full/part_routes.csv"
ROUTE_OUTPUT = "data/processed/full/route_summary.csv"

MIN_PRIMARY_N = 500
MIN_FLAG_N = 100

print("Loading Line 3 events...")

events = pd.read_csv(
    EVENT_FILE,
    dtype={
        "Id": "int32",
        "Station": "int16",
        "Timestamp": "float32",
        "Response": "int8",
    },
)

# ----------------------------------
# BUILD ONE ROW PER PART
# ----------------------------------

part_routes = (
    events.sort_values(["Id", "Timestamp", "Station"])
    .groupby("Id")
    .agg(
        Route=(
            "Station",
            lambda x: " -> ".join(x.astype(str)),
        ),
        Response=("Response", "first"),
        FirstTimestamp=("Timestamp", "min"),
        LastTimestamp=("Timestamp", "max"),
        StationsVisited=("Station", "count"),
    )
    .reset_index()
)

part_routes["Line3FlowTime"] = (
    part_routes["LastTimestamp"]
    - part_routes["FirstTimestamp"]
).round(2)

# Save the part-level table for later Power BI analysis
part_routes.to_csv(
    PART_ROUTE_OUTPUT,
    index=False,
)

# ----------------------------------
# SUMMARIZE ROUTES
# ----------------------------------

route_summary = (
    part_routes.groupby("Route")
    .agg(
        PartCount=("Id", "count"),
        AverageFlowTime=("Line3FlowTime", "mean"),
        MedianFlowTime=("Line3FlowTime", "median"),
        P95FlowTime=(
            "Line3FlowTime",
            lambda x: x.quantile(0.95),
        ),
        AverageStationsVisited=("StationsVisited", "mean"),
        FailureRate=("Response", "mean"),
    )
    .reset_index()
    .sort_values("PartCount", ascending=False)
)

route_summary["FailureRate (%)"] = (
    route_summary["FailureRate"] * 100
)

route_summary["Share (%)"] = (
    route_summary["PartCount"]
    / route_summary["PartCount"].sum()
    * 100
)

route_summary["Confidence"] = np.select(
    [
        route_summary["PartCount"] >= MIN_PRIMARY_N,
        route_summary["PartCount"] >= MIN_FLAG_N,
    ],
    [
        "Primary",
        "Flagged (low volume)",
    ],
    default="Exclude from ranking",
)

route_summary["MaterialRoute"] = np.where(
    route_summary["Share (%)"] >= 1.0,
    "Material route",
    "Minor route",
)

os.makedirs(
    "data/processed/full",
    exist_ok=True,
)

route_summary.to_csv(
    ROUTE_OUTPUT,
    index=False,
)

print("\nTop 15 Routes")
print("----------------------------")

print(
    route_summary.head(15).to_string(index=False)
)

print("\nSaved:")
print(PART_ROUTE_OUTPUT)
print(ROUTE_OUTPUT)