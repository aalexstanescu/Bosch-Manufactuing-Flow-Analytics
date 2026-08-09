import os
import pandas as pd

EVENT_FILE = "data/processed/full/bosch_line3_processed.csv"
FLOW_FILE = "data/processed/full/bosch_line3_flow_metrics.csv"

STATION_OUTPUT = "data/processed/full/station_summary.csv"
TRANSITION_OUTPUT = "data/processed/full/transition_summary.csv"

print("Loading event and flow data...")

events = pd.read_csv(EVENT_FILE)
flows = pd.read_csv(FLOW_FILE)

# --------------------------------------------------
# 1. STATION-LEVEL METRICS
# Based on every part that visited each station
# --------------------------------------------------

station_summary = (
    events.groupby("Station")
    .agg(
        PartsObserved=("Id", "nunique"),
        FailedParts=("Response", "sum"),
        FailureRate=("Response", "mean"),
    )
    .reset_index()
)

station_summary["FailureRate"] *= 100

# Count how often each station has an outgoing transition.
outgoing_counts = (
    flows.groupby("Station")["Id"]
    .nunique()
    .reset_index(name="PartsWithOutgoingTransition")
)

station_summary = station_summary.merge(
    outgoing_counts,
    on="Station",
    how="left"
)

station_summary["PartsWithOutgoingTransition"] = (
    station_summary["PartsWithOutgoingTransition"]
    .fillna(0)
    .astype(int)
)

# How often the station was the final observed station.
station_summary["PartsEndingAtStation"] = (
    station_summary["PartsObserved"]
    - station_summary["PartsWithOutgoingTransition"]
)

station_summary["EndingAtStationRate"] = (
    station_summary["PartsEndingAtStation"]
    / station_summary["PartsObserved"]
    * 100
)

# Confidence based on all station visits.
station_summary["Confidence"] = "High"

station_summary.loc[
    station_summary["PartsObserved"] < 500,
    "Confidence"
] = "Medium"

station_summary.loc[
    station_summary["PartsObserved"] < 100,
    "Confidence"
] = "Low"

station_summary = station_summary.sort_values("Station")

# --------------------------------------------------
# 2. TRANSITION-LEVEL METRICS
# Based only on actual station-to-station movements
# --------------------------------------------------

transition_summary = (
    flows.groupby(["Station", "NextStation"])
    .agg(
        TransitionCount=("Id", "nunique"),
        AvgTransit=("TransitTime", "mean"),
        MedianTransit=("TransitTime", "median"),
        P95Transit=("TransitTime", lambda x: x.quantile(0.95)),
        MaxTransit=("TransitTime", "max"),
        ZeroTransitRate=(
            "TransitTime",
            lambda x: (x == 0).mean() * 100
        ),
        FailureRate=("Response", "mean"),
    )
    .reset_index()
)

transition_summary["FailureRate"] *= 100

transition_summary["Confidence"] = "High"

transition_summary.loc[
    transition_summary["TransitionCount"] < 500,
    "Confidence"
] = "Medium"

transition_summary.loc[
    transition_summary["TransitionCount"] < 100,
    "Confidence"
] = "Low"

transition_summary = transition_summary.sort_values(
    ["Station", "NextStation"]
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

station_summary.to_csv(
    STATION_OUTPUT,
    index=False
)

transition_summary.to_csv(
    TRANSITION_OUTPUT,
    index=False
)

print("\nStation Summary")
print("----------------")
print(station_summary.to_string(index=False))

print("\nTop transitions by median transit time")
print("--------------------------------------")

reliable_transitions = transition_summary[
    transition_summary["TransitionCount"] >= 100
].sort_values(
    ["MedianTransit", "P95Transit"],
    ascending=False
)

print(
    reliable_transitions.head(15).to_string(index=False)
)

print("\nSaved:")
print(STATION_OUTPUT)
print(TRANSITION_OUTPUT)

print("\nStations ranked by failure rate")
print("--------------------------------")

print(
    station_summary[
        ["Station", "PartsObserved", "FailedParts", "FailureRate"]
    ]
    .sort_values("FailureRate", ascending=False)
    .to_string(index=False)
)