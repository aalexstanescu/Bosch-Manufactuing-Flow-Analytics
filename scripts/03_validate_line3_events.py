import pandas as pd

FILE = "data/processed/bosch_line3_processed.csv"

df = pd.read_csv(FILE)

# Count distinct timestamps for every part-station pair
event_counts = (
    df.groupby(["Id", "Station"])["Timestamp"]
    .nunique()
    .reset_index(name="DistinctTimestamps")
)

multiple_events = event_counts[
    event_counts["DistinctTimestamps"] > 1
].copy()

print("Validation summary")
print("------------------")
print(f"Processed rows: {len(df):,}")
print(f"Unique parts: {df['Id'].nunique():,}")
print(f"Unique stations: {df['Station'].nunique():,}")
print(
    "Part-station combinations with multiple timestamps: "
    f"{len(multiple_events):,}"
)

if len(multiple_events) > 0:
    print("\nExamples:")
    print(multiple_events.head(20).to_string(index=False))
else:
    print("\nEvery part-station pair has exactly one distinct timestamp.")

print("\nObservations per station:")
station_counts = (
    df.groupby("Station")["Id"]
    .nunique()
    .reset_index(name="PartsObserved")
    .sort_values("Station")
)

print(station_counts.to_string(index=False))