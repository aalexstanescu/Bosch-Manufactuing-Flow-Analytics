import pandas as pd

# Read only ONE production record
df = pd.read_csv(
    "data/raw/train_date.csv",
    nrows=1
)

row = df.iloc[0]

print(f"Part ID: {int(row['Id'])}")
print("\nNon-missing timestamps:\n")

for column, value in row.items():
    if column == "Id":
        continue

    if pd.notna(value):
        print(f"{column}: {value}")
