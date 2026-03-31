import pandas as pd
from pathlib import Path
from src.features.lob_reconstruction import reconstruct_day

# Load first raw file
raw_files = sorted(Path("data/raw/").glob("*.parquet"))
if not raw_files:
    raise FileNotFoundError("No parquet files found")

f = raw_files[0]
print(f"Processing {f}")

df_raw = pd.read_parquet(f)

# Convert index to CET
df_raw = df_raw.reset_index()
df_raw["ts"] = pd.to_datetime(df_raw["ts_recv"]).dt.tz_convert("Europe/Berlin")
df_raw = df_raw.set_index("ts").sort_index()

# Collect fill_order_ids
fill_order_ids = set(df_raw.loc[df_raw["action"] == "F", "order_id"])

# Filter actions
df_raw = df_raw[df_raw["action"].isin(["A", "C", "T"])]
df_raw = df_raw[df_raw["side"].isin(["B", "A"])]
df_raw = df_raw[~((df_raw["action"] == "T") & (df_raw["side"] == "N"))]

# Reconstruct
df_events = reconstruct_day(df_raw, fill_order_ids)

if df_events.empty:
    print("No events")
else:
    print(f"Events: {len(df_events)}")

    # Check spreads
    spread = df_events['p_mid'] - df_events['p_ref']
    print('Spread statistics:')
    print(spread.describe())
    print()

    # Count differ
    differ = (spread != 0).sum()
    print(f'Events where p_ref != p_mid: {differ:,} ({100*differ/len(df_events):.2f}%)')
    print()

    # Unique spreads
    print('Unique spread values (first 10):')
    unique_spreads = sorted(spread.unique())
    for s in unique_spreads[:10]:
        count = (spread == s).sum()
        print(f'  {s:>8.6f}: {count:>7,} events')

    print()
    print("Expected: small positive spreads for odd, 0 for even.")