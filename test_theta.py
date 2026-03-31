#!/usr/bin/env python
import pandas as pd
import numpy as np
from src.features.qr_transforms import calibrate_theta, compute_p_ref_series, quantize_queue_sizes, assign_period_ids, compute_aes

# Load event flow
df = pd.read_parquet('data/processed/FGBL_event_flow.parquet')
print(f"Loaded {len(df)} events")

# Apply filtering
df = df[abs(df['p_mid'] - df['p_ref']) <= 0.05]
print(f"After filtering: {len(df)} events")

# Apply transformations in order
aes = compute_aes(df)
df = quantize_queue_sizes(df, aes)
df = assign_period_ids(df)

print(f"\nDepletions: {df['depletion'].sum()}")
depl = df[df['depletion']][['p_mid','depletion_side']]
print(f"Depletion events:\n{depl.head(20)}")

# Try to calibrate theta
print("\nTesting calibrate_theta()...")
try:
    theta = calibrate_theta(df)
    print(f"✓ Calibrated theta = {theta:.6f}")
except ValueError as e:
    print(f"✗ ValueError: {e}")
except Exception as e:
    print(f"✗ Error ({type(e).__name__}): {e}")
    import traceback
    traceback.print_exc()
