import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/FGBL_qr_features.parquet')

# Check spread
spread = df['p_mid'] - df['p_ref']
print('Spread (p_mid - p_ref) statistics:')
print(spread.describe())
print()

# Count events where p_ref != p_mid
differ = (spread != 0).sum()
print(f'Events where p_ref != p_mid: {differ:,} ({100*differ/len(df):.2f}%)')
print(f'Events where p_ref == p_mid: {(spread == 0).sum():,} ({100*(spread == 0).sum()/len(df):.2f}%)')
print()

# Show unique spread values
print('Unique spread values (sorted):')
unique_spreads = sorted(spread.unique())
print(unique_spreads[:20])
print()

# Verify the relationship: for spread = 0.5 tick (odd spread), p_ref should be best_bid + 0.5
# For spread = 0 or 1.0 tick (even spread), p_ref should be mid
mask_half = np.abs(spread - 0.005) < 1e-6
print(f'Events with 0.5 tick spread: {mask_half.sum():,}')
print(f'Sample of 0.5 tick events:')
print(df[mask_half][['p_ref', 'p_mid']].head(10).to_string())
