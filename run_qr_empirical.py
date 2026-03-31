from src.features.qr_empirical import build_qr_intensity_tables


if __name__ == "__main__":
    curves_df, size_curves_df, diagnostics_df, aes = build_qr_intensity_tables(
        event_flow_path="data/processed/FGBL_event_flow.parquet",
        output_path="data/processed/qr_intensities.parquet",
        size_output_path="data/processed/qr_intensities_size.parquet",
        raw_dir="data/raw",
        level=1,
        min_obs=50,
    )

    print(f"Saved {len(curves_df):,} intensity rows → data/processed/qr_intensities.parquet")
    print(f"Saved {len(size_curves_df):,} size-intensity rows → data/processed/qr_intensities_size.parquet")
    print(f"Days processed: {diagnostics_df['date'].nunique()}")
    print(
        "[QR Empirical] old_events_used="
        f"{int(diagnostics_df['events_used_old'].sum()):,}, "
        f"old_discarded_pct={diagnostics_df['discarded_pct_old'].mean():.2f}%, "
        f"state_rows_used={int(diagnostics_df['state_rows_used'].sum()):,}, "
        f"state_retained_pct={diagnostics_df['state_retained_pct'].mean():.2f}%, "
        f"n_range=({diagnostics_df['n_min'].min()}, {diagnostics_df['n_max'].max()})"
    )
