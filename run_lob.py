from src.features.lob_reconstruction import build_event_flow

if __name__ == "__main__":
    build_event_flow(
        raw_dir="data/raw/",
        output_path="data/processed/FGBL_event_flow.parquet",
    )
