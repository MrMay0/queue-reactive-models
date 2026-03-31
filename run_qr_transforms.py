from src.features.qr_transforms import build_qr_features

if __name__ == "__main__":
    build_qr_features(
        event_flow_path="data/processed/FGBL_event_flow.parquet",
        output_path="data/processed/FGBL_qr_features.parquet")
