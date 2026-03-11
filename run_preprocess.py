from src.data.preprocess import build_processed_dataset

if __name__ == "__main__":
    build_processed_dataset(
        raw_dir="data/raw/",
        output_path="data/processed/FGBL_clean.parquet",
    )
