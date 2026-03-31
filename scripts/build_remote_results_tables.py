from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact results tables from remote QR simulation artifacts.")
    parser.add_argument("--input-dir", default="data/processed/remote_results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-day", default=None)
    parser.add_argument("--plot-start", default=None)
    parser.add_argument("--plot-end", default=None)
    parser.add_argument("--stats-days", nargs="*", default=None)
    return parser.parse_args()


PERIODS = {
    "Full day": ("09:00:00", "18:00:00"),
    "Calm": ("10:00:00", "14:00:00"),
    "Active": ("15:00:00", "18:00:00"),
}
MODEL_ORDER = ["QRU", "QR", "FTQR", "SAQR"]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing precomputed file {path}")
    return path


def rolling_vol(mid: pd.Series, annualization_factor: float, window_seconds: int = 600) -> pd.Series:
    returns = np.log(mid).diff()
    return returns.rolling(window_seconds).std() * annualization_factor


def period_volatility(df: pd.DataFrame, start_time: str, end_time: str, annualization_factor: float) -> float:
    window = df.set_index("timestamp").between_time(start_time, end_time, inclusive="left")
    returns = np.log(window["mid_price"]).diff().dropna()
    if returns.empty:
        return np.nan
    return float(returns.std() * annualization_factor)


def load_midprice(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "mid_price"]]


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(require_file(input_dir / "price_dynamics_metadata.json").read_text())
    annualization_factor = float(metadata["annualization_factor"])
    plot_day = args.plot_day or metadata["plot_day"]
    plot_start = args.plot_start or metadata["plot_start"]
    plot_end = args.plot_end or metadata["plot_end"]
    stats_days = args.stats_days or metadata["stats_days"]
    price_dir = require_file(input_dir / "price_dynamics")

    rows = []
    for day in stats_days:
        real_df = load_midprice(require_file(price_dir / f"real_{day}.parquet"))
        for period_name, (start_time, end_time) in PERIODS.items():
            sigma_real = period_volatility(real_df, start_time, end_time, annualization_factor)
            for model in MODEL_ORDER:
                sim_df = load_midprice(require_file(price_dir / f"{model.lower()}_{day}.parquet"))
                sigma_sim = period_volatility(sim_df, start_time, end_time, annualization_factor)
                rows.append(
                    {
                        "model": model,
                        "day": str(day),
                        "period": period_name,
                        "sigma_real": sigma_real,
                        "sigma_sim": sigma_sim,
                        "relative_difference_pct": 100.0 * (sigma_sim - sigma_real) / sigma_real if pd.notna(sigma_real) and sigma_real != 0 else np.nan,
                        "quadratic_error": (sigma_sim - sigma_real) ** 2 if pd.notna(sigma_real) and pd.notna(sigma_sim) else np.nan,
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "volatility_summary.csv", index=False)

    aggregated = (
        summary.groupby(["period", "model"], sort=False)
        .agg(
            avg_relative_difference_pct=("relative_difference_pct", "mean"),
            avg_quadratic_error=("quadratic_error", "mean"),
            mean_sigma_real=("sigma_real", "mean"),
            mean_sigma_sim=("sigma_sim", "mean"),
        )
        .reset_index()
    )
    aggregated.to_csv(output_dir / "volatility_summary_aggregated.csv", index=False)

    plot_rows = []
    for model in ["Real"] + MODEL_ORDER:
        df = load_midprice(require_file(price_dir / f"{model.lower()}_{plot_day}.parquet"))
        df = df[(df["timestamp"] >= pd.Timestamp(f"{plot_day} {plot_start}", tz=metadata["timezone"])) & (df["timestamp"] <= pd.Timestamp(f"{plot_day} {plot_end}", tz=metadata["timezone"]))]
        vol = rolling_vol(df["mid_price"], annualization_factor, window_seconds=600)
        plot_rows.append(pd.DataFrame({"timestamp": df["timestamp"], "model": model, "mid_price": df["mid_price"], "rolling_vol": vol}))
    pd.concat(plot_rows, ignore_index=True).to_parquet(output_dir / "plot_window_series.parquet", index=False)

    print(json.dumps({"status": "ok", "rows": len(summary), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
