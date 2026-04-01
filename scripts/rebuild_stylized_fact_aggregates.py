from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_remote_stylized_fact_artifacts import (
    ccdf_tail_table,
    interarrival_outputs,
    queue_size_outputs,
    size_distribution_table,
    traded_volume_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild stylized-fact aggregate tables from already saved simulation files."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/remote_results/stylized_facts",
        help="Stylized-facts result directory containing simulations/<model>/<day>.parquet",
    )
    parser.add_argument(
        "--stats-days",
        nargs="*",
        default=None,
        help="Optional explicit list of YYYY-MM-DD days. Defaults to metadata stats_days.",
    )
    return parser.parse_args()


def require_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing directory {path}")
    return path


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file {path}")
    return path


def collect_sim_paths(sim_root: Path, stats_days: list[str]) -> dict[str, list[Path]]:
    sim_paths: dict[str, list[Path]] = {}
    for model_dir in sorted(sim_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = "Real" if model_dir.name == "real" else model_dir.name.upper()
        paths = []
        for day in stats_days:
            path = model_dir / f"{day}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing simulation file {path}")
            paths.append(path)
        sim_paths[model_name] = paths
    if "Real" not in sim_paths:
        raise FileNotFoundError("Missing real market simulation cache under simulations/real/")
    return sim_paths


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    sim_root = require_dir(out_dir / "simulations")

    metadata_path = require_file(out_dir / "stylized_facts_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    stats_days = args.stats_days or metadata["stats_days"]
    stats_days = [str(day) for day in stats_days]

    sim_paths = collect_sim_paths(sim_root, stats_days)

    size_dist = size_distribution_table(sim_paths)
    size_dist.to_parquet(out_dir / "order_size_distribution.parquet", index=False)

    tail_ccdf, tail_fit = ccdf_tail_table(sim_paths)
    tail_ccdf.to_parquet(out_dir / "order_size_tail_ccdf.parquet", index=False)
    tail_fit.to_csv(out_dir / "order_size_tail_fit.csv", index=False)

    traded_windows, traded_summary_daily = traded_volume_outputs(sim_paths, stats_days)
    traded_windows.to_parquet(out_dir / "traded_volume_10min.parquet", index=False)
    traded_summary_daily.to_csv(out_dir / "traded_volume_summary_daily.csv", index=False)
    (
        traded_summary_daily.groupby(["period", "model"], sort=False)
        .agg(
            avg_relative_difference_pct=("relative_difference_pct", "mean"),
            avg_quadratic_error=("quadratic_error", "mean"),
            mean_real_traded_volume=("real_traded_volume", "mean"),
            mean_sim_traded_volume=("sim_traded_volume", "mean"),
        )
        .reset_index()
        .to_csv(out_dir / "traded_volume_summary.csv", index=False)
    )

    ia_hist, ia_fit, ia_pdf = interarrival_outputs(sim_paths)
    ia_hist.to_parquet(out_dir / "trade_interarrival_hist.parquet", index=False)
    ia_fit.to_csv(out_dir / "trade_weibull_fit.csv", index=False)
    ia_pdf.to_parquet(out_dir / "trade_weibull_pdf.parquet", index=False)

    queue_dist, queue_gamma, queue_ks_daily, queue_ks_summary = queue_size_outputs(sim_paths, stats_days)
    queue_dist.to_parquet(out_dir / "queue_size_distribution.parquet", index=False)
    queue_gamma.to_parquet(out_dir / "queue_size_gamma_fit.parquet", index=False)
    queue_ks_daily.to_csv(out_dir / "queue_size_ks_daily.csv", index=False)
    queue_ks_summary.to_csv(out_dir / "queue_size_ks_summary.csv", index=False)

    metadata["models"] = list(sim_paths.keys())
    metadata["stats_days"] = stats_days
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(out_dir),
                "models": metadata["models"],
                "stats_days": stats_days,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
