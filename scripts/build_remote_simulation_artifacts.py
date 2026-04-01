from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnose_ftqr_small_n import build_streaming_diagnostics, simulate_model_effects
from models.common import calibrate_common
from models.ftqr import FTQRSimulator, calibrate_ftqr
from models.qr import QRSimulator, calibrate_qr
from models.qru import QRUSimulator, calibrate_qru
from models.saqr import SAQRSimulator, calibrate_saqr
from src.features.qr_empirical import build_qr_intensity_tables, load_day, trading_days


TZ = "Europe/Berlin"
TRADING_SECONDS = 9 * 3600
BASE_SEED = 20260331


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact remote artifacts for QR result notebooks.")
    parser.add_argument("--event-flow", default=str(ROOT / "data/processed/FGBL_event_flow.parquet"))
    parser.add_argument("--raw-dir", default=str(ROOT / "data/raw"))
    parser.add_argument("--models", nargs="+", default=["QRU", "QR", "FTQR"])
    parser.add_argument("--days", nargs="*", default=None, help="Explicit day list for sampled price series.")
    parser.add_argument("--plot-day", default=None, help="Representative day for Figure 9-style plots.")
    parser.add_argument("--plot-start", default="10:00:00")
    parser.add_argument("--plot-end", default="12:30:00")
    parser.add_argument("--stats-days", nargs="*", default=None, help="Days used in the volatility benchmark.")
    parser.add_argument("--output-dir", default=str(ROOT / "data/processed/remote_results"))
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def calendar_bounds(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{day} 09:00:00", tz=TZ)
    end = pd.Timestamp(f"{day} 18:00:00", tz=TZ)
    return start, end


def normalize_days(args: argparse.Namespace, all_days: list) -> tuple[list, object]:
    if args.days:
        selected_days = [pd.Timestamp(day).date() for day in args.days]
    else:
        selected_days = []

    if args.stats_days:
        stats_days = [pd.Timestamp(day).date() for day in args.stats_days]
    else:
        stats_days = all_days[:5]

    if args.plot_day:
        plot_day = pd.Timestamp(args.plot_day).date()
    else:
        counts = []
        dataset = ds.dataset(args.event_flow, format="parquet")
        for day in all_days:
            counts.append({"date": day, "events": dataset.to_table(columns=["date"], filter=ds.field("date") == day).num_rows})
        counts_df = pd.DataFrame(counts)
        target = counts_df["events"].median()
        plot_day = counts_df.iloc[(counts_df["events"] - target).abs().argmin()]["date"]

    needed = list(dict.fromkeys(selected_days + stats_days + [plot_day]))
    return needed, plot_day


def sample_real_mid_1s(dataset: ds.Dataset, day) -> pd.DataFrame:
    cols = ["ts", "date", "p_mid"]
    df = load_day(dataset, day, cols).copy()
    if "ts" not in getattr(df.index, "names", []):
        if "ts" in df.columns:
            df = df.sort_values("ts").set_index("ts")
        else:
            df = df.reset_index().sort_values("ts").set_index("ts")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.to_datetime(df.index)
    start, end = calendar_bounds(day)
    idx = pd.date_range(start, end, freq="1s", inclusive="left")
    frame = df[["p_mid"]].rename(columns={"p_mid": "mid_price"})
    if frame.empty:
        raise ValueError(f"No real market rows found for {day}")
    if frame.index.min() > start:
        init_row = frame.iloc[[0]].copy()
        init_row.index = pd.DatetimeIndex([start])
        frame = pd.concat([init_row, frame])
    sampled = frame.reindex(frame.index.union(idx)).sort_index().ffill().reindex(idx)
    sampled.index.name = "timestamp"
    return sampled.reset_index()[["timestamp", "mid_price"]]


def load_open_state(dataset: ds.Dataset, day) -> tuple[float, int, int]:
    cols = ["ts", "date", "p_mid", "best_bid_int", "best_ask_int"]
    df = load_day(dataset, day, cols).copy()
    if "ts" not in getattr(df.index, "names", []):
        if "ts" in df.columns:
            df = df.sort_values("ts").set_index("ts")
        else:
            df = df.reset_index().sort_values("ts").set_index("ts")
    df = df.sort_index()
    if df.empty:
        raise ValueError(f"No opening state found for {day}")
    row = df.iloc[0]
    return float(row["p_mid"]), int(row["best_bid_int"]), int(row["best_ask_int"])


def simulate_mid_1s(simulator, day, start_bid: int, start_ask: int, q0: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    current_bid = int(start_bid)
    current_ask = int(start_ask)
    queue_size = int(q0)
    elapsed = 0.0
    next_second = 0
    mid_prices = np.empty(TRADING_SECONDS, dtype=np.float64)

    while next_second < TRADING_SECONDS:
        n, rates = simulator._lookup_rates(queue_size)
        lambda_total = float(rates.sum())
        if lambda_total <= 0:
            mid_prices[next_second:] = (current_bid + current_ask) / 200.0
            break

        delta_t = float(rng.exponential(1.0 / lambda_total))
        event_time = elapsed + delta_t
        fill_until = min(TRADING_SECONDS, int(np.floor(event_time)) + 1)
        if fill_until > next_second:
            mid_prices[next_second:fill_until] = (current_bid + current_ask) / 200.0
            next_second = fill_until
        if event_time >= TRADING_SECONDS:
            break

        eta, size = simulator._sample_event(n, queue_size, rng)
        queue_after = simulator._apply_event(queue_size, eta, size)
        if queue_after == 0:
            shift = int(rng.choice([-1, 1]))
            current_bid += shift
            current_ask += shift
            queue_size = simulator.calibration.sample_reset_queue(rng)
        else:
            queue_size = queue_after
        elapsed = event_time

    if next_second < TRADING_SECONDS:
        mid_prices[next_second:] = (current_bid + current_ask) / 200.0

    start, _ = calendar_bounds(day)
    out = pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta(np.arange(TRADING_SECONDS), unit="s"),
            "mid_price": mid_prices,
        }
    )
    return out


def save_artifact(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def export_calibration_artifacts(common, ftqr_cal, qru_cal, qr_model_dir: Path, overwrite: bool, event_flow_path: str, raw_dir: str) -> None:
    qr_model_dir.mkdir(parents=True, exist_ok=True)

    summary_path = qr_model_dir / "calibration_summary.json"
    if overwrite or not summary_path.exists():
        summary = {
            "level": int(common.level),
            "aes_level": float(common.aes_level),
            "n_bins": int(len(common.intensity_df)),
            "ftqr_bins": int(len(ftqr_cal.intensity_df)),
            "saqr_joint_rows": int(len(common.joint_size_df)),
            "qru_unit_size": int(qru_cal.unit_size),
        }
        summary_path.write_text(json.dumps(summary, indent=2))

    save_artifact(common.intensity_df, qr_model_dir / "common_intensity.parquet", overwrite)
    save_artifact(ftqr_cal.intensity_df, qr_model_dir / "ftqr_intensity.parquet", overwrite)
    save_artifact(common.size_distribution, qr_model_dir / "size_distribution.parquet", overwrite)
    save_artifact(common.eta_size_distribution, qr_model_dir / "eta_size_distribution.parquet", overwrite)
    save_artifact(common.joint_size_df, qr_model_dir / "joint_size.parquet", overwrite)

    saqr_aes = common.joint_size_df.copy()
    saqr_aes["size_aes"] = np.ceil(saqr_aes["size"] / common.aes_level).astype(int)
    saqr_aes = (
        saqr_aes.groupby(["eta", "n", "size_aes"], as_index=False)
        .agg(
            count=("count", "sum"),
            lambda_eta_size=("lambda_eta_size", "sum"),
            n_obs=("n_obs", "first"),
            Lambda=("Lambda", "first"),
        )
    )
    saqr_aes["row_prob"] = saqr_aes.groupby(["eta", "n"])["lambda_eta_size"].transform(lambda x: x / x.sum())
    save_artifact(saqr_aes, qr_model_dir / "saqr_aes.parquet", overwrite)

    build_qr_intensity_tables(
        event_flow_path=event_flow_path,
        output_path=str(qr_model_dir / "qr_intensities.parquet"),
        size_output_path=str(qr_model_dir / "qr_intensities_size.parquet"),
        raw_dir=raw_dir,
        level=common.level,
        min_obs=50,
    )

    qr_df = common.intensity_df.copy().rename(
        columns={"lambda_L": "lambda_L_qr", "lambda_C": "lambda_C_qr", "lambda_M": "lambda_M_qr"}
    )
    ft_df = ftqr_cal.intensity_df.copy().rename(
        columns={"lambda_L": "lambda_L_ftqr", "lambda_C": "lambda_C_ftqr", "lambda_M": "lambda_M_ftqr"}
    )
    table = qr_df.merge(
        ft_df[["n", "lambda_L_ftqr", "lambda_C_ftqr", "lambda_M_ftqr", "lambda_C_all", "lambda_M_all", "lambda_global"]],
        on="n",
        how="left",
    ).rename(columns={"lambda_C_all": "lambda_C_all_ftqr", "lambda_M_all": "lambda_M_all_ftqr"})
    table["lambda_M_total_ftqr"] = table["lambda_M_ftqr"] + table["lambda_M_all_ftqr"]
    table["lambda_C_total_ftqr"] = table["lambda_C_ftqr"] + table["lambda_C_all_ftqr"]
    table["lambda_L_qru"] = table["lambda_L_qr"]
    table["lambda_C_qru"] = table["lambda_C_qr"]
    table["lambda_M_qru"] = table["lambda_M_qr"]

    full_df, _, cond_df = build_streaming_diagnostics(event_flow_path, raw_dir, common.level)
    full_pivot = full_df.pivot_table(index=["n", "eta"], columns="is_full", values="count", fill_value=0).reset_index()
    full_pivot = full_pivot.rename(columns={False: "count_partial", True: "count_full"})
    full_pivot["count_total"] = full_pivot["count_partial"] + full_pivot["count_full"]
    full_pivot["p_full"] = np.where(full_pivot["count_total"] > 0, full_pivot["count_full"] / full_pivot["count_total"], np.nan)
    prob_df = full_pivot.pivot(index="n", columns="eta", values="p_full").reset_index().rename(columns={"C": "p_full_C", "M": "p_full_M"})
    table = table.merge(prob_df, on="n", how="left")

    table[table["n"] <= 20].sort_values("n").to_csv(qr_model_dir / "ftqr_small_n_table.csv", index=False)
    full_pivot.to_csv(qr_model_dir / "ftqr_full_consumption_probabilities.csv", index=False)
    cond_df.to_csv(qr_model_dir / "ftqr_conditional_size_distributions.csv", index=False)
    simulate_model_effects(common).to_csv(qr_model_dir / "qr_qru_ftqr_simulation_summary.csv", index=False)


def main() -> int:
    args = parse_args()
    event_flow_path = str(Path(args.event_flow).resolve())
    raw_dir = str(Path(args.raw_dir).resolve())
    output_dir = Path(args.output_dir).resolve()
    price_dir = output_dir / "price_dynamics"
    qr_model_dir = output_dir / "qr_model_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    price_dir.mkdir(parents=True, exist_ok=True)
    qr_model_dir.mkdir(parents=True, exist_ok=True)

    all_days = trading_days(raw_dir)
    needed_days, plot_day = normalize_days(args, all_days)
    stats_days = [pd.Timestamp(day).date() for day in (args.stats_days or [str(day) for day in all_days[:5]])]

    dataset = ds.dataset(event_flow_path, format="parquet")
    common = calibrate_common(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs)
    qr_cal = calibrate_qr(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs, common=common)
    qru_cal = calibrate_qru(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs, common=common)
    ftqr_cal = calibrate_ftqr(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs, common=common)
    export_calibration_artifacts(common, ftqr_cal, qru_cal, qr_model_dir, args.overwrite, event_flow_path, raw_dir)

    simulators = {
        "QRU": QRUSimulator(qru_cal),
        "QR": QRSimulator(qr_cal),
        "FTQR": FTQRSimulator(ftqr_cal),
    }
    if "SAQR" in args.models:
        saqr_cal = calibrate_saqr(
            event_flow_path,
            raw_dir=raw_dir,
            level=args.level,
            min_obs=args.min_obs,
            smoothing_alpha=25.0,
            common=common,
        )
        simulators["SAQR"] = SAQRSimulator(saqr_cal)
    selected_models = [m for m in args.models if m in simulators]

    for day_idx, day in enumerate(needed_days):
        real_out = price_dir / f"real_{day}.parquet"
        if args.overwrite or not real_out.exists():
            real_df = sample_real_mid_1s(dataset, day)
            real_df.to_parquet(real_out, index=False)
            del real_df
            gc.collect()

        _, start_bid, start_ask = load_open_state(dataset, day)
        q0 = common.sample_reset_queue(np.random.default_rng(BASE_SEED + day_idx))
        for model_idx, model_name in enumerate(selected_models):
            out_path = price_dir / f"{model_name.lower()}_{day}.parquet"
            if out_path.exists() and not args.overwrite:
                continue
            sim_df = simulate_mid_1s(
                simulators[model_name],
                day=day,
                start_bid=start_bid,
                start_ask=start_ask,
                q0=q0,
                seed=BASE_SEED + 10_000 * (model_idx + 1) + day_idx,
            )
            sim_df.to_parquet(out_path, index=False)
            del sim_df
            gc.collect()

    metadata = {
        "plot_day": str(plot_day),
        "plot_start": args.plot_start,
        "plot_end": args.plot_end,
        "stats_days": [str(day) for day in stats_days],
        "available_days": [str(day) for day in needed_days],
        "models": selected_models,
        "annualization_factor": float(np.sqrt(252 * TRADING_SECONDS)),
        "base_seed": BASE_SEED,
        "timezone": TZ,
    }
    (output_dir / "price_dynamics_metadata.json").write_text(json.dumps(metadata, indent=2))
    manifest = {
        "price_dynamics_dir": str(price_dir),
        "qr_model_validation_dir": str(qr_model_dir),
        "files": sorted([str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()]),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "files": len(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
