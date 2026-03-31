from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common import calibrate_common
from models.ftqr import FTQRSimulator, calibrate_ftqr
from models.qr import QRSimulator, calibrate_qr
from models.qru import QRUSimulator, calibrate_qru
from models.saqr import SAQRSimulator, calibrate_saqr
from src.features.qr_empirical import load_day


EVENT_FLOW_PATH = ROOT / "data/processed/FGBL_event_flow.parquet"
RAW_DIR = ROOT / "data/raw"
DEFAULT_OUT_DIR = ROOT / "data/processed/price_dynamics_volatility_light"
BASE_SEED = 20260331


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a minimal 1-second mid-price cache for one model, one day, one window.",
    )
    parser.add_argument("--model", required=True, choices=["real", "qru", "qr", "ftqr", "saqr"])
    parser.add_argument("--day", required=True, help="Trading day in YYYY-MM-DD format.")
    parser.add_argument("--start-time", required=True, help="Window start in HH:MM or HH:MM:SS.")
    parser.add_argument("--end-time", required=True, help="Window end in HH:MM or HH:MM:SS.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output parquet path. Defaults to data/processed/price_dynamics_volatility_light/<model>_<day>.parquet",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def normalize_clock(value: str) -> str:
    parts = value.split(":")
    if len(parts) == 2:
        return value + ":00"
    return value


def output_path(model: str, day: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUT_DIR / f"{model}_{day}.parquet"


def window_bounds(day: str, start_time: str, end_time: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{day} {normalize_clock(start_time)}", tz="Europe/Berlin")
    end = pd.Timestamp(f"{day} {normalize_clock(end_time)}", tz="Europe/Berlin")
    if end <= start:
        raise ValueError("end-time must be after start-time")
    return start, end


def sample_real_mid(day: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dataset = ds.dataset(EVENT_FLOW_PATH, format="parquet")
    df = load_day(
        dataset,
        pd.Timestamp(day).date(),
        ["ts", "date", "p_mid"],
    ).copy()
    if "ts" not in getattr(df.index, "names", []):
        if "ts" in df.columns:
            df = df.sort_values("ts").set_index("ts")
        else:
            df = df.reset_index().sort_values("ts").set_index("ts")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.to_datetime(df.index)
    frame = df[["p_mid"]].rename(columns={"p_mid": "mid_price"})
    idx = pd.date_range(start, end, freq="1s", inclusive="left")
    if frame.empty:
        raise ValueError(f"No real-market rows found for {day}")
    if frame.index.min() > start:
        prior = frame.iloc[[0]].copy()
        prior.index = pd.DatetimeIndex([start])
        frame = pd.concat([prior, frame])
    sampled = frame.reindex(frame.index.union(idx)).sort_index().ffill().reindex(idx)
    sampled.index.name = "timestamp"
    return sampled[["mid_price"]]


def build_simulator(model: str):
    common = calibrate_common(str(EVENT_FLOW_PATH), raw_dir=str(RAW_DIR), level=1, min_obs=50)
    if model == "qru":
        cal = calibrate_qru(str(EVENT_FLOW_PATH), raw_dir=str(RAW_DIR), level=1, min_obs=50, common=common)
        return QRUSimulator(cal), common
    if model == "qr":
        cal = calibrate_qr(str(EVENT_FLOW_PATH), raw_dir=str(RAW_DIR), level=1, min_obs=50, common=common)
        return QRSimulator(cal), common
    if model == "ftqr":
        cal = calibrate_ftqr(str(EVENT_FLOW_PATH), raw_dir=str(RAW_DIR), level=1, min_obs=50, common=common)
        return FTQRSimulator(cal), common
    if model == "saqr":
        cal = calibrate_saqr(
            str(EVENT_FLOW_PATH),
            raw_dir=str(RAW_DIR),
            level=1,
            min_obs=50,
            smoothing_alpha=25.0,
            common=common,
        )
        return SAQRSimulator(cal), common
    raise ValueError(f"Unsupported model: {model}")


def simulate_mid_window(
    model: str,
    day: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    simulator, common = build_simulator(model)
    real_start = sample_real_mid(day, start, start + pd.Timedelta(seconds=1)).iloc[0]["mid_price"]
    duration_seconds = float((end - start).total_seconds())
    day_seed = BASE_SEED + int(pd.Timestamp(day).strftime("%d"))
    rng = np.random.default_rng(day_seed + 1000 * ["qru", "qr", "ftqr", "saqr"].index(model))
    queue_size = int(common.sample_reset_queue(rng))
    elapsed = 0.0
    price_shift = 0
    rows = [{"timestamp": start, "mid_price": float(real_start)}]
    steps = 0
    max_steps = 500_000

    while elapsed < duration_seconds and steps < max_steps:
        n, rates = simulator._lookup_rates(queue_size)
        lambda_total = float(rates.sum())
        if lambda_total <= 0:
            break
        dt = float(rng.exponential(1.0 / lambda_total))
        if elapsed + dt > duration_seconds:
            break
        eta, size = simulator._sample_event(n, queue_size, rng)
        queue_after = simulator._apply_event(queue_size, eta, size)
        if queue_after == 0:
            price_shift += int(rng.choice([-1, 1]))
            queue_size = int(simulator.calibration.sample_reset_queue(rng))
        else:
            queue_size = int(queue_after)
        elapsed += dt
        rows.append(
            {
                "timestamp": start + pd.to_timedelta(elapsed, unit="s"),
                "mid_price": float(real_start + 0.01 * price_shift),
            }
        )
        steps += 1

    df = pd.DataFrame(rows).set_index("timestamp")
    idx = pd.date_range(start, end, freq="1s", inclusive="left")
    df = df.reindex(df.index.union(idx)).sort_index().ffill().reindex(idx)
    df.index.name = "timestamp"
    return df[["mid_price"]]


def main() -> None:
    args = parse_args()
    out = output_path(args.model, args.day, args.output)
    if out.exists() and not args.force:
        print(f"Output already exists: {out}")
        return

    start, end = window_bounds(args.day, args.start_time, args.end_time)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "real":
        df = sample_real_mid(args.day, start, end)
    else:
        df = simulate_mid_window(args.model, args.day, start, end)

    df.to_parquet(out)
    print(f"Wrote {len(df):,} rows to {out}")
    del df
    gc.collect()


if __name__ == "__main__":
    main()
