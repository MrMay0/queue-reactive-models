from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from models.common import calibrate_common
from models.ftqr import FTQRSimulator, calibrate_ftqr
from models.qr import QRSimulator, calibrate_qr
from models.qru import QRUSimulator, calibrate_qru
from models.saqr import SAQRSimulator, calibrate_saqr
from src.features.qr_empirical import load_day, trading_days

TRADING_SECONDS = 9 * 3600
TZ = "Europe/Berlin"
BASE_SEED = 20260331


@dataclass
class Thresholds:
    max_runtime_seconds: float
    max_rss_mb: float


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def calendar_bounds(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{day} 09:00:00", tz=TZ)
    end = pd.Timestamp(f"{day} 18:00:00", tz=TZ)
    return start, end


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
    second_index = pd.date_range(start, end, freq="1s", inclusive="left")
    frame = df[["p_mid"]].rename(columns={"p_mid": "mid_price"})
    if frame.empty:
        raise ValueError(f"Empty real-market day: {day}")
    if frame.index.min() > start:
        init_row = frame.iloc[[0]].copy()
        init_row.index = pd.DatetimeIndex([start])
        frame = pd.concat([init_row, frame])

    sampled = frame.reindex(frame.index.union(second_index)).sort_index().ffill().reindex(second_index)
    sampled = sampled.reset_index().rename(columns={"index": "timestamp"})
    sampled["timestamp"] = pd.to_datetime(sampled["timestamp"])
    return sampled[["timestamp", "mid_price"]]


def simulate_mid_1s(
    simulator,
    start_mid: float,
    q0: int,
    seed: int,
    thresholds: Thresholds,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seconds = np.arange(TRADING_SECONDS, dtype=np.int32)
    mid_prices = np.empty(TRADING_SECONDS, dtype=np.float64)

    current_mid = float(start_mid)
    queue_size = int(q0)
    elapsed = 0.0
    next_second = 0
    started = time.perf_counter()

    while next_second < TRADING_SECONDS:
        if time.perf_counter() - started > thresholds.max_runtime_seconds:
            raise TimeoutError(f"runtime exceeded {thresholds.max_runtime_seconds:.1f}s")
        if peak_rss_mb() > thresholds.max_rss_mb:
            raise MemoryError(f"peak RSS exceeded {thresholds.max_rss_mb:.1f} MB")

        n, rates = simulator._lookup_rates(queue_size)
        lambda_total = float(rates.sum())
        if lambda_total <= 0:
            mid_prices[next_second:] = current_mid
            break

        delta_t = float(rng.exponential(1.0 / lambda_total))
        event_time = elapsed + delta_t

        fill_until = min(TRADING_SECONDS, int(np.floor(event_time)) + 1)
        if event_time < 0:
            fill_until = next_second
        if fill_until > next_second:
            mid_prices[next_second:fill_until] = current_mid
            next_second = fill_until

        if event_time >= TRADING_SECONDS:
            break

        eta, size = simulator._sample_event(n, queue_size, rng)
        queue_after = simulator._apply_event(queue_size, eta, size)
        if queue_after == 0:
            current_mid += 0.01 * int(rng.choice([-1, 1]))
            queue_size = simulator.calibration.sample_reset_queue(rng)
        else:
            queue_size = queue_after
        elapsed = event_time

    if next_second < TRADING_SECONDS:
        mid_prices[next_second:] = current_mid

    return pd.DataFrame({"offset_s": seconds, "mid_price": mid_prices})


def save_mid_cache(df: pd.DataFrame, out_path: Path, day) -> None:
    start, _ = calendar_bounds(day)
    out = df.copy()
    out["timestamp"] = start + pd.to_timedelta(out["offset_s"], unit="s")
    out = out[["timestamp", "mid_price"]]
    out.to_parquet(out_path, index=False)


def load_open_mid(dataset: ds.Dataset, day) -> float:
    cols = ["ts", "date", "p_mid"]
    df = load_day(dataset, day, cols)
    if "ts" not in getattr(df.index, "names", []):
        if "ts" in df.columns:
            df = df.sort_values("ts").set_index("ts")
        else:
            df = df.reset_index().sort_values("ts").set_index("ts")
    df = df.sort_index()
    if df.empty:
        raise ValueError(f"No open state for {day}")
    return float(df.iloc[0]["p_mid"])


def day_cache_dir(cache_root: Path, day) -> Path:
    out = cache_root / str(day)
    out.mkdir(parents=True, exist_ok=True)
    return out


def log_result(log_rows: list[dict], **kwargs) -> None:
    kwargs["logged_at"] = pd.Timestamp.utcnow().isoformat()
    log_rows.append(kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 1-second price cache for real market and QR models.")
    parser.add_argument("--event-flow", default="data/processed/FGBL_event_flow.parquet")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cache-dir", default="data/processed/price_dynamics_cache")
    parser.add_argument("--days", nargs="*", help="Optional list of YYYY-MM-DD trading days to build.")
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float, default=30.0)
    parser.add_argument("--max-rss-mb", type=float, default=4096.0)
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--level", type=int, default=1)
    args = parser.parse_args()

    event_flow_path = Path(args.event_flow)
    raw_dir = Path(args.raw_dir)
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    log_path = cache_root / "build_log.jsonl"

    all_days = trading_days(str(raw_dir))
    if args.days:
        keep = {pd.Timestamp(day).date() for day in args.days}
        days = [day for day in all_days if day in keep]
    else:
        days = all_days
    if args.limit_days is not None:
        days = days[: args.limit_days]

    dataset = ds.dataset(event_flow_path, format="parquet")
    thresholds = Thresholds(args.max_runtime_seconds, args.max_rss_mb)

    common = calibrate_common(
        event_flow_path=str(event_flow_path),
        raw_dir=str(raw_dir),
        level=args.level,
        min_obs=args.min_obs,
    )
    simulators = {
        "QRU": QRUSimulator(calibrate_qru(str(event_flow_path), raw_dir=str(raw_dir), level=args.level, min_obs=args.min_obs, common=common)),
        "QR": QRSimulator(calibrate_qr(str(event_flow_path), raw_dir=str(raw_dir), level=args.level, min_obs=args.min_obs, common=common)),
        "FTQR": FTQRSimulator(calibrate_ftqr(str(event_flow_path), raw_dir=str(raw_dir), level=args.level, min_obs=args.min_obs, common=common)),
        "SAQR": SAQRSimulator(calibrate_saqr(str(event_flow_path), raw_dir=str(raw_dir), level=args.level, min_obs=args.min_obs, smoothing_alpha=25.0, common=common)),
    }

    log_rows: list[dict] = []
    failed = False

    for day_idx, day in enumerate(days):
        q0 = common.sample_reset_queue(np.random.default_rng(BASE_SEED + day_idx))
        open_mid = load_open_mid(dataset, day)
        out_dir = day_cache_dir(cache_root, day)

        real_path = out_dir / "real.parquet"
        if args.overwrite or not real_path.exists():
            started = time.perf_counter()
            try:
                real_df = sample_real_mid_1s(dataset, day)
                real_df.to_parquet(real_path, index=False)
                log_result(
                    log_rows,
                    day=str(day),
                    model="Real",
                    status="ok",
                    runtime_seconds=time.perf_counter() - started,
                    peak_rss_mb=peak_rss_mb(),
                    rows=len(real_df),
                )
            except Exception as exc:
                failed = True
                log_result(
                    log_rows,
                    day=str(day),
                    model="Real",
                    status="failed",
                    runtime_seconds=time.perf_counter() - started,
                    peak_rss_mb=peak_rss_mb(),
                    error=str(exc),
                )
                break
            finally:
                del real_df
                gc.collect()

        for model_idx, (model_name, simulator) in enumerate(simulators.items()):
            out_path = out_dir / f"{model_name}.parquet"
            if out_path.exists() and not args.overwrite:
                log_result(
                    log_rows,
                    day=str(day),
                    model=model_name,
                    status="cached",
                    runtime_seconds=0.0,
                    peak_rss_mb=peak_rss_mb(),
                    rows=pd.read_parquet(out_path, columns=["mid_price"]).shape[0],
                )
                continue

            started = time.perf_counter()
            try:
                sim_df = simulate_mid_1s(
                    simulator=simulator,
                    start_mid=open_mid,
                    q0=q0,
                    seed=BASE_SEED + 10_000 * (model_idx + 1) + day_idx,
                    thresholds=thresholds,
                )
                save_mid_cache(sim_df, out_path, day)
                log_result(
                    log_rows,
                    day=str(day),
                    model=model_name,
                    status="ok",
                    runtime_seconds=time.perf_counter() - started,
                    peak_rss_mb=peak_rss_mb(),
                    rows=len(sim_df),
                )
            except Exception as exc:
                failed = True
                log_result(
                    log_rows,
                    day=str(day),
                    model=model_name,
                    status="failed",
                    runtime_seconds=time.perf_counter() - started,
                    peak_rss_mb=peak_rss_mb(),
                    error=str(exc),
                )
                break
            finally:
                if "sim_df" in locals():
                    del sim_df
                gc.collect()

        with log_path.open("a", encoding="utf-8") as fh:
            for row in log_rows:
                fh.write(json.dumps(row) + "\n")
            log_rows.clear()

        if failed:
            break

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
