from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import gamma, ks_2samp, weibull_min

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common import calibrate_common
from models.ftqr import FTQRSimulator, calibrate_ftqr
from models.qr import QRSimulator, calibrate_qr
from models.qru import QRUSimulator, calibrate_qru
from models.saqr import SAQRSimulator, calibrate_saqr
from src.features.qr_empirical import load_day, trading_days


TZ = "Europe/Berlin"
FULL_TRADING_SECONDS = 9 * 3600
BASE_SEED = 20260331
PERIODS = {
    "09:00-18:00": ("09:00:00", "18:00:00"),
    "10:00-14:00": ("10:00:00", "14:00:00"),
    "15:00-18:00": ("15:00:00", "18:00:00"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stylized-fact artifacts for the final results notebook.")
    parser.add_argument("--event-flow", default=str(ROOT / "data/processed/FGBL_event_flow.parquet"))
    parser.add_argument("--raw-dir", default=str(ROOT / "data/raw"))
    parser.add_argument("--stats-days", nargs="*", default=None, help="Explicit list of YYYY-MM-DD days.")
    parser.add_argument("--models", nargs="+", default=["QRU", "QR", "FTQR"])
    parser.add_argument("--output-dir", default=str(ROOT / "data/processed/remote_results/stylized_facts"))
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Recompute artifacts even if output files already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the execution plan and exit without computing.")
    parser.add_argument("--test-days", type=int, default=None, help="Keep only the first N selected days after filtering.")
    parser.add_argument("--test-models", nargs="*", default=None, help="Override --models for a tiny smoke test.")
    parser.add_argument(
        "--max-sim-seconds",
        type=float,
        default=FULL_TRADING_SECONDS,
        help="Simulate only the first N seconds of the session. Useful for smoke tests.",
    )
    return parser.parse_args()


class Logger:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.stage_times: dict[str, float] = {}
        self.day_times: dict[str, float] = {}
        self.model_times: dict[str, float] = {}

    def _elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = float(seconds)
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, sec = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m{sec:04.1f}s"
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"

    def log(self, message: str) -> None:
        now = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{now}] +{self._fmt(self._elapsed())} | {message}", flush=True)

    @contextmanager
    def stage(self, name: str):
        self.log(f"START {name}")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stage_times[name] = self.stage_times.get(name, 0.0) + elapsed
            self.log(f"END   {name} ({self._fmt(elapsed)})")

    def record_day(self, day: str, seconds: float) -> None:
        self.day_times[day] = self.day_times.get(day, 0.0) + seconds

    def record_model(self, model: str, seconds: float) -> None:
        self.model_times[model] = self.model_times.get(model, 0.0) + seconds


def calendar_start(day) -> pd.Timestamp:
    return pd.Timestamp(f"{day} 09:00:00", tz=TZ)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_paths(out_dir: Path) -> dict[str, Path]:
    sim_dir = ensure_dir(out_dir / "simulations")
    return {
        "root": out_dir,
        "sim": sim_dir,
        "logs": ensure_dir(out_dir / "logs"),
    }


def should_skip(path: Path, force: bool) -> bool:
    return path.exists() and not force


def write_json(path: Path, payload: dict, logger: Logger) -> None:
    path.write_text(json.dumps(payload, indent=2))
    logger.log(f"WROTE {path}")


def write_df(df: pd.DataFrame, path: Path, logger: Logger) -> None:
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")
    logger.log(f"WROTE {path}")


def select_days(raw_dir: str, explicit_days: list[str] | None, test_days: int | None) -> list:
    all_days = trading_days(raw_dir)
    if explicit_days:
        keep = {pd.Timestamp(day).date() for day in explicit_days}
        days = [day for day in all_days if day in keep]
    else:
        days = all_days[:5]
    if test_days is not None:
        days = days[:test_days]
    return days


def descriptive_stats_real(dataset: ds.Dataset, all_days: list, logger: Logger) -> pd.DataFrame:
    event_counts = defaultdict(int)
    size_sums = defaultdict(float)
    ait_sums = defaultdict(float)
    ait_counts = defaultdict(int)
    cols = ["date", "side", "level", "eta", "size", "delta_t"]
    for idx, day in enumerate(all_days, start=1):
        logger.log(f"Descriptive stats day {idx}/{len(all_days)}: {day}")
        df = load_day(dataset, day, cols)
        grouped = df.groupby(["side", "level", "eta"]).agg(count=("size", "count"), size_sum=("size", "sum"))
        for (side, level, eta), row in grouped.iterrows():
            event_counts[(side, level, eta)] += int(row["count"])
            size_sums[(side, level)] += float(row["size_sum"])
        dt = df[df["delta_t"].notna() & (df["delta_t"] > 0)].groupby(["side", "level"])["delta_t"].agg(["sum", "count"])
        for (side, level), row in dt.iterrows():
            ait_sums[(side, level)] += float(row["sum"])
            ait_counts[(side, level)] += int(row["count"])
        del df, grouped, dt
        gc.collect()

    rows = []
    for side in ["bid", "ask"]:
        for level in range(1, 6):
            total_events = sum(event_counts.get((side, level, eta), 0) for eta in ["L", "C", "M"])
            aes = size_sums.get((side, level), 0.0) / total_events if total_events else np.nan
            ait = ait_sums.get((side, level), 0.0) / ait_counts.get((side, level), 0) if ait_counts.get((side, level), 0) else np.nan
            rows.append(
                {
                    "Side": side,
                    "Level": level,
                    "Limit events": event_counts.get((side, level, "L"), 0),
                    "Cancel events": event_counts.get((side, level, "C"), 0),
                    "Market events": event_counts.get((side, level, "M"), 0),
                    "AES": aes,
                    "AIT_seconds": ait,
                }
            )
    return pd.DataFrame(rows)


def simulate_day_events(simulator, day, seed: int, q0: int, max_sim_seconds: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lookup_rates = simulator._lookup_rates
    sample_event = simulator._sample_event
    apply_event = simulator._apply_event
    sample_reset = simulator.calibration.sample_reset_queue

    start = calendar_start(day)
    elapsed = 0.0
    queue_size = int(q0)
    rows = []
    append = rows.append
    horizon = min(float(max_sim_seconds), FULL_TRADING_SECONDS)

    while elapsed < horizon:
        n, rates = lookup_rates(queue_size)
        lambda_total = float(rates.sum())
        if lambda_total <= 0:
            break
        dt = float(rng.exponential(1.0 / lambda_total))
        if elapsed + dt > horizon:
            break
        eta, size = sample_event(n, queue_size, rng)
        append(
            {
                "timestamp": start + pd.to_timedelta(elapsed + dt, unit="s"),
                "eta_raw": eta,
                "eta": "L" if eta == "L" else ("M" if eta in {"M", "M_all"} else "C"),
                "size": int(size),
                "q_before": int(queue_size),
                "delta_t": dt,
            }
        )
        queue_after = apply_event(queue_size, eta, size)
        queue_size = int(sample_reset(rng)) if queue_after == 0 else int(queue_after)
        elapsed += dt
    return pd.DataFrame(rows)


def real_day_level1(dataset: ds.Dataset, day, level: int) -> pd.DataFrame:
    cols = ["ts", "date", "side", "level", "eta", "size", "delta_t", "q_before"]
    df = load_day(dataset, day, cols).copy()
    return df[df["level"] == level].copy()


def ensure_real_day_file(dataset: ds.Dataset, day, level: int, out_path: Path, force: bool, logger: Logger) -> None:
    if should_skip(out_path, force):
        logger.log(f"SKIP real day already cached: {out_path}")
        return
    with logger.stage(f"real-day {day}"):
        df = real_day_level1(dataset, day, level)
        df = df.assign(timestamp=pd.to_datetime(df.index), day=str(day))
        write_df(df[["timestamp", "day", "eta", "size", "delta_t", "q_before"]], out_path, logger)
        del df
        gc.collect()


def ensure_simulated_day_file(
    model: str,
    simulator,
    day,
    seed: int,
    q0: int,
    max_sim_seconds: float,
    out_path: Path,
    force: bool,
    logger: Logger,
) -> None:
    if should_skip(out_path, force):
        logger.log(f"SKIP simulation already cached: {out_path}")
        return
    with logger.stage(f"simulate {model} {day}"):
        t0 = time.perf_counter()
        df = simulate_day_events(simulator, day, seed=seed, q0=q0, max_sim_seconds=max_sim_seconds)
        df["day"] = str(day)
        write_df(df, out_path, logger)
        elapsed = time.perf_counter() - t0
        logger.record_model(model, elapsed)
        logger.record_day(str(day), elapsed)
        del df
        gc.collect()


def load_saved_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported cached frame format: {path}")


def size_distribution_table(sim_paths: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    for model, paths in sim_paths.items():
        counts = defaultdict(int)
        total = 0
        for path in paths:
            df = load_saved_frame(path)
            grouped = df.groupby("size").size()
            for size, count in grouped.items():
                counts[int(size)] += int(count)
                total += int(count)
        for size in sorted(counts):
            rows.append({"size": size, "count": counts[size], "model": model, "probability": counts[size] / total if total else np.nan})
    return pd.DataFrame(rows)


def ccdf_tail_table(sim_paths: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tail_rows = []
    fit_rows = []
    for model, paths in sim_paths.items():
        series = []
        for path in paths:
            df = load_saved_frame(path)
            series.append(df["size"].to_numpy(dtype=float))
        if not series:
            continue
        sizes = np.sort(np.concatenate(series))
        if len(sizes) == 0:
            continue
        threshold = np.quantile(sizes, 0.9)
        tail = sizes[sizes >= threshold]
        unique_sizes = np.unique(tail)
        ccdf = np.array([(tail >= s).mean() for s in unique_sizes], dtype=float)
        tail_rows.append(pd.DataFrame({"model": model, "size": unique_sizes, "ccdf": ccdf}))
        valid = (unique_sizes > 0) & (ccdf > 0)
        if valid.sum() >= 3:
            x = np.log(unique_sizes[valid])
            y = np.log(ccdf[valid])
            slope, intercept = np.polyfit(x, y, 1)
            y_hat = slope * x + intercept
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            slope, r2 = np.nan, np.nan
        fit_rows.append({"model": model, "tail_threshold": threshold, "tail_slope": slope, "tail_r2": r2, "tail_n": len(tail)})
    return pd.concat(tail_rows, ignore_index=True), pd.DataFrame(fit_rows)


def traded_volume_outputs(sim_paths: dict[str, list[Path]], stats_days: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_windows = []
    summary_rows = []
    sim_models = [model for model in sim_paths if model != "Real"]
    for day in stats_days:
        day_frames = {}
        for model, paths in sim_paths.items():
            path = next(path for path in paths if path.stem == str(day))
            df = load_saved_frame(path)
            trades = df[df["eta"] == "M"].copy()
            if trades.empty:
                day_frames[model] = pd.DataFrame(columns=["window_start", "traded_volume"])
                continue
            trades["window_start"] = pd.to_datetime(trades["timestamp"]).dt.floor("10min")
            agg = trades.groupby("window_start", as_index=False)["size"].sum().rename(columns={"size": "traded_volume"})
            agg["model"] = model
            agg["day"] = str(day)
            all_windows.append(agg)
            day_frames[model] = agg
        real_day = day_frames["Real"]
        for period_name, (start_time, end_time) in PERIODS.items():
            real_period = real_day.set_index("window_start").between_time(start_time, end_time, inclusive="left")["traded_volume"].sum()
            for model in sim_models:
                sim_period = day_frames[model].set_index("window_start").between_time(start_time, end_time, inclusive="left")["traded_volume"].sum()
                summary_rows.append(
                    {
                        "day": str(day),
                        "period": period_name,
                        "model": model,
                        "real_traded_volume": real_period,
                        "sim_traded_volume": sim_period,
                        "relative_difference_pct": 100.0 * (sim_period - real_period) / real_period if real_period else np.nan,
                        "quadratic_error": (sim_period - real_period) ** 2,
                    }
                )
    return pd.concat(all_windows, ignore_index=True), pd.DataFrame(summary_rows)


def interarrival_outputs(sim_paths: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist_rows = []
    fit_rows = []
    sample_rows = []
    bins = np.logspace(-4, 2, 60)
    for model, paths in sim_paths.items():
        parts = []
        for path in paths:
            df = load_saved_frame(path)
            trades = df[df["eta"] == "M"].copy().sort_values("timestamp")
            if trades.empty:
                continue
            if model == "Real":
                trades["interarrival"] = pd.to_datetime(trades["timestamp"]).diff().dt.total_seconds()
            else:
                trades["interarrival"] = trades["delta_t"]
            x = trades["interarrival"].dropna()
            x = x[x > 0]
            if not x.empty:
                parts.append(x.to_numpy(dtype=float))
        if not parts:
            continue
        x = np.concatenate(parts)
        hist, edges = np.histogram(x, bins=bins, density=True)
        hist_rows.append(pd.DataFrame({"model": model, "bin_left": edges[:-1], "bin_right": edges[1:], "density": hist}))
        c, loc, scale = weibull_min.fit(x, floc=0)
        fit_grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        sample_rows.append(pd.DataFrame({"model": model, "interarrival": fit_grid, "weibull_pdf": weibull_min.pdf(fit_grid, c, loc=loc, scale=scale)}))
        fit_rows.append({"model": model, "shape_k": c, "scale_lambda": scale, "n_obs": len(x), "mean_interarrival": x.mean()})
    return pd.concat(hist_rows, ignore_index=True), pd.DataFrame(fit_rows), pd.concat(sample_rows, ignore_index=True)


def queue_size_outputs(sim_paths: dict[str, list[Path]], stats_days: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist_rows = []
    ks_rows = []
    real_q = []
    sim_models = [model for model in sim_paths if model != "Real"]
    for model, paths in sim_paths.items():
        counts = defaultdict(int)
        total = 0
        for path in paths:
            df = load_saved_frame(path)
            q = df["q_before"].dropna()
            q = q[q > 0]
            if model == "Real":
                real_q.append(q.to_numpy(dtype=float))
            grouped = q.value_counts().sort_index()
            for queue_size, count in grouped.items():
                counts[int(queue_size)] += int(count)
                total += int(count)
        for queue_size in sorted(counts):
            dist_rows.append(
                {
                    "queue_size": queue_size,
                    "count": counts[queue_size],
                    "probability": counts[queue_size] / total if total else np.nan,
                    "model": model,
                }
            )

    real_q_all = np.concatenate(real_q)
    a, loc, scale = gamma.fit(real_q_all, floc=0)
    grid = np.linspace(real_q_all.min(), np.quantile(real_q_all, 0.99), 200)
    gamma_fit_df = pd.DataFrame({"queue_size": grid, "gamma_pdf": gamma.pdf(grid, a, loc=loc, scale=scale), "shape": a, "scale": scale})

    for day in stats_days:
        real_path = next(path for path in sim_paths["Real"] if path.stem == str(day))
        real_day = load_saved_frame(real_path)["q_before"].dropna()
        real_day = real_day[real_day > 0]
        if real_day.empty:
            continue
        for model in sim_models:
            sim_path = next(path for path in sim_paths[model] if path.stem == str(day))
            sim_day = load_saved_frame(sim_path)["q_before"].dropna()
            sim_day = sim_day[sim_day > 0]
            if sim_day.empty:
                continue
            ks_rows.append({"day": str(day), "model": model, "ks_stat": ks_2samp(real_day.to_numpy(), sim_day.to_numpy()).statistic})

    ks_daily = pd.DataFrame(ks_rows)
    ks_summary = ks_daily.groupby("model")["ks_stat"].describe().reset_index()
    return pd.DataFrame(dist_rows), gamma_fit_df, ks_daily, ks_summary


def save_timing_summary(out_dir: Path, logger: Logger, extra: dict) -> None:
    payload = {
        "total_runtime_seconds": time.perf_counter() - logger.t0,
        "stage_runtime_seconds": logger.stage_times,
        "model_runtime_seconds": logger.model_times,
        "day_runtime_seconds": logger.day_times,
        **extra,
    }
    (out_dir / "logs" / "timing_summary.json").write_text(json.dumps(payload, indent=2))
    logger.log(f"WROTE {out_dir / 'logs' / 'timing_summary.json'}")


def memoize_block(fn):
    cache = {}

    def wrapper():
        if "value" not in cache:
            cache["value"] = fn()
        return cache["value"]

    return wrapper


def main() -> int:
    args = parse_args()
    logger = Logger()
    event_flow_path = str(Path(args.event_flow).resolve())
    raw_dir = str(Path(args.raw_dir).resolve())
    out_dir = Path(args.output_dir).resolve()
    paths = output_paths(out_dir)

    models = args.test_models if args.test_models else args.models
    stats_days = select_days(raw_dir, args.stats_days, args.test_days)

    plan = {
        "event_flow_path": event_flow_path,
        "raw_dir": raw_dir,
        "output_dir": str(out_dir),
        "models": models,
        "stats_days": [str(day) for day in stats_days],
        "level": args.level,
        "min_obs": args.min_obs,
        "max_sim_seconds": float(args.max_sim_seconds),
        "force": bool(args.force),
        "dry_run": bool(args.dry_run),
    }
    logger.log(f"Execution plan: {json.dumps(plan)}")
    write_json(paths["logs"] / "execution_plan.json", plan, logger)

    if args.dry_run:
        logger.log("Dry run requested; exiting before computation.")
        return 0

    with logger.stage("loading event-flow dataset"):
        dataset = ds.dataset(event_flow_path, format="parquet")

    with logger.stage("selecting days"):
        all_days = trading_days(raw_dir)
        logger.log(f"Selected {len(stats_days)} day(s): {', '.join(map(str, stats_days))}")
        desc_days = stats_days if args.test_days is not None else all_days

    desc_path = out_dir / "descriptive_stats_level.csv"
    if should_skip(desc_path, args.force):
        logger.log(f"SKIP descriptive stats already exist: {desc_path}")
    else:
        with logger.stage("descriptive stats computation"):
            desc_df = descriptive_stats_real(dataset, desc_days, logger)
            write_df(desc_df, desc_path, logger)

    need_simulations = False
    sim_paths: dict[str, list[Path]] = {}
    for model in ["Real", *models]:
        model_dir = ensure_dir(paths["sim"] / model.lower())
        sim_paths[model] = [model_dir / f"{day}.parquet" for day in stats_days]
        if any(not should_skip(path, args.force) for path in sim_paths[model]):
            need_simulations = True

    common = None
    simulators = {}
    if need_simulations or args.force:
        with logger.stage("common calibration"):
            common = calibrate_common(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs)

        model_builders = {
            "QRU": lambda: QRUSimulator(calibrate_qru(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs, common=common)),
            "QR": lambda: QRSimulator(calibrate_qr(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs, common=common)),
            "FTQR": lambda: FTQRSimulator(calibrate_ftqr(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs, common=common)),
            "SAQR": lambda: SAQRSimulator(calibrate_saqr(event_flow_path, raw_dir=raw_dir, level=args.level, min_obs=args.min_obs, smoothing_alpha=25.0, common=common)),
        }
        for model in models:
            with logger.stage(f"calibration {model}"):
                simulators[model] = model_builders[model]()

    for idx, day in enumerate(stats_days):
        ensure_real_day_file(dataset, day, args.level, sim_paths["Real"][idx], args.force, logger)

    if common is not None:
        for model_idx, model in enumerate(models):
            for day_idx, day in enumerate(stats_days):
                q0 = common.sample_reset_queue(np.random.default_rng(BASE_SEED + day_idx))
                ensure_simulated_day_file(
                    model=model,
                    simulator=simulators[model],
                    day=day,
                    seed=BASE_SEED + 10_000 * (model_idx + 1) + day_idx,
                    q0=q0,
                    max_sim_seconds=args.max_sim_seconds,
                    out_path=sim_paths[model][day_idx],
                    force=args.force,
                    logger=logger,
                )

    tail_block = memoize_block(lambda: ccdf_tail_table(sim_paths))
    traded_block = memoize_block(lambda: traded_volume_outputs(sim_paths, stats_days))
    interarrival_block = memoize_block(lambda: interarrival_outputs(sim_paths))
    queue_block = memoize_block(lambda: queue_size_outputs(sim_paths, stats_days))

    output_builders = {
        "order_size_distribution.parquet": lambda: size_distribution_table(sim_paths),
        "order_size_tail_ccdf.parquet": lambda: tail_block()[0],
        "order_size_tail_fit.csv": lambda: tail_block()[1],
        "traded_volume_10min.parquet": lambda: traded_block()[0],
        "traded_volume_summary_daily.csv": lambda: traded_block()[1],
        "traded_volume_summary.csv": lambda: traded_block()[1]
        .groupby(["period", "model"], sort=False)
        .agg(
            avg_relative_difference_pct=("relative_difference_pct", "mean"),
            avg_quadratic_error=("quadratic_error", "mean"),
            mean_real_traded_volume=("real_traded_volume", "mean"),
            mean_sim_traded_volume=("sim_traded_volume", "mean"),
        )
        .reset_index(),
        "trade_interarrival_hist.parquet": lambda: interarrival_block()[0],
        "trade_weibull_fit.csv": lambda: interarrival_block()[1],
        "trade_weibull_pdf.parquet": lambda: interarrival_block()[2],
        "queue_size_distribution.parquet": lambda: queue_block()[0],
        "queue_size_gamma_fit.parquet": lambda: queue_block()[1],
        "queue_size_ks_daily.csv": lambda: queue_block()[2],
        "queue_size_ks_summary.csv": lambda: queue_block()[3],
    }

    for name, builder in output_builders.items():
        path = out_dir / name
        if should_skip(path, args.force):
            logger.log(f"SKIP output already exists: {path}")
            continue
        stage_name = name.replace("_", " ").replace(".parquet", "").replace(".csv", "")
        with logger.stage(stage_name):
            df = builder()
            write_df(df, path, logger)
            del df
            gc.collect()

    metadata_path = out_dir / "stylized_facts_metadata.json"
    if should_skip(metadata_path, args.force):
        logger.log(f"SKIP metadata already exists: {metadata_path}")
    else:
        metadata = {
            "stats_days": [str(day) for day in stats_days],
            "level": args.level,
            "models": ["Real"] + list(models),
            "notes": "All comparisons in this notebook use level-1 event flow and level-1 queue-reactive simulations on the selected subset of trading days.",
            "max_sim_seconds": float(args.max_sim_seconds),
        }
        write_json(metadata_path, metadata, logger)

    save_timing_summary(out_dir, logger, extra={"models": list(models), "stats_days": [str(day) for day in stats_days]})
    logger.log("Batch completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
