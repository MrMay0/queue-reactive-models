from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.features.qr_empirical import (
    _add_state_columns,
    _collapse_state_process,
    compute_global_aes_streaming,
    load_day,
    trading_days,
)
from src.features.qr_transforms import quantize_queue_sizes


@dataclass
class CalibrationResult:
    level: int
    aes_by_side: pd.Series
    aes_level: float
    intensity_df: pd.DataFrame
    ft_intensity_df: pd.DataFrame
    size_distribution: pd.DataFrame
    eta_size_distribution: pd.DataFrame
    joint_size_df: pd.DataFrame
    reset_distribution: pd.DataFrame

    def __post_init__(self) -> None:
        self.intensity_df = self.intensity_df.sort_values("n").reset_index(drop=True)
        self.ft_intensity_df = self.ft_intensity_df.sort_values("n").reset_index(drop=True)
        self.size_distribution = self.size_distribution.sort_values("size").reset_index(drop=True)
        self.eta_size_distribution = self.eta_size_distribution.sort_values(["eta", "size"]).reset_index(drop=True)
        self.joint_size_df = self.joint_size_df.sort_values(["n", "eta", "size"]).reset_index(drop=True)
        self.reset_distribution = self.reset_distribution.sort_values("queue_size").reset_index(drop=True)

        self.support_n = self.intensity_df["n"].to_numpy(dtype=int)
        self._intensity_lookup = self.intensity_df.set_index("n")
        self._ft_lookup = self.ft_intensity_df.set_index("n")
        self._unconditional_sizes = _df_to_distribution(self.size_distribution, "size", "prob")
        self._reset_sizes = _df_to_distribution(self.reset_distribution, "queue_size", "prob")
        self._eta_size_probs = {
            eta: _df_to_distribution(df_eta, "size", "prob")
            for eta, df_eta in self.eta_size_distribution.groupby("eta", sort=False)
        }
        self._joint_probs = {
            int(n): _joint_distribution(df_n)
            for n, df_n in self.joint_size_df.groupby("n", sort=False)
        }
        self._joint_prior = _joint_distribution(self.joint_size_df)
        self._joint_sampler_tables_cache: dict[float, dict[int, dict[str, np.ndarray | int]]] = {}

    def nearest_n(self, queue_size: int) -> int:
        queue_size = max(int(queue_size), 1)
        n = int(np.ceil(queue_size / self.aes_level))
        idx = int(np.abs(self.support_n - n).argmin())
        return int(self.support_n[idx])

    def sample_reset_queue(self, rng: np.random.Generator) -> int:
        sizes, probs = self._reset_sizes
        return int(rng.choice(sizes, p=probs))

    def unconditional_size_sample(self, rng: np.random.Generator) -> int:
        sizes, probs = self._unconditional_sizes
        return int(rng.choice(sizes, p=probs))

    def eta_size_sample(self, eta: str, rng: np.random.Generator) -> int:
        sizes, probs = self._eta_size_probs.get(eta, self._unconditional_sizes)
        return int(rng.choice(sizes, p=probs))

    def joint_sample(
        self,
        n: int,
        rng: np.random.Generator,
        smoothing_alpha: float = 25.0,
    ) -> tuple[str, int]:
        tables = self.build_joint_sampler_tables(smoothing_alpha)
        table = tables.get(int(n), tables[-1])
        idx = int(np.searchsorted(table["cdf"], rng.random(), side="left"))
        return str(table["etas"][idx]), int(table["sizes"][idx])

    def build_joint_sampler_tables(self, smoothing_alpha: float = 25.0) -> dict[int, dict[str, np.ndarray | int]]:
        alpha_key = float(smoothing_alpha)
        cached = self._joint_sampler_tables_cache.get(alpha_key)
        if cached is not None:
            return cached

        prior_pairs, prior_probs, prior_count = self._joint_prior
        prior_table = _build_joint_sampler_table(prior_pairs, prior_probs, prior_count)
        tables: dict[int, dict[str, np.ndarray | int]] = {-1: prior_table}
        for n in self.support_n:
            pairs_n, probs_n, count_n = self._joint_probs.get(int(n), self._joint_prior)
            pairs, probs = _blend_joint_distribution(
                pairs_n,
                probs_n,
                count_n,
                prior_pairs,
                prior_probs,
                smoothing_alpha=alpha_key,
            )
            tables[int(n)] = _build_joint_sampler_table(pairs, probs, count_n)

        self._joint_sampler_tables_cache[alpha_key] = tables
        return tables


@dataclass
class SimulationResult:
    path: pd.DataFrame
    summary: dict[str, float]


def _df_to_distribution(df: pd.DataFrame, value_col: str, prob_col: str) -> tuple[np.ndarray, np.ndarray]:
    if df.empty:
        return np.array([1]), np.array([1.0])
    values = df[value_col].to_numpy()
    probs = df[prob_col].to_numpy(dtype=float)
    probs = probs / probs.sum()
    return values, probs


def _joint_distribution(df: pd.DataFrame) -> tuple[list[tuple[str, int]], np.ndarray, int]:
    if df.empty:
        return [("L", 1)], np.array([1.0]), 0
    pairs = list(zip(df["eta"], df["size"].astype(int)))
    weight_col = "prob_joint" if "prob_joint" in df.columns else "count"
    weights = df[weight_col].to_numpy(dtype=float)
    weights = weights / weights.sum()
    total = int(df["count"].sum()) if "count" in df.columns else len(df)
    return pairs, weights, total


def _blend_joint_distribution(
    pairs_n: list[tuple[str, int]],
    probs_n: np.ndarray,
    count_n: int,
    prior_pairs: list[tuple[str, int]],
    prior_probs: np.ndarray,
    smoothing_alpha: float,
) -> tuple[list[tuple[str, int]], np.ndarray]:
    if count_n <= 0:
        return prior_pairs, prior_probs

    weight = count_n / (count_n + smoothing_alpha)
    prior_map = {pair: prob for pair, prob in zip(prior_pairs, prior_probs)}
    local_map = {pair: prob for pair, prob in zip(pairs_n, probs_n)}
    pairs = list(dict.fromkeys(list(pairs_n) + list(prior_pairs)))
    probs = np.array(
        [
            weight * local_map.get(pair, 0.0) + (1.0 - weight) * prior_map.get(pair, 0.0)
            for pair in pairs
        ],
        dtype=float,
    )
    total_prob = probs.sum()
    if total_prob <= 0:
        return prior_pairs, prior_probs
    probs /= total_prob
    return pairs, probs


def _build_joint_sampler_table(
    pairs: list[tuple[str, int]],
    probs: np.ndarray,
    count: int,
) -> dict[str, np.ndarray | int]:
    etas = np.array([eta for eta, _ in pairs], dtype=object)
    sizes = np.array([size for _, size in pairs], dtype=np.int64)
    cdf = np.cumsum(probs, dtype=float)
    cdf[-1] = 1.0
    return {
        "etas": etas,
        "sizes": sizes,
        "probs": np.array(probs, dtype=float),
        "cdf": cdf,
        "count": int(count),
    }


def _normalize_counter(counter: dict, names: list[str], count_name: str = "count") -> pd.DataFrame:
    if not counter:
        return pd.DataFrame(columns=[*names, count_name])
    rows = []
    for key, value in counter.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append((*key_tuple, value))
    return pd.DataFrame(rows, columns=[*names, count_name])


def _event_kind_series(df: pd.DataFrame) -> np.ndarray:
    eta = df["eta"].to_numpy()
    size = df["size"].to_numpy()
    q_before = df["q_before"].to_numpy()
    return np.select(
        [
            eta == "L",
            (eta == "C") & (size >= q_before),
            eta == "C",
            (eta == "M") & (size >= q_before),
            eta == "M",
        ],
        ["L", "C_all", "C", "M_all", "M"],
        default="L",
    )


def calibrate_common(
    event_flow_path: str,
    raw_dir: str = "data/raw",
    level: int = 1,
    min_obs: int = 50,
) -> CalibrationResult:
    dataset = ds.dataset(event_flow_path, format="parquet")
    days = trading_days(raw_dir)
    aes = compute_global_aes_streaming(event_flow_path, raw_dir=raw_dir)
    aes_by_side = aes.xs(level, level="level").sort_index()
    aes_level = float(aes_by_side.mean())

    state_n_obs = defaultdict(int)
    state_sum_dt = defaultdict(float)
    eta_counts = defaultdict(int)
    ft_counts = defaultdict(int)
    size_counts = defaultdict(int)
    eta_size_counts = defaultdict(int)
    joint_counts = defaultdict(int)
    reset_counts = defaultdict(int)

    cols = ["ts", "date", "side", "level", "eta", "q_before", "size", "delta_t"]

    for day in days:
        df_day = load_day(dataset, day, cols)
        df_day = quantize_queue_sizes(df_day, aes)
        df_day = df_day[df_day["level"] == level].copy()
        if df_day.empty:
            continue

        df_day = _add_state_columns(df_day, aes)
        state_df = _collapse_state_process(df_day)

        occ_mask = (
            state_df["delta_t_state"].notna()
            & (state_df["delta_t_state"] > 0)
            & (state_df["q_after_aes"] > 0)
        )
        occ = (
            state_df.loc[occ_mask]
            .groupby("q_after_aes")["delta_t_state"]
            .agg(n_obs="count", sum_dt="sum")
        )
        for n, row in occ.iterrows():
            state_n_obs[int(n)] += int(row["n_obs"])
            state_sum_dt[int(n)] += float(row["sum_dt"])

        event_df = df_day[df_day["q_before_aes"] > 0].copy()
        if event_df.empty:
            del df_day, state_df, occ
            gc.collect()
            continue

        eta_grouped = event_df.groupby(["q_before_aes", "eta"]).size()
        for key, count in eta_grouped.items():
            state_n, eta_name = key
            eta_counts[(int(state_n), str(eta_name))] += int(count)

        event_df["ft_type"] = _event_kind_series(event_df)
        ft_grouped = event_df.groupby(["q_before_aes", "ft_type"]).size()
        for key, count in ft_grouped.items():
            state_n, eta_name = key
            ft_counts[(int(state_n), str(eta_name))] += int(count)

        for size, count in event_df.groupby("size").size().items():
            size_counts[int(size)] += int(count)

        eta_size_grouped = event_df.groupby(["eta", "size"]).size()
        for key, count in eta_size_grouped.items():
            eta_name, size = key
            eta_size_counts[(str(eta_name), int(size))] += int(count)

        joint_grouped = event_df.groupby(["q_before_aes", "eta", "size"]).size()
        for key, count in joint_grouped.items():
            state_n, eta_name, size = key
            joint_counts[(int(state_n), str(eta_name), int(size))] += int(count)

        for queue_size, count in event_df.groupby("q_before").size().items():
            reset_counts[int(queue_size)] += int(count)

        del df_day, state_df, occ, event_df, eta_grouped, ft_grouped, eta_size_grouped, joint_grouped
        gc.collect()

    intensity_df = (
        _normalize_counter({n: state_n_obs[n] for n in state_n_obs}, ["n"], "n_obs")
        .merge(_normalize_counter({n: state_sum_dt[n] for n in state_sum_dt}, ["n"], "sum_dt"), on="n", how="outer")
        .fillna(0)
    )
    intensity_df = intensity_df[(intensity_df["n_obs"] >= min_obs) & (intensity_df["sum_dt"] > 0)].copy()
    intensity_df["n"] = intensity_df["n"].astype(int)
    intensity_df["ait"] = intensity_df["sum_dt"] / intensity_df["n_obs"]
    intensity_df["Lambda"] = intensity_df["n_obs"] / intensity_df["sum_dt"]
    t_map = intensity_df.set_index("n")["sum_dt"].to_dict()
    lambda_map = intensity_df.set_index("n")["Lambda"].to_dict()
    n_obs_map = intensity_df.set_index("n")["n_obs"].to_dict()
    eta_df = _normalize_counter(eta_counts, ["n", "eta"])
    if not eta_df.empty:
        eta_df = eta_df[eta_df["n"].isin(intensity_df["n"])].copy()
        eta_pivot = eta_df.pivot(index="n", columns="eta", values="count").fillna(0)
    else:
        eta_pivot = pd.DataFrame(index=intensity_df["n"])
    for eta_name in ["L", "C", "M"]:
        counts = eta_pivot.get(eta_name, pd.Series(0, index=intensity_df["n"]))
        intensity_df[f"count_{eta_name}"] = intensity_df["n"].map(counts.to_dict()).fillna(0).astype(int)
        intensity_df[f"lambda_{eta_name}"] = intensity_df["n"].map(
            {n: lambda_map[n] * counts.get(n, 0) / n_obs_map[n] for n in intensity_df["n"]}
        )

    ft_df = intensity_df[["n", "n_obs", "sum_dt", "ait", "Lambda"]].copy()
    ft_events_df = _normalize_counter(ft_counts, ["n", "eta"])
    if not ft_events_df.empty:
        ft_events_df = ft_events_df[ft_events_df["n"].isin(ft_df["n"])].copy()
        ft_pivot = ft_events_df.pivot(index="n", columns="eta", values="count").fillna(0)
    else:
        ft_pivot = pd.DataFrame(index=ft_df["n"])
    for eta_name in ["L", "C", "M", "C_all", "M_all"]:
        counts = ft_pivot.get(eta_name, pd.Series(0, index=ft_df["n"]))
        ft_df[f"count_{eta_name}"] = ft_df["n"].map(counts.to_dict()).fillna(0).astype(int)
        ft_df[f"lambda_{eta_name}"] = ft_df["n"].map(
            {n: lambda_map[n] * counts.get(n, 0) / n_obs_map[n] for n in ft_df["n"]}
        )
    ft_df["lambda_global"] = ft_df[
        ["lambda_L", "lambda_C", "lambda_M", "lambda_C_all", "lambda_M_all"]
    ].sum(axis=1)

    size_df = _normalize_counter(size_counts, ["size"])
    size_df["prob"] = size_df["count"] / size_df["count"].sum()

    eta_size_df = _normalize_counter(eta_size_counts, ["eta", "size"])
    if not eta_size_df.empty:
        eta_totals = eta_size_df.groupby("eta")["count"].transform("sum")
        eta_size_df["prob"] = eta_size_df["count"] / eta_totals
    else:
        eta_size_df["prob"] = pd.Series(dtype=float)

    joint_df = _normalize_counter(joint_counts, ["n", "eta", "size"])
    joint_df = joint_df[joint_df["n"].isin(intensity_df["n"])].copy()
    if not joint_df.empty:
        joint_df["total_n"] = joint_df.groupby("n")["count"].transform("sum")
        joint_df["prob_joint"] = joint_df["count"] / joint_df["total_n"]
        joint_df["sum_dt"] = joint_df["n"].map(t_map)
        joint_df["Lambda"] = joint_df["n"].map(lambda_map)
        joint_df["n_obs"] = joint_df["n"].map(n_obs_map)
        joint_df["lambda_eta_size"] = joint_df["Lambda"] * joint_df["count"] / joint_df["n_obs"]
    else:
        joint_df["total_n"] = pd.Series(dtype=float)
        joint_df["prob_joint"] = pd.Series(dtype=float)
        joint_df["sum_dt"] = pd.Series(dtype=float)
        joint_df["Lambda"] = pd.Series(dtype=float)
        joint_df["n_obs"] = pd.Series(dtype=float)
        joint_df["lambda_eta_size"] = pd.Series(dtype=float)

    reset_df = _normalize_counter(reset_counts, ["queue_size"])
    reset_df["prob"] = reset_df["count"] / reset_df["count"].sum()

    return CalibrationResult(
        level=level,
        aes_by_side=aes_by_side,
        aes_level=aes_level,
        intensity_df=intensity_df,
        ft_intensity_df=ft_df,
        size_distribution=size_df,
        eta_size_distribution=eta_size_df,
        joint_size_df=joint_df,
        reset_distribution=reset_df,
    )


class BaseQRSimulator(ABC):
    def __init__(self, calibration: CalibrationResult):
        self.calibration = calibration

    def _lookup_rates(self, queue_size: int) -> tuple[int, pd.Series]:
        n = self.calibration.nearest_n(queue_size)
        return n, self._rate_table(n)

    @abstractmethod
    def _rate_table(self, n: int) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def _sample_event(self, n: int, queue_size: int, rng: np.random.Generator) -> tuple[str, int]:
        raise NotImplementedError

    def _apply_event(self, queue_size: int, eta: str, size: int) -> int:
        if eta == "L":
            return queue_size + size
        if eta in {"C", "M"}:
            return max(0, queue_size - size)
        if eta in {"C_all", "M_all"}:
            return 0
        raise ValueError(f"Unknown event type {eta!r}")

    def simulate(
        self,
        steps: int = 50_000,
        seed: int = 0,
        q0: int | None = None,
    ) -> SimulationResult:
        rng = np.random.default_rng(seed)
        queue_size = int(q0) if q0 is not None else self.calibration.sample_reset_queue(rng)
        time = 0.0
        price = 0
        rows: list[dict[str, float | int | str | bool]] = []

        for step in range(steps):
            n, rates = self._lookup_rates(queue_size)
            lambda_total = float(rates.sum())
            if lambda_total <= 0:
                break

            delta_t = float(rng.exponential(1.0 / lambda_total))
            eta, size = self._sample_event(n, queue_size, rng)
            queue_after = self._apply_event(queue_size, eta, size)
            depleted = queue_after == 0
            if depleted:
                price += int(rng.choice([-1, 1]))
                next_queue = self.calibration.sample_reset_queue(rng)
            else:
                next_queue = queue_after

            time += delta_t
            rows.append(
                {
                    "step": step,
                    "time": time,
                    "delta_t": delta_t,
                    "n": n,
                    "queue_size": queue_size,
                    "eta": eta,
                    "size": int(size),
                    "queue_after": queue_after,
                    "depleted": depleted,
                    "price": price,
                }
            )
            queue_size = next_queue

        path = pd.DataFrame(rows)
        summary = {
            "steps": float(len(path)),
            "mean_queue": float(path["queue_size"].mean()) if not path.empty else np.nan,
            "depletion_rate": float(path["depleted"].mean()) if not path.empty else np.nan,
            "price_vol": float(path["price"].diff().fillna(0).std()) if not path.empty else np.nan,
            "mean_dt": float(path["delta_t"].mean()) if not path.empty else np.nan,
        }
        return SimulationResult(path=path, summary=summary)
