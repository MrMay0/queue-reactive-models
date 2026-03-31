from __future__ import annotations

import gc
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.features.qr_transforms import estimate_queue_intensities, quantize_queue_sizes


def trading_days(raw_dir: str = "data/raw") -> list[date]:
    days = []
    for path in sorted(Path(raw_dir).glob("*.parquet")):
        days.append(pd.Timestamp(path.stem.split("_")[-1]).date())
    return days


def load_day(dataset: ds.Dataset, day: date, columns: list[str]) -> pd.DataFrame:
    table = dataset.to_table(columns=columns, filter=ds.field("date") == day)
    df_day = table.to_pandas()
    if "ts" in df_day.columns:
        df_day = df_day.sort_values("ts").set_index("ts")
        df_day.index.name = "ts"
    return df_day


def compute_global_aes_streaming(
    event_flow_path: str,
    raw_dir: str = "data/raw",
) -> pd.Series:
    dataset = ds.dataset(event_flow_path, format="parquet")
    aes_sum = defaultdict(float)
    aes_count = defaultdict(int)

    for day in trading_days(raw_dir):
        df_day = load_day(dataset, day, ["date", "side", "level", "size"])
        grouped = df_day.groupby(["side", "level"])["size"].agg(["sum", "count"])
        for (side, level), row in grouped.iterrows():
            aes_sum[(side, level)] += float(row["sum"])
            aes_count[(side, level)] += int(row["count"])
        del df_day, grouped
        gc.collect()

    return (
        pd.Series({k: aes_sum[k] / aes_count[k] for k in aes_sum}, name="aes")
        .rename_axis(["side", "level"])
        .sort_index()
    )


def _add_state_columns(df: pd.DataFrame, aes: pd.Series) -> pd.DataFrame:
    df = df.copy()
    aes_col = (
        df[["side", "level"]]
        .merge(aes.reset_index(), on=["side", "level"], how="left")["aes"]
        .to_numpy()
    )
    q_after = df["q_before"].to_numpy().copy()
    add_mask = df["eta"].to_numpy() == "L"
    q_after[add_mask] = q_after[add_mask] + df.loc[add_mask, "size"].to_numpy()
    q_after[~add_mask] = np.maximum(0, q_after[~add_mask] - df.loc[~add_mask, "size"].to_numpy())
    df["q_after"] = q_after
    df["q_after_aes"] = np.ceil(df["q_after"].to_numpy() / aes_col).astype(int)
    df.loc[df["q_after"] <= 0, "q_after_aes"] = 0
    return df


def _collapse_state_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one post-event state per timestamp within each (side, level).
    """
    state_df = (
        df.reset_index()
        .sort_values(["side", "level", "ts"])
        .groupby(["side", "level", "ts"], sort=False)
        .tail(1)
        .sort_values(["side", "level", "ts"])
        .set_index("ts")
    )
    group_keys = [state_df["side"], state_df["level"]]
    state_df["delta_t_state"] = state_df.index.to_series().groupby(group_keys).diff().dt.total_seconds()
    return state_df


def estimate_queue_intensities_state_duration(
    df: pd.DataFrame,
    aes: pd.Series,
    queue_col: str = "q_before_aes",
    min_obs: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Estimate QR intensities using state durations rather than event rows.

    Occupancy:
      - states are post-event queue sizes after collapsing same-timestamp bursts
      - T(n) = sum duration spent in state n
      - N(n) = number of state observations with positive duration

    Event counts:
      - counted on original event flow conditioned on pre-event queue size
    """
    required = {"side", "level", "eta", "q_before", "q_before_aes", "size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for state-duration intensities: {sorted(missing)}")

    df = _add_state_columns(df, aes)
    state_df = _collapse_state_process(df)
    state_valid = state_df["delta_t_state"].notna() & (state_df["delta_t_state"] > 0) & (state_df["q_after_aes"] > 0)
    state_occ = state_df.loc[state_valid, ["q_after_aes", "delta_t_state"]].copy()
    if state_occ.empty:
        raise ValueError("No positive-duration states available for intensity estimation.")

    occ_stats = (
        state_occ.groupby("q_after_aes")["delta_t_state"]
        .agg(n_obs="count", sum_dt="sum", ait="mean")
        .reset_index()
        .rename(columns={"q_after_aes": "n"})
    )
    occ_stats = occ_stats[occ_stats["n_obs"] >= min_obs].copy()
    if occ_stats.empty:
        raise ValueError(f"No state bins with at least {min_obs} positive-duration observations.")
    occ_stats["Lambda"] = occ_stats["n_obs"] / occ_stats["sum_dt"]

    event_df = df[(df[queue_col].notna()) & (df[queue_col] > 0)].copy()
    eta_counts = (
        event_df.groupby([queue_col, "eta"]).size()
        .unstack(fill_value=0)
        .reindex(columns=["L", "C", "M"], fill_value=0)
    )
    eta_counts = eta_counts.reindex(occ_stats["n"], fill_value=0)
    lambda_base = occ_stats.set_index("n")["Lambda"]
    n_obs_map = occ_stats.set_index("n")["n_obs"]

    occ_stats["count_L"] = occ_stats["n"].map(eta_counts["L"].to_dict())
    occ_stats["count_C"] = occ_stats["n"].map(eta_counts["C"].to_dict())
    occ_stats["count_M"] = occ_stats["n"].map(eta_counts["M"].to_dict())
    occ_stats["lambda_L"] = occ_stats["n"].map((lambda_base * eta_counts["L"] / n_obs_map).to_dict())
    occ_stats["lambda_C"] = occ_stats["n"].map((lambda_base * eta_counts["C"] / n_obs_map).to_dict())
    occ_stats["lambda_M"] = occ_stats["n"].map((lambda_base * eta_counts["M"] / n_obs_map).to_dict())

    size_counts = (
        event_df.groupby([queue_col, "eta", "size"])
        .size()
        .reset_index(name="n_eta_size")
        .rename(columns={queue_col: "n", "size": "event_size"})
    )
    size_counts = size_counts[size_counts["n"].isin(occ_stats["n"])].copy()
    size_counts["n_obs"] = size_counts["n"].map(n_obs_map.to_dict())
    size_counts["Lambda"] = size_counts["n"].map(lambda_base.to_dict())
    size_counts["lambda_eta_size"] = size_counts["Lambda"] * size_counts["n_eta_size"] / size_counts["n_obs"]

    return occ_stats.sort_values("n").reset_index(drop=True), size_counts, state_df


def build_qr_intensity_tables(
    event_flow_path: str,
    output_path: str,
    size_output_path: str | None = "data/processed/qr_intensities_size.parquet",
    raw_dir: str = "data/raw",
    level: int = 1,
    min_obs: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build streaming QR intensity tables for the full month and time splits.

    Returns
    -------
    curves_df       aggregated intensity curves for full/half samples
    size_curves_df  aggregated size-dependent intensity curves
    diagnostics_df  daily diagnostics used for filtering/plotting
    aes             global AES per (side, level)
    """
    dataset = ds.dataset(event_flow_path, format="parquet")
    days = trading_days(raw_dir)
    aes = compute_global_aes_streaming(event_flow_path, raw_dir=raw_dir)

    midpoint = len(days) // 2
    first_half = set(days[:midpoint])
    second_half = set(days[midpoint:])

    curve_rows: list[pd.DataFrame] = []
    size_curve_rows: list[pd.DataFrame] = []
    diagnostics: list[dict] = []

    cols = ["ts", "date", "side", "level", "eta", "q_before", "size", "delta_t"]

    for day in days:
        df_day = load_day(dataset, day, cols)
        df_day = quantize_queue_sizes(df_day, aes)
        df_day = df_day[df_day["level"] == level].copy()

        total_level_events = len(df_day)
        dt_valid = (
            df_day["delta_t"].notna()
            & (df_day["delta_t"] > 0)
            & df_day["q_before_aes"].notna()
        )
        old_used_events = int(dt_valid.sum())
        old_discarded_pct = 100 * (1 - old_used_events / total_level_events) if total_level_events else 0.0

        old_intensity_df, old_dt_df = estimate_queue_intensities(df_day, min_obs=min_obs)
        old_intensity_df["estimator"] = "event_based"
        old_intensity_df["date"] = str(day)
        old_intensity_df["segment"] = "daily"
        curve_rows.append(old_intensity_df)

        intensity_df, size_counts, state_df = estimate_queue_intensities_state_duration(df_day, aes, min_obs=min_obs)
        intensity_df["estimator"] = "state_duration"
        intensity_df["date"] = str(day)
        intensity_df["segment"] = "daily"
        curve_rows.append(intensity_df)

        n_obs_map = intensity_df.set_index("n")["n_obs"].to_dict()
        lambda_map = intensity_df.set_index("n")["Lambda"].to_dict()
        size_counts = size_counts[size_counts["n"].isin(intensity_df["n"])].copy()
        size_counts["date"] = str(day)
        size_counts["segment"] = "daily"
        size_counts["estimator"] = "state_duration"
        size_curve_rows.append(size_counts)

        state_valid = (
            state_df["delta_t_state"].notna()
            & (state_df["delta_t_state"] > 0)
            & (state_df["q_after_aes"] > 0)
        )
        state_rows = int(state_valid.sum())
        state_retained_pct = 100.0 if state_rows > 0 else 0.0

        diagnostics.append(
            {
                "date": str(day),
                "events_level": total_level_events,
                "events_used_old": old_used_events,
                "discarded_pct_old": old_discarded_pct,
                "state_rows_used": state_rows,
                "state_retained_pct": state_retained_pct,
                "n_min": int(intensity_df["n"].min()) if not intensity_df.empty else pd.NA,
                "n_max": int(intensity_df["n"].max()) if not intensity_df.empty else pd.NA,
                "n_bins": len(intensity_df),
            }
        )

        del df_day, old_dt_df, old_intensity_df, intensity_df, size_counts, state_df
        gc.collect()

    daily_curves = pd.concat(curve_rows, ignore_index=True)
    daily_size_curves = pd.concat(size_curve_rows, ignore_index=True)
    diagnostics_df = pd.DataFrame(diagnostics)

    def aggregate_segment(name: str, day_set: set[date] | None) -> pd.DataFrame:
        outputs = []
        for estimator in daily_curves["estimator"].unique():
            subset = daily_curves[daily_curves["estimator"] == estimator]
            if day_set is not None:
                keep = {str(day) for day in day_set}
                subset = subset[subset["date"].isin(keep)]
            agg = (
                subset.groupby("n")[["n_obs", "ait", "Lambda", "lambda_L", "lambda_C", "lambda_M"]]
                .mean()
                .reset_index()
                .sort_values("n")
            )
            agg["segment"] = name
            agg["date"] = pd.NA
            agg["estimator"] = estimator
            outputs.append(agg)
        return pd.concat(outputs, ignore_index=True)

    curves_df = pd.concat(
        [
            aggregate_segment("full_month", None),
            aggregate_segment("first_half", first_half),
            aggregate_segment("second_half", second_half),
            daily_curves,
        ],
        ignore_index=True,
    )

    def aggregate_size_segment(name: str, day_set: set[date] | None) -> pd.DataFrame:
        outputs = []
        for estimator in daily_size_curves["estimator"].unique():
            subset = daily_size_curves[daily_size_curves["estimator"] == estimator]
            if day_set is not None:
                keep = {str(day) for day in day_set}
                subset = subset[subset["date"].isin(keep)]
            agg = (
                subset.groupby(["n", "eta", "event_size"])[["n_eta_size", "lambda_eta_size"]]
                .mean()
                .reset_index()
                .sort_values(["n", "eta", "event_size"])
            )
            agg["segment"] = name
            agg["date"] = pd.NA
            agg["estimator"] = estimator
            outputs.append(agg)
        return pd.concat(outputs, ignore_index=True)

    size_curves_df = pd.concat(
        [
            aggregate_size_segment("full_month", None),
            aggregate_size_segment("first_half", first_half),
            aggregate_size_segment("second_half", second_half),
            daily_size_curves,
        ],
        ignore_index=True,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    curves_df.to_parquet(output_path)
    if size_output_path:
        Path(size_output_path).parent.mkdir(parents=True, exist_ok=True)
        size_curves_df.to_parquet(size_output_path)

    return curves_df, size_curves_df, diagnostics_df, aes
