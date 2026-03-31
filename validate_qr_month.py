from __future__ import annotations

import gc
import time
import resource
from collections import defaultdict
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.features.qr_transforms import (
    calibrate_theta,
    compute_p_ref_series,
    estimate_queue_intensities,
    quantize_queue_sizes,
    theta_sensitivity_analysis,
)


EVENT_FLOW_PATH = "data/processed/FGBL_event_flow.parquet"


def memory_gb() -> float:
    # On macOS ru_maxrss is bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def trading_days() -> list[date]:
    days = []
    for path in sorted(Path("data/raw").glob("*.parquet")):
        days.append(pd.Timestamp(path.stem.split("_")[-1]).date())
    return days


def load_day(dataset: ds.Dataset, day: date, columns: list[str]) -> pd.DataFrame:
    table = dataset.to_table(columns=columns, filter=ds.field("date") == day)
    df_day = table.to_pandas()
    if "ts" in df_day.columns:
        df_day = df_day.sort_values("ts").set_index("ts")
        df_day.index.name = "ts"
    return df_day


def monotonic_share(series: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    diffs = series.diff().dropna()
    return float((diffs <= 0).mean() * 100)


def main() -> None:
    started = time.time()
    peak_before = memory_gb()

    dataset = ds.dataset(EVENT_FLOW_PATH, format="parquet")
    days = trading_days()

    aes_sum = defaultdict(float)
    aes_count = defaultdict(int)

    max_rows_materialized = 0

    # Pass 1: global AES without loading the month at once.
    aes_cols = ["date", "side", "level", "size"]
    for day in days:
        df_day = load_day(dataset, day, aes_cols)
        max_rows_materialized = max(max_rows_materialized, len(df_day))
        grouped = df_day.groupby(["side", "level"])["size"].agg(["sum", "count"])
        for (side, level), row in grouped.iterrows():
            aes_sum[(side, level)] += float(row["sum"])
            aes_count[(side, level)] += int(row["count"])
        del df_day, grouped
        gc.collect()

    aes = (
        pd.Series({k: aes_sum[k] / aes_count[k] for k in aes_sum}, name="aes")
        .rename_axis(["side", "level"])
        .sort_index()
    )

    day_rows = []
    intensity_rows = []

    day_cols = [
        "ts",
        "date",
        "side",
        "level",
        "eta",
        "q_before",
        "size",
        "delta_t",
        "spread_ticks",
        "depletion",
        "depletion_side",
        "p_mid",
        "p_ref",
        "best_bid_int",
        "best_ask_int",
    ]

    # Pass 2: daily validation metrics.
    for day in days:
        df_day = load_day(dataset, day, day_cols)
        max_rows_materialized = max(max_rows_materialized, len(df_day))
        df_day = quantize_queue_sizes(df_day, aes)

        total_events = len(df_day)
        eta_counts = df_day["eta"].value_counts()

        dep = df_day[df_day["depletion"].fillna(False)]
        dep_counts = dep["depletion_side"].value_counts()

        spread_valid = df_day["spread_ticks"].notna() & (df_day["spread_ticks"] > 0)
        spread_series = df_day.loc[spread_valid, "spread_ticks"].astype(int)
        even_spread_mask = spread_valid & (df_day["spread_ticks"] % 2 == 0)
        odd_spread_mask = spread_valid & (df_day["spread_ticks"] % 2 == 1)
        even_dep = dep[dep["spread_ticks"].notna() & (dep["spread_ticks"] > 0) & (dep["spread_ticks"] % 2 == 0)]

        theta_table = theta_sensitivity_analysis(df_day, horizons=(1, 5, 10))
        theta_tf = theta_sensitivity_analysis(
            df_day,
            horizons=(1, 5, 10),
            min_next_dt=pd.Timedelta(microseconds=500),
        )

        theta_1 = calibrate_theta(df_day, horizon=1, verbose=False)
        p_ref = compute_p_ref_series(df_day, theta_1)
        dev = p_ref - df_day["p_mid"]
        tick = 0.01
        allowed = np.array([-0.5 * tick, 0.0, 0.5 * tick])
        valid_dev = np.isclose(dev.to_numpy()[:, None], allowed[None, :], atol=1e-9).any(axis=1)

        dep_dev = dev[df_day["depletion"].fillna(False)].to_numpy()
        ask_mask = dep["depletion_side"].eq("ask").to_numpy()
        bid_mask = dep["depletion_side"].eq("bid").to_numpy()

        intensity_df, _ = estimate_queue_intensities(df_day, min_obs=30)
        intensity_df["date"] = str(day)
        intensity_rows.append(intensity_df)

        day_rows.append(
            {
                "date": str(day),
                "events": total_events,
                "L": int(eta_counts.get("L", 0)),
                "C": int(eta_counts.get("C", 0)),
                "M": int(eta_counts.get("M", 0)),
                "depl": len(dep),
                "depl_bid": int(dep_counts.get("bid", 0)),
                "depl_ask": int(dep_counts.get("ask", 0)),
                "depl_rate_10k": 10000 * len(dep) / total_events,
                "even_spread_pct": float(even_spread_mask.mean() * 100),
                "odd_spread_pct": float(odd_spread_mask.mean() * 100),
                "spread_1_pct": float((spread_series == 1).mean() * 100) if not spread_series.empty else np.nan,
                "spread_2_pct": float((spread_series == 2).mean() * 100) if not spread_series.empty else np.nan,
                "even_dep": len(even_dep),
                "even_dep_pct": 100 * len(even_dep) / len(dep) if len(dep) else np.nan,
                "theta_1": float(theta_table.loc[theta_table["horizon"] == 1, "theta"].iloc[0]),
                "theta_5": float(theta_table.loc[theta_table["horizon"] == 5, "theta"].iloc[0]),
                "theta_10": float(theta_table.loc[theta_table["horizon"] == 10, "theta"].iloc[0]),
                "theta_1_tf0_5ms": float(theta_tf.loc[theta_tf["horizon"] == 1, "theta"].iloc[0]),
                "theta_5_tf0_5ms": float(theta_tf.loc[theta_tf["horizon"] == 5, "theta"].iloc[0]),
                "theta_10_tf0_5ms": float(theta_tf.loc[theta_tf["horizon"] == 10, "theta"].iloc[0]),
                "theta_tf_n": int(theta_tf.loc[theta_tf["horizon"] == 1, "n_total"].iloc[0]),
                "pref_dev_pct": float((dev != 0).mean() * 100),
                "pref_valid_pct": float(valid_dev.mean() * 100),
                "ask_pref_up_pct": float((dep_dev[ask_mask] > 0).mean() * 100) if ask_mask.any() else np.nan,
                "bid_pref_down_pct": float((dep_dev[bid_mask] < 0).mean() * 100) if bid_mask.any() else np.nan,
                "eligible_pref_events": len(even_dep),
                "intensity_n_min": int(intensity_df["n"].min()),
                "intensity_n_max": int(intensity_df["n"].max()),
            }
        )

        del df_day, dep, even_dep, intensity_df, p_ref, dev, valid_dev, theta_table, theta_tf
        gc.collect()

    day_df = pd.DataFrame(day_rows)
    intensity_all = pd.concat(intensity_rows, ignore_index=True)
    agg_intensity = (
        intensity_all.groupby("n")[["Lambda", "lambda_L", "lambda_C", "lambda_M"]]
        .mean()
        .reset_index()
        .sort_values("n")
    )

    theta_summary = (
        day_df[["theta_1", "theta_5", "theta_10"]]
        .agg(["mean", "std", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )

    outlier_flags = []
    for col in ["theta_1", "depl_rate_10k", "even_spread_pct"]:
        mu = day_df[col].mean()
        sigma = day_df[col].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            continue
        flagged = day_df.loc[(day_df[col] - mu).abs() > 2 * sigma, ["date", col]]
        for _, row in flagged.iterrows():
            outlier_flags.append({"metric": col, "date": row["date"], "value": row[col]})
    outlier_df = pd.DataFrame(outlier_flags)

    summary = pd.DataFrame(
        [
            {"metric": "days", "value": len(day_df)},
            {"metric": "events_total", "value": int(day_df["events"].sum())},
            {"metric": "avg_depletion_rate_10k", "value": day_df["depl_rate_10k"].mean()},
            {"metric": "avg_even_spread_pct", "value": day_df["even_spread_pct"].mean()},
            {"metric": "avg_even_dep_pct", "value": day_df["even_dep_pct"].mean()},
            {"metric": "avg_theta_1", "value": day_df["theta_1"].mean()},
            {"metric": "avg_theta_5", "value": day_df["theta_5"].mean()},
            {"metric": "avg_theta_10", "value": day_df["theta_10"].mean()},
            {"metric": "avg_theta_1_tf0_5ms", "value": day_df["theta_1_tf0_5ms"].mean()},
            {"metric": "avg_pref_dev_pct", "value": day_df["pref_dev_pct"].mean()},
            {"metric": "avg_pref_valid_pct", "value": day_df["pref_valid_pct"].mean()},
            {"metric": "avg_ask_pref_up_pct", "value": day_df["ask_pref_up_pct"].mean()},
            {"metric": "avg_bid_pref_down_pct", "value": day_df["bid_pref_down_pct"].mean()},
            {"metric": "agg_lambda_L_monotone_pct", "value": monotonic_share(agg_intensity["lambda_L"])},
            {"metric": "agg_lambda_M_monotone_pct", "value": monotonic_share(agg_intensity["lambda_M"])},
            {"metric": "agg_lambda_C_monotone_pct", "value": monotonic_share(agg_intensity["lambda_C"])},
            {"metric": "peak_memory_gb", "value": memory_gb()},
            {"metric": "max_rows_materialized", "value": max_rows_materialized},
            {"metric": "runtime_min", "value": (time.time() - started) / 60.0},
        ]
    )

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nTHETA_SUMMARY")
    print(theta_summary.to_string(index=False))
    print("\nDAILY_STATS")
    print(
        day_df[
            [
                "date",
                "events",
                "L",
                "C",
                "M",
                "depl",
                "depl_bid",
                "depl_ask",
                "depl_rate_10k",
                "even_spread_pct",
                "even_dep",
                "even_dep_pct",
                "theta_1",
                "theta_5",
                "theta_10",
                "theta_1_tf0_5ms",
                "theta_tf_n",
                "pref_dev_pct",
                "pref_valid_pct",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nDAILY_P_REF")
    print(
        day_df[
            [
                "date",
                "eligible_pref_events",
                "pref_dev_pct",
                "ask_pref_up_pct",
                "bid_pref_down_pct",
                "spread_1_pct",
                "spread_2_pct",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nAGG_INTENSITY_HEAD")
    print(agg_intensity.head(20).round(6).to_string(index=False))
    print("\nAGG_INTENSITY_TAIL")
    print(agg_intensity.tail(20).round(6).to_string(index=False))
    print("\nOUTLIER_DAYS")
    if outlier_df.empty:
        print("none")
    else:
        print(outlier_df.round(4).to_string(index=False))
    print("\nENGINEERING")
    print("Processed one trading day at a time from the monthly parquet via pyarrow.dataset filters.")
    print("No full-month DataFrame was materialized.")


if __name__ == "__main__":
    main()
