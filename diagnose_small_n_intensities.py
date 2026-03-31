from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.features.qr_empirical import compute_global_aes_streaming, load_day, trading_days


EVENT_FLOW_PATH = "data/processed/FGBL_event_flow.parquet"
OUT_DIR = Path("data/processed/qr_small_n_diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_quantization_columns(df: pd.DataFrame, aes: pd.Series) -> pd.DataFrame:
    df = df.copy()
    aes_col = (
        df[["side", "level"]]
        .merge(aes.reset_index(), on=["side", "level"], how="left")["aes"]
        .to_numpy()
    )
    df["aes_value"] = aes_col
    df["q_before_aes"] = np.ceil(df["q_before"].to_numpy() / aes_col).astype(int)
    df["q_before_floor"] = np.floor(df["q_before"].to_numpy() / aes_col).astype(int)
    return df


def scenario_summary(df: pd.DataFrame, queue_col: str, scenario: str, side_label: str) -> pd.DataFrame:
    valid = (
        df[queue_col].notna()
        & (df[queue_col] > 0)
        & df["delta_t"].notna()
        & (df["delta_t"] > 0)
    )
    if not valid.any():
        return pd.DataFrame(columns=["scenario", "side_group", "n", "count", "sum_dt", "count_L", "count_C", "count_M"])
    work = df.loc[valid, [queue_col, "delta_t", "eta"]].copy()
    work = work.rename(columns={queue_col: "n"})
    grouped = work.groupby("n")
    base = grouped["delta_t"].agg(count="count", sum_dt="sum").reset_index()
    eta_counts = (
        work.groupby(["n", "eta"]).size()
        .unstack(fill_value=0)
        .reindex(columns=["L", "C", "M"], fill_value=0)
        .reset_index()
    )
    out = base.merge(eta_counts, on="n", how="left").rename(columns={"L": "count_L", "C": "count_C", "M": "count_M"})
    out["scenario"] = scenario
    out["side_group"] = side_label
    return out


def finalize_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary_df.groupby(["scenario", "side_group", "n"], as_index=False)[["count", "sum_dt", "count_L", "count_C", "count_M"]]
        .sum()
        .sort_values(["scenario", "side_group", "n"])
    )
    grouped["mean_dt"] = grouped["sum_dt"] / grouped["count"]
    grouped["Lambda"] = 1.0 / grouped["mean_dt"]
    grouped["lambda_L"] = grouped["Lambda"] * grouped["count_L"] / grouped["count"]
    grouped["lambda_C"] = grouped["Lambda"] * grouped["count_C"] / grouped["count"]
    grouped["lambda_M"] = grouped["Lambda"] * grouped["count_M"] / grouped["count"]
    grouped["prop"] = grouped["count"] / grouped.groupby(["scenario", "side_group"])["count"].transform("sum")
    return grouped


def collapse_same_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    collapsed = (
        df.reset_index()
        .sort_values(["side", "ts"])
        .groupby(["side", "ts"], sort=False)
        .tail(1)
        .sort_values(["side", "ts"])
        .set_index("ts")
    )
    collapsed.index.name = "ts"
    collapsed["delta_t"] = (
        collapsed.index.to_series().groupby(collapsed["side"]).diff().dt.total_seconds()
    )
    return collapsed


def update_transition_counts(df: pd.DataFrame, transition_counts: defaultdict) -> None:
    for side in ["B", "A"]:
        side_df = df[df["side"] == side].sort_index()
        next_q = side_df["q_before_aes"].shift(-1)
        for n in range(1, 21):
            mask = side_df["q_before_aes"] == n
            if not mask.any():
                continue
            nxt = next_q[mask].dropna()
            transition_counts[(side, n, "to_zero")] += int((nxt == 0).sum())
            transition_counts[(side, n, "to_lower")] += int((nxt < n).sum())
            transition_counts[(side, n, "to_same")] += int((nxt == n).sum())
            transition_counts[(side, n, "to_higher")] += int((nxt > n).sum())


def append_limited(target: list[float], values: np.ndarray, limit: int = 200_000) -> None:
    remaining = limit - len(target)
    if remaining <= 0:
        return
    target.extend(values[:remaining].tolist())


def main() -> None:
    dataset = ds.dataset(EVENT_FLOW_PATH, format="parquet")
    days = trading_days()
    aes = compute_global_aes_streaming(EVENT_FLOW_PATH)

    summary_frames: list[pd.DataFrame] = []
    q_hist_counts = defaultdict(int)
    floor_hist_counts = defaultdict(int)
    raw_q_hist_counts = defaultdict(int)
    eta_small_counts = defaultdict(int)
    dt_validity = defaultdict(int)
    transition_counts = defaultdict(int)
    dt_samples_small: list[float] = []
    dt_samples_large: list[float] = []

    cols = ["ts", "date", "side", "level", "eta", "q_before", "size", "delta_t"]
    for day in days:
        df_day = load_day(dataset, day, cols)
        df_day = df_day[df_day["level"] == 1].copy()
        df_day = add_quantization_columns(df_day, aes)

        q_hist = df_day["q_before_aes"].value_counts()
        floor_hist = df_day["q_before_floor"].value_counts()
        raw_hist = df_day["q_before"].value_counts()
        for n in range(1, 21):
            q_hist_counts[n] += int(q_hist.get(n, 0))
            floor_hist_counts[n] += int(floor_hist.get(n, 0))
        for q in range(1, 201):
            raw_q_hist_counts[q] += int(raw_hist.get(q, 0))

        dt_validity["total"] += len(df_day)
        dt_validity["nan"] += int(df_day["delta_t"].isna().sum())
        dt_validity["nonpositive"] += int((df_day["delta_t"].fillna(0) <= 0).sum())

        small_mask = df_day["q_before_aes"].between(1, 20)
        small_eta = df_day.loc[small_mask, "eta"].value_counts()
        for eta in ["L", "C", "M"]:
            eta_small_counts[eta] += int(small_eta.get(eta, 0))

        valid_dt = df_day["delta_t"].notna() & (df_day["delta_t"] > 0)
        append_limited(dt_samples_small, df_day.loc[valid_dt & df_day["q_before_aes"].between(1, 5), "delta_t"].to_numpy())
        append_limited(dt_samples_large, df_day.loc[valid_dt & df_day["q_before_aes"].between(30, 50), "delta_t"].to_numpy())

        update_transition_counts(df_day, transition_counts)

        summary_frames.append(scenario_summary(df_day, "q_before_aes", "aes_raw", "all"))
        summary_frames.append(scenario_summary(df_day[df_day["side"] == "B"], "q_before_aes", "aes_raw", "bid"))
        summary_frames.append(scenario_summary(df_day[df_day["side"] == "A"], "q_before_aes", "aes_raw", "ask"))
        summary_frames.append(scenario_summary(df_day, "q_before", "raw_queue", "all"))
        summary_frames.append(scenario_summary(df_day, "q_before_floor", "aes_floor", "all"))

        df_ts = collapse_same_timestamp(df_day)
        summary_frames.append(scenario_summary(df_ts, "q_before_aes", "aes_ts_last", "all"))
        summary_frames.append(scenario_summary(df_ts[df_ts["side"] == "B"], "q_before_aes", "aes_ts_last", "bid"))
        summary_frames.append(scenario_summary(df_ts[df_ts["side"] == "A"], "q_before_aes", "aes_ts_last", "ask"))

        del df_day, df_ts
        gc.collect()

    summary_df = finalize_summary(pd.concat(summary_frames, ignore_index=True))

    small_n_table = (
        summary_df[(summary_df["scenario"] == "aes_raw") & (summary_df["side_group"] == "all") & (summary_df["n"].between(1, 20))]
        [["n", "count", "mean_dt", "Lambda", "lambda_L", "lambda_C", "lambda_M", "prop"]]
        .copy()
    )
    small_n_table["q_before_aes_hist"] = small_n_table["n"].map(q_hist_counts)
    small_n_table["q_before_floor_hist"] = small_n_table["n"].map(floor_hist_counts)

    bid_small = summary_df[(summary_df["scenario"] == "aes_raw") & (summary_df["side_group"] == "bid") & (summary_df["n"].between(1, 50))]
    ask_small = summary_df[(summary_df["scenario"] == "aes_raw") & (summary_df["side_group"] == "ask") & (summary_df["n"].between(1, 50))]
    ts_small = summary_df[(summary_df["scenario"] == "aes_ts_last") & (summary_df["side_group"] == "all") & (summary_df["n"].between(1, 50))]
    aes_small = summary_df[(summary_df["scenario"] == "aes_raw") & (summary_df["side_group"] == "all") & (summary_df["n"].between(1, 50))]
    floor_small = summary_df[(summary_df["scenario"] == "aes_floor") & (summary_df["side_group"] == "all") & (summary_df["n"].between(1, 50))]

    transition_rows = []
    for side in ["B", "A"]:
        for n in range(1, 21):
            total = sum(transition_counts[(side, n, key)] for key in ["to_zero", "to_lower", "to_same", "to_higher"])
            if total == 0:
                continue
            transition_rows.append(
                {
                    "side": side,
                    "n": n,
                    "to_zero_pct": 100 * transition_counts[(side, n, "to_zero")] / total,
                    "to_lower_pct": 100 * transition_counts[(side, n, "to_lower")] / total,
                    "to_same_pct": 100 * transition_counts[(side, n, "to_same")] / total,
                    "to_higher_pct": 100 * transition_counts[(side, n, "to_higher")] / total,
                    "transitions": total,
                }
            )
    transition_df = pd.DataFrame(transition_rows)

    eta_prop_df = small_n_table[["n", "count", "lambda_L", "lambda_C", "lambda_M"]].copy()
    eta_prop_df["count_L"] = summary_df.set_index(["scenario", "side_group", "n"]).loc[("aes_raw", "all")]["count_L"].reindex(eta_prop_df["n"]).to_numpy()
    eta_prop_df["count_C"] = summary_df.set_index(["scenario", "side_group", "n"]).loc[("aes_raw", "all")]["count_C"].reindex(eta_prop_df["n"]).to_numpy()
    eta_prop_df["count_M"] = summary_df.set_index(["scenario", "side_group", "n"]).loc[("aes_raw", "all")]["count_M"].reindex(eta_prop_df["n"]).to_numpy()
    eta_prop_df["prop_L"] = eta_prop_df["count_L"] / eta_prop_df["count"]
    eta_prop_df["prop_C"] = eta_prop_df["count_C"] / eta_prop_df["count"]
    eta_prop_df["prop_M"] = eta_prop_df["count_M"] / eta_prop_df["count"]

    small_n_table.to_csv(OUT_DIR / "small_n_table.csv", index=False)
    transition_df.to_csv(OUT_DIR / "transition_small_n.csv", index=False)
    eta_prop_df.to_csv(OUT_DIR / "eta_props_small_n.csv", index=False)
    summary_df.to_parquet(OUT_DIR / "intensity_scenarios.parquet", index=False)

    # Plot 1: q_before_aes histogram.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(list(range(1, 21)), [q_hist_counts[n] for n in range(1, 21)], color="#1f77b4", alpha=0.8, label="ceil(q/AES)")
    ax.plot(list(range(1, 21)), [floor_hist_counts[n] for n in range(1, 21)], color="#d62728", marker="o", label="floor(q/AES)")
    ax.set_xlabel("Queue size bin")
    ax.set_ylabel("Observation count")
    ax.set_title("Level-1 queue size histogram, AES-normalized")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q_before_aes_hist.png", dpi=160)
    plt.close(fig)

    # Plot 2: delta_t distribution small vs large n.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(np.log10(np.array(dt_samples_small) + 1e-9), bins=80, alpha=0.6, label="n in [1,5]", density=True)
    ax.hist(np.log10(np.array(dt_samples_large) + 1e-9), bins=80, alpha=0.6, label="n in [30,50]", density=True)
    ax.set_xlabel("log10(delta_t + 1e-9)")
    ax.set_ylabel("Density")
    ax.set_title("delta_t distribution by queue-size regime")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dt_small_vs_large.png", dpi=160)
    plt.close(fig)

    # Plot 3: event-type proportions.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eta_prop_df["n"], eta_prop_df["prop_L"], label="L", linewidth=2)
    ax.plot(eta_prop_df["n"], eta_prop_df["prop_C"], label="C", linewidth=2)
    ax.plot(eta_prop_df["n"], eta_prop_df["prop_M"], label="M", linewidth=2)
    ax.set_xlabel("Queue size n (AES)")
    ax.set_ylabel("Event-type proportion")
    ax.set_title("Event-type mix for small queue sizes")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eta_props_small_n.png", dpi=160)
    plt.close(fig)

    # Plot 4: bid vs ask side comparison.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    for ax, col, title in zip(axes, ["lambda_L", "lambda_C"], [r"$\lambda_L(n)$", r"$\lambda_C(n)$"]):
        ax.plot(bid_small["n"], bid_small[col], label="Bid", linewidth=2)
        ax.plot(ask_small["n"], ask_small[col], label="Ask", linewidth=2)
        ax.set_title(f"{title}: bid vs ask")
        ax.set_xlabel("Queue size n (AES)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Intensity")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lambda_side_compare.png", dpi=160)
    plt.close(fig)

    # Plot 5: raw vs timestamp aggregated.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    for ax, col, title in zip(axes, ["lambda_L", "lambda_C"], [r"$\lambda_L(n)$", r"$\lambda_C(n)$"]):
        ax.plot(aes_small["n"], aes_small[col], label="Raw rows", linewidth=2)
        ax.plot(ts_small["n"], ts_small[col], label="Last row per timestamp", linewidth=2, linestyle="--")
        ax.set_title(f"{title}: raw vs timestamp-aggregated")
        ax.set_xlabel("Queue size n (AES)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Intensity")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lambda_timestamp_compare.png", dpi=160)
    plt.close(fig)

    # Plot 6: AES vs raw/floor comparison.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    for ax, col, title in zip(axes, ["lambda_L", "lambda_C"], [r"$\lambda_L(n)$", r"$\lambda_C(n)$"]):
        ax.plot(aes_small["n"], aes_small[col], label="AES ceil", linewidth=2)
        ax.plot(floor_small["n"], floor_small[col], label="AES floor", linewidth=2, linestyle=":")
        ax.set_title(f"{title}: ceil vs floor AES bins")
        ax.set_xlabel("Queue size bin")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Intensity")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lambda_aes_floor_compare.png", dpi=160)
    plt.close(fig)

    print("SMALL_N_TABLE")
    print(small_n_table.round(6).to_string(index=False))
    print("\nDT_VALIDITY")
    print(
        {
            "total_rows": dt_validity["total"],
            "nan_delta_t_pct": 100 * dt_validity["nan"] / dt_validity["total"],
            "nonpositive_delta_t_pct": 100 * dt_validity["nonpositive"] / dt_validity["total"],
        }
    )
    print("\nETA_SMALL_COUNTS")
    print(eta_small_counts)
    print("\nTRANSITION_HEAD")
    print(transition_df.head(20).round(4).to_string(index=False))
    print(f"\nSaved diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
