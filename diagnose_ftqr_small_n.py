from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from models.common import calibrate_common
from models.ftqr import FTQRSimulator, calibrate_ftqr
from models.qr import QRSimulator, calibrate_qr
from models.qru import QRUSimulator, calibrate_qru
from src.features.qr_empirical import compute_global_aes_streaming, load_day, trading_days
from src.features.qr_transforms import quantize_queue_sizes


OUT_DIR = Path("data/processed/ftqr_diagnostics")


def build_streaming_diagnostics(
    event_flow_path: str,
    raw_dir: str,
    level: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = ds.dataset(event_flow_path, format="parquet")
    aes = compute_global_aes_streaming(event_flow_path, raw_dir=raw_dir)

    full_counts = defaultdict(int)
    eta_counts = defaultdict(int)
    cond_size_counts = defaultdict(int)
    uncond_size_counts = defaultdict(int)

    cols = ["ts", "date", "side", "level", "eta", "q_before", "size"]

    for day in trading_days(raw_dir):
        df_day = load_day(dataset, day, cols)
        df_day = quantize_queue_sizes(df_day, aes)
        df_day = df_day[df_day["level"] == level].copy()
        if df_day.empty:
            continue

        event_df = df_day[df_day["q_before_aes"] > 0].copy()
        event_df["is_full"] = event_df["size"] >= event_df["q_before"]

        subset = event_df[event_df["eta"].isin(["C", "M"])].copy()
        grouped_eta = subset.groupby(["q_before_aes", "eta"]).size()
        for key, count in grouped_eta.items():
            n, eta = key
            eta_counts[(int(n), str(eta))] += int(count)

        grouped_full = subset.groupby(["q_before_aes", "eta", "is_full"]).size()
        for key, count in grouped_full.items():
            n, eta, is_full = key
            full_counts[(int(n), str(eta), bool(is_full))] += int(count)

        for size, count in event_df.groupby("size").size().items():
            uncond_size_counts[int(size)] += int(count)

        selected = event_df[event_df["q_before_aes"].isin([3, 5, 10]) & event_df["eta"].isin(["C", "M"])].copy()
        grouped_cond = selected.groupby(["q_before_aes", "eta", "size"]).size()
        for key, count in grouped_cond.items():
            n, eta, size = key
            cond_size_counts[(int(n), str(eta), int(size))] += int(count)

        del df_day, event_df, subset, selected, grouped_eta, grouped_full, grouped_cond
        gc.collect()

    full_df = pd.DataFrame(
        [
            {"n": n, "eta": eta, "is_full": is_full, "count": count}
            for (n, eta, is_full), count in full_counts.items()
        ]
    )
    eta_df = pd.DataFrame(
        [{"n": n, "eta": eta, "count": count} for (n, eta), count in eta_counts.items()]
    )
    cond_df = pd.DataFrame(
        [
            {"n": n, "eta": eta, "size": size, "count": count}
            for (n, eta, size), count in cond_size_counts.items()
        ]
    )
    uncond_df = pd.DataFrame([{"size": size, "count": count} for size, count in uncond_size_counts.items()])
    return full_df, eta_df, cond_df.merge(uncond_df, on="size", how="left", suffixes=("", "_uncond"))


def plot_lambda_m_comparison(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = table[table["n"] <= 25].copy()
    ax.plot(plot_df["n"], plot_df["lambda_M_qr"], linewidth=2, label="QR lambda_M")
    ax.plot(plot_df["n"], plot_df["lambda_M_total_ftqr"], linewidth=2, label="FTQR lambda_M_total")
    ax.plot(plot_df["n"], plot_df["lambda_M_all_ftqr"], linewidth=2, label="FTQR lambda_M_all")
    ax.set_xlabel("Queue size n (AES)")
    ax.set_ylabel("Intensity")
    ax.set_title("lambda_M: QR vs FTQR")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lambda_m_qr_vs_ftqr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_event_decomposition(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    plot_df = table[table["n"] <= 25].copy()
    for ax, partial_col, full_col, title in [
        (axes[0], "lambda_C_ftqr", "lambda_C_all_ftqr", "Cancels"),
        (axes[1], "lambda_M_ftqr", "lambda_M_all_ftqr", "Trades"),
    ]:
        ax.plot(plot_df["n"], plot_df[partial_col], linewidth=2, label=partial_col)
        ax.plot(plot_df["n"], plot_df[full_col], linewidth=2, label=full_col)
        ax.set_title(title)
        ax.set_xlabel("Queue size n (AES)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Intensity")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ftqr_full_event_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qr_qru_intensity_comparison(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    plot_df = table[table["n"] <= 25].copy()
    for ax, col, title in zip(axes, ["lambda_L_qr", "lambda_C_qr", "lambda_M_qr"], ["lambda_L", "lambda_C", "lambda_M"]):
        ax.plot(plot_df["n"], plot_df[col], linewidth=2, label="QR")
        ax.plot(plot_df["n"], plot_df[col], linewidth=2, linestyle="--", label="QRU")
        ax.set_title(title)
        ax.set_xlabel("Queue size n (AES)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Intensity")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qr_vs_qru_intensities.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_small_n_bias(qr_intensity_path: str) -> None:
    df = pd.read_parquet(qr_intensity_path)
    if "estimator" not in df.columns:
        raise ValueError("qr_intensities.parquet is missing the estimator column.")
    plot_df = df[(df["segment"] == "full_month") & (df["estimator"].isin(["event_based", "state_duration"]))].copy()
    plot_df = plot_df[plot_df["n"] <= 25].sort_values(["estimator", "n"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for estimator in ["event_based", "state_duration"]:
        sub = plot_df[plot_df["estimator"] == estimator]
        ax.plot(sub["n"], sub["lambda_M"], linewidth=2, label=estimator)
    ax.set_xlabel("Queue size n (AES)")
    ax.set_ylabel("lambda_M")
    ax.set_title("Small-n bias check: raw-row vs state-duration")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lambda_m_bias_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_conditional_size_distributions(cond_df: pd.DataFrame) -> None:
    for eta in ["M", "C"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
        eta_df = cond_df[cond_df["eta"] == eta].copy()
        for ax, n in zip(axes, [3, 5, 10]):
            sub = eta_df[eta_df["n"] == n].copy()
            if sub.empty:
                continue
            sub["prob_cond"] = sub["count"] / sub["count"].sum()
            sub["prob_uncond"] = sub["count_uncond"] / sub["count_uncond"].sum()
            sub = sub.sort_values("size").head(40)
            ax.plot(sub["size"], sub["prob_uncond"], label="P(s)", linewidth=2)
            ax.plot(sub["size"], sub["prob_cond"], label=f"P(s | q={n})", linewidth=2)
            ax.set_title(f"{eta}, n={n}")
            ax.set_xlabel("Size")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Probability")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"size_distribution_{eta.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def simulate_model_effects(common) -> pd.DataFrame:
    qr = QRSimulator(calibrate_qr("", common=common))
    qru = QRUSimulator(calibrate_qru("", common=common))
    ftqr = FTQRSimulator(calibrate_ftqr("", common=common))

    rows = []
    for seed, (name, sim) in enumerate([("QR", qr), ("QRU", qru), ("FTQR", ftqr)], start=1):
        result = sim.simulate(steps=25_000, seed=seed)
        rows.append({"model": name, **result.summary})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    event_flow_path = "data/processed/FGBL_event_flow.parquet"
    raw_dir = "data/raw"
    level = 1

    common = calibrate_common(event_flow_path, raw_dir=raw_dir, level=level, min_obs=50)
    ftqr_cal = calibrate_ftqr(event_flow_path, raw_dir=raw_dir, level=level, min_obs=50, common=common)

    qr_df = common.intensity_df.copy().rename(
        columns={
            "lambda_L": "lambda_L_qr",
            "lambda_C": "lambda_C_qr",
            "lambda_M": "lambda_M_qr",
        }
    )
    ft_df = ftqr_cal.intensity_df.copy().rename(
        columns={
            "lambda_L": "lambda_L_ftqr",
            "lambda_C": "lambda_C_ftqr",
            "lambda_M": "lambda_M_ftqr",
        }
    )
    table = qr_df.merge(
        ft_df[["n", "lambda_L_ftqr", "lambda_C_ftqr", "lambda_M_ftqr", "lambda_C_all", "lambda_M_all", "lambda_global"]],
        on="n",
        how="left",
    )
    table = table.rename(columns={"lambda_C_all": "lambda_C_all_ftqr", "lambda_M_all": "lambda_M_all_ftqr"})
    table["lambda_M_total_ftqr"] = table["lambda_M_ftqr"] + table["lambda_M_all_ftqr"]
    table["lambda_C_total_ftqr"] = table["lambda_C_ftqr"] + table["lambda_C_all_ftqr"]
    table["lambda_L_qru"] = table["lambda_L_qr"]
    table["lambda_C_qru"] = table["lambda_C_qr"]
    table["lambda_M_qru"] = table["lambda_M_qr"]

    full_df, eta_df, cond_df = build_streaming_diagnostics(event_flow_path, raw_dir, level)
    full_pivot = full_df.pivot_table(index=["n", "eta"], columns="is_full", values="count", fill_value=0).reset_index()
    full_pivot = full_pivot.rename(columns={False: "count_partial", True: "count_full"})
    full_pivot["count_total"] = full_pivot["count_partial"] + full_pivot["count_full"]
    full_pivot["p_full"] = np.where(full_pivot["count_total"] > 0, full_pivot["count_full"] / full_pivot["count_total"], np.nan)
    prob_df = full_pivot.pivot(index="n", columns="eta", values="p_full").reset_index().rename(
        columns={"C": "p_full_C", "M": "p_full_M"}
    )
    table = table.merge(prob_df, on="n", how="left")

    small_n = table[table["n"] <= 20].copy().sort_values("n")
    small_n.to_csv(OUT_DIR / "ftqr_small_n_table.csv", index=False)
    full_pivot.to_csv(OUT_DIR / "ftqr_full_consumption_probabilities.csv", index=False)
    cond_df.to_csv(OUT_DIR / "ftqr_conditional_size_distributions.csv", index=False)

    sim_df = simulate_model_effects(common)
    sim_df.to_csv(OUT_DIR / "qr_qru_ftqr_simulation_summary.csv", index=False)

    plot_lambda_m_comparison(table)
    plot_full_event_decomposition(table)
    plot_qr_qru_intensity_comparison(table)
    plot_small_n_bias("data/processed/qr_intensities.parquet")
    plot_conditional_size_distributions(cond_df)

    qr_hump = small_n["lambda_M_qr"].idxmax() == small_n["n"].idxmin()
    ftqr_hump = small_n["lambda_M_total_ftqr"].idxmax() not in [small_n["n"].idxmin()]

    print("Saved diagnostics to", OUT_DIR)
    print(small_n[[
        "n",
        "lambda_L_qr",
        "lambda_C_qr",
        "lambda_M_qr",
        "lambda_C_ftqr",
        "lambda_M_ftqr",
        "lambda_C_all_ftqr",
        "lambda_M_all_ftqr",
        "lambda_M_total_ftqr",
        "p_full_C",
        "p_full_M",
    ]].head(10).to_string(index=False))
    print()
    print("QR small-n lambda_M peak at n=", int(small_n.loc[small_n["lambda_M_qr"].idxmax(), "n"]))
    print("FTQR total small-n lambda_M peak at n=", int(small_n.loc[small_n["lambda_M_total_ftqr"].idxmax(), "n"]))
    print(sim_df.to_string(index=False))


if __name__ == "__main__":
    main()
