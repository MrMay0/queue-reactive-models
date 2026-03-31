"""
QR Model Preprocessing — Step 3
=================================
Implements the mandatory transformations from Huang et al. (2015) and
Bodor & Carlier (2024) to prepare the event flow for queue-reactive
model calibration.

Transformations (applied in order):
  1. Compute AES per (side, level): mean event size across all types.
  2. Quantize queue sizes: q_before_aes = ceil(q_before / AES_{side,level}).
  3. Assign period IDs: each contiguous block of constant p_ref gets a unique
     integer period_id.  delta_t is already reset to 0 at period boundaries
     by lob_reconstruction.py.
  4. Calibrate theta: empirical P(p_ref moves | best queue fully consumed).

Input : event flow parquet produced by lob_reconstruction.build_event_flow()
Output: enriched DataFrame with columns `q_before_aes`, `period_id`; plus
        the AES Series and the scalar theta.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _collapse_to_timestamp_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw event rows into one state row per timestamp.

    The retained row is the last emitted row at that timestamp, which reflects
    the fully updated book after all atomic events sharing that timestamp.
    Depletions are aggregated at timestamp resolution so same-timestamp atomic
    rows from one aggressive order do not create artificial t -> t+1 steps.
    """
    if df.empty:
        return df.copy()

    df = df.sort_index().copy()

    grouped = df.groupby(level=0, sort=False)
    state_df = grouped.tail(1).copy()

    depletion_any = grouped["depletion"].any()
    dep_side_counts = (
        df.loc[df["depletion"].fillna(False)]
        .groupby([df.loc[df["depletion"].fillna(False)].index, "depletion_side"])
        .size()
        .unstack(fill_value=0)
    )

    state_df["depletion"] = depletion_any.reindex(state_df.index, fill_value=False).to_numpy()
    state_df["depletion_side"] = None

    if not dep_side_counts.empty:
        dep_side_counts = dep_side_counts.reindex(state_df.index, fill_value=0)
        ask_only = (dep_side_counts.get("ask", 0) > 0) & (dep_side_counts.get("bid", 0) == 0)
        bid_only = (dep_side_counts.get("bid", 0) > 0) & (dep_side_counts.get("ask", 0) == 0)
        both_sides = (dep_side_counts.get("bid", 0) > 0) & (dep_side_counts.get("ask", 0) > 0)

        state_df.loc[ask_only.to_numpy(), "depletion_side"] = "ask"
        state_df.loc[bid_only.to_numpy(), "depletion_side"] = "bid"
        # Ambiguous timestamps with both-side depletion are not usable for a
        # directional continuation statistic.
        state_df.loc[both_sides.to_numpy(), "depletion"] = False
        state_df.loc[both_sides.to_numpy(), "depletion_side"] = None

    return state_df


def _collapse_to_bursts(df: pd.DataFrame, threshold: pd.Timedelta) -> pd.DataFrame:
    """
    Collapse consecutive timestamps into bursts separated by at least threshold.

    The input is assumed to already be one row per timestamp. The last
    timestamp in each burst is retained.
    """
    if df.empty:
        return df.copy()

    df = df.sort_index().copy()
    ts = pd.Series(df.index, index=df.index)
    dt = ts.diff()
    burst_start = dt.isna() | (dt >= threshold)
    burst_id = burst_start.cumsum()
    return df.groupby(burst_id, sort=False).tail(1).copy()


def prepare_theta_dataset(
    df: pd.DataFrame,
    min_next_dt: pd.Timedelta | None = None,
    burst_threshold: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """
    Prepare the dataset used for theta calibration.

    Steps:
      1. Collapse raw rows to one state per timestamp.
      2. Optionally collapse adjacent timestamps into bursts and keep only the
         last timestamp of each burst.
      3. Optionally drop depletion rows whose next timestamp arrives too fast.
    """
    df_theta = _collapse_to_timestamp_states(df)

    if burst_threshold is not None:
        df_theta = _collapse_to_bursts(df_theta, burst_threshold)

    if min_next_dt is not None and not df_theta.empty:
        next_dt = pd.Series(df_theta.index, index=df_theta.index).shift(-1) - pd.Series(df_theta.index, index=df_theta.index)
        too_fast = df_theta["depletion"].fillna(False) & next_dt.notna() & (next_dt < min_next_dt)
        df_theta = df_theta.loc[~too_fast].copy()

    return df_theta


# ---------------------------------------------------------------------------
# 1. Average Event Size
# ---------------------------------------------------------------------------

def compute_aes(df: pd.DataFrame) -> pd.Series:
    """
    Compute Average Event Size (AES) for each (side, level).

    From the paper (Section 2.1):
        AES_i = mean of all event sizes at level i, regardless of type.

    This scalar is the normalisation unit: all intensities are estimated
    as functions of queue size expressed in AES units, not raw lots.

    Returns
    -------
    pd.Series indexed by (side, level) with float values, named 'aes'.
    """
    return df.groupby(["side", "level"])["size"].mean().rename("aes")


# ---------------------------------------------------------------------------
# 2. Quantize queue sizes
# ---------------------------------------------------------------------------

def quantize_queue_sizes(df: pd.DataFrame, aes: pd.Series) -> pd.DataFrame:
    """
    Replace raw queue sizes with their AES-normalised, ceiling-rounded
    equivalent:

        q_before_aes = ceil(q_before / AES_{side, level})

    From the paper (Section 3.1):
        "Queue sizes are quantized based on the average event size.
         If the size of queue i is denoted by q_i and its average event
         size is AES_i, then the updated queue size is given by
         q_i <- ceil(q_i / AES_i)."

    A new column `q_before_aes` is added; the original `q_before` is kept
    for traceability.

    Parameters
    ----------
    df  : event flow DataFrame
    aes : Series indexed by (side, level), from compute_aes()
    """
    df = df.copy()

    # Vectorised merge: map AES onto every row via (side, level) key
    aes_col = (
        df[["side", "level"]]
        .merge(aes.reset_index(), on=["side", "level"], how="left")["aes"]
        .values
    )

    df["q_before_aes"] = np.ceil(df["q_before"].values / aes_col).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. Constant-reference-price periods
# ---------------------------------------------------------------------------

def assign_period_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign an integer `period_id` that increments each time p_ref changes.

    From the paper (Section 3.1):
        "The data is divided into periods with a constant reference price.
         The inter-arrival time counter for events is reset to 0 for all
         queues when there is a change in the reference price."

    lob_reconstruction.py already sets delta_t = 0 at period boundaries;
    this function makes the period boundaries explicit via a column so that
    downstream intensity calibration can group by period.

    Parameters
    ----------
    df : event flow DataFrame, sorted by timestamp (index).
    """
    df = df.copy()
    # Detect every row where p_ref differs from the previous row
    p_ref = df["p_ref"]
    df["period_id"] = (p_ref != p_ref.shift(fill_value=p_ref.iloc[0])).cumsum()
    return df


# ---------------------------------------------------------------------------
# 4. Calibrate theta
# ---------------------------------------------------------------------------

def calibrate_theta(
    df: pd.DataFrame,
    horizon: int = 1,
    verbose: bool = True,
    min_next_dt: pd.Timedelta | None = None,
    burst_threshold: pd.Timedelta | None = None,
) -> float:
    """
    Calibrate theta from all queue depletion events, independent of spread.

    Continuation is defined per depletion side:
      - ask depletion → continuation if p_mid(t+horizon) > p_mid(t)
      - bid depletion → continuation if p_mid(t+horizon) < p_mid(t)

    theta = continuations / total_depletion_events_with_observed_horizon
    """
    if df.empty:
        raise ValueError("Dataframe is empty; cannot calibrate theta.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    n_total_before = int(df["depletion"].fillna(False).sum())
    df = prepare_theta_dataset(df, min_next_dt=min_next_dt, burst_threshold=burst_threshold)
    valid_mask = df["depletion"].fillna(False)

    if valid_mask.sum() == 0:
        raise ValueError("No depletion events found; cannot calibrate theta.")

    p_mid_horizon = df["p_mid"].shift(-horizon)

    valid_df = df[valid_mask].copy()
    valid_df["p_mid_horizon"] = p_mid_horizon[valid_mask]
    valid_df = valid_df.dropna(subset=["p_mid_horizon"])

    if len(valid_df) == 0:
        raise ValueError("No depletion events with an observed horizon.")

    continuation_mask = (
        ((valid_df["depletion_side"] == "ask") & (valid_df["p_mid_horizon"] > valid_df["p_mid"])) |
        ((valid_df["depletion_side"] == "bid") & (valid_df["p_mid_horizon"] < valid_df["p_mid"]))
    )
    n_total = len(valid_df)
    n_continuation = int(continuation_mask.sum())
    theta_hat = n_continuation / n_total

    if theta_hat <= 0.0 or theta_hat >= 1.0:
        raise ValueError(
            f"Calibrated theta = {theta_hat:.6f} is degenerate (must be in (0, 1)); "
            f"total events = {n_total}, continuations = {n_continuation}. "
            f"Check depletion detection or data quality."
        )

    if verbose:
        ask_df = valid_df[valid_df["depletion_side"] == "ask"]
        bid_df = valid_df[valid_df["depletion_side"] == "bid"]
        ask_up = int((ask_df["p_mid_horizon"] > ask_df["p_mid"]).sum())
        ask_down = int((ask_df["p_mid_horizon"] < ask_df["p_mid"]).sum())
        ask_flat = int((ask_df["p_mid_horizon"] == ask_df["p_mid"]).sum())
        bid_down = int((bid_df["p_mid_horizon"] < bid_df["p_mid"]).sum())
        bid_up = int((bid_df["p_mid_horizon"] > bid_df["p_mid"]).sum())
        bid_flat = int((bid_df["p_mid_horizon"] == bid_df["p_mid"]).sum())
        print(
            f"[theta calibration] horizon={horizon}, "
            f"N_total_before={n_total_before}, "
            f"N_total={n_total}, "
            f"N_continuation={n_continuation}, theta={theta_hat:.6f}"
        )
        print(
            f"[theta calibration] horizon={horizon}, "
            f"ask: up={ask_up}, down={ask_down}, flat={ask_flat}; "
            f"bid: down={bid_down}, up={bid_up}, flat={bid_flat}"
        )

    return float(theta_hat)


def theta_sensitivity_analysis(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    min_next_dt: pd.Timedelta | None = None,
    burst_threshold: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """
    Compute theta across multiple event horizons.
    """
    rows = []
    n_total_before = int(df["depletion"].fillna(False).sum())
    df = prepare_theta_dataset(df, min_next_dt=min_next_dt, burst_threshold=burst_threshold)
    depletion_mask = df["depletion"].fillna(False)
    depletion_side = df["depletion_side"].to_numpy()
    p_mid = df["p_mid"].to_numpy()

    for horizon in horizons:
        p_mid_horizon = pd.Series(p_mid).shift(-horizon).to_numpy()
        valid = depletion_mask.to_numpy() & ~pd.isna(p_mid_horizon)
        continuation = (
            ((depletion_side == "ask") & (p_mid_horizon > p_mid))
            | ((depletion_side == "bid") & (p_mid_horizon < p_mid))
        )
        n_total = int(valid.sum())
        n_continuation = int((valid & continuation).sum())
        if n_total == 0:
            theta = np.nan
        else:
            theta = n_continuation / n_total
        rows.append(
            {
                "horizon": horizon,
                "n_total_before": n_total_before,
                "n_total": n_total,
                "n_continuation": n_continuation,
                "theta": theta,
            }
        )

    return pd.DataFrame(rows)


def p_ref_diagnostics(
    df: pd.DataFrame,
    theta_empirical: float,
    theta_fixed: float = 0.7,
    tick: float = 0.01,
) -> dict:
    """
    Validate p_ref range, frequency, and directional consistency.
    """
    p_ref_empirical = compute_p_ref_series(df, theta_empirical)
    p_ref_fixed = compute_p_ref_series(df, theta_fixed)

    dev_empirical = p_ref_empirical - df["p_mid"]
    dev_fixed = p_ref_fixed - df["p_mid"]
    allowed = np.array([-0.5 * tick, 0.0, 0.5 * tick])

    empirical_valid = np.isclose(dev_empirical.to_numpy()[:, None], allowed[None, :], atol=1e-9).any(axis=1)
    fixed_valid = np.isclose(dev_fixed.to_numpy()[:, None], allowed[None, :], atol=1e-9).any(axis=1)

    dep = df[df["depletion"].fillna(False)].copy()
    dep["p_ref_empirical"] = p_ref_empirical[df["depletion"].fillna(False)].to_numpy()
    dep["dev_empirical"] = dep["p_ref_empirical"] - dep["p_mid"]

    ask_dep = dep[dep["depletion_side"] == "ask"]
    bid_dep = dep[dep["depletion_side"] == "bid"]

    return {
        "p_ref_empirical": p_ref_empirical,
        "p_ref_fixed": p_ref_fixed,
        "valid_dev_pct_empirical": float(empirical_valid.mean() * 100),
        "valid_dev_pct_fixed": float(fixed_valid.mean() * 100),
        "invalid_dev_count_empirical": int((~empirical_valid).sum()),
        "invalid_dev_count_fixed": int((~fixed_valid).sum()),
        "dev_freq_empirical_pct": float((dev_empirical != 0).mean() * 100),
        "dev_freq_fixed_pct": float((dev_fixed != 0).mean() * 100),
        "ask_dep_above_mid_pct": float((ask_dep["dev_empirical"] > 0).mean() * 100) if not ask_dep.empty else np.nan,
        "bid_dep_below_mid_pct": float((bid_dep["dev_empirical"] < 0).mean() * 100) if not bid_dep.empty else np.nan,
    }


def estimate_queue_intensities(
    df: pd.DataFrame,
    queue_col: str = "q_before_aes",
    min_obs: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate AIT, Lambda(n), and per-type intensities from positive delta_t.

    Returns
    -------
    intensity_df : one row per queue size n
    dt_df        : underlying positive-delta_t observations for traceability
    """
    required = {queue_col, "delta_t", "eta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for intensity estimation: {sorted(missing)}")

    dt_df = df[[queue_col, "delta_t", "eta", "size"]].copy()
    dt_df = dt_df[dt_df[queue_col].notna()]
    dt_df = dt_df[dt_df["delta_t"] > 0].copy()
    if dt_df.empty:
        raise ValueError("No positive delta_t observations available for intensity estimation.")

    n_counts = dt_df.groupby(queue_col).size().rename("n_obs")
    ait = dt_df.groupby(queue_col)["delta_t"].mean().rename("ait")
    intensity_df = pd.concat([n_counts, ait], axis=1).reset_index().rename(columns={queue_col: "n"})
    intensity_df = intensity_df[intensity_df["n_obs"] >= min_obs].copy()
    if intensity_df.empty:
        raise ValueError(f"No queue sizes with at least {min_obs} positive delta_t observations.")
    intensity_df["Lambda"] = 1.0 / intensity_df["ait"]

    eta_counts = (
        dt_df.groupby([queue_col, "eta"]).size()
        .unstack(fill_value=0)
        .reindex(columns=["L", "C", "M"], fill_value=0)
    )
    eta_counts = eta_counts.reindex(intensity_df["n"], fill_value=0)
    eta_props = eta_counts.div(eta_counts.sum(axis=1), axis=0)
    lambda_base = intensity_df.set_index("n")["Lambda"]

    intensity_df["lambda_L"] = intensity_df["n"].map((lambda_base * eta_props["L"]).to_dict())
    intensity_df["lambda_C"] = intensity_df["n"].map((lambda_base * eta_props["C"]).to_dict())
    intensity_df["lambda_M"] = intensity_df["n"].map((lambda_base * eta_props["M"]).to_dict())

    return intensity_df.sort_values("n").reset_index(drop=True), dt_df


def compute_p_ref_series(df: pd.DataFrame, theta: float, seed: int = 0) -> pd.Series:
    """
    Build p_ref row-by-row from the current state only.

    Rule:
      - no depletion -> p_ref = p_mid
      - depletion + even spread -> shift with probability theta
      - all other rows -> p_ref = p_mid
    """
    if df.empty:
        return pd.Series(dtype=float)

    p_ref_values = []
    tick = 0.01
    rng = np.random.default_rng(seed)

    for _, row in df.iterrows():
        p_mid = row["p_mid"]
        if pd.isna(p_mid):
            p_ref_values.append(np.nan)
            continue

        spread_ticks = row["spread_ticks"]
        is_even_spread = pd.notna(spread_ticks) and spread_ticks > 0 and int(spread_ticks) % 2 == 0
        if row.get("depletion", False) and is_even_spread:
            if rng.random() < theta:
                if row["depletion_side"] == "ask":
                    new_p_ref = p_mid + 0.5 * tick
                elif row["depletion_side"] == "bid":
                    new_p_ref = p_mid - 0.5 * tick
                else:
                    new_p_ref = p_mid
            else:
                new_p_ref = p_mid
        else:
            new_p_ref = p_mid

        p_ref_values.append(new_p_ref)

    return pd.Series(p_ref_values, index=df.index)
# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Apply all QR preprocessing transformations in order.

    Parameters
    ----------
    df : event flow DataFrame from lob_reconstruction.build_event_flow(),
         with columns: side, level, eta, q_before, size, delta_t, p_ref, p_mid.

    Returns
    -------
    df_out : enriched DataFrame with additional columns:
               - q_before_aes  (int)  : queue size in AES units
               - period_id     (int)  : constant-p_ref period index
    aes    : pd.Series indexed by (side, level), AES in lots
    theta  : float, calibrated reference-price-change probability
    """
    df = df.sort_index()

    # Step 1 & 2: AES then quantize queue sizes
    aes = compute_aes(df)
    df = quantize_queue_sizes(df, aes)

    # Step 3: Estimate theta FIRST (depends only on depletion/p_mid, not p_ref)
    theta_empirical = calibrate_theta(df, horizon=1)

    # Step 4: Compute p_ref time series (requires theta; no future lookahead)
    df["p_ref_empirical"] = compute_p_ref_series(df, theta_empirical)
    df["p_ref_fixed"] = compute_p_ref_series(df, 0.7)

    # Step 5: Set canonical p_ref, then assign period_ids based on actual p_ref
    df["p_ref"] = df["p_ref_empirical"]
    df = assign_period_ids(df)

    return df, aes, theta_empirical


def build_qr_features(
    event_flow_path: str,
    output_path: str,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Load the event flow from disk, apply all QR transformations, and save.

    Parameters
    ----------
    event_flow_path : path to FGBL_event_flow.parquet
    output_path     : destination, e.g. data/processed/FGBL_qr_features.parquet

    Returns
    -------
    (df_out, aes, theta)
    """
    df = pd.read_parquet(event_flow_path)
    df_out, aes, theta = transform(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path)

    theta_sensitivity = theta_sensitivity_analysis(df)

    # Diagnostics: p_ref deviations from p_mid
    tick = 0.01
    tick_half = 0.5 * tick
    
    dev_empirical = df_out['p_ref_empirical'] - df_out['p_mid']
    dev_fixed = df_out['p_ref_fixed'] - df_out['p_mid']
    
    n_diff_empirical = (dev_empirical != 0).sum()
    n_diff_fixed = (dev_fixed != 0).sum()
    
    # Check bounds: deviations should be in {-0.5*tick, 0, +0.5*tick}
    valid_range = np.isclose(np.abs(dev_empirical[dev_empirical != 0]), tick_half, atol=1e-9).sum()
    valid_range_fixed = np.isclose(np.abs(dev_fixed[dev_fixed != 0]), tick_half, atol=1e-9).sum()
    p_ref_changes = df_out["p_ref_empirical"].diff().fillna(0) != 0
    allowed_change_mask = (
        df_out["depletion"].fillna(False)
        & df_out["spread_ticks"].notna()
        & (df_out["spread_ticks"] > 0)
        & (df_out["spread_ticks"] % 2 == 0)
    )
    invalid_changes = int((p_ref_changes & ~allowed_change_mask).sum())
    
    print(f"\nAES per (side, level):\n{aes.to_string()}\n")
    print(f"[QR Features] Calibrated theta = {theta:.4f}  (paper value: 0.70)")
    print(f"[QR Features] Theta sensitivity:\n{theta_sensitivity.to_string(index=False)}")
    print(f"[QR Features] p_ref_empirical: {n_diff_empirical:,} events differ from p_mid ({100*n_diff_empirical/len(df_out):.2f}%)")
    print(f"[QR Features] p_ref_fixed: {n_diff_fixed:,} events differ from p_mid ({100*n_diff_fixed/len(df_out):.2f}%)")
    print(f"[QR Features] Deviations in valid range (±{tick_half}): empirical={valid_range}/{n_diff_empirical}, fixed={valid_range_fixed}/{n_diff_fixed}")
    print(f"[QR Features] p_ref change violations outside depletion+even-spread rows: {invalid_changes}")
    print(f"[QR Features] Saved {len(df_out):,} events → {output_path}")

    return df_out, aes, theta
