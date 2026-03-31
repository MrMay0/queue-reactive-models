"""
LOB Reconstruction — Step 2
============================
Converts clean MBO data into the event flow required by the queue-reactive model.

Output columns per row (one row = one order book event):
    ts        : event timestamp (CET)
    date      : trading date
    side      : 'B' or 'A'
    level     : 1 = best, 2 = second best, ..., up to K=5
    eta       : 'L' (Limit/Add), 'C' (Cancel), 'M' (Market/Trade)
    q_before  : queue size in lots BEFORE this event
    size      : size of this event in lots
    delta_t   : seconds since last event at this (side, level), within same p_ref period
    p_ref     : reference price at time of event (= mid-price for 1-tick spread)
    p_mid     : mid-price at time of event
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import time
from bisect import bisect_left, insort
from tqdm import tqdm


TICK      = 0.01   # FGBL minimum price increment
MAX_LEVEL = 5      # paper uses K=5 levels per side
ACTION_TO_ETA = {"A": "L", "C": "C", "T": "M"}


def _price_to_int(price: float) -> int:
    """Convert float price to integer ticks to avoid floating-point errors."""
    return round(price / TICK)


def _price_rank(price_levels: list[int], price_int: int, side: str) -> int | None:
    """
    Compute ordinal level from active price levels on one side of the book.

    Bid levels are ranked from highest to lowest price.
    Ask levels are ranked from lowest to highest price.
    """
    idx = bisect_left(price_levels, price_int)
    if idx >= len(price_levels) or price_levels[idx] != price_int:
        return None
    if side == "B":
        return len(price_levels) - idx
    return idx + 1


def reconstruct_day(df_raw_day: pd.DataFrame, fill_cancel_keys: set[tuple]) -> pd.DataFrame:
    """
    Reconstruct the event flow for a single trading day.

    Args:
        df_raw_day:     raw MBO dataframe for ONE day (actions A/C/T only).
                        Must have columns: action, side, price, size, order_id
                        and a datetime index named 'ts' (CET).
        fill_cancel_keys: set of keys identifying trade-induced C events that
                          are exact mirrors of passive F events. The T handler
                          already decremented the passive queue, so replaying
                          only those mirrored C rows would double-count.

    Returns:
        DataFrame with event flow for the 09:00–18:00 session only.
    """
    # --- LOB state ---
    # queues[(side, price_int)] = total size at that price level
    queues: dict = defaultdict(int)

    # Best bid/ask tracked as integers
    best_bid_int: int | None = None
    best_ask_int: int | None = None
    bid_prices: list[int] = []
    ask_prices: list[int] = []

    # p_ref tracked as integer x2 (= (best_bid_int + best_ask_int)) to stay integer
    p_ref_int_x2: int | None = None
    order_state: dict[int, tuple[str, int, int]] = {}

    # Last event timestamp per (side, level) — reset when p_ref changes
    last_ts: dict = {}   # (side, level) -> pd.Timestamp

    # Output rows
    rows = []

    session_start = time(9, 0)
    session_end = time(18, 0)
    def add_price_level(price_levels: list[int], price_int: int) -> None:
        idx = bisect_left(price_levels, price_int)
        if idx >= len(price_levels) or price_levels[idx] != price_int:
            insort(price_levels, price_int)

    def remove_price_level(price_levels: list[int], price_int: int) -> None:
        idx = bisect_left(price_levels, price_int)
        if idx < len(price_levels) and price_levels[idx] == price_int:
            price_levels.pop(idx)

    def update_order_state(order_id: int | None, side: str, price_int: int, delta: int) -> None:
        if order_id is None:
            return
        prev_side, prev_price_int, prev_size = order_state.get(order_id, (side, price_int, 0))
        new_size = prev_size + delta
        if new_size <= 0:
            order_state.pop(order_id, None)
        else:
            order_state[order_id] = (prev_side, prev_price_int, new_size)

    def process_atomic_event(
        ts,
        action: str,
        queue_side: str,
        price_int: int,
        size: int,
        eta: str,
        in_session: bool,
        order_id: int | None = None,
    ) -> None:
        nonlocal best_bid_int, best_ask_int, p_ref_int_x2

        if size <= 0:
            return

        q_before = queues[(queue_side, price_int)]

        old_p_ref_int_x2 = p_ref_int_x2
        old_best_bid_int = best_bid_int
        old_best_ask_int = best_ask_int
        old_bid_prices = bid_prices.copy()
        old_ask_prices = ask_prices.copy()
        old_spread_ticks = None
        old_p_mid = None
        if old_best_bid_int is not None and old_best_ask_int is not None:
            old_spread_ticks = old_best_ask_int - old_best_bid_int
            old_p_mid = (old_best_bid_int + old_best_ask_int) / 2 * TICK

        best_bid_qty_before = queues.get(("B", best_bid_int), 0) if best_bid_int is not None else 0
        best_ask_qty_before = queues.get(("A", best_ask_int), 0) if best_ask_int is not None else 0

        if action == "A":
            was_empty = queues[(queue_side, price_int)] == 0
            queues[(queue_side, price_int)] += size
            if was_empty:
                if queue_side == "B":
                    add_price_level(bid_prices, price_int)
                else:
                    add_price_level(ask_prices, price_int)
            update_order_state(order_id, queue_side, price_int, size)
        else:
            queues[(queue_side, price_int)] = max(0, queues[(queue_side, price_int)] - size)
            if queues[(queue_side, price_int)] == 0:
                if queue_side == "B":
                    remove_price_level(bid_prices, price_int)
                else:
                    remove_price_level(ask_prices, price_int)
            if action == "C":
                update_order_state(order_id, queue_side, price_int, -size)

        depletion = False
        depletion_side = None

        if action == "A":
            best_bid_int = bid_prices[-1] if bid_prices else None
            best_ask_int = ask_prices[0] if ask_prices else None
        else:
            if queue_side == "B" and old_best_bid_int == price_int:
                best_bid_qty_after = queues.get(("B", price_int), 0)
                if best_bid_qty_before > 0 and best_bid_qty_after == 0:
                    depletion = True
                    depletion_side = "bid"
            elif queue_side == "A" and old_best_ask_int == price_int:
                best_ask_qty_after = queues.get(("A", price_int), 0)
                if best_ask_qty_before > 0 and best_ask_qty_after == 0:
                    depletion = True
                    depletion_side = "ask"
            best_bid_int = bid_prices[-1] if bid_prices else None
            best_ask_int = ask_prices[0] if ask_prices else None

        if best_bid_int is not None and best_ask_int is not None:
            p_ref_int_x2 = best_bid_int + best_ask_int
        else:
            p_ref_int_x2 = None

        book_valid_before = (
            old_best_bid_int is not None
            and old_best_ask_int is not None
            and old_best_ask_int > old_best_bid_int
        )
        if in_session and old_p_mid is not None and old_p_ref_int_x2 is not None and book_valid_before:
            level = _price_rank(
                old_bid_prices if queue_side == "B" else old_ask_prices,
                price_int,
                queue_side,
            )
            if level is not None and 1 <= level <= MAX_LEVEL:
                key = (queue_side, level)
                if p_ref_int_x2 != old_p_ref_int_x2 or key not in last_ts:
                    delta_t = 0.0
                else:
                    delta_t = (ts - last_ts[key]).total_seconds()

                rows.append({
                    "ts"             : ts,
                    "date"           : ts.date(),
                    "side"           : queue_side,
                    "level"          : level,
                    "eta"            : eta,
                    "q_before"       : q_before,
                    "size"           : size,
                    "delta_t"        : delta_t,
                    "best_bid_int"   : old_best_bid_int,
                    "best_ask_int"   : old_best_ask_int,
                    "spread_ticks"   : old_spread_ticks,
                    "depletion"      : depletion,
                    "depletion_side" : depletion_side,
                    "p_mid"          : old_p_mid,
                    "p_ref"          : old_p_mid,
                })
                last_ts[key] = ts

        if p_ref_int_x2 != old_p_ref_int_x2:
            last_ts.clear()

    def walk_passive_book(
        ts,
        passive_side: str,
        remaining_size: int,
        in_session: bool,
        limit_price_int: int | None = None,
    ) -> int:
        """
        Consume passive liquidity from the current best outward.

        If ``limit_price_int`` is provided, stop once the next best passive
        price would violate the incoming limit price. This lets marketable
        limit orders execute aggressively first and leave any residual size to
        rest on their own side.
        """
        while remaining_size > 0:
            best_price = ask_prices[0] if passive_side == "A" and ask_prices else None
            if passive_side == "B" and bid_prices:
                best_price = bid_prices[-1]
            if best_price is None:
                break

            if limit_price_int is not None:
                crosses_limit = (
                    passive_side == "A" and best_price <= limit_price_int
                ) or (
                    passive_side == "B" and best_price >= limit_price_int
                )
                if not crosses_limit:
                    break

            available = queues[(passive_side, best_price)]
            if available <= 0:
                if passive_side == "B":
                    remove_price_level(bid_prices, best_price)
                else:
                    remove_price_level(ask_prices, best_price)
                continue

            consumed = min(remaining_size, available)
            process_atomic_event(
                ts=ts,
                action="T",
                queue_side=passive_side,
                price_int=best_price,
                size=consumed,
                eta="M",
                in_session=in_session,
                order_id=None,
            )
            remaining_size -= consumed

        return remaining_size

    for row in df_raw_day.itertuples():
        ts = row.Index
        action = row.action
        side = row.side
        price = row.price
        size = int(row.size)

        if pd.isna(price) or side not in ("B", "A"):
            continue

        price_int = _price_to_int(price)
        in_session = session_start <= ts.time() <= session_end

        if action == "T":
            passive_side = "A" if side == "B" else "B"
            walk_passive_book(
                ts=ts,
                passive_side=passive_side,
                remaining_size=size,
                in_session=in_session,
            )
            continue

        queue_side = side

        order_id = row.order_id

        if action == "M":
            prev_state = order_state.get(order_id)
            if prev_state is None:
                continue

            prev_side, prev_price_int, prev_size = prev_state
            if prev_side != side or prev_price_int != price_int:
                continue

            delta = size - prev_size
            order_state[order_id] = (side, price_int, size)
            if delta == 0:
                continue

            action = "A" if delta > 0 else "C"
            size = abs(delta)
            queue_side = side

        is_marketable_add = (
            action == "A"
            and (
                (side == "B" and best_ask_int is not None and price_int >= best_ask_int)
                or (side == "A" and best_bid_int is not None and price_int <= best_bid_int)
            )
        )
        if is_marketable_add:
            passive_side = "A" if side == "B" else "B"
            residual_size = walk_passive_book(
                ts=ts,
                passive_side=passive_side,
                remaining_size=size,
                in_session=in_session,
                limit_price_int=price_int,
            )
            if residual_size <= 0:
                continue
            size = residual_size

        cancel_key = (
            order_id,
            ts,
            row.sequence,
            side,
            price_int,
            size,
        )
        is_trade_cancel = action == "C" and cancel_key in fill_cancel_keys
        if is_trade_cancel:
            update_order_state(order_id, side, price_int, -size)
            continue

        process_atomic_event(
            ts=ts,
            action=action,
            queue_side=queue_side,
            price_int=price_int,
            size=size,
            eta=ACTION_TO_ETA[action],
            in_session=in_session,
            order_id=order_id,
        )

    if not rows:
        return pd.DataFrame()

    df_events = pd.DataFrame(rows).set_index("ts")
    df_events.index.name = "ts"
    return df_events


def build_event_flow(raw_dir: str, output_path: str) -> pd.DataFrame:
    """
    Run LOB reconstruction on all daily raw files and save event flow.

    Args:
        raw_dir:     path to data/raw/
        output_path: e.g. data/processed/FGBL_event_flow.parquet
    """
    raw_files = sorted(Path(raw_dir).glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    daily_flows = []
    for f in tqdm(raw_files, desc="Reconstructing LOB"):
        df_raw = pd.read_parquet(f)

        # Replay on exchange event time, not receive time.
        df_raw = df_raw.reset_index()
        df_raw["ts_event"] = pd.to_datetime(df_raw["ts_event"]).dt.tz_convert("Europe/Berlin")
        df_raw = df_raw.sort_values(["ts_event", "sequence", "ts_recv"]).set_index("ts_event")
        df_raw.index.name = "ts"

        # Collect the exact passive-fill mirrors BEFORE filtering F out.
        # In Eurex MBO data, every trade produces T (aggressor) → F (passive fill)
        # → C (passive order removed). We keep T for queue decrement and skip only
        # the exact mirrored C rows to avoid double-counting while still allowing
        # later true cancellations of the same order_id to reduce the queue.
        fill_cancel_keys = set(
            zip(
                df_raw.loc[df_raw["action"] == "F", "order_id"],
                df_raw.loc[df_raw["action"] == "F"].index,
                df_raw.loc[df_raw["action"] == "F", "sequence"],
                df_raw.loc[df_raw["action"] == "F", "side"],
                df_raw.loc[df_raw["action"] == "F", "price"].map(_price_to_int),
                df_raw.loc[df_raw["action"] == "F", "size"].astype(int),
            )
        )

        # Keep all state-changing messages needed for replay.
        df_raw = df_raw[df_raw["action"].isin(["A", "C", "T", "M"])]
        df_raw = df_raw[df_raw["side"].isin(["B", "A"])]
        df_raw = df_raw[~((df_raw["action"] == "T") & (df_raw["side"] == "N"))]

        df_flow = reconstruct_day(df_raw, fill_cancel_keys)
        if not df_flow.empty:
            daily_flows.append(df_flow)

    df_all = pd.concat(daily_flows).sort_index()
    df_all.to_parquet(output_path)
    print(f"Saved {len(df_all):,} events → {output_path}")
    return df_all
