from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common import _blend_joint_distribution, calibrate_common
from models.qr import QRSimulator, calibrate_qr
from models.saqr import SAQRSimulator, calibrate_saqr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short SAQR performance benchmark.")
    parser.add_argument("--event-flow", default=str(ROOT / "data/processed/FGBL_event_flow.parquet"))
    parser.add_argument("--raw-dir", default=str(ROOT / "data/raw"))
    parser.add_argument("--day", default="2025-11-03")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--smoothing-alpha", type=float, default=25.0)
    return parser.parse_args()


def simulate_events(simulator, q0: int, horizon_seconds: float, seed: int) -> tuple[int, float]:
    rng = np.random.default_rng(seed)
    lookup_rates = simulator._lookup_rates
    sample_event = simulator._sample_event
    apply_event = simulator._apply_event
    sample_reset = simulator.calibration.sample_reset_queue

    elapsed = 0.0
    queue_size = int(q0)
    n_events = 0

    t0 = time.perf_counter()
    while elapsed < horizon_seconds:
        n, rates = lookup_rates(queue_size)
        lambda_total = float(rates.sum())
        if lambda_total <= 0:
            break
        dt = float(rng.exponential(1.0 / lambda_total))
        if elapsed + dt > horizon_seconds:
            break
        eta, size = sample_event(n, queue_size, rng)
        queue_after = apply_event(queue_size, eta, size)
        queue_size = int(sample_reset(rng)) if queue_after == 0 else int(queue_after)
        elapsed += dt
        n_events += 1

    runtime = time.perf_counter() - t0
    return n_events, runtime


def compare_cached_vs_reference(common, smoothing_alpha: float) -> list[dict[str, float | int]]:
    tables = common.build_joint_sampler_tables(smoothing_alpha)
    support = [int(n) for n in common.support_n[:3]]
    if common.support_n.size > 0:
        support.append(int(common.support_n[-1]))

    prior_pairs, prior_probs, _ = common._joint_prior
    rows = []
    for n in dict.fromkeys(support):
        pairs_n, probs_n, count_n = common._joint_probs.get(int(n), common._joint_prior)
        ref_pairs, ref_probs = _blend_joint_distribution(
            pairs_n,
            probs_n,
            count_n,
            prior_pairs,
            prior_probs,
            smoothing_alpha=smoothing_alpha,
        )
        ref_map = {pair: prob for pair, prob in zip(ref_pairs, ref_probs)}

        table = tables.get(int(n), tables[-1])
        cached_map = {
            (str(eta), int(size)): float(prob)
            for eta, size, prob in zip(table["etas"], table["sizes"], table["probs"])
        }
        keys = sorted(set(ref_map) | set(cached_map))
        max_abs_prob_diff = max(abs(ref_map.get(key, 0.0) - cached_map.get(key, 0.0)) for key in keys)

        ref_eta = {"L": 0.0, "C": 0.0, "M": 0.0}
        cached_eta = {"L": 0.0, "C": 0.0, "M": 0.0}
        for (eta, _), prob in ref_map.items():
            ref_eta[str(eta)] += float(prob)
        for (eta, _), prob in cached_map.items():
            cached_eta[str(eta)] += float(prob)

        max_abs_eta_diff = max(abs(ref_eta[k] - cached_eta[k]) for k in ref_eta)
        rows.append(
            {
                "n": int(n),
                "support_size": len(keys),
                "max_abs_joint_prob_diff": float(max_abs_prob_diff),
                "max_abs_eta_marginal_diff": float(max_abs_eta_diff),
            }
        )
    return rows


def main() -> int:
    args = parse_args()

    common = calibrate_common(
        args.event_flow,
        raw_dir=args.raw_dir,
        level=args.level,
        min_obs=args.min_obs,
    )
    q0 = common.sample_reset_queue(np.random.default_rng(args.seed))

    qr_sim = QRSimulator(
        calibrate_qr(
            args.event_flow,
            raw_dir=args.raw_dir,
            level=args.level,
            min_obs=args.min_obs,
            common=common,
        )
    )
    saqr_cal = calibrate_saqr(
        args.event_flow,
        raw_dir=args.raw_dir,
        level=args.level,
        min_obs=args.min_obs,
        smoothing_alpha=args.smoothing_alpha,
        common=common,
    )
    saqr_sim = SAQRSimulator(saqr_cal)

    comparison_rows = compare_cached_vs_reference(common, args.smoothing_alpha)
    print("Distribution consistency check")
    for row in comparison_rows:
        print(
            f"n={row['n']:>3} support={row['support_size']:>4} "
            f"max|p_new-p_ref|={row['max_abs_joint_prob_diff']:.3e} "
            f"max|eta_new-eta_ref|={row['max_abs_eta_marginal_diff']:.3e}"
        )

    qr_events, qr_runtime = simulate_events(qr_sim, q0=q0, horizon_seconds=args.seconds, seed=args.seed)
    saqr_events, saqr_runtime = simulate_events(saqr_sim, q0=q0, horizon_seconds=args.seconds, seed=args.seed)

    qr_eps = qr_events / qr_runtime if qr_runtime > 0 else float("nan")
    saqr_eps = saqr_events / saqr_runtime if saqr_runtime > 0 else float("nan")
    ratio = qr_eps / saqr_eps if saqr_eps > 0 else float("inf")

    print("\nPerformance benchmark")
    print(f"QR   : events={qr_events} runtime={qr_runtime:.3f}s events_per_sec={qr_eps:.1f}")
    print(f"SAQR : events={saqr_events} runtime={saqr_runtime:.3f}s events_per_sec={saqr_eps:.1f}")
    print(f"QR/SAQR speed ratio: {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
