#!/usr/bin/env python
"""Paired sequence-cluster comparison of two LHR zero-shot artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np

from e_jepa_ttc.utils.io import read_structured, write_structured


def _identity(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("sequence_id", "")),
        str(row.get("sample_token", "")),
        str(row.get("track_id", "")),
        int(row.get("timestamp_us", 0)),
    )


def _rows(payload: dict[str, Any]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Comparison inputs require per-sample predictions.")
    result = {}
    for raw in rows:
        row = dict(raw)
        identity = _identity(row)
        if identity in result:
            raise ValueError(f"Duplicate prediction in one artifact: {identity}")
        result[identity] = row
    return result


def compare(
    control: dict[str, Any], candidate: dict[str, Any], *, iterations: int, seed: int
) -> dict[str, Any]:
    left, right = _rows(control), _rows(candidate)
    if set(left) != set(right):
        raise ValueError(
            f"Paired sample sets differ: control_only={len(set(left) - set(right))}, "
            f"candidate_only={len(set(right) - set(left))}."
        )
    pairs = []
    for identity in sorted(left):
        c, x = left[identity], right[identity]
        truth_c, truth_x = float(c["target_ttc_s"]), float(x["target_ttc_s"])
        if not np.isclose(truth_c, truth_x, atol=1e-6):
            raise ValueError(f"Target mismatch for {identity}.")
        c_abs = abs(float(c["predicted_ttc_s"]) - truth_c)
        x_abs = abs(float(x["predicted_ttc_s"]) - truth_c)
        denom = max(abs(truth_c), 0.25)
        pairs.append(
            {
                "sequence_id": identity[0],
                "delta_mae_s": x_abs - c_abs,
                "delta_relative_error": (x_abs - c_abs) / denom,
            }
        )
    by_sequence: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in pairs:
        by_sequence[row["sequence_id"]].append(row)
    sequence_ids = sorted(by_sequence)

    def estimate(selected: list[str]) -> tuple[float, float, float]:
        rows = [row for seq in selected for row in by_sequence[seq]]
        dmae = float(np.mean([row["delta_mae_s"] for row in rows]))
        dmre = float(np.mean([row["delta_relative_error"] for row in rows]))
        return dmae, dmre, dmre + 0.25 * dmae

    point = estimate(sequence_ids)
    rng = np.random.default_rng(seed)
    boot = [
        estimate([str(x) for x in rng.choice(sequence_ids, len(sequence_ids), replace=True)])
        for _ in range(iterations)
    ]
    names = (
        "candidate_minus_control_mae_s",
        "candidate_minus_control_mre",
        "candidate_minus_control_score",
    )
    return {
        "artifact_type": "eap_lhr_zero_shot_paired_comparison_v3",
        "sample_count": len(pairs),
        "sequence_count": len(sequence_ids),
        "bootstrap_iterations": iterations,
        "comparisons": {
            name: {
                "estimate": point[index],
                "lower": float(np.quantile([row[index] for row in boot], 0.025)),
                "upper": float(np.quantile([row[index] for row in boot], 0.975)),
                "candidate_better": float(np.quantile([row[index] for row in boot], 0.975)) < 0.0,
            }
            for index, name in enumerate(names)
        },
        "benchmark10_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = compare(
        read_structured(args.control),
        read_structured(args.candidate),
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_structured(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
