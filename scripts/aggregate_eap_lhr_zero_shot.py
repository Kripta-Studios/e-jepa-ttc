#!/usr/bin/env python
"""Aggregate disjoint LHR zero-shot predictions with sequence-cluster bootstrap."""

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


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    truth = np.asarray([float(row["target_ttc_s"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row["predicted_ttc_s"]) for row in rows], dtype=np.float64)
    error = np.abs(pred - truth)
    relative = error / np.maximum(np.abs(truth), 0.25)
    mae = float(error.mean())
    mre = float(relative.mean())
    return {
        "mae_s": mae,
        "mean_relative_error": mre,
        "selection_score": mre + 0.25 * mae,
    }


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {
        name: {"sample_count": len(values), **_metrics(values)}
        for name, values in sorted(grouped.items())
    }


def _cluster_bootstrap(rows: list[dict[str, Any]], *, iterations: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence_id"])].append(row)
    sequence_ids = sorted(grouped)
    if len(sequence_ids) < 2:
        return {"status": "insufficient_sequences", "sequence_count": len(sequence_ids)}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("mae_s", "mean_relative_error", "selection_score")}
    for _ in range(iterations):
        selected = rng.choice(sequence_ids, size=len(sequence_ids), replace=True)
        boot = [row for sequence in selected for row in grouped[str(sequence)]]
        metrics = _metrics(boot)
        for name in samples:
            samples[name].append(metrics[name])
    return {
        "status": "sequence_cluster_bootstrap",
        "iterations": iterations,
        "sequence_count": len(sequence_ids),
        "confidence": 0.95,
        **{
            name: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for name, values in samples.items()
        },
    }


def aggregate_payloads(
    payloads: list[dict[str, Any]], *, iterations: int = 2000, seed: int = 7
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("At least one zero-shot payload is required.")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for payload in payloads:
        if payload.get("artifact_type") not in {
            "eap_lhr_object_jepa_ttc_zero_shot_v3",
            "eap_lhr_object_jepa_ttc_zero_shot_v2",
        }:
            raise ValueError(
                f"Unexpected zero-shot artifact type: {payload.get('artifact_type')!r}."
            )
        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("Every input must contain per-sample predictions.")
        for raw in predictions:
            row = dict(raw)
            identity = _identity(row)
            if identity in seen:
                raise ValueError(f"Duplicate OOF prediction: {identity}")
            seen.add(identity)
            rows.append(row)
    result = {
        "artifact_type": "eap_lhr_zero_shot_oof_aggregate_v3",
        "input_count": len(payloads),
        "sample_count": len(rows),
        "sequence_count": len({str(row["sequence_id"]) for row in rows}),
        "metrics": _metrics(rows),
        "per_sequence": _group_metrics(rows, "sequence_id"),
        "per_category": _group_metrics(rows, "category"),
        "per_sampling_group": _group_metrics(rows, "sampling_group"),
        "bootstrap": _cluster_bootstrap(rows, iterations=iterations, seed=seed),
        "benchmark10_opened": False,
        "predictions": rows,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    payload = aggregate_payloads(
        [read_structured(path) for path in args.inputs],
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_structured(args.output, payload)
    print(
        json.dumps({key: value for key, value in payload.items() if key != "predictions"}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
