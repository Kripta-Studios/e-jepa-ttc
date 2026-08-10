#!/usr/bin/env python
"""Summarize preregistered A5-CORR seed7/13/23 replication without test access."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    if value.get("official_test_opened") is not False:
        raise ValueError(f"replication summary refuses test-opened run: {path}")
    return value


def _pearson(node: object, level: str) -> float:
    if not isinstance(node, dict) or not isinstance(node.get(level), dict):
        return float("nan")
    return float(node[level].get("pearson", float("nan")))


def _metrics(summary: dict[str, Any]) -> dict[str, float]:
    vm = summary.get("validation_metrics")
    if not isinstance(vm, dict):
        raise ValueError("summary lacks validation_metrics")
    gd = vm.get("geometry_diagnostics")
    if not isinstance(gd, dict):
        raise ValueError("summary lacks geometry diagnostics")
    signed = vm.get("signed") if isinstance(vm.get("signed"), dict) else {}
    macro = vm.get("sequence_macro") if isinstance(vm.get("sequence_macro"), dict) else {}
    return {
        "log_ratio_pearson": float(vm.get("log_ratio_pearson", float("nan"))),
        "delta_height_global": _pearson(gd.get("delta_log_height_vs_physical"), "global"),
        "delta_height_macro": _pearson(gd.get("delta_log_height_vs_physical"), "macro_by_sequence"),
        "absolute_height_macro": _pearson(gd.get("absolute_log_height"), "macro_by_sequence"),
        "MiD_macro": float(macro.get("mean_interval_distance", float("nan"))),
        "failure_pct": float(signed.get("failure_rate_pct", float("nan"))),
        "known_coverage": float(vm.get("known_coverage", float("nan"))),
    }


def run(paths: list[Path], output: Path) -> dict[str, Any]:
    summaries = [_read(path) for path in paths]
    rows = []
    for path, summary in zip(paths, summaries, strict=True):
        training = summary.get("training_config")
        if not isinstance(training, dict):
            raise ValueError("summary lacks training_config")
        row = {"seed": int(training["seed"]), **_metrics(summary), "path": str(path)}
        rows.append(row)
    seeds = sorted(int(row["seed"]) for row in rows)
    if seeds != [7, 13, 23]:
        raise ValueError(f"replication requires seeds [7,13,23], got {seeds}")

    seed7 = summaries[[int(s["training_config"]["seed"]) for s in summaries].index(7)]
    contract = seed7.get("decision_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("transport_gate"), dict):
        raise ValueError("seed7 lacks A5 transport gate")
    gate = contract["transport_gate"]
    thresholds = {
        "log_ratio_pearson": float(gate["parent_a4_log_ratio_pearson"]) + float(gate["log_ratio_pearson_min_gain_vs_a4"]),
        "delta_height_global": float(gate["parent_a4_delta_log_height_vs_physical_pearson"]) + float(gate["delta_log_height_vs_physical_min_gain_vs_a4"]),
        "delta_height_macro": float(gate["parent_a4_delta_log_height_vs_physical_macro_pearson"]) + float(gate["delta_log_height_vs_physical_macro_min_gain_vs_a4"]),
        "absolute_height_macro": float(gate["parent_a4_absolute_log_height_macro_pearson"]) - float(gate["absolute_log_height_macro_max_drop_vs_a4"]),
    }
    per_seed_support = {}
    for row in rows:
        support = all(
            math.isfinite(float(row[name])) and float(row[name]) >= threshold
            for name, threshold in thresholds.items()
        )
        per_seed_support[str(row["seed"])] = support
    support_count = sum(per_seed_support.values())

    numeric = [key for key in rows[0] if key not in {"seed", "path"}]
    aggregate = {}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
    payload = {
        "artifact_type": "a5_corr_seed_replication_summary_v1",
        "seeds": rows,
        "aggregate": aggregate,
        "fixed_absolute_mechanistic_thresholds": thresholds,
        "per_seed_support": per_seed_support,
        "support_count": support_count,
        "replication_supported": support_count >= 2,
        "interpretation": (
            "Seeds 13/23 are replication only. The absolute threshold reuses the preregistered A4 seed7 gate; "
            "it is not a seed-specific A4 causal comparison."
        ),
        "official_test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run([p.resolve() for p in args.summary], args.output.resolve())
    print(json.dumps({"replication_supported": payload["replication_supported"], "support_count": payload["support_count"]}, sort_keys=True))
    return 0 if payload["replication_supported"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
