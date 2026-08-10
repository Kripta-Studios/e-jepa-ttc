#!/usr/bin/env python
"""Evaluate the frozen post-A4 A4D temporal-delta mechanism gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _pearson(node: object, level: str) -> float:
    if not isinstance(node, dict):
        raise ValueError("geometry diagnostic is missing")
    level_node = node.get(level)
    if not isinstance(level_node, dict):
        raise ValueError(f"geometry diagnostic lacks {level}")
    return float(level_node["pearson"])


def run(run_dir: Path, parent_summary_path: Path, output_path: Path) -> dict[str, Any]:
    child = _read(run_dir / "summary.json")
    parent = _read(parent_summary_path)
    if child.get("official_test_opened") is not False:
        raise ValueError("A4D diagnosis refuses a run that opened official test")

    training = child.get("training_config")
    if not isinstance(training, dict) or training.get("representation_supervision") != (
        "dinov3_local_relational_temporal_delta"
    ):
        raise ValueError("run is not the frozen A4D temporal-delta arm")
    contract = child.get("decision_contract")
    if not isinstance(contract, dict):
        raise ValueError("A4D run lacks decision_contract")
    gate = contract.get("temporal_delta_gate")
    if not isinstance(gate, dict):
        raise ValueError("A4D run lacks temporal_delta_gate")

    metrics = child.get("validation_metrics")
    parent_metrics = parent.get("validation_metrics")
    if not isinstance(metrics, dict) or not isinstance(parent_metrics, dict):
        raise ValueError("run summaries lack validation_metrics")
    geometry = metrics.get("geometry_diagnostics")
    parent_geometry = parent_metrics.get("geometry_diagnostics")
    if not isinstance(geometry, dict) or not isinstance(parent_geometry, dict):
        raise ValueError("run summaries lack geometry_diagnostics")

    ratio = float(metrics["log_ratio_pearson"])
    parent_ratio = float(parent_metrics["log_ratio_pearson"])
    delta_h = _pearson(geometry.get("delta_log_height_vs_physical"), "global")
    parent_delta_h = _pearson(
        parent_geometry.get("delta_log_height_vs_physical"), "global"
    )
    delta_h_macro = _pearson(
        geometry.get("delta_log_height_vs_physical"), "macro_by_sequence"
    )
    parent_delta_h_macro = _pearson(
        parent_geometry.get("delta_log_height_vs_physical"), "macro_by_sequence"
    )
    absolute_h_macro = _pearson(
        geometry.get("absolute_log_height"), "macro_by_sequence"
    )
    parent_absolute_h_macro = _pearson(
        parent_geometry.get("absolute_log_height"), "macro_by_sequence"
    )
    delta_w_macro = _pearson(
        geometry.get("delta_log_width_vs_physical"), "macro_by_sequence"
    )
    parent_delta_w_macro = _pearson(
        parent_geometry.get("delta_log_width_vs_physical"), "macro_by_sequence"
    )

    width_node = geometry.get("delta_log_width_vs_physical")
    if not isinstance(width_node, dict) or not isinstance(width_node.get("per_sequence"), dict):
        raise ValueError("A4D width dynamics lack per-sequence diagnostics")
    width_per_sequence = {
        str(sequence): float(values["pearson"])
        for sequence, values in width_node["per_sequence"].items()
        if isinstance(values, dict)
    }
    positive_width_sequences = sum(
        math.isfinite(value) and value > 0.0 for value in width_per_sequence.values()
    )

    ratio_threshold = float(gate["parent_a4_log_ratio_pearson"]) + float(
        gate["log_ratio_pearson_min_gain_vs_a4"]
    )
    delta_h_threshold = float(
        gate["parent_a4_delta_log_height_vs_physical_pearson"]
    ) + float(gate["delta_log_height_vs_physical_min_gain_vs_a4"])
    delta_h_macro_threshold = float(
        gate["parent_a4_delta_log_height_vs_physical_macro_pearson"]
    ) + float(gate["delta_log_height_vs_physical_macro_min_gain_vs_a4"])
    absolute_h_macro_floor = float(
        gate["parent_a4_absolute_log_height_macro_pearson"]
    ) - float(gate["absolute_log_height_macro_max_drop_vs_a4"])
    checks = {
        "log_ratio_gain": ratio >= ratio_threshold,
        "delta_height_gain": delta_h >= delta_h_threshold,
        "delta_height_macro_gain": delta_h_macro >= delta_h_macro_threshold,
        "absolute_height_macro_non_regression": (
            absolute_h_macro >= absolute_h_macro_floor
        ),
    }
    gate_pass = all(checks.values())

    history = child.get("history")
    temporal_curve: list[dict[str, float | int]] = []
    if isinstance(history, list):
        for row in history:
            if not isinstance(row, dict) or not isinstance(row.get("train"), dict):
                continue
            train = row["train"]
            raw = train.get("dinov3_relational_temporal_delta_raw")
            weighted = train.get("dinov3_relational_temporal_delta_weighted")
            if raw is not None and weighted is not None:
                temporal_curve.append(
                    {
                        "epoch": int(row["epoch"]),
                        "raw": float(raw),
                        "weighted": float(weighted),
                    }
                )

    selection = child.get("selection", {})
    parent_selection = parent.get("selection", {})
    result: dict[str, Any] = {
        "artifact_type": "a4d_temporal_delta_postrun_diagnosis_v1",
        "gate_pass": gate_pass,
        "checks": checks,
        "thresholds": {
            "log_ratio_pearson": ratio_threshold,
            "delta_log_height_vs_physical_pearson": delta_h_threshold,
            "delta_log_height_vs_physical_macro_pearson": (
                delta_h_macro_threshold
            ),
            "absolute_log_height_macro_pearson_floor": absolute_h_macro_floor,
        },
        "a4_parent": {
            "log_ratio_pearson": parent_ratio,
            "delta_log_height_vs_physical_pearson": parent_delta_h,
            "delta_log_height_vs_physical_macro_pearson": parent_delta_h_macro,
            "absolute_log_height_macro_pearson": parent_absolute_h_macro,
            "delta_log_width_vs_physical_macro_pearson": parent_delta_w_macro,
            "sequence_macro_MiD": parent_selection.get("sequence_macro_MiD"),
            "failure_rate_pct": parent_selection.get("failure_rate_pct"),
        },
        "a4d": {
            "log_ratio_pearson": ratio,
            "log_ratio_gain": ratio - parent_ratio,
            "delta_log_height_vs_physical_pearson": delta_h,
            "delta_log_height_gain": delta_h - parent_delta_h,
            "delta_log_height_vs_physical_macro_pearson": delta_h_macro,
            "delta_log_height_macro_gain": delta_h_macro - parent_delta_h_macro,
            "absolute_log_height_macro_pearson": absolute_h_macro,
            "absolute_log_height_macro_change": (
                absolute_h_macro - parent_absolute_h_macro
            ),
            "delta_log_width_vs_physical_macro_pearson": delta_w_macro,
            "delta_log_width_macro_gain": delta_w_macro - parent_delta_w_macro,
            "delta_log_width_vs_physical_per_sequence": width_per_sequence,
            "positive_width_dynamic_sequences": positive_width_sequences,
            "sequence_macro_MiD": selection.get("sequence_macro_MiD"),
            "failure_rate_pct": selection.get("failure_rate_pct"),
            "best_epoch": selection.get("best_epoch"),
        },
        "temporal_delta_training_curve": temporal_curve,
        "decision": (
            "TEMPORAL_RELATION_CHANGE_SUPPORTED"
            if gate_pass
            else "TEMPORAL_RELATION_CHANGE_NOT_YET_SUPPORTED"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.run_dir / "diagnostics" / "a4d_decision.json")
    run(args.run_dir.resolve(), args.parent_summary.resolve(), output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
