#!/usr/bin/env python
"""Evaluate the frozen A5-CORR mechanistic gate against A4 seed7."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be mapping: {path}")
    return payload


def _pearson(node: object, level: str) -> float:
    if not isinstance(node, dict):
        raise ValueError("missing diagnostic node")
    child = node.get(level)
    if not isinstance(child, dict):
        raise ValueError(f"diagnostic lacks {level}")
    return float(child["pearson"])


def _per_sequence_pearson(node: object) -> dict[str, float]:
    if not isinstance(node, dict) or not isinstance(node.get("per_sequence"), dict):
        raise ValueError("diagnostic lacks per_sequence")
    return {
        str(sequence): float(metrics["pearson"])
        for sequence, metrics in node["per_sequence"].items()
    }


def run(child_path: Path, parent_path: Path, output_path: Path) -> dict[str, Any]:
    child = _read(child_path)
    parent = _read(parent_path)
    if child.get("official_test_opened") is not False or parent.get("official_test_opened") is not False:
        raise ValueError("A5 gate refuses summaries that opened official test")
    model_cfg = child.get("model_config")
    training = child.get("training_config")
    contract = child.get("decision_contract")
    if not isinstance(contract, dict):
        raise ValueError("A5 summary lacks decision_contract")
    gate = contract.get("transport_gate")
    if not isinstance(gate, dict):
        raise ValueError("A5 summary lacks transport_gate")
    if not isinstance(training, dict) or training.get("representation_supervision") != "dinov3_local_relational":
        raise ValueError("A5-CORR must keep A4 endpoint DINO and remove A4D temporal loss")
    # model_config in summary is metadata path/hash; architecture truth is also in checkpoint.
    # Require the registered A5 representation contract instead of inferring from filename.
    change = contract.get("representation_change")
    if not isinstance(change, dict) or change.get("type") != "a4_endpoint_dino_plus_event_native_local_cross_time_transport":
        raise ValueError("summary is not registered A5-CORR")

    cm = child.get("validation_metrics")
    pm = parent.get("validation_metrics")
    if not isinstance(cm, dict) or not isinstance(pm, dict):
        raise ValueError("summaries lack validation_metrics")
    cg = cm.get("geometry_diagnostics")
    pg = pm.get("geometry_diagnostics")
    if not isinstance(cg, dict) or not isinstance(pg, dict):
        raise ValueError("summaries lack geometry_diagnostics")

    ratio = float(cm["log_ratio_pearson"])
    parent_ratio = float(pm["log_ratio_pearson"])
    delta_h = _pearson(cg.get("delta_log_height_vs_physical"), "global")
    parent_delta_h = _pearson(pg.get("delta_log_height_vs_physical"), "global")
    delta_h_macro = _pearson(cg.get("delta_log_height_vs_physical"), "macro_by_sequence")
    parent_delta_h_macro = _pearson(pg.get("delta_log_height_vs_physical"), "macro_by_sequence")
    abs_h_macro = _pearson(cg.get("absolute_log_height"), "macro_by_sequence")
    parent_abs_h_macro = _pearson(pg.get("absolute_log_height"), "macro_by_sequence")

    child_per = _per_sequence_pearson(cg.get("delta_log_height_vs_physical"))
    parent_per = _per_sequence_pearson(pg.get("delta_log_height_vs_physical"))
    if set(child_per) != set(parent_per):
        raise ValueError("A4/A5 validation sequence sets differ")
    per_delta = {seq: child_per[seq] - parent_per[seq] for seq in sorted(child_per)}
    improved = sum(value > 0.0 for value in per_delta.values())
    worst_regression = min(per_delta.values()) if per_delta else float("nan")

    thresholds = {
        "ratio": parent_ratio + float(gate["log_ratio_pearson_min_gain_vs_a4"]),
        "delta_h": parent_delta_h + float(gate["delta_log_height_vs_physical_min_gain_vs_a4"]),
        "delta_h_macro": parent_delta_h_macro + float(gate["delta_log_height_vs_physical_macro_min_gain_vs_a4"]),
        "abs_h_macro": parent_abs_h_macro - float(gate["absolute_log_height_macro_max_drop_vs_a4"]),
        "min_improved_sequences": int(gate["minimum_improved_validation_sequences_for_delta_height"]),
        "worst_sequence_min_delta": -float(gate["worst_sequence_delta_height_max_regression"]),
    }
    checks = {
        "log_ratio": ratio >= thresholds["ratio"],
        "delta_height_global": delta_h >= thresholds["delta_h"],
        "delta_height_macro": delta_h_macro >= thresholds["delta_h_macro"],
        "absolute_height_non_regression": abs_h_macro >= thresholds["abs_h_macro"],
        "sequence_improvement_count": improved >= thresholds["min_improved_sequences"],
        "sequence_worst_regression": worst_regression >= thresholds["worst_sequence_min_delta"],
    }
    passed = all(checks.values())

    transport = cm.get("transport_diagnostics")
    transport_summary: dict[str, Any] | None = None
    if isinstance(transport, dict):
        physical = transport.get("against_physical_log_ratio")
        quality = transport.get("quality")
        transport_summary = {
            "against_physical_log_ratio": physical,
            "quality": quality,
        }

    signed = cm.get("signed") if isinstance(cm.get("signed"), dict) else {}
    sequence_macro = cm.get("sequence_macro") if isinstance(cm.get("sequence_macro"), dict) else {}
    parent_signed = pm.get("signed") if isinstance(pm.get("signed"), dict) else {}
    parent_sequence_macro = pm.get("sequence_macro") if isinstance(pm.get("sequence_macro"), dict) else {}
    result: dict[str, Any] = {
        "artifact_type": "a5_corr_mechanistic_gate_v1",
        "passed": passed,
        "checks": checks,
        "thresholds": thresholds,
        "metrics": {
            "log_ratio_pearson": ratio,
            "parent_log_ratio_pearson": parent_ratio,
            "delta_log_height_vs_physical": delta_h,
            "parent_delta_log_height_vs_physical": parent_delta_h,
            "delta_log_height_vs_physical_macro": delta_h_macro,
            "parent_delta_log_height_vs_physical_macro": parent_delta_h_macro,
            "absolute_log_height_macro": abs_h_macro,
            "parent_absolute_log_height_macro": parent_abs_h_macro,
            "per_sequence_delta_height_pearson_gain": per_delta,
            "improved_sequence_count": improved,
            "worst_sequence_gain": worst_regression,
            "MiD_macro": sequence_macro.get("mean_interval_distance"),
            "parent_MiD_macro": parent_sequence_macro.get("mean_interval_distance"),
            "failure_pct": signed.get("failure_rate_pct"),
            "parent_failure_pct": parent_signed.get("failure_rate_pct"),
        },
        "transport_diagnostics": transport_summary,
        "decision": "PROMOTE_TO_REPLICATION" if passed else "STOP_AND_ANALYZE_A5",
        "benchmark_metrics_secondary": True,
        "private_test_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-summary", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.child_summary.resolve(), args.parent_summary.resolve(), args.output.resolve())
    print(json.dumps({"passed": result["passed"], "decision": result["decision"], "checks": result["checks"]}, sort_keys=True))
    return 0 if result["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
