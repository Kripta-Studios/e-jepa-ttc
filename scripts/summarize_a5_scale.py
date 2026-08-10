#!/usr/bin/env python
"""Summarize A5 data scaling (8k -> 16k) without inventing a new promotion gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("official_test_opened") is not False:
        raise ValueError(f"invalid/open-test summary: {path}")
    return value


def _p(node: object, level: str) -> float:
    if not isinstance(node, dict) or not isinstance(node.get(level), dict):
        return float("nan")
    return float(node[level]["pearson"])


def _metrics(summary: dict[str, Any]) -> dict[str, float]:
    vm = summary["validation_metrics"]
    gd = vm["geometry_diagnostics"]
    return {
        "log_ratio_pearson": float(vm["log_ratio_pearson"]),
        "delta_height_global": _p(gd["delta_log_height_vs_physical"], "global"),
        "delta_height_macro": _p(gd["delta_log_height_vs_physical"], "macro_by_sequence"),
        "absolute_height_macro": _p(gd["absolute_log_height"], "macro_by_sequence"),
        "MiD_macro": float(vm["sequence_macro"]["mean_interval_distance"]),
        "failure_pct": float(vm["signed"]["failure_rate_pct"]),
    }


def run(base_path: Path, scaled_path: Path, output: Path) -> dict[str, Any]:
    base = _read(base_path)
    scaled = _read(scaled_path)
    b = _metrics(base)
    s = _metrics(scaled)
    delta = {key: s[key] - b[key] for key in b}
    payload = {
        "artifact_type": "a5_data_scale_comparison_v1",
        "base_summary": str(base_path),
        "scaled_summary": str(scaled_path),
        "base_metrics": b,
        "scaled_metrics": s,
        "delta_scaled_minus_base": delta,
        "interpretation_contract": {
            "this_is_not_a_new_posthoc_promotion_gate": True,
            "private_test_remains_closed": True,
            "same_validation_should_be_frozen_across_scale": True,
        },
        "official_test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--scaled-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.base_summary.resolve(), args.scaled_summary.resolve(), args.output.resolve())
    print(json.dumps(payload["delta_scaled_minus_base"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
