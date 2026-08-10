#!/usr/bin/env python
"""Compare preregistered A5 base/capacity arms after the transport gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("official_test_opened") is not False:
        raise ValueError(path)
    return value


def _p(node: object, level: str) -> float:
    if not isinstance(node, dict) or not isinstance(node.get(level), dict):
        return float("nan")
    return float(node[level]["pearson"])


def _m(summary: dict[str, Any]) -> dict[str, float]:
    vm = summary["validation_metrics"]
    gd = vm["geometry_diagnostics"]
    return {
        "log_ratio": float(vm["log_ratio_pearson"]),
        "delta_height_global": _p(gd["delta_log_height_vs_physical"], "global"),
        "delta_height_macro": _p(gd["delta_log_height_vs_physical"], "macro_by_sequence"),
        "absolute_height_macro": _p(gd["absolute_log_height"], "macro_by_sequence"),
        "MiD_macro": float(vm["sequence_macro"]["mean_interval_distance"]),
        "failure_pct": float(vm["signed"]["failure_rate_pct"]),
    }


def run(base_path: Path, cap_s_path: Path, cap_m_path: Path, output: Path) -> dict[str, Any]:
    entries = {
        "base": _read(base_path),
        "cap_s": _read(cap_s_path),
        "cap_m": _read(cap_m_path),
    }
    metrics = {name: _m(summary) for name, summary in entries.items()}
    params = {name: int(summary["parameter_count"]) for name, summary in entries.items()}
    base = metrics["base"]
    deltas = {
        name: {key: value - base[key] for key, value in row.items()}
        for name, row in metrics.items()
        if name != "base"
    }
    # Frozen capacity criterion: require +0.01 macro delta-height without >0.01
    # log-ratio regression, then prefer the smallest qualifying model.
    qualifying = []
    for name in ("cap_s", "cap_m"):
        if (
            deltas[name]["delta_height_macro"] >= 0.01
            and deltas[name]["log_ratio"] >= -0.01
            and deltas[name]["absolute_height_macro"] >= -0.01
        ):
            qualifying.append(name)
    selected = min(qualifying, key=lambda name: params[name]) if qualifying else "base"
    payload = {
        "artifact_type": "a5_capacity_comparison_v1",
        "metrics": metrics,
        "parameter_count": params,
        "delta_vs_base": deltas,
        "selection_contract": {
            "delta_height_macro_min_gain": 0.01,
            "log_ratio_max_regression": 0.01,
            "absolute_height_macro_max_regression": 0.01,
            "prefer_smallest_qualifying_model": True,
        },
        "qualifying_capacity_arms": qualifying,
        "selected_for_future_scaling": selected,
        "note": "This capacity comparison is separate from the A5-CORR causal transport test.",
        "official_test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--cap-s-summary", type=Path, required=True)
    parser.add_argument("--cap-m-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.base_summary.resolve(), args.cap_s_summary.resolve(), args.cap_m_summary.resolve(), args.output.resolve())
    print(json.dumps({"selected_for_future_scaling": payload["selected_for_future_scaling"], "qualifying": payload["qualifying_capacity_arms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
