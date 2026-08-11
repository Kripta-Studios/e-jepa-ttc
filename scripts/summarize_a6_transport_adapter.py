#!/usr/bin/env python
"""Gate A6 transport-adapter runs against A4 geometry and replicated free-A5 temporal gain."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a mapping")
    return value

def metric(summary: dict[str, Any]) -> dict[str, float]:
    vm = summary["validation_metrics"]
    geo = vm["geometry_diagnostics"]
    transport = vm["transport_diagnostics"]["against_physical_log_ratio"]["foreground_divergence_y"]
    per_sequence = transport["per_sequence"]
    return {
        "log_ratio_pearson": float(vm["log_ratio_pearson"]),
        "sequence_macro_MiD": float(summary["selection"]["sequence_macro_MiD"]),
        "failure_rate_pct": float(summary["selection"]["failure_rate_pct"]),
        "delta_log_height_vs_physical": float(geo["delta_log_height_vs_physical"]["global"]["pearson"]),
        "delta_log_height_vs_physical_macro": float(geo["delta_log_height_vs_physical"]["macro_by_sequence"]["pearson"]),
        "absolute_log_height_macro": float(geo["absolute_log_height"]["macro_by_sequence"]["pearson"]),
        "foreground_divergence_y_global": float(transport["global"]["pearson"]),
        "foreground_divergence_y_macro": float(transport["macro_by_sequence"]["pearson"]),
        "positive_foreground_divergence_y_sequences": float(sum(float(v["pearson"]) > 0.0 for v in per_sequence.values())),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/experiment/e_jepa_garl_event_causal_scale_a6_transport_adapter_v1.yaml")
    ap.add_argument("--summary", action="append", required=True)
    ap.add_argument("--required-passes", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    protocol = read_yaml(ROOT / args.protocol)
    parent = protocol["parent_metrics"]
    free = protocol["replicated_free_a5_metrics"]
    gate = protocol["adapter_gate"]
    contract = protocol["adapter_contract"]

    parent_ratio = float(parent["log_ratio_pearson"])
    free_ratio = float(free["log_ratio_pearson_mean"])
    parent_mid = float(parent["sequence_macro_MiD"])
    free_mid = float(free["sequence_macro_MiD_mean"])
    ratio_den = free_ratio - parent_ratio
    mid_den = parent_mid - free_mid
    if ratio_den <= 0 or mid_den <= 0:
        raise ValueError("A6 preregistration has invalid parent/free-A5 ordering")

    runs = []
    passing = 0
    for ref in args.summary:
        summary = json.loads((ROOT / ref).read_text(encoding="utf-8"))
        values = metric(summary)
        ratio_recovery = (values["log_ratio_pearson"] - parent_ratio) / ratio_den
        mid_recovery = (parent_mid - values["sequence_macro_MiD"]) / mid_den
        init = summary.get("initialization", {})
        model = summary.get("model_config", summary.get("model_architecture", {}))
        checks = {
            "temporal_gain_recovery": ratio_recovery >= float(gate["temporal_gain_recovery_fraction_min"]),
            "MiD_gain_recovery": mid_recovery >= float(gate["MiD_gain_recovery_fraction_min"]),
            "failure": values["failure_rate_pct"] <= float(gate["maximum_failure_rate_pct"]),
            "delta_height_preserved": abs(values["delta_log_height_vs_physical"] - float(parent["delta_log_height_vs_physical"])) <= float(gate["geometry_absolute_tolerance"]),
            "delta_height_macro_preserved": abs(values["delta_log_height_vs_physical_macro"] - float(parent["delta_log_height_vs_physical_macro"])) <= float(gate["geometry_absolute_tolerance"]),
            "absolute_height_macro_preserved": abs(values["absolute_log_height_macro"] - float(parent["absolute_log_height_macro"])) <= float(gate["geometry_absolute_tolerance"]),
            "foreground_divergence_y_global": values["foreground_divergence_y_global"] >= float(gate["foreground_divergence_y_global_min"]),
            "foreground_divergence_y_macro": values["foreground_divergence_y_macro"] >= float(gate["foreground_divergence_y_macro_min"]),
            "foreground_divergence_y_sequences": int(values["positive_foreground_divergence_y_sequences"]) >= int(gate["minimum_positive_foreground_divergence_y_sequences"]),
            "encoder_frozen": init.get("freeze_encoder") is True,
            "complete_encoder_loaded": init.get("complete_encoder_loaded") is True,
            "adapter_enabled": bool(model.get("transport_adapter_enabled", False)) is True,
            "adapter_depth": int(model.get("transport_adapter_depth", -1)) == int(contract["transport_adapter_depth"]),
        }
        passed = all(checks.values())
        passing += int(passed)
        runs.append({
            "seed": int(summary["training_config"]["seed"]),
            "metrics": values,
            "temporal_gain_recovery_fraction": ratio_recovery,
            "MiD_gain_recovery_fraction": mid_recovery,
            "checks": checks,
            "passed": passed,
            "initialization": init,
        })
    overall = passing >= args.required_passes
    artifact = {
        "artifact_type": "a6_transport_adapter_gate_v1",
        "passed": overall,
        "passing_seeds": passing,
        "required_passing_seeds": args.required_passes,
        "runs": runs,
        "private_test_opened": False,
        "interpretation": "A6 must preserve frozen A4 geometry while recovering at least half of replicated free-A5 temporal gain",
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": overall, "passing_seeds": passing, "required": args.required_passes}, sort_keys=True))
    return 0 if overall else 3

if __name__ == "__main__":
    raise SystemExit(main())
