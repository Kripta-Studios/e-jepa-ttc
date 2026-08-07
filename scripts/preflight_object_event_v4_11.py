#!/usr/bin/env python3
"""Preflight for the v4.11 train-only sign-router development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

COMPLETE_V49_STATUSES = {"fusion_screen_passed", "fusion_screen_failed"}
REQUIRED_PREDICTION_COLUMNS = {
    "sequence_id",
    "sample_token",
    "track_id",
    "delta_t_s",
    "target_ttc_s",
    "target_expansion",
    "base_prediction_expansion",
    "dense_prediction_expansion",
    "base_zero_events_expansion",
    "dense_zero_events_expansion",
    "base_shuffled_mean_expansion",
    "dense_shuffled_mean_expansion",
    "fused_prediction_expansion",
    "fused_zero_events_expansion",
    "fused_shuffled_mean_expansion",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v410-summary",
        type=Path,
        default=Path("artifacts/debug/object_event_v4_10_multiseed/summary.json"),
    )
    parser.add_argument(
        "--v49-run-root",
        type=Path,
        default=Path("artifacts/debug/object_event_v4_9_fixed_fusion"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiment/e_jepa_garl_object_event_sign_router_v4_11.yaml"
        ),
    )
    args = parser.parse_args()

    config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    seeds = tuple(int(seed) for seed in config_payload["router"]["seeds"])
    v410 = json.loads(args.v410_summary.read_text(encoding="utf-8"))
    if v410.get("artifact_type") != "object_event_v4_10_true_seed_fixed_fusion_robustness":
        raise RuntimeError("v4.10 summary has the wrong artifact type")
    if v410.get("status") not in {"robust_passed", "robust_failed"}:
        raise RuntimeError("v4.10 aggregate is incomplete")
    gates = v410.get("gates", {})
    required_positive_gates = {
        "ensemble_pearson",
        "ensemble_track_bootstrap_lower",
        "ensemble_weighted_mid",
        "ensemble_balanced_sign",
        "ensemble_negative_accuracy",
        "ensemble_expansion_mae",
        "ensemble_min_sequence_pearson",
        "pairwise_prediction_pearson",
        "mean_sample_prediction_std",
        "zero_event_dependence",
        "shuffled_event_dependence",
    }
    failed_required = sorted(name for name in required_positive_gates if not bool(gates.get(name)))
    if failed_required:
        raise RuntimeError(f"v4.10 is too weak for v4.11 routing: {failed_required}")
    if bool(gates.get("ensemble_min_sequence_negative_accuracy")):
        raise RuntimeError("v4.10 already solved the target v4.11 failure")

    seed_status: dict[str, object] = {}
    for seed in seeds:
        seed_root = args.v49_run_root / f"seed-{seed}"
        summary_path = seed_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("artifact_type") != "object_event_v4_9_fixed_event_fusion":
            raise RuntimeError(f"seed {seed} is not a v4.9 artifact")
        if summary.get("status") not in COMPLETE_V49_STATUSES:
            raise RuntimeError(f"seed {seed} is incomplete")
        if float(summary.get("fusion_config", {}).get("alpha", -1.0)) != 0.5:
            raise RuntimeError(f"seed {seed} does not use alpha=0.5")
        for split in ("train", "validation"):
            path = seed_root / f"{split}_predictions.csv"
            columns = set(pd.read_csv(path, nrows=2).columns)
            missing = sorted(REQUIRED_PREDICTION_COLUMNS - columns)
            if missing:
                raise RuntimeError(f"seed {seed} {split} predictions miss {missing}")
        seed_status[str(seed)] = {
            "status": summary["status"],
            "passed": bool(summary.get("passed")),
            "root": str(seed_root.resolve()),
        }

    result = {
        "status": "passed",
        "v410_status": v410["status"],
        "v410_failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "seeds": list(seeds),
        "seed_status": seed_status,
        "scientific_contract": {
            "v411_targets_systematic_negative_sign_failure_only": True,
            "validation_labels_are_not_used_for_router_fitting": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
