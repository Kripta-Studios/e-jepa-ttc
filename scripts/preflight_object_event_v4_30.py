#!/usr/bin/env python3
"""Strict, metadata-only v4.30 preflight; it never materializes sealed splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.train_e_jepa_object_event_v4_12 import _load_backbone  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_protocol_sha256(raw: dict[str, Any]) -> str:
    """Hash the normalized locked mapping separately from the authored YAML bytes."""
    return hashlib.sha256(yaml.safe_dump(raw, sort_keys=True).encode("utf-8")).hexdigest()


def parse_seed_paths(items: list[str]) -> dict[int, Path]:
    parsed = {int(item.split("=", 1)[0]): Path(item.split("=", 1)[1]) for item in items}
    if sorted(parsed) != [7, 13, 23] or len(parsed) != len(items):
        raise ValueError("v4.30 requires exactly one adapted checkpoint for seeds 7,13,23")
    return parsed


def validate_config(raw: dict[str, Any]) -> None:
    """Reject protocol drift before any output directory is touched."""
    if set(raw) != {
        "arms",
        "stabilization",
        "controls",
        "train",
        "loss",
        "selection",
        "arm_b_margin",
        "development",
    }:
        raise ValueError("v4.30 config top-level schema differs from the locked protocol")
    expected_arms = ["stable_multiscale_similarity", "stable_multiscale_similarity_normal_flow"]
    if sorted(raw["arms"]) != expected_arms:
        raise ValueError("v4.30 requires exactly the two stable similarity arms")
    expected_arm = {
        "correlation_dim": 48,
        "scales": [1, 2, 4],
        "temperature": 0.07,
        "ridge": 0.01,
        "huber_delta": 0.08,
        "huber_passes": 3,
        "tile_size": 4,
        "batch_size": 8,
    }
    if any(arm != expected_arm for arm in raw["arms"].values()):
        raise ValueError("v4.30 arm controls are not exactly locked")
    if raw["stabilization"] != {
        "geometry_checkpoint_seeds": [7, 13, 23],
        "student_seeds": [7, 13, 23],
        "checkpoint_ema_epochs": [8, 9, 10],
        "whitening_shrinkage": 0.10,
        "js_median_max": 0.02,
        "js_p95_max": 0.08,
        "displacement_disagreement_p95_max_feature_px": 0.5,
    }:
        raise ValueError("v4.30 stabilization protocol changed")
    if raw["controls"] != {
        "zero_event": True,
        "temporal_shuffle_permutation": [2, 0, 1],
        "endpoint_swap_permutation": [0, 2, 1],
    }:
        raise ValueError("v4.30 stage-2 control protocol changed")
    expected_train = {
        "fold_count": 3,
        "epochs": 10,
        "final_epochs": 12,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "max_grad_norm": 1.0,
        "num_workers": 0,
        "optimization_seed_by_fold": {0: 43000, 1: 43001, 2: 43002},
        "final_training_seed_by_student": {7: 430107, 13: 430113, 23: 430123},
        "magnitude_buckets": [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, float("inf")]],
    }
    if raw["train"] != expected_train:
        raise ValueError("v4.30 grouped OOF RNG/training protocol changed")
    if raw["loss"] != {
        "g_weight": 1.0,
        "ell_weight": 1.0,
        "bucket_ratio_weight": 0.5,
        "sign_weight": 0.25,
        "track_weight": 0.25,
        "support_weight": 0.25,
        "distill_weight": 0.25,
        "cycle_weight": 0.1,
        "normal_flow_weight": 0.25,
        "smooth_l1_beta": 0.004,
        "sign_temperature": 0.015,
        "max_abs_g": 0.24975,
    }:
        raise ValueError("v4.30 loss weights are not frozen")
    expected_selection = {
        "oof_rows_per_constituent": 2048,
        "pearson": 0.777,
        "log_eta_pearson": 0.758,
        "minimum_sequence_pearson": 0.50,
        "negative_accuracy": 0.84,
        "balanced_sign_accuracy": 0.89,
        "negative_track_macro_accuracy": 0.857,
        "minimum_negative_track_accuracy": 0.50,
        "eligible_negative_track_p10": 0.65,
        "high_bucket_pearson": 0.45,
        "seed_prediction_p95_range": 0.02,
        "seed_prediction_max_range": 0.08,
        "seed_sign_disagreement": 0.02,
        "seed_pearson_range": 0.02,
        "std_ratio": [0.90, 1.10],
        "calibration_slope": [0.90, 1.10],
        "magnitude_ratio": [0.85, 1.15],
        "shuffle_ratio_max": 0.50,
        "endpoint_swap_pearson_max": 0.15,
        "bottom_support_seed_p95_range": 0.05,
    }
    if raw["selection"] != expected_selection:
        raise ValueError("v4.30 selection gates are not exactly locked")
    if raw["arm_b_margin"] != {
        "paired_sequence_pearson": 0.01,
        "high_bucket_pearson": 0.10,
        "negative_track_macro_accuracy": 0.02,
        "shuffle_ratio_reduction": 0.10,
    }:
        raise ValueError("v4.30 Arm-B margins are not exactly locked")
    if raw["development"] != {
        "minimum_pearson_gain_over_v410": 0.005,
        "minimum_negative_accuracy_gain_over_v410": 0.020,
        "minimum_balanced_sign_gain_over_v410": 0.010,
        "minimum_log_eta_pearson": 0.758,
        "minimum_negative_track_macro_gain_over_v410": 0.020,
        "minimum_relative_paper_weighted_mid_improvement_over_v410": 0.020,
        "paper_mid_weights": {"crucial": 0.5, "small": 0.3, "large": 0.1, "negative": 0.1},
    }:
        raise ValueError("v4.30 development decision protocol changed")


def validate_checkpoints(checkpoints: dict[int, Path], v48_config: Path) -> dict[str, str]:
    """Strict-load every checkpoint before any destructive output handling."""
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for seed, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source = payload.get("source_checkpoint") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("artifact_type") != "object_event_v4_22_adapted_v48"
            or int(payload.get("seed", -1)) != seed
            or not isinstance(payload.get("model_state_dict"), dict)
            or not isinstance(source, str)
            or not source.replace("\\", "/").endswith(f"/screen-seed-{seed}/best_gate_passing.pt")
        ):
            raise ValueError(f"checkpoint seed {seed} has invalid v4.22 provenance")
        _load_backbone(v48_config_path=v48_config, checkpoint_path=path)
        digest = sha256(path)
        if digest in seen:
            raise ValueError("duplicate checkpoint content is forbidden")
        seen.add(digest)
        hashes[str(seed)] = digest
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v429-summary", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
    args = parser.parse_args()
    for path in (args.cache_manifest, args.v48_config, args.config, args.v429_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    v429 = json.loads(args.v429_summary.read_text(encoding="utf-8"))
    if v429.get("status") != "completed_oof_gate_failed" or not v429.get(
        "scientific_contract", {}
    ).get("development_validation_not_materialized_after_oof_failure"):
        raise ValueError("v4.29 sealed OOF-failure provenance is required")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(raw)
    hashes = validate_checkpoints(parse_seed_paths(args.adapted_checkpoint), args.v48_config)
    manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "passed",
                "config_file_sha256": sha256(args.config),
                "canonical_protocol_sha256": canonical_protocol_sha256(raw),
                "v48_config_file_sha256": sha256(args.v48_config),
                "v429_summary_file_sha256": sha256(args.v429_summary),
                "cache_manifest_sha256": sha256(args.cache_manifest),
                "cache_manifest_keys": sorted(manifest),
                "checkpoint_sha256": hashes,
                "development_validation_materialized": False,
                "official_eap_test_opened": False,
                "evttc_opened": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
