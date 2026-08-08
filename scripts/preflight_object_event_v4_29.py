#!/usr/bin/env python3
"""Validate v4.29 inputs without materialising development validation or tests."""

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


def parse_seed_paths(items: list[str]) -> dict[int, Path]:
    parsed = {int(item.split("=", 1)[0]): Path(item.split("=", 1)[1]) for item in items}
    if sorted(parsed) != [7, 13, 23]:
        raise ValueError("v4.29 requires exactly checkpoint seeds 7,13,23")
    if len(parsed) != len(items):
        raise ValueError("duplicate checkpoint seed")
    return parsed


def validate_config(raw: dict[str, Any]) -> None:
    if set(raw) != {"arms", "loss", "train", "selection", "final_decision"}:
        raise ValueError("v4.29 config top-level keys differ from preregistration")
    expected_arm = {
        "correlation_dim": 48,
        "adjacent_radius": 4,
        "direct_radius": 7,
        "temperature": 0.07,
        "ridge": 0.001,
        "huber_delta": 0.08,
        "foreground_floor": 0.05,
        "activity_floor": 0.05,
        "support_threshold": 0.02,
        "min_effective_mass": 4.0,
        "max_condition_number": 100.0,
        "min_determinant": 0.05,
        "batch_size": 8,
    }
    if sorted(raw.get("arms", {})) != ["local_affine_geom_teacher", "local_affine_lhr"]:
        raise ValueError("config must predeclare exactly the two v4.29 arms")
    train = raw.get("train", {})
    if train.get("seeds") != [7, 13, 23] or train.get("matcher_init_seeds") != [7, 13, 23]:
        raise ValueError("fixed v4.29 seed matrix must be [7,13,23] x [7,13,23]")
    if (
        train.get("num_workers") != 0
        or train.get("epochs") != 10
        or train.get("final_epochs") != 12
    ):
        raise ValueError("v4.29 requires no workers and 10/12 epoch schedule")
    for name, arm in raw["arms"].items():
        if arm != expected_arm:
            raise ValueError(f"{name} does not exactly match the locked arm protocol")
        locked = {
            "adjacent_radius": 4,
            "direct_radius": 7,
            "temperature": 0.07,
            "ridge": 0.001,
            "huber_delta": 0.08,
            "max_condition_number": 100.0,
            "min_determinant": 0.05,
            "batch_size": 8,
        }
        if any(float(arm.get(k, float("nan"))) != float(v) for k, v in locked.items()):
            raise ValueError(f"{name} violates a locked local-affine control")
    gates = raw.get("selection", {}).get("gates", {})
    required = {
        "pearson",
        "negative_accuracy",
        "balanced_sign_accuracy",
        "log_eta_pearson",
        "negative_track_macro_accuracy",
        "minimum_sequence_pearson",
        "prediction_std_ratio",
        "calibration_slope_intercept",
        "invalid_affine_fraction",
        "high_magnitude_ratio",
        "gain_over_v428",
    }
    if set(gates) != required:
        raise ValueError("v4.29 config lacks preregistered OOF gates")
    exact = {
        "pearson": 0.635,
        "negative_accuracy": 0.652021,
        "balanced_sign_accuracy": 0.775,
        "log_eta_pearson": 0.615,
        "negative_track_macro_accuracy": 0.697674,
        "minimum_sequence_pearson": 0.430,
        "invalid_affine_fraction": 0.02,
    }
    if any(abs(float(gates[key]) - value) > 1e-9 for key, value in exact.items()):
        raise ValueError("v4.29 absolute gates do not match preregistration")
    if (
        gates["prediction_std_ratio"] != [0.75, 1.25]
        or gates["calibration_slope_intercept"] != [0.70, 1.30]
        or gates["high_magnitude_ratio"] != [0.70, 1.30]
        or gates["gain_over_v428"]
        != {
            "pearson": 0.025,
            "log_eta_pearson": 0.015,
            "negative_accuracy": 0.020,
            "minimum_sequence_pearson": 0.100,
        }
    ):
        raise ValueError("v4.29 range/gain gates do not match preregistration")
    if raw.get("final_decision", {}) != {
        "minimum_pearson_gain_over_v410": 0.005,
        "minimum_negative_accuracy_gain_over_v410": 0.020,
        "minimum_balanced_sign_gain_over_v410": 0.010,
        "minimum_log_eta_pearson": 0.450,
        "minimum_negative_track_macro_gain_over_v410": 0.020,
    }:
        raise ValueError("v4.29 final decision thresholds do not match preregistration")
    if raw.get("loss", {}) != {
        "lhr_weight": 4.0,
        "expansion_weight": 1.0,
        "correlation_weight": 1.0,
        "sign_weight": 1.0,
        "confidence_weight": 0.05,
        "composition_weight": 0.20,
        "residual_weight": 0.05,
        "invalid_weight": 2.0,
        "geometry_teacher_weight": 0.50,
        "smooth_l1_beta": 0.004,
        "sign_temperature": 0.015,
        "max_abs_expansion": 0.25,
    }:
        raise ValueError("v4.29 loss protocol does not match preregistration")
    if train != {
        "fold_count": 3,
        "seeds": [7, 13, 23],
        "matcher_init_seeds": [7, 13, 23],
        "epochs": 10,
        "final_epochs": 12,
        "geometry_tail_tensors": 8,
        "projection_learning_rate": 0.0001,
        "geometry_learning_rate": 0.00001,
        "weight_decay": 0.0001,
        "geometry_anchor_weight": 0.01,
        "max_grad_norm": 1.0,
        "optimization_seed_by_fold": {0: 42900, 1: 42901, 2: 42902},
        "num_workers": 0,
    }:
        raise ValueError("v4.29 train protocol does not match preregistration")
    if raw.get("selection", {}).get("objective") != "pearson" or raw.get("selection", {}).get(
        "tie_break"
    ) != ["minimum_sequence_pearson", "negative_accuracy", "lexical_arm_name"]:
        raise ValueError("v4.29 selection objective/tie-break differs from preregistration")
    if set(raw["selection"]) != {"objective", "tie_break", "gates"}:
        raise ValueError("v4.29 selection keys differ from preregistration")


def validate_checkpoints(checkpoints: dict[int, Path], v48_config: Path) -> dict[str, str]:
    """Validate exact v4.22 seed provenance and reject duplicate artifacts."""
    hashes: dict[str, str] = {}
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for seed, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source = payload.get("source_checkpoint") if isinstance(payload, dict) else None
        normalized_source = str(source).replace("\\", "/")
        if (
            not isinstance(payload, dict)
            or payload.get("artifact_type") != "object_event_v4_22_adapted_v48"
            or int(payload.get("seed", -1)) != seed
            or not isinstance(payload.get("model_state_dict"), dict)
            or not isinstance(source, str)
            or not normalized_source.endswith(f"/screen-seed-{seed}/best_gate_passing.pt")
        ):
            raise ValueError(f"checkpoint seed {seed} has unexpected artifact/provenance")
        # Exercise the exact strict loader before any analyzer output can be removed.
        _load_backbone(v48_config_path=v48_config, checkpoint_path=path)
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError("duplicate checkpoint path")
        seen_paths.add(resolved)
        digest = sha256(path)
        if digest in seen_hashes:
            raise ValueError("duplicate checkpoint hash")
        seen_hashes.add(digest)
        hashes[str(seed)] = digest
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v427-summary", type=Path, required=True)
    parser.add_argument("--v428-summary", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v410-summary", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.cache_manifest,
        args.v48_config,
        args.config,
        args.v427_summary,
        args.v428_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Reading manifest metadata is an audit only: no split/shard is materialised.
    manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    v427 = json.loads(args.v427_summary.read_text(encoding="utf-8"))
    v428 = json.loads(args.v428_summary.read_text(encoding="utf-8"))
    for name, summary, artifact in (
        ("v4.27", v427, "object_event_v4_27_scale_correlation_lhr"),
        ("v4.28", v428, "object_event_v4_28_multiscale_posterior"),
    ):
        if (
            summary.get("artifact_type") != artifact
            or summary.get("status") != "completed_oof_gate_failed"
        ):
            raise ValueError(f"{name} must be the completed OOF-gate-failed reference")
        contract = summary.get("scientific_contract", {})
        if (
            not contract.get("development_validation_not_materialized_after_oof_failure")
            or not contract.get("official_eap_test_not_opened")
            or not contract.get("evttc_not_opened")
        ):
            raise ValueError(f"{name} sealed-state contract is incomplete")
    anchors = {
        "v427": {
            "pearson": 0.6026512358857619,
            "log_eta_pearson": 0.5924083144210808,
            "negative_accuracy": 0.6520210896309314,
            "balanced_sign_accuracy": 0.7621160214888937,
            "minimum_sequence_pearson": 0.40563893157090014,
        },
        "v428": {
            "pearson": 0.6076134981874503,
            "log_eta_pearson": 0.6005657522084517,
            "negative_accuracy": 0.6098418277680141,
            "balanced_sign_accuracy": 0.7623245649996258,
            "minimum_sequence_pearson": 0.26129261040754553,
            "prediction_std_ratio": 0.5118979370522713,
            "scale_entropy_mean": 0.7482269205793273,
        },
        "v428_spatial": {
            "pearson": 0.3798278783162915,
            "log_eta_pearson": 0.3773204851728421,
            "negative_accuracy": 0.6133567662565905,
            "balanced_sign_accuracy": 0.7197953540545968,
            "minimum_sequence_pearson": 0.16664803264149447,
            "prediction_std_ratio": 1.098591166424655,
            "scale_entropy_mean": 0.9148598624160513,
        },
    }
    actual_v428 = v428["arm_results"][v428["champion"]]["oof_metrics"]
    for label, actual in (("v427", v427["oof_train_metrics"]), ("v428", actual_v428)):
        if any(
            abs(float(actual[key]) - expected) > 1e-9 for key, expected in anchors[label].items()
        ):
            raise ValueError(f"{label} numeric reference anchor changed")
    if v428.get("champion") != "profile_posterior":
        raise ValueError("v4.28 champion must be profile_posterior")
    spatial = v428["arm_results"].get("spatial_rotation_posterior", {}).get("oof_metrics", {})
    if any(
        abs(float(spatial[key]) - expected) > 1e-9
        for key, expected in anchors["v428_spatial"].items()
    ):
        raise ValueError("v4.28 spatial numeric anchor changed")
    if (
        abs(float(v427["oof_track_metrics"]["negative_track_macro_accuracy"]) - 0.6744186046511628)
        > 1e-9
        or abs(
            float(
                v428["arm_results"]["profile_posterior"]["oof_track_metrics"][
                    "negative_track_macro_accuracy"
                ]
            )
            - 0.6976744186046512
        )
        > 1e-9
    ):
        raise ValueError("track reference anchor changed")
    validate_config(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    checkpoints = parse_seed_paths(args.adapted_checkpoint)
    hashes = validate_checkpoints(checkpoints, args.v48_config)
    for path in (args.ensemble_validation, args.v410_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint_sha256": hashes,
                "cache_manifest_keys": sorted(manifest.keys()),
                "cuda_available": torch.cuda.is_available(),
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
