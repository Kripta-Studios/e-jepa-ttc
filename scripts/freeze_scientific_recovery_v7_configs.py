#!/usr/bin/env python
"""Freeze the V7 protocol and twelve seed-7 fold configurations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402

BASE_CONFIG_DIR = ROOT / "configs/experiment/scientific_recovery_v6_fold_chain"
OUTPUT_CONFIG_DIR = ROOT / "configs/experiment/scientific_recovery_v7_fold_chain"
GROUPED_PROTOCOL = ROOT / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
V7_PROTOCOL = ROOT / "configs/protocol/scientific_recovery_v7_balanced_oof.json"
BASE_CACHE = ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
T20_CACHE = ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_t20_v1/manifest.json"

ARMS: dict[str, dict[str, Any]] = {
    "soft": {
        "model_config": "configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_causal.yaml",
        "expected_parameter_count": 424274,
        "single_difference": "fold_local_a4_soft_geometry_distillation",
    },
    "c2f": {
        "model_config": "configs/model/e_jepa_causal_scale_event_v7_c2f.yaml",
        "expected_parameter_count": 424293,
        "single_difference": "adaptive_fine_r1_coarse_r2_transport",
    },
    "t20": {
        "model_config": "configs/model/e_jepa_causal_scale_event_v7_t20.yaml",
        "expected_parameter_count": 432274,
        "single_difference": "ten_temporal_bins_per_polarity",
    },
    "cap_s": {
        "model_config": "configs/model/e_jepa_causal_scale_event_v7_cap_s.yaml",
        "expected_parameter_count": 1106786,
        "single_difference": "hidden96_geometry192_depth3",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _read_json(path: Path, *, signed: bool = False) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not a mapping: {path}")
    if signed and not verify_artifact_hash(payload):
        raise ValueError(f"signed artifact is invalid: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _cache_identity(path: Path, *, expected_channels: int) -> dict[str, Any]:
    manifest = _read_json(path, signed=True)
    extension = manifest.get("object_lhr_extension", {})
    shape = extension.get("event_v4_common_roi_shape")
    if shape != [3, expected_channels, 128, 128]:
        raise ValueError(f"cache {path} declares unexpected V4 shape: {shape}")
    if int(manifest.get("split_counts", {}).get("train", -1)) != 8192:
        raise ValueError(f"cache {path} does not contain 8192 train rows")
    if expected_channels == 22 and int(
        manifest.get("split_counts", {}).get("validation", -1)
    ) != 0:
        raise ValueError("T20 cache must not materialize public validation rows")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "file_sha256": _sha256(path),
        "artifact_sha256": manifest["artifact_sha256"],
        "shape": shape,
    }


def _fold_teacher(fold: int) -> dict[str, str]:
    path = (
        ROOT
        / f"artifacts/runs/scientific_recovery_v5_a4_parent_grouped_fold{fold}_seed7/model_best.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}


def freeze(*, require_t20: bool) -> dict[str, Any]:
    """Write the signed protocol and exact runnable configs."""

    grouped = _read_json(GROUPED_PROTOCOL, signed=True)
    base_cache = _cache_identity(BASE_CACHE, expected_channels=12)
    t20_cache = (
        _cache_identity(T20_CACHE, expected_channels=22)
        if T20_CACHE.is_file()
        else None
    )
    if require_t20 and t20_cache is None:
        raise FileNotFoundError(
            "T20 cache is absent. Materialize it before freezing runnable V7 configs."
        )
    protocol: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v7_balanced_oof_protocol_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_v7_training",
        "git_base_commit": "28c1efb50622255719f239622ba07858ce704535",
        "git_branch": "scientific-recovery-v7-balanced-oof",
        "git_commit_at_freeze": _git("rev-parse", "HEAD"),
        "sample_contract": {
            "rows": 8192,
            "sequences": 9,
            "folds": 3,
            "sorted_sample_tokens_sha256": grouped["sorted_sample_tokens_sha256"],
            "fold_definitions": grouped["folds"],
        },
        "training_contract": {
            "exploratory_seed": 7,
            "confirmation_seeds": [7, 13, 23],
            "epochs": 18,
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "batch_size": 32,
            "initial_arms": list(ARMS),
            "confirm_one_winner_only": True,
        },
        "evaluation_contract": {
            "point_prediction_column": "point_prediction_ttc_s",
            "selective_prediction_column": "prediction_ttc_s",
            "guard_margin": "min(abs(log_height_ratio)/0.002,sensor_support/0.0001)",
            "risk_coverage_levels": [1.0, 0.99, 0.975, 0.95, 0.90, 0.80, 0.70, 0.50],
            "bootstrap_resamples_exploratory": 5000,
            "bootstrap_resamples_confirmation": 10000,
            "bootstrap_seed": 20260813,
        },
        "gates": {
            "mechanism_positive": {
                "delta_point_MiD_vs_a5_max": -5.0,
                "probability_delta_below_zero_min": 0.90,
                "selective_coverage_drop_max_pp": 1.0,
            },
            "geometry_positive": {
                "minimum_retention": 0.60,
                "positive_sign_required": True,
                "measures": [
                    "bbox_slope",
                    "physical_slope",
                    "bbox_std_ratio",
                    "physical_std_ratio",
                ],
            },
            "confirmation_candidate": {
                "delta_point_MiD_vs_a5_max": -3.0,
                "probability_delta_below_zero_min": 0.90,
                "geometry_positive_required": True,
            },
        },
        "sources": {
            "grouped_protocol": {
                "path": GROUPED_PROTOCOL.relative_to(ROOT).as_posix(),
                "file_sha256": _sha256(GROUPED_PROTOCOL),
                "artifact_sha256": grouped["artifact_sha256"],
            },
            "base_cache": base_cache,
            "t20_cache": t20_cache,
        },
        "closed_evaluation": {
            "allowed_splits": ["train_fold", "outer_dev_fold"],
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
        "claims": {
            "jepa_objective_active": False,
            "jepa_attribution_allowed": False,
            "architecture_only_attribution_allowed": False,
            "sota_claim_allowed": False,
        },
    }
    sign_artifact(protocol)
    _write_json(V7_PROTOCOL, protocol)

    if t20_cache is None:
        return protocol
    OUTPUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for fold in range(3):
        base_path = BASE_CONFIG_DIR / f"a5_causal_fold{fold}.yaml"
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        teacher = _fold_teacher(fold)
        fold_contract = grouped["folds"][fold]
        for arm, arm_contract in ARMS.items():
            config = copy.deepcopy(base)
            config["experiment"].update(
                {
                    "name": f"scientific_recovery_v7_{arm}_fold{fold}_seed7",
                    "protocol_version": "scientific_recovery_v7_balanced_oof_v1",
                    "single_scientific_difference": arm_contract["single_difference"],
                    "grouped_dev_role": "balanced_v7_exploratory_arm",
                }
            )
            config["model_config"] = arm_contract["model_config"]
            config["decision_contract"]["expected_parameter_count"] = arm_contract[
                "expected_parameter_count"
            ]
            config["decision_contract"].update(
                {
                    "v7_protocol_path": V7_PROTOCOL.relative_to(ROOT).as_posix(),
                    "v7_protocol_artifact_sha256": protocol["artifact_sha256"],
                    "jepa_objective": False,
                    "jepa_attribution_allowed": False,
                    "public_validation_used_for_selection": False,
                    "private_test_remains_closed": True,
                    "evttc_test_remains_closed": True,
                    "codabench_remains_closed": True,
                    "point_prediction_finite_required": True,
                }
            )
            config["data"]["development_protocol"].update(
                {
                    "path": GROUPED_PROTOCOL.relative_to(ROOT).as_posix(),
                    "file_sha256": _sha256(GROUPED_PROTOCOL),
                    "artifact_sha256": grouped["artifact_sha256"],
                    "fold": fold,
                }
            )
            config["decision_contract"]["grouped_dev_protocol"].update(
                {
                    "artifact_sha256": grouped["artifact_sha256"],
                    "fold": fold,
                    "train_rows": fold_contract["train_rows"],
                    "dev_rows": fold_contract["dev_rows"],
                }
            )
            config["training"].update(
                {
                    "seed": 7,
                    "epochs": 18,
                    "batch_size": 32,
                    "learning_rate": 0.0003,
                    "initialization_checkpoint": None,
                    "initialization_checkpoint_sha256": None,
                    "initialization_mode": "none",
                    "freeze_encoder": False,
                    "freeze_encoder_stages": 0,
                    "soft_geometry_teacher_checkpoint": None,
                    "soft_geometry_teacher_checkpoint_sha256": None,
                    "soft_dense_cosine_weight": 0.0,
                    "soft_geometry_weight": 0.0,
                }
            )
            if arm == "soft":
                config["training"].update(
                    {
                        "soft_geometry_teacher_checkpoint": teacher["path"],
                        "soft_geometry_teacher_checkpoint_sha256": teacher["sha256"],
                        "soft_dense_cosine_weight": 1.0,
                        "soft_geometry_weight": 1.0,
                    }
                )
                config["decision_contract"]["soft_teacher_contract"] = {
                    "fold": fold,
                    "checkpoint": teacher["path"],
                    "checkpoint_sha256": teacher["sha256"],
                    "teacher_tokens_equal_fold_train": True,
                    "teacher_tokens_intersect_outer_dev": False,
                    "teacher_frozen_eval": True,
                    "teacher_excluded_from_optimizer": True,
                    "teacher_absent_at_inference": True,
                    "dense_cosine_weight": 1.0,
                    "geometry_smooth_l1_weight": 1.0,
                }
            elif arm == "c2f":
                change = config["decision_contract"]["representation_change"]
                change["transport_mode"] = "adaptive_pyramid"
                change["transport_fine_radius"] = 1
                change["transport_coarse_radius"] = 2
                change["transport_coarse_downsample"] = 2
                change["router_inputs"] = [
                    "event_count",
                    "event_rate",
                    "flow_magnitude",
                    "confidence_margin",
                    "entropy",
                    "cycle_error",
                ]
                change["router_forbidden_inputs"] = [
                    "ttc",
                    "sequence_id",
                    "track_id",
                    "bbox",
                    "ttc_bucket",
                ]
                change["initial_fine_weight"] = 0.9
            elif arm == "t20":
                config["data"].update(
                    {
                        "cache_manifest": t20_cache["path"],
                        "cache_manifest_sha256": t20_cache["file_sha256"],
                        "cache_artifact_sha256": t20_cache["artifact_sha256"],
                    }
                )
                config["decision_contract"]["temporal_resolution_ablation"] = {
                    "bins_per_polarity": 10,
                    "channels_per_step": 22,
                    "storage_dtype": "float16",
                    "load_dtype": "float32",
                    "not_exact_garl_parity": True,
                    "garl_intervals": 2,
                    "garl_planes_per_interval": 20,
                    "garl_polarity_agnostic": True,
                }
            elif arm == "cap_s":
                config["decision_contract"]["capacity_ablation"] = {
                    "hidden_dim": 96,
                    "geometry_dim": 192,
                    "residual_depth": 3,
                    "transport_radius": 1,
                    "cap_m_conditional_on_delta_mid": -5.0,
                    "cap_m_conditional_probability": 0.90,
                }
            output = OUTPUT_CONFIG_DIR / f"v7_{arm}_fold{fold}_seed7.yaml"
            output.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
                newline="\n",
            )

    manifest: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v7_frozen_config_manifest_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_v7_training",
        "protocol_artifact_sha256": protocol["artifact_sha256"],
        "configs": {
            path.stem: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for path in sorted(OUTPUT_CONFIG_DIR.glob("v7_*_fold*_seed7.yaml"))
        },
        "closed_evaluation": protocol["closed_evaluation"],
    }
    if len(manifest["configs"]) != 12:
        raise RuntimeError("V7 freeze did not produce twelve initial configs")
    sign_artifact(manifest)
    _write_json(OUTPUT_CONFIG_DIR / "frozen_manifest.json", manifest)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="Freeze the protocol before T20 exists; write configs after materialization.",
    )
    args = parser.parse_args()
    try:
        protocol = freeze(require_t20=not args.protocol_only)
    except Exception as error:
        parser.exit(2, f"V7 freeze failed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {
                "protocol": str(V7_PROTOCOL),
                "artifact_sha256": protocol["artifact_sha256"],
                "configs_written": not args.protocol_only,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
