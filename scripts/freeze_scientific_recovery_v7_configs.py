#!/usr/bin/env python
"""Freeze the V7 protocol, initial screen, and preregistered control configs."""

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
from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheDataset  # noqa: E402

BASE_CONFIG_DIR = ROOT / "configs/experiment/scientific_recovery_v6_fold_chain"
OUTPUT_CONFIG_DIR = ROOT / "configs/experiment/scientific_recovery_v7_fold_chain"
GROUPED_PROTOCOL = ROOT / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
V7_PROTOCOL = ROOT / "configs/protocol/scientific_recovery_v7_balanced_oof.json"
BASE_CACHE = ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
T20_CACHE = ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_t20_v1/manifest.json"
INITIAL_MANIFEST = OUTPUT_CONFIG_DIR / "frozen_manifest.json"
PARTIAL_FREEZE_MANIFEST = OUTPUT_CONFIG_DIR / "soft_partial_freeze_manifest.json"
INITIAL_RESULT_DIR = ROOT / "artifacts/scientific_recovery_v7/results"

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


def _sorted_values_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cache_identity(
    path: Path,
    *,
    expected_channels: int,
    expected_token_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json(path, signed=True)
    extension = manifest.get("object_lhr_extension", {})
    shape = extension.get("event_v4_common_roi_shape")
    if shape != [3, expected_channels, 128, 128]:
        raise ValueError(f"cache {path} declares unexpected V4 shape: {shape}")
    if int(manifest.get("split_counts", {}).get("train", -1)) != 8192:
        raise ValueError(f"cache {path} does not contain 8192 train rows")
    if expected_channels == 22 and int(manifest.get("split_counts", {}).get("validation", -1)) != 0:
        raise ValueError("T20 cache must not materialize public validation rows")
    token_sha256 = None
    if expected_token_sha256 is not None:
        dataset = GarlTTCLHRCacheDataset(path, splits=("train",))
        tokens = [str(dataset[index]["sample_token"]) for index in range(len(dataset))]
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"cache {path} contains duplicate sample tokens")
        token_sha256 = _sorted_values_sha256(tokens)
        if token_sha256 != expected_token_sha256:
            raise ValueError(f"cache {path} token universe differs from the protocol")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "file_sha256": _sha256(path),
        "artifact_sha256": manifest["artifact_sha256"],
        "shape": shape,
        "sorted_sample_tokens_sha256": token_sha256,
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
    expected_token_sha256 = str(grouped["sorted_sample_tokens_sha256"])
    base_cache = _cache_identity(BASE_CACHE, expected_channels=12)
    t20_cache = (
        _cache_identity(
            T20_CACHE,
            expected_channels=22,
            expected_token_sha256=expected_token_sha256,
        )
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
    if V7_PROTOCOL.is_file():
        frozen_protocol = _read_json(V7_PROTOCOL, signed=True)
        if frozen_protocol.get("status") != "frozen_before_v7_training":
            raise ValueError("existing V7 protocol has an incompatible status")
        if frozen_protocol.get("sample_contract") != protocol["sample_contract"]:
            raise ValueError("existing V7 protocol sample contract differs")
        protocol = frozen_protocol
    else:
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

    initial_paths = [
        OUTPUT_CONFIG_DIR / f"v7_{arm}_fold{fold}_seed7.yaml" for fold in range(3) for arm in ARMS
    ]
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
            for path in sorted(initial_paths)
        },
        "closed_evaluation": protocol["closed_evaluation"],
    }
    if len(manifest["configs"]) != 12:
        raise RuntimeError("V7 freeze did not produce twelve initial configs")
    sign_artifact(manifest)
    _write_json(INITIAL_MANIFEST, manifest)
    return protocol


def freeze_soft_partial_control() -> dict[str, Any]:
    """Freeze the sole post-screen control allowed by the V7 decision tree."""

    protocol = _read_json(V7_PROTOCOL, signed=True)
    initial_manifest = _read_json(INITIAL_MANIFEST, signed=True)
    if initial_manifest.get("protocol_artifact_sha256") != protocol["artifact_sha256"]:
        raise ValueError("initial config manifest is bound to a different V7 protocol")

    screen: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        path = INITIAL_RESULT_DIR / f"{arm}_seed7_oof.json"
        result = _read_json(path, signed=True)
        if result.get("status") != "completed_seed7_oof_gate":
            raise ValueError(f"initial result is not complete: {arm}")
        gates = result.get("gates")
        if not isinstance(gates, dict) or any(bool(gates.get(gate)) for gate in gates):
            raise ValueError(f"initial result unexpectedly passed a gate: {arm}")
        screen[arm] = {
            "path": path.relative_to(ROOT).as_posix(),
            "file_sha256": _sha256(path),
            "artifact_sha256": result["artifact_sha256"],
            "gates": gates,
        }
    if screen["soft"]["gates"].get("geometry_positive") is not False:
        raise ValueError("partial-freeze control requires SOFT geometry_positive=false")

    if PARTIAL_FREEZE_MANIFEST.is_file():
        frozen = _read_json(PARTIAL_FREEZE_MANIFEST, signed=True)
        if frozen.get("trigger_results") != screen:
            raise ValueError("existing partial-freeze manifest has different trigger results")
        for entry in frozen.get("configs", {}).values():
            path = ROOT / str(entry["path"])
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"frozen partial-freeze config changed: {path}")
        return frozen

    configs: dict[str, dict[str, str]] = {}
    for fold in range(3):
        source = OUTPUT_CONFIG_DIR / f"v7_soft_fold{fold}_seed7.yaml"
        source_key = source.stem
        expected_source_sha = initial_manifest["configs"][source_key]["sha256"]
        if _sha256(source) != expected_source_sha:
            raise ValueError(f"initial SOFT config hash mismatch: {source}")
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        if (
            config["training"].get("soft_dense_cosine_weight") != 1.0
            or config["training"].get("soft_geometry_weight") != 1.0
            or config["training"].get("initialization_mode") != "none"
            or config["training"].get("freeze_encoder") is not False
        ):
            raise ValueError(f"initial SOFT contract is incompatible in fold {fold}")
        config["experiment"].update(
            {
                "name": (f"scientific_recovery_v7_soft_partial_freeze_fold{fold}_seed7"),
                "single_scientific_difference": "freeze_encoder_features_0_3",
                "grouped_dev_role": "preregistered_soft_partial_freeze_control",
            }
        )
        config["training"]["freeze_encoder_stages"] = 3
        config["decision_contract"]["partial_freeze_control"] = {
            "trigger": "initial_soft_failed_geometry_positive",
            "frozen_module_slice": "encoder.features[0:3]",
            "freeze_encoder_stages": 3,
            "student_initialized_from_scratch": True,
            "student_remaining_layers_trainable": True,
            "same_fold_local_teacher": True,
            "same_soft_loss_weights": True,
            "teacher_dense_cosine_weight": 1.0,
            "teacher_geometry_smooth_l1_weight": 1.0,
            "layer_or_weight_sweep_allowed": False,
            "initial_screen_artifacts": {
                arm: entry["artifact_sha256"] for arm, entry in screen.items()
            },
        }
        output = OUTPUT_CONFIG_DIR / f"v7_soft_partial_freeze_fold{fold}_seed7.yaml"
        output.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        configs[output.stem] = {
            "path": output.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output),
            "source_soft_config_sha256": expected_source_sha,
        }

    manifest: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v7_soft_partial_freeze_manifest_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_after_negative_initial_screen_before_control_training",
        "protocol_artifact_sha256": protocol["artifact_sha256"],
        "initial_config_manifest_artifact_sha256": initial_manifest["artifact_sha256"],
        "decision": {
            "reason": "SOFT failed geometry and no initial arm became a candidate",
            "only_authorized_control": "freeze encoder.features[0:3] with unchanged SOFT losses",
            "layer_or_weight_sweep_allowed": False,
        },
        "trigger_results": screen,
        "configs": configs,
        "closed_evaluation": protocol["closed_evaluation"],
    }
    sign_artifact(manifest)
    _write_json(PARTIAL_FREEZE_MANIFEST, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="Freeze the protocol before T20 exists; write configs after materialization.",
    )
    parser.add_argument(
        "--soft-partial-freeze-control",
        action="store_true",
        help="Freeze the sole control activated after a negative initial screen.",
    )
    args = parser.parse_args()
    if args.protocol_only and args.soft_partial_freeze_control:
        parser.error("--protocol-only and --soft-partial-freeze-control are exclusive")
    try:
        if args.soft_partial_freeze_control:
            manifest = freeze_soft_partial_control()
            print(
                json.dumps(
                    {
                        "manifest": str(PARTIAL_FREEZE_MANIFEST),
                        "artifact_sha256": manifest["artifact_sha256"],
                        "configs_written": len(manifest["configs"]),
                    }
                )
            )
            return 0
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
