#!/usr/bin/env python
"""Train the event causal-scale arm on a bounded public eAP/Garl validation screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (  # noqa: E402
    DINOv3RelationalTeacherDataset,
)
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.data.sam_teacher_cache import SAMTeacherMaskDataset  # noqa: E402
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleTTC,
    CausalScaleTTCConfig,
)
from e_jepa_ttc.reproducibility import environment_snapshot, resolve_device  # noqa: E402
from e_jepa_ttc.training.causal_scale_eap import (  # noqa: E402
    CausalScaleEAPTrainingConfig,
    checkpoint_payload,
    train_real_causal_scale,
)
from scripts.evaluate_causal_scale_v5_operator import (  # noqa: E402
    _classify_worktree_status,
)

DEFAULT_CONFIG = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def _resolve(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("path references must be strings")
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _model_config(path: Path) -> CausalScaleTTCConfig:
    raw = _read_yaml(path)
    model_name = raw.pop("model", None)
    if model_name not in {"e_jepa_causal_scale_event_v8", "e_jepa_causal_scale_event_v9_transport"}:
        raise ValueError("real screen requires causal-scale event v8 or preregistered v9 transport")
    thresholds = raw.get("risk_thresholds_s")
    if not isinstance(thresholds, list):
        raise ValueError("risk_thresholds_s must be a list")
    raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
    return CausalScaleTTCConfig(**raw)


def _validate_bbox_geometry_loss(
    training_config: CausalScaleEAPTrainingConfig,
    loss_config: CausalScaleTTCLossConfig,
    decision_contract: dict[str, Any],
) -> None:
    if training_config.foreground_supervision not in {
        "bbox_geometry",
        "bbox_geometry_sam_teacher",
    }:
        return
    uses_sam = training_config.foreground_supervision == "bbox_geometry_sam_teacher"
    if not uses_sam and (
        loss_config.foreground_bce_weight != 0.0
        or loss_config.foreground_dice_weight != 0.0
    ):
        raise ValueError("bbox_geometry supervision requires BCE/Dice weights to be zero")
    if uses_sam and min(
        loss_config.foreground_bce_weight, loss_config.foreground_dice_weight
    ) <= 0.0:
        raise ValueError("SAM teacher supervision requires positive BCE/Dice weights")
    if min(
        loss_config.foreground_extent_weight,
        loss_config.foreground_width_weight,
        loss_config.foreground_center_weight,
    ) <= 0.0:
        raise ValueError("bbox_geometry supervision requires positive h/w/center weights")
    pair_weight = loss_config.foreground_pair_ratio_weight
    if pair_weight == 0.0:
        return
    if decision_contract.get("pair_ratio_target_source") != "numeric_bbox_height_training_only":
        raise ValueError("bbox_geometry pair-ratio requires a numeric training-only target")
    if decision_contract.get("pair_ratio_target_uses_dense_mask") is not False:
        raise ValueError("bbox_geometry pair-ratio must declare dense-mask use false")
    if float(decision_contract.get("pair_ratio_weight", float("nan"))) != pair_weight:
        raise ValueError("bbox_geometry pair-ratio weight differs from decision contract")
    if decision_contract.get("pair_ratio_disabled_during_three_epoch_geometry_warmup") is not True:
        raise ValueError("bbox_geometry pair-ratio must remain disabled during warm-up")


def _validate_representation_change(
    training_config: CausalScaleEAPTrainingConfig,
    decision_contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Fail closed on the post-A4 temporal-delta calibration contract."""

    if (
        training_config.representation_supervision
        != "dinov3_local_relational_temporal_delta"
    ):
        return None

    change = decision_contract.get("representation_change")
    if not isinstance(change, dict):
        raise ValueError("A4D requires decision_contract.representation_change")
    if change.get("type") != "dinov3_local_relational_plus_temporal_delta":
        raise ValueError("A4D representation_change.type is not frozen")
    if float(change.get("parent_endpoint_distillation_weight", float("nan"))) != (
        training_config.representation_distillation_weight
    ):
        raise ValueError("A4D must preserve the frozen A4 endpoint distillation weight")
    if change.get("no_bbox_mask_on_temporal_delta_loss") is not True:
        raise ValueError("A4D temporal delta must remain unmasked by bbox")
    if change.get("same_cached_teacher_as_a4") is not True:
        raise ValueError("A4D must reuse the exact A4 teacher cache")

    calibration = change.get("temporal_delta_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("A4D temporal_delta_calibration contract is required")
    if calibration.get("method") != (
        "quarter_weighted_endpoint_equivalence_on_random_init_train_only"
    ):
        raise ValueError("A4D temporal-delta calibration method differs from protocol")
    if int(calibration.get("samples", -1)) != 64 or int(
        calibration.get("seed", -1)
    ) != 7:
        raise ValueError("A4D temporal-delta calibration must use 64 samples and seed 7")
    if float(calibration.get("target_fraction_of_weighted_endpoint", float("nan"))) != 0.25:
        raise ValueError("A4D temporal-delta calibration target fraction must be 0.25")
    if calibration.get("clamp_range") != [0.25, 4.0]:
        raise ValueError("A4D temporal-delta calibration clamp must be [0.25, 4.0]")

    artifact_path = _resolve(calibration.get("artifact"))
    expected_file_sha = str(calibration.get("file_sha256", ""))
    if len(expected_file_sha) != 64 or _sha256(artifact_path) != expected_file_sha:
        raise ValueError("A4D temporal-delta calibration file hash differs from protocol")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError("A4D temporal-delta calibration artifact signature is invalid")
    if payload.get("artifact_type") != "a4d_dinov3_temporal_delta_weight_calibration_v1":
        raise ValueError("unexpected A4D temporal-delta calibration artifact type")
    expected_signed_sha = str(calibration.get("artifact_sha256", ""))
    if payload.get("artifact_sha256") != expected_signed_sha:
        raise ValueError("A4D temporal-delta calibration signed identity differs from protocol")
    if (
        training_config.representation_temporal_delta_calibration_artifact_sha256
        != expected_signed_sha
    ):
        raise ValueError(
            "A4D training config is not bound to the signed calibration artifact"
        )
    scope = payload.get("scope", {})
    if not isinstance(scope, dict):
        raise ValueError("A4D temporal-delta calibration scope is malformed")
    if scope.get("public_train_only") is not True:
        raise ValueError("A4D temporal-delta calibration must be public_train_only")
    if scope.get("validation_or_test_opened") is not False:
        raise ValueError("A4D temporal-delta calibration may not open validation/test")
    if int(scope.get("optimizer_steps", -1)) != 0:
        raise ValueError("A4D temporal-delta calibration may not take optimizer steps")
    if int(payload.get("samples_collected", -1)) != 64 or int(
        payload.get("seed", -1)
    ) != 7:
        raise ValueError("A4D temporal-delta calibration artifact sample contract differs")
    if payload.get("teacher_artifact_sha256") != (
        training_config.representation_teacher_cache_artifact_sha256
    ):
        raise ValueError("A4D calibration and training teacher identities differ")
    selected = float(payload.get("selected_weight", float("nan")))
    if not math.isfinite(selected) or not math.isclose(
        selected,
        training_config.representation_temporal_delta_weight,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("A4D temporal-delta weight differs from signed calibration")

    return {
        "path": artifact_path.relative_to(ROOT).as_posix(),
        "file_sha256": expected_file_sha,
        "artifact_sha256": expected_signed_sha,
        "selected_weight": selected,
        "teacher_artifact_sha256": payload.get("teacher_artifact_sha256"),
        "scope": scope,
        "median_endpoint_relation_error": payload.get(
            "median_endpoint_relation_error"
        ),
        "median_temporal_delta_error": payload.get("median_temporal_delta_error"),
        "median_teacher_temporal_delta_abs": payload.get(
            "median_teacher_temporal_delta_abs"
        ),
    }



def _validate_a5_transport_change(
    training_config: CausalScaleEAPTrainingConfig,
    model_config: CausalScaleTTCConfig,
    decision_contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Fail closed on A5 event-native transport and its train-only preflight."""

    change = decision_contract.get("representation_change")
    if not isinstance(change, dict) or change.get("type") != (
        "a4_endpoint_dino_plus_event_native_local_cross_time_transport"
    ):
        if model_config.transport_enabled:
            raise ValueError(
                "transport-enabled models require the preregistered A5 representation_change"
            )
        return None

    if not model_config.transport_enabled:
        raise ValueError("A5 representation_change requires transport_enabled=true")
    if training_config.representation_supervision != "dinov3_local_relational":
        raise ValueError("A5 must keep A4 endpoint DINO and remove A4D temporal delta")
    if training_config.representation_temporal_delta_weight != 0.0:
        raise ValueError("A5 must not train the A4D temporal-delta objective")
    if change.get("dino_endpoint_teacher_unchanged_from_a4") is not True:
        raise ValueError("A5 must preserve the A4 endpoint DINO teacher")
    if change.get("dino_temporal_delta_removed") is not True:
        raise ValueError("A5 must explicitly remove the A4D temporal delta")
    if change.get("transport_model_input") != "event_dense_features_only":
        raise ValueError("A5 transport must consume event dense features only")
    if change.get("bbox_used_by_transport") is not False:
        raise ValueError("bbox may not enter A5 transport")
    if change.get("rgb_used_at_inference") is not False:
        raise ValueError("RGB may not enter A5 inference")
    if change.get("jepa_objective") is not False:
        raise ValueError("A5-CORR-V1 does not introduce a JEPA objective")
    if change.get("direct_ttc_regressor_from_flow") is not False:
        raise ValueError("A5-CORR-V1 may not regress TTC directly from flow")
    if change.get("analytic_height_ratio_remains_primary_backbone") is not True:
        raise ValueError("A5 must preserve the analytic height-ratio backbone")
    if int(change.get("transport_radius", -1)) != model_config.transport_radius:
        raise ValueError("A5 transport radius differs from model config")
    if change.get("transport_pairs") != ["t0_to_t1", "t1_to_t2"]:
        raise ValueError("A5 transport must cover both t0->t1 and t1->t2")

    preflight = decision_contract.get("preflight_contract")
    if not isinstance(preflight, dict):
        raise ValueError("A5 requires decision_contract.preflight_contract")
    artifact_ref = preflight.get("artifact")
    if not isinstance(artifact_ref, str):
        raise ValueError("A5 preflight contract must name its signed artifact")
    artifact_path = _resolve(artifact_ref)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError("A5 preflight artifact signature is invalid")
    if payload.get("artifact_type") != "a5_transport_preflight_train_only_v1":
        raise ValueError("unexpected A5 preflight artifact type")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("A5 preflight scope is malformed")
    if scope.get("public_train_only") is not True:
        raise ValueError("A5 preflight must be train-only")
    if scope.get("validation_or_test_opened") is not False:
        raise ValueError("A5 preflight may not open validation/test")
    if int(scope.get("optimizer_steps", -1)) != 0:
        raise ValueError("A5 preflight may not take optimizer steps")
    if list(scope.get("radii", [])) != list(preflight.get("radii", [])):
        raise ValueError("A5 preflight radii differ from preregistration")
    if payload.get("a5_corr_authorized") is not True:
        raise ValueError("A5-CORR training is blocked because preflight did not pass")

    thresholds = payload.get("decision_thresholds")
    expected_thresholds = {
        "teacher_r4_global_error_reduction_min": float(
            preflight["teacher_r4_global_error_reduction_min"]
        ),
        "teacher_r4_foreground_error_reduction_min": float(
            preflight["teacher_r4_foreground_error_reduction_min"]
        ),
        "student_r4_entropy_max": float(preflight["student_r4_entropy_max"]),
        "student_r4_confidence_margin_min": float(
            preflight["student_r4_confidence_margin_min"]
        ),
    }
    if thresholds != expected_thresholds:
        raise ValueError("A5 preflight thresholds differ from preregistration")

    return {
        "path": artifact_path.relative_to(ROOT).as_posix(),
        "file_sha256": _sha256(artifact_path),
        "artifact_sha256": payload.get("artifact_sha256"),
        "a5_corr_authorized": True,
        "scope": scope,
        "decision_checks": payload.get("decision_checks"),
        "teacher_transport_r4": (
            payload.get("teacher_transport", {}).get("4")
            if isinstance(payload.get("teacher_transport"), dict)
            else None
        ),
    }


def _finite_json(value: object) -> object:
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return {str(key): _finite_json(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_finite_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _reset_peak_memory_stats(device: torch.device) -> None:
    """Reset CUDA peak accounting without passing a device to the Windows API."""

    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()


def run(
    config_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    """Execute a train/validation-only run; no benchmark test labels are opened."""

    raw = _read_yaml(config_path)
    experiment = raw.get("experiment")
    data = raw.get("data")
    if not isinstance(experiment, dict) or not isinstance(data, dict):
        raise ValueError("experiment and data sections are required")
    if data.get("opened_splits") != ["train", "validation"]:
        raise ValueError("this screen may open train and validation only")
    forbidden = ("official_test_opened", "codabench_opened", "evttc_test_opened")
    if any(data.get(key) is not False for key in forbidden):
        raise ValueError("private/CodaBench/EvTTC test access must remain false")
    manifest_path = _resolve(data["cache_manifest"])
    expected_manifest_hash = str(data["cache_manifest_sha256"])
    actual_manifest_hash = _sha256(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError("cache manifest hash differs from the frozen protocol")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_sha256") != data.get("cache_artifact_sha256"):
        raise ValueError("cache artifact identity differs from the frozen protocol")

    expected_train_rows = int(data.get("expected_train_rows", 2048))
    expected_validation_rows = int(data.get("expected_validation_rows", 2048))
    if min(expected_train_rows, expected_validation_rows) <= 0:
        raise ValueError("expected train/validation row counts must be positive")
    split_counts = manifest.get("split_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("cache manifest split_counts must be a mapping")
    if int(split_counts.get("train", -1)) != expected_train_rows:
        raise ValueError(
            "train cache row count differs from frozen protocol: "
            f"{split_counts.get('train')} != {expected_train_rows}"
        )

    validation_manifest_path = manifest_path
    validation_manifest_hash = actual_manifest_hash
    validation_manifest = manifest
    if "validation_cache_manifest" in data:
        validation_manifest_path = _resolve(data["validation_cache_manifest"])
        validation_manifest_hash = _sha256(validation_manifest_path)
        if validation_manifest_hash != str(data["validation_cache_manifest_sha256"]):
            raise ValueError(
                "validation cache manifest hash differs from the frozen protocol"
            )
        validation_manifest = json.loads(
            validation_manifest_path.read_text(encoding="utf-8")
        )
        if validation_manifest.get("artifact_sha256") != data.get(
            "validation_cache_artifact_sha256"
        ):
            raise ValueError(
                "validation cache artifact identity differs from the frozen protocol"
            )
    validation_split_counts = validation_manifest.get("split_counts")
    if not isinstance(validation_split_counts, dict):
        raise ValueError("validation cache split_counts must be a mapping")
    if int(validation_split_counts.get("validation", -1)) != expected_validation_rows:
        raise ValueError(
            "validation cache row count differs from frozen protocol: "
            f"{validation_split_counts.get('validation')} != "
            f"{expected_validation_rows}"
        )
    train_sequences = {str(value) for value in data.get("train_sequence_ids", [])}
    validation_sequences = {
        str(value) for value in data.get("validation_sequence_ids", [])
    }
    if len(train_sequences) != 9 or len(validation_sequences) != 3:
        raise ValueError("frozen protocol requires 9 train and 3 validation sequences")
    if train_sequences & validation_sequences:
        raise ValueError("train and validation sequence IDs overlap")

    status_lines = _git(
        "-c", "core.quotepath=false", "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    worktree = _classify_worktree_status(status_lines)
    code_dirty = bool(worktree["tracked_dirty"] or worktree["untracked_code_paths"])
    if code_dirty:
        raise RuntimeError("representative real screen requires clean tracked/code state")

    model_path = _resolve(raw["model_config"])
    model_config = _model_config(model_path)
    training_raw = raw.get("training")
    loss_raw = raw.get("loss")
    if not isinstance(training_raw, dict) or not isinstance(loss_raw, dict):
        raise ValueError("training and loss mappings are required")
    training_config = CausalScaleEAPTrainingConfig(**training_raw)
    loss_config = CausalScaleTTCLossConfig(**loss_raw)
    decision_contract = raw.get("decision_contract")
    if not isinstance(decision_contract, dict):
        raise ValueError("decision_contract mapping is required")
    _validate_bbox_geometry_loss(training_config, loss_config, decision_contract)
    representation_calibration = _validate_representation_change(
        training_config, decision_contract
    )
    a5_preflight = _validate_a5_transport_change(
        training_config, model_config, decision_contract
    )
    parameter_count = sum(
        parameter.numel() for parameter in CausalScaleTTC(model_config).parameters()
    )
    expected_parameter_count = decision_contract.get("expected_parameter_count")
    if expected_parameter_count is not None and parameter_count != int(
        expected_parameter_count
    ):
        raise ValueError(
            f"model parameter count changed: {parameter_count} != {expected_parameter_count}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    train_dataset = GarlTTCObjectEventV4Dataset(
        str(manifest_path), splits=("train",)
    )
    validation_dataset = GarlTTCObjectEventV4Dataset(
        str(validation_manifest_path), splits=("validation",)
    )
    teacher_metadata: dict[str, Any] | None = None
    if training_config.foreground_supervision == "bbox_geometry_sam_teacher":
        teacher = data.get("sam_teacher")
        if not isinstance(teacher, dict):
            raise ValueError("A3 requires data.sam_teacher")
        teacher_manifest = _resolve(teacher["manifest"])
        train_dataset = SAMTeacherMaskDataset(
            train_dataset,
            manifest_path=teacher_manifest,
            expected_artifact_sha256=str(teacher["artifact_sha256"]),
            expected_manifest_sha256=str(teacher["manifest_sha256"]),
        )
        if training_config.teacher_cache_artifact_sha256 != teacher["artifact_sha256"]:
            raise ValueError("training and data teacher identities differ")
        teacher_metadata = {
            "manifest": teacher_manifest.relative_to(ROOT).as_posix(),
            "manifest_sha256": str(teacher["manifest_sha256"]),
            "artifact_sha256": str(teacher["artifact_sha256"]),
            "scope": "public_train_only",
            "validation_teacher_loaded": False,
        }
    # --- A4: DINO relational teacher ---
    representation_teacher_metadata: dict[str, Any] | None = None
    if training_config.representation_supervision in {
        "dinov3_local_relational",
        "dinov3_local_relational_temporal_delta",
    }:
        dino_teacher = data.get("dinov3_relational_teacher")
        if not isinstance(dino_teacher, dict):
            raise ValueError("A4 requires data.dinov3_relational_teacher")
        dino_manifest = _resolve(dino_teacher["manifest"])
        train_dataset = DINOv3RelationalTeacherDataset(
            train_dataset,
            manifest_path=dino_manifest,
            expected_artifact_sha256=str(dino_teacher["artifact_sha256"]),
            expected_manifest_sha256=str(dino_teacher["manifest_sha256"]),
        )
        if (
            training_config.representation_teacher_cache_artifact_sha256
            != dino_teacher["artifact_sha256"]
        ):
            raise ValueError(
                "training and data representation teacher identities differ"
            )
        representation_teacher_metadata = {
            "manifest": dino_manifest.relative_to(ROOT).as_posix(),
            "manifest_sha256": str(dino_teacher["manifest_sha256"]),
            "artifact_sha256": str(dino_teacher["artifact_sha256"]),
            "scope": "public_train_only",
            "validation_teacher_loaded": False,
            "teacher_type": "dinov3_local_relational",
        }
    device = resolve_device(device_name)
    _reset_peak_memory_stats(device)
    result = train_real_causal_scale(
        model_config,
        training_config,
        loss_config,
        train_dataset,
        validation_dataset,
        device,
        checkpoint_dir=state_dir,
        resume=resume,
    )
    checkpoint_path = output_dir / "model_best.pt"
    temporary_checkpoint = output_dir / ".model_best.pt.tmp"
    torch.save(
        checkpoint_payload(result, training_config, loss_config), temporary_checkpoint
    )
    os.replace(temporary_checkpoint, checkpoint_path)
    validation = result.best_validation
    predictions = pd.DataFrame(
        {
            "sample_token": validation["sample_tokens"],
            "sequence_id": validation["sequence_ids"],
            "target_ttc_s": validation["target_ttc_s"],
            "prediction_ttc_s": validation["prediction_ttc_s"],
        }
    )
    predictions_path = output_dir / "validation_predictions.csv"
    predictions.to_csv(predictions_path, index=False, lineterminator="\n")
    metrics = {
        key: value
        for key, value in validation.items()
        if key not in {"sample_tokens", "sequence_ids", "target_ttc_s", "prediction_ttc_s"}
    }
    payload: dict[str, Any] = {
        "artifact_type": "causal_scale_eap_public_validation_screen_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed_public_validation_only",
        "selectable": False,
        "sota_claim_authorized": False,
        "official_test_opened": False,
        "garl_comparison_pending": True,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": code_dirty,
        "worktree": worktree,
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(config_path),
        },
        "model_config": {
            "path": model_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(model_path),
        },
        "model_architecture": asdict(model_config),
        "cache": {
            "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": actual_manifest_hash,
            "artifact_sha256": manifest["artifact_sha256"],
            "split_counts": manifest["split_counts"],
            "effective_train_rows": expected_train_rows,
            "validation_manifest_path": (
                validation_manifest_path.relative_to(ROOT).as_posix()
            ),
            "validation_manifest_sha256": validation_manifest_hash,
            "validation_artifact_sha256": validation_manifest["artifact_sha256"],
            "validation_split_counts": validation_manifest["split_counts"],
            "effective_validation_rows": expected_validation_rows,
            "train_sequence_ids": sorted(train_sequences),
            "validation_sequence_ids": sorted(validation_sequences),
        },
        "model_input_contract": {
            "forward_inputs": ["event_v4_common_roi", "garl_delta_t_s"],
            "foreground_supervision": training_config.foreground_supervision,
            "weak_bbox_supervision_only": (
                training_config.foreground_supervision == "weak_box"
            ),
            "weak_bbox_rasterized_for_loss": (
                training_config.foreground_supervision == "weak_box"
            ),
            "bbox_geometry_training_only": (
                training_config.foreground_supervision
                in {"bbox_geometry", "bbox_geometry_sam_teacher"}
            ),
            "bbox_is_not_segmentation_ground_truth": True,
            "bbox_is_model_input": False,
            "sam_teacher_is_model_input": False,
            "sam_teacher_train_only": teacher_metadata is not None,
            "validation_teacher_loaded": False,
            "t0_proxy_box_excluded": training_config.mask_t0_as_proxy,
            "representation_supervision": training_config.representation_supervision,
            "dinov3_teacher_is_model_input": False,
            "dinov3_teacher_train_only": representation_teacher_metadata is not None,
            "validation_dinov3_teacher_loaded": False,
            "cross_time_transport_enabled": model_config.transport_enabled,
            "cross_time_transport_inputs": (
                ["event_dense_features_t_previous", "event_dense_features_t_current"]
                if model_config.transport_enabled
                else []
            ),
            "bbox_is_transport_input": False,
            "rgb_is_transport_input": False,
            "dinov3_is_transport_input": False,
        },
        "sam_teacher": teacher_metadata,
        "representation_teacher": representation_teacher_metadata,
        "representation_temporal_delta_calibration": representation_calibration,
        "a5_transport_preflight": a5_preflight,
        "parameter_count": parameter_count,
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "selection": {
            "split": "validation",
            "best_epoch": result.best_epoch,
            **result.best_selection,
        },
        "validation_metrics": metrics,
        "history": result.history,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else None
        ),
        "checkpoint": {"path": checkpoint_path.name, "sha256": _sha256(checkpoint_path)},
        "predictions": {"path": predictions_path.name, "sha256": _sha256(predictions_path)},
        "environment": environment_snapshot(),
        "device": str(device),
        "decision_contract": decision_contract,
    }
    payload = cast(dict[str, Any], _finite_json(payload))
    sign_artifact(payload)
    _atomic_json(output_dir / "summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(
            args.config.resolve(),
            args.output_dir.resolve(),
            device_name=args.device,
            resume=args.resume,
        )
    except Exception as error:
        parser.exit(2, f"causal-scale eAP screen failed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection": payload["selection"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "peak_vram_mb": payload["peak_vram_mb"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
