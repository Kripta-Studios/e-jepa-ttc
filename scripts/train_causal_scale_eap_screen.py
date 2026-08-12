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
from e_jepa_ttc.data.scientific_recovery_v5 import SequenceIndexedView  # noqa: E402
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


def _porcelain_path(line: str) -> str:
    """Extract a normalized path from a Git porcelain-v1 status line."""

    payload = line[3:] if len(line) >= 4 else line
    if " -> " in payload:
        payload = payload.split(" -> ", 1)[1]
    return payload.strip().replace("\\", "/")


def _is_ignored_operational_tracked_path(path: str) -> bool:
    """Return True only for observer scripts that cannot affect model numerics.

    The external progress monitor is intentionally allowed to be edited while a
    long experiment is running (for example to change its refresh interval).
    Training/orchestration/model/config code remains fail-closed.
    """

    normalized = path.replace("\\", "/")
    name = Path(normalized).name
    return (
        normalized.startswith("scripts/")
        and name.startswith("monitor_scientific_recovery_v")
        and name.endswith(".ps1")
    )


def _blocking_worktree_state(status_lines: list[str]) -> dict[str, object]:
    """Classify worktree state while allowing observer-only tracked edits."""

    base = dict(_classify_worktree_status(status_lines))
    tracked_dirty_paths = [
        _porcelain_path(line) for line in status_lines if not line.startswith("?? ")
    ]
    ignored = [
        path for path in tracked_dirty_paths if _is_ignored_operational_tracked_path(path)
    ]
    blocking = [path for path in tracked_dirty_paths if path not in ignored]
    base["tracked_dirty_paths"] = tracked_dirty_paths
    base["ignored_operational_dirty_paths"] = ignored
    base["blocking_tracked_dirty_paths"] = blocking
    base["science_code_dirty"] = bool(blocking or base["untracked_code_paths"])
    return base


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sorted_values_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
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
    allowed_models = {
        "e_jepa_causal_scale_event_v8",
        "e_jepa_causal_scale_event_v9_transport",
        "e_jepa_causal_scale_event_v10_transport_adapter",
        "e_jepa_causal_scale_event_v11_dual_transport",
    }
    if model_name not in allowed_models:
        raise ValueError(f"real screen requires an audited causal-scale model, got {model_name!r}")
    thresholds = raw.get("risk_thresholds_s")
    if not isinstance(thresholds, list):
        raise ValueError("risk_thresholds_s must be a list")
    raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
    return CausalScaleTTCConfig(**raw)


def _load_grouped_development_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Load one frozen V5 train-only fold and reject any broader data scope."""

    reference = data.get("development_protocol")
    if not isinstance(reference, dict):
        raise ValueError("grouped development requires data.development_protocol")
    protocol_path = _resolve(reference.get("path"))
    observed_file_sha = _sha256(protocol_path)
    if observed_file_sha != reference.get("file_sha256"):
        raise ValueError("grouped-development protocol file SHA256 differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not verify_artifact_hash(protocol):
        raise ValueError("grouped-development protocol signature is invalid")
    if protocol.get("artifact_sha256") != reference.get("artifact_sha256"):
        raise ValueError("grouped-development artifact SHA256 differs")
    if protocol.get("artifact_type") != (
        "scientific_recovery_v5_train_only_grouped_dev_v1"
    ):
        raise ValueError("grouped-development artifact type is incompatible")
    if protocol.get("status") != "frozen_before_a8_results":
        raise ValueError("grouped-development protocol was not frozen before A8")
    checks = protocol.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("grouped-development checks are missing")
    required_true = (
        "train_only_grouped_dev",
        "sequence_disjoint_folds",
        "sample_token_unique",
        "same_cache_universe",
    )
    if any(checks.get(key) is not True for key in required_true):
        raise ValueError("grouped-development protocol did not pass its split checks")
    if checks.get("public_validation_used_for_selection") is not False:
        raise ValueError("grouped development may not use public validation for selection")
    if checks.get("private_test_opened") is not False:
        raise ValueError("grouped development may not open private/test")

    fold_index = reference.get("fold")
    if not isinstance(fold_index, int):
        raise ValueError("grouped-development fold must be an integer")
    folds = protocol.get("folds")
    if not isinstance(folds, list):
        raise ValueError("grouped-development folds are malformed")
    matches = [fold for fold in folds if fold.get("fold") == fold_index]
    if len(matches) != 1:
        raise ValueError(f"grouped-development fold {fold_index} is unavailable")
    fold = matches[0]
    train_sequences = {str(value) for value in fold.get("train_sequence_ids", [])}
    dev_sequences = {str(value) for value in fold.get("dev_sequence_ids", [])}
    universe = {str(value) for value in protocol.get("sequence_ids", [])}
    if not train_sequences or not dev_sequences:
        raise ValueError("grouped-development fold contains an empty partition")
    if train_sequences & dev_sequences or train_sequences | dev_sequences != universe:
        raise ValueError("grouped-development fold is not a disjoint universe partition")
    if int(fold.get("train_rows", 0)) + int(fold.get("dev_rows", 0)) != int(
        protocol.get("sample_count", -1)
    ):
        raise ValueError("grouped-development fold row counts are not exhaustive")
    return {
        "path": protocol_path,
        "file_sha256": observed_file_sha,
        "artifact_sha256": protocol["artifact_sha256"],
        "protocol": protocol,
        "fold": fold,
        "fold_index": fold_index,
        "train_sequences": train_sequences,
        "dev_sequences": dev_sequences,
    }


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
    change_type = change.get("type") if isinstance(change, dict) else None
    supported_a5_types = {
        "a4_endpoint_dino_plus_event_native_local_cross_time_transport",
        "a4_frozen_endpoint_plus_event_native_local_cross_time_transport",
        "a4_frozen_endpoint_plus_adaptive_transport_adapter",
        "a4_frozen_geometry_plus_trainable_transport_encoder",
    }
    if change_type not in supported_a5_types:
        if model_config.transport_enabled:
            raise ValueError(
                "transport-enabled models require a preregistered A5 representation_change"
            )
        return None

    if not model_config.transport_enabled:
        raise ValueError("A5 representation_change requires transport_enabled=true")
    if training_config.representation_supervision != "dinov3_local_relational":
        raise ValueError("A5 must keep A4 endpoint DINO and remove A4D temporal delta")
    if change_type in {
        "a4_frozen_endpoint_plus_event_native_local_cross_time_transport",
        "a4_frozen_endpoint_plus_adaptive_transport_adapter",
        "a4_frozen_geometry_plus_trainable_transport_encoder",
    }:
        if change_type == "a4_frozen_endpoint_plus_adaptive_transport_adapter":
            contract_name, label = "adapter_contract", "A6-ADAPTER"
        elif change_type == "a4_frozen_geometry_plus_trainable_transport_encoder":
            contract_name, label = "dual_stream_contract", "A7-DUAL-STREAM"
        else:
            contract_name, label = "anchor_contract", "A5-ANCHOR"
        anchor = decision_contract.get(contract_name)
        if not isinstance(anchor, dict):
            raise ValueError(f"{label} requires decision_contract.{contract_name}")
        if training_config.initialization_mode != "shape_compatible":
            raise ValueError(f"{label} requires shape-compatible A4 initialization")
        if training_config.freeze_encoder is not True:
            raise ValueError(f"{label} requires the inherited A4 endpoint encoder to remain frozen")
        if training_config.foreground_warmup_epochs != 0:
            raise ValueError(f"{label} requires foreground_warmup_epochs=0")
        if anchor.get("parent_encoder_frozen_for_entire_run") is not True:
            raise ValueError(f"{label} must freeze the parent endpoint encoder for the full run")
        if anchor.get("geometry_must_equal_parent_by_construction") is not True:
            raise ValueError(f"{label} geometry preservation contract is missing")
        if anchor.get("initialization_mode") != "shape_compatible":
            raise ValueError(f"{label} initialization mode differs from frozen contract")
        if anchor.get("initialization_checkpoint") != training_config.initialization_checkpoint:
            raise ValueError(f"{label} initialization checkpoint differs from training config")
        if anchor.get("initialization_checkpoint_sha256") != (
            training_config.initialization_checkpoint_sha256
        ):
            raise ValueError(
                f"{label} initialization checkpoint SHA256 differs from training config"
            )
        if contract_name == "adapter_contract":
            if model_config.transport_adapter_enabled is not True:
                raise ValueError("A6-ADAPTER requires transport_adapter_enabled=true")
            if int(anchor.get("transport_adapter_depth", -1)) != (
                model_config.transport_adapter_depth
            ):
                raise ValueError("A6-ADAPTER depth differs from frozen contract")
            if anchor.get("adapter_is_transport_only") is not True:
                raise ValueError("A6-ADAPTER may only adapt the cost-volume feature path")
            if anchor.get("adapter_identity_initialized") is not True:
                raise ValueError("A6-ADAPTER must be identity-initialized")
        elif contract_name == "dual_stream_contract":
            if model_config.transport_encoder_copy_enabled is not True:
                raise ValueError("A7-DUAL-STREAM requires transport_encoder_copy_enabled=true")
            if model_config.transport_adapter_enabled is True:
                raise ValueError("A7-DUAL-STREAM may not also enable the A6 adapter")
            if anchor.get("dual_stream_is_transport_only") is not True:
                raise ValueError("A7 trainable encoder may only feed the transport branch")
            if anchor.get("transport_encoder_initialized_from_parent") is not True:
                raise ValueError("A7 transport encoder must initialize from the frozen A4 encoder")
    elif training_config.initialization_mode != "none" or training_config.freeze_encoder:
        raise ValueError("baseline A5-CORR may not silently inherit/freeze an A4 checkpoint")
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

    artifact_type = payload.get("artifact_type")
    expected_artifact_type = preflight.get("artifact_type")
    if expected_artifact_type is not None and artifact_type != expected_artifact_type:
        raise ValueError("A5 preflight artifact type differs from frozen contract")
    supported_artifact_types = {
        "a5_transport_preflight_train_only_v1",
        "a5_transport_preflight_train_only_v2",
        "a5_transport_preflight_train_only_v3_confirmation",
    }
    if artifact_type not in supported_artifact_types:
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

    observed_file_sha = _sha256(artifact_path)
    expected_file_sha = preflight.get("file_sha256")
    if expected_file_sha is not None and observed_file_sha != expected_file_sha:
        raise ValueError("A5 preflight file hash differs from frozen contract")
    expected_signed_sha = preflight.get("artifact_sha256")
    if expected_signed_sha is not None and payload.get("artifact_sha256") != expected_signed_sha:
        raise ValueError("A5 preflight signed hash differs from frozen contract")

    decision_checks: object = None
    teacher_transport_r4: object = None
    selected_radius: int | None = None
    selected_temperature: float | None = None

    if artifact_type == "a5_transport_preflight_train_only_v1":
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
        decision_checks = payload.get("decision_checks")
        teacher_transport_r4 = (
            payload.get("teacher_transport", {}).get("4")
            if isinstance(payload.get("teacher_transport"), dict)
            else None
        )
    else:
        decision = payload.get("decision")
        if not isinstance(decision, dict):
            raise ValueError("A5 V2/V3 preflight decision is malformed")
        if decision.get("a5_corr_authorized") is not True:
            raise ValueError("A5-CORR training is blocked because preflight did not pass")

        selected_radius = int(decision.get("selected_radius", -1))
        selected_temperature = float(decision.get("selected_temperature", float("nan")))
        if selected_radius not in (1, 2, 4):
            raise ValueError("A5 preflight selected an invalid transport radius")
        if not math.isfinite(selected_temperature) or selected_temperature not in (
            0.02,
            0.04,
            0.07,
            0.10,
        ):
            raise ValueError("A5 preflight selected an invalid transport temperature")
        if selected_radius != model_config.transport_radius:
            raise ValueError("A5 selected radius differs from runtime model config")
        if not math.isclose(
            selected_temperature,
            model_config.transport_temperature,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("A5 selected temperature differs from runtime model config")
        if int(preflight.get("selected_radius", -1)) != selected_radius:
            raise ValueError("A5 selected radius differs from frozen preflight contract")
        if not math.isclose(
            float(preflight.get("selected_temperature", float("nan"))),
            selected_temperature,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("A5 selected temperature differs from frozen preflight contract")

        decision_checks = decision.get("checks")

        if artifact_type == "a5_transport_preflight_train_only_v3_confirmation":
            if int(scope.get("v2_v3_index_overlap", -1)) != 0:
                raise ValueError("A5 V3 confirmation must be disjoint from V2 discovery rows")
            if int(scope.get("v3_confirmation_rows", 0)) <= 0:
                raise ValueError("A5 V3 confirmation must contain held-out train rows")
            discovery = payload.get("discovery_contract")
            if not isinstance(discovery, dict):
                raise ValueError("A5 V3 discovery contract is malformed")
            if int(discovery.get("candidate_radius", -1)) != selected_radius:
                raise ValueError("A5 V3 candidate radius differs from authorized radius")
            if not math.isclose(
                float(discovery.get("candidate_temperature", float("nan"))),
                selected_temperature,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("A5 V3 candidate temperature differs from authorized temperature")
            if discovery.get("no_candidate_reselection_in_v3") is not True:
                raise ValueError("A5 V3 must not reselect radius/temperature on confirmation rows")
            interpretation = payload.get("interpretation_contract")
            required_v3_flags = (
                "no_radius_or_temperature_search_in_v3",
                "no_training_or_optimizer_steps",
                "ttc_labels_are_not_used",
                "v2_rejection_is_preserved_and_not_overwritten",
                "v3_candidate_was_selected_from_v2_train_only_discovery",
                "v3_confirmation_indices_are_disjoint_from_v2_indices",
            )
            if not isinstance(interpretation, dict) or any(
                interpretation.get(key) is not True for key in required_v3_flags
            ):
                raise ValueError("A5 V3 interpretation contract is incomplete")

    return {
        "path": artifact_path.relative_to(ROOT).as_posix(),
        "file_sha256": observed_file_sha,
        "artifact_sha256": payload.get("artifact_sha256"),
        "artifact_type": artifact_type,
        "a5_corr_authorized": True,
        "scope": scope,
        "decision_checks": decision_checks,
        "teacher_transport_r4": teacher_transport_r4,
        "selected_radius": selected_radius,
        "selected_temperature": selected_temperature,
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
    grouped = data.get("development_protocol") is not None
    expected_opened_splits = ["train"] if grouped else ["train", "validation"]
    if data.get("opened_splits") != expected_opened_splits:
        raise ValueError(
            f"opened_splits must be exactly {expected_opened_splits} for this protocol"
        )
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

    grouped_contract = _load_grouped_development_contract(data) if grouped else None
    if grouped_contract is not None:
        fold = grouped_contract["fold"]
        expected_train_rows = int(fold["train_rows"])
        expected_validation_rows = int(fold["dev_rows"])
        expected_source_rows = int(grouped_contract["protocol"]["sample_count"])
        if data.get("validation_cache_manifest") is not None:
            raise ValueError("grouped development may not reference a validation cache")
    else:
        expected_train_rows = int(data.get("expected_train_rows", 2048))
        expected_validation_rows = int(data.get("expected_validation_rows", 2048))
        expected_source_rows = expected_train_rows
    if min(expected_train_rows, expected_validation_rows, expected_source_rows) <= 0:
        raise ValueError("expected train/dev row counts must be positive")
    split_counts = manifest.get("split_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("cache manifest split_counts must be a mapping")
    if int(split_counts.get("train", -1)) != expected_source_rows:
        raise ValueError(
            "train cache row count differs from frozen protocol: "
            f"{split_counts.get('train')} != {expected_source_rows}"
        )

    validation_manifest_path: Path | None = None
    validation_manifest_hash: str | None = None
    validation_manifest: dict[str, Any] | None = None
    if not grouped:
        validation_manifest_path = manifest_path
        validation_manifest_hash = actual_manifest_hash
        validation_manifest = manifest
    if not grouped and "validation_cache_manifest" in data:
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
    if grouped_contract is not None:
        train_sequences = cast(set[str], grouped_contract["train_sequences"])
        validation_sequences = cast(set[str], grouped_contract["dev_sequences"])
        if set(data.get("train_sequence_ids", [])) != train_sequences:
            raise ValueError("config train sequences differ from frozen grouped fold")
        if set(data.get("dev_sequence_ids", [])) != validation_sequences:
            raise ValueError("config dev sequences differ from frozen grouped fold")
    else:
        assert validation_manifest is not None
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
        raise ValueError("train and dev sequence IDs overlap")

    status_lines = _git(
        "-c", "core.quotepath=false", "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    worktree = _blocking_worktree_state(status_lines)
    code_dirty = bool(worktree["science_code_dirty"])
    if code_dirty:
        blocking = worktree.get("blocking_tracked_dirty_paths", [])
        untracked_code = worktree.get("untracked_code_paths", [])
        raise RuntimeError(
            "representative real screen requires clean scientific code state; "
            f"blocking_tracked={blocking}, untracked_code={untracked_code}"
        )

    model_path = _resolve(raw["model_config"])
    model_config = _model_config(model_path)
    training_raw = raw.get("training")
    loss_raw = raw.get("loss")
    if not isinstance(training_raw, dict) or not isinstance(loss_raw, dict):
        raise ValueError("training and loss mappings are required")
    training_config = CausalScaleEAPTrainingConfig(**training_raw)
    loss_config = CausalScaleTTCLossConfig(**loss_raw)
    initialization_metadata: dict[str, Any] | None = None
    if training_config.initialization_mode == "shape_compatible":
        if training_config.initialization_checkpoint is None:
            raise ValueError("initialization checkpoint path is missing")
        initialization_path = _resolve(training_config.initialization_checkpoint)
        observed_initialization_sha = _sha256(initialization_path)
        if observed_initialization_sha != training_config.initialization_checkpoint_sha256:
            raise ValueError("initialization checkpoint SHA256 differs from frozen training config")
        initialization_payload = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        if initialization_payload.get("artifact_type") != (
            "causal_scale_eap_public_validation_checkpoint_v1"
        ):
            raise ValueError("initialization checkpoint artifact type is incompatible")
        source_model_config = initialization_payload.get("model_config")
        if not isinstance(source_model_config, dict):
            raise ValueError("initialization checkpoint model_config is malformed")
        if bool(source_model_config.get("transport_enabled", False)):
            raise ValueError("A5-ANCHOR must initialize from a transport-disabled A4 checkpoint")
        initialization_metadata = {
            "path": initialization_path.relative_to(ROOT).as_posix(),
            "sha256": observed_initialization_sha,
            "artifact_type": initialization_payload.get("artifact_type"),
            "source_transport_enabled": False,
        }
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
    if grouped:
        validation_dataset = GarlTTCObjectEventV4Dataset(
            str(manifest_path), splits=("train",)
        )
    else:
        assert validation_manifest_path is not None
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
    fold_identity: dict[str, Any] | None = None
    if grouped_contract is not None:
        train_dataset = SequenceIndexedView(
            train_dataset, sequence_ids=train_sequences
        )
        validation_dataset = SequenceIndexedView(
            validation_dataset, sequence_ids=validation_sequences
        )
        train_identity = train_dataset.identity_frame()
        dev_identity = validation_dataset.identity_frame()
        observed_train_tokens = _sorted_values_sha256(
            train_identity["sample_token"].astype(str).tolist()
        )
        observed_dev_tokens = _sorted_values_sha256(
            dev_identity["sample_token"].astype(str).tolist()
        )
        expected_fold = grouped_contract["fold"]
        if len(train_dataset) != expected_train_rows or len(validation_dataset) != (
            expected_validation_rows
        ):
            raise ValueError("grouped-development view row counts differ from frozen fold")
        if observed_train_tokens != expected_fold["train_sample_tokens_sha256"]:
            raise ValueError("grouped-development train token hash differs")
        if observed_dev_tokens != expected_fold["dev_sample_tokens_sha256"]:
            raise ValueError("grouped-development dev token hash differs")
        fold_identity = {
            "fold": grouped_contract["fold_index"],
            "train_rows": len(train_dataset),
            "dev_rows": len(validation_dataset),
            "train_sample_tokens_sha256": observed_train_tokens,
            "dev_sample_tokens_sha256": observed_dev_tokens,
            "sequence_disjoint": train_sequences.isdisjoint(validation_sequences),
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
        checkpoint_payload(
            result,
            training_config,
            loss_config,
            artifact_type=(
                "causal_scale_eap_grouped_dev_checkpoint_v1"
                if grouped
                else "causal_scale_eap_public_validation_checkpoint_v1"
            ),
        ),
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, checkpoint_path)
    validation = result.best_validation
    predictions = pd.DataFrame(
        {
            "sample_token": validation["sample_tokens"],
            "sequence_id": validation["sequence_ids"],
            "track_id": validation["track_ids"],
            "target_ttc_s": validation["target_ttc_s"],
            "prediction_ttc_s": validation["prediction_ttc_s"],
        }
    )
    evaluation_split = "dev" if grouped else "validation"
    predictions_path = output_dir / f"{evaluation_split}_predictions.csv"
    predictions.to_csv(predictions_path, index=False, lineterminator="\n")
    metrics = {
        key: value
        for key, value in validation.items()
        if key
        not in {
            "sample_tokens",
            "sequence_ids",
            "track_ids",
            "target_ttc_s",
            "prediction_ttc_s",
        }
    }
    cache_payload: dict[str, Any] = {
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": actual_manifest_hash,
        "artifact_sha256": manifest["artifact_sha256"],
        "split_counts": manifest["split_counts"],
        "source_train_rows": expected_source_rows,
        "effective_train_rows": expected_train_rows,
        "effective_dev_rows": expected_validation_rows,
        "train_sequence_ids": sorted(train_sequences),
        "dev_sequence_ids": sorted(validation_sequences),
    }
    if not grouped:
        assert validation_manifest_path is not None
        assert validation_manifest_hash is not None
        assert validation_manifest is not None
        cache_payload.update(
            {
                "validation_manifest_path": (
                    validation_manifest_path.relative_to(ROOT).as_posix()
                ),
                "validation_manifest_sha256": validation_manifest_hash,
                "validation_artifact_sha256": validation_manifest["artifact_sha256"],
                "validation_split_counts": validation_manifest["split_counts"],
                "effective_validation_rows": expected_validation_rows,
                "validation_sequence_ids": sorted(validation_sequences),
            }
        )
    development_protocol_metadata = None
    if grouped_contract is not None:
        development_protocol_metadata = {
            "path": grouped_contract["path"].relative_to(ROOT).as_posix(),
            "file_sha256": grouped_contract["file_sha256"],
            "artifact_sha256": grouped_contract["artifact_sha256"],
            "fold_identity": fold_identity,
            "train_only_grouped_dev": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
        }
    payload: dict[str, Any] = {
        "artifact_type": (
            "causal_scale_eap_train_only_grouped_dev_run_v1"
            if grouped
            else "causal_scale_eap_public_validation_screen_v1"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "status": (
            "completed_train_only_grouped_dev"
            if grouped
            else "completed_public_validation_only"
        ),
        "selectable": False,
        "development_selectable": grouped,
        "sota_claim_authorized": False,
        "official_test_opened": False,
        "public_validation_opened": not grouped,
        "public_validation_used_for_selection": False if grouped else True,
        "private_test_opened": False,
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
        "cache": cache_payload,
        "development_protocol": development_protocol_metadata,
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
        "initialization": {
            **result.initialization,
            "validated_source": initialization_metadata,
        },
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "selection": {
            "split": evaluation_split,
            "best_epoch": result.best_epoch,
            **result.best_selection,
        },
        f"{evaluation_split}_metrics": metrics,
        "history": result.history,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else None
        ),
        "peak_vram_reserved_mb": (
            float(torch.cuda.max_memory_reserved(device) / 2**20)
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
