"""Checkpoint provenance helpers for downstream experiment ledgers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.data.carla_looming import CARLA_LOOMING_DATASET_ID
from e_jepa_ttc.utils.io import read_structured


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8192 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def checkpoint_provenance(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return explicit seed and best-vs-last provenance, including legacy files."""

    path = Path(checkpoint_path)
    role = checkpoint.get("checkpoint_role")
    if role is None:
        stem = path.stem.lower()
        if stem.endswith("_best") or "encoder_best" in stem:
            role = "best"
        elif stem.endswith("_last") or "encoder_last" in stem:
            role = "last"
        else:
            role = "unspecified"
    selected_by = checkpoint.get("checkpoint_selected_by")
    if selected_by is None:
        selected_by = (
            "validation_loss"
            if role == "best"
            else ("final_epoch" if role == "last" else "unspecified")
        )
    return {
        "path": path.as_posix(),
        "checkpoint_sha256": _file_sha256(path),
        "source_epoch": checkpoint.get("epoch"),
        "source_seed": checkpoint.get("seed"),
        "checkpoint_role": str(role),
        "checkpoint_selected_by": str(selected_by),
        "recommended_for_downstream": role == "best" and selected_by == "validation_loss",
        "selection_warning": (
            "last checkpoint is not validation-selected; justify it with a frozen ablation"
            if role == "last"
            else None
        ),
    }


def _validate_external_carla_checkpoint(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    source_split_path: str | Path,
) -> dict[str, Any]:
    """Validate provenance shared by approved CARLA encoder checkpoints."""

    path = Path(checkpoint_path)
    if checkpoint.get("external_pretraining") is not True:
        raise ValueError("External SSL checkpoint must declare external_pretraining=true.")
    if checkpoint.get("pretraining_dataset_id") != CARLA_LOOMING_DATASET_ID:
        raise ValueError("External SSL checkpoint dataset is not approved by the closed protocol.")
    if checkpoint.get("model_name") != "event-tubelet-transformer":
        raise ValueError("External SSL checkpoint must use event-tubelet-transformer.")
    if int(checkpoint.get("in_channels", -1)) != 21 or int(checkpoint.get("bins", -1)) != 5:
        raise ValueError("External SSL checkpoint is incompatible with 21-channel EvTTC BASE.")
    state = checkpoint.get("encoder_state_dict")
    if not isinstance(state, Mapping) or "event_embed.weight" not in state:
        raise ValueError("External SSL checkpoint has no compatible encoder_state_dict.")
    source_split = Path(source_split_path)
    if not source_split.is_file():
        raise FileNotFoundError(f"External SSL source split is missing: {source_split}.")
    source_split_payload = read_structured(source_split)
    if not verify_artifact_hash(source_split_payload):
        raise ValueError("External SSL source split artifact signature is invalid.")
    checkpoint_split_artifact = checkpoint.get("split_artifact_sha256")
    if checkpoint_split_artifact is not None:
        if checkpoint_split_artifact != source_split_payload["artifact_sha256"]:
            raise ValueError("External SSL checkpoint source split artifact does not match.")
    elif checkpoint.get("split_manifest_sha256") not in {
        _file_sha256(source_split),
        source_split_payload.get("legacy_file_sha256"),
    }:
        raise ValueError("External SSL checkpoint legacy source split hash does not match.")
    provenance = checkpoint_provenance(path, checkpoint)
    if provenance["recommended_for_downstream"] is not True:
        raise ValueError("External SSL checkpoint must be the validation-selected best encoder.")
    return {
        **provenance,
        "pretraining_dataset_id": checkpoint["pretraining_dataset_id"],
        "source_split": source_split.as_posix(),
        "source_split_sha256": checkpoint["split_manifest_sha256"],
        "source_split_artifact_sha256": source_split_payload["artifact_sha256"],
        "uses_evttc_pretraining_events": False,
        "benchmark10_opened": False,
    }


def validate_external_ssl_checkpoint(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    source_split_path: str | Path,
) -> dict[str, Any]:
    """Validate a label-free external SSL encoder before EvTTC fine-tuning."""

    forbidden_truthy = (
        "uses_ttc_labels",
        "uses_collision_labels",
        "uses_velocity_feature",
        "uses_object_diameter_feature",
        "benchmark10_opened",
    )
    violations = [field for field in forbidden_truthy if checkpoint.get(field) is not False]
    if violations:
        raise ValueError(f"External SSL checkpoint violates provenance fields: {violations}.")
    result = _validate_external_carla_checkpoint(
        checkpoint_path,
        checkpoint,
        source_split_path=source_split_path,
    )
    return {**result, "pretraining_regime": "ssl", "uses_ttc_labels": False}


def validate_external_ttc_checkpoint(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    source_split_path: str | Path,
) -> dict[str, Any]:
    """Validate the separate CARLA JEPA+synthetic-TTC ablation checkpoint."""

    if checkpoint.get("pretraining_regime") != "ssl_plus_synthetic_ttc":
        raise ValueError("External TTC checkpoint has the wrong pretraining regime.")
    if checkpoint.get("uses_ttc_labels") is not True:
        raise ValueError("External TTC checkpoint must disclose synthetic TTC supervision.")
    if checkpoint.get("uses_collision_labels") is not True:
        raise ValueError("External TTC checkpoint must disclose collision-label use.")
    forbidden_truthy = (
        "uses_velocity_feature",
        "uses_object_diameter_feature",
        "benchmark10_opened",
    )
    violations = [field for field in forbidden_truthy if checkpoint.get(field) is not False]
    if violations:
        raise ValueError(f"External TTC checkpoint violates provenance fields: {violations}.")
    if checkpoint.get("synthetic_ttc_definition") != (
        "collision_end_timestamp_minus_causal_reference_timestamp"
    ):
        raise ValueError("External TTC checkpoint has an unknown synthetic TTC definition.")
    result = _validate_external_carla_checkpoint(
        checkpoint_path,
        checkpoint,
        source_split_path=source_split_path,
    )
    return {
        **result,
        "pretraining_regime": "ssl_plus_synthetic_ttc",
        "uses_ttc_labels": True,
        "synthetic_ttc_definition": checkpoint["synthetic_ttc_definition"],
    }


def validate_external_eap_checkpoint(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    source_split_path: str | Path,
    expected_regime: str,
) -> dict[str, Any]:
    """Validate a public eAP train-only SSL or weak-geometry encoder."""

    if expected_regime not in {"eap_ssl", "eap_geo"}:
        raise ValueError("Expected eAP regime must be eap_ssl or eap_geo.")
    if checkpoint.get("external_pretraining") is not True:
        raise ValueError("eAP checkpoint must declare external_pretraining=true.")
    if checkpoint.get("pretraining_dataset_id") != "EAP_PUBLIC_TRAIN40":
        raise ValueError("eAP checkpoint dataset is not the approved public train-40 release.")
    if checkpoint.get("pretraining_regime") != expected_regime:
        raise ValueError("eAP checkpoint pretraining regime differs from the requested arm.")
    if checkpoint.get("model_name") != "event-tubelet-transformer":
        raise ValueError("eAP checkpoint must use event-tubelet-transformer.")
    if int(checkpoint.get("in_channels", -1)) != 21 or int(checkpoint.get("bins", -1)) != 5:
        raise ValueError("eAP checkpoint is incompatible with 21-channel EvTTC BASE.")
    state = checkpoint.get("encoder_state_dict")
    if not isinstance(state, Mapping) or "event_embed.weight" not in state:
        raise ValueError("eAP checkpoint has no compatible encoder_state_dict.")
    forbidden_truthy = (
        "uses_ttc_labels",
        "uses_collision_labels",
        "uses_rgb",
        "uses_evttc_pretraining_events",
        "benchmark10_opened",
    )
    violations = [field for field in forbidden_truthy if checkpoint.get(field) is not False]
    if violations:
        raise ValueError(f"eAP checkpoint violates provenance fields: {violations}.")
    if checkpoint.get("uses_labels_for_window_sampling") is not False:
        raise ValueError("eAP checkpoint must use label-independent window sampling.")
    geometry_expected = expected_regime == "eap_geo"
    for field in ("uses_object_bboxes", "uses_depth_track_derivatives"):
        if checkpoint.get(field) is not geometry_expected:
            raise ValueError(f"eAP checkpoint has inconsistent {field} provenance.")
    source_split = Path(source_split_path)
    if not source_split.is_file():
        raise FileNotFoundError(f"eAP source split is missing: {source_split}.")
    split_payload = read_structured(source_split)
    if not verify_artifact_hash(split_payload):
        raise ValueError("eAP source split artifact signature is invalid.")
    if checkpoint.get("split_artifact_sha256") != split_payload.get("artifact_sha256"):
        raise ValueError("eAP checkpoint source split artifact does not match.")
    if checkpoint.get("inventory_artifact_sha256") != split_payload.get(
        "inventory_artifact_sha256"
    ):
        raise ValueError("eAP checkpoint inventory provenance does not match its split.")
    provenance = checkpoint_provenance(checkpoint_path, checkpoint)
    if provenance["recommended_for_downstream"] is not True:
        raise ValueError("eAP checkpoint must be the validation-selected best encoder.")
    return {
        **provenance,
        "pretraining_dataset_id": checkpoint["pretraining_dataset_id"],
        "pretraining_regime": expected_regime,
        "source_split": source_split.as_posix(),
        "source_split_sha256": _file_sha256(source_split),
        "source_split_artifact_sha256": split_payload["artifact_sha256"],
        "inventory_artifact_sha256": split_payload["inventory_artifact_sha256"],
        "uses_ttc_labels": False,
        "uses_object_bboxes": geometry_expected,
        "uses_depth_track_derivatives": geometry_expected,
        "uses_labels_for_window_sampling": False,
        "uses_rgb": False,
        "uses_evttc_pretraining_events": False,
        "benchmark10_opened": False,
    }


__all__ = [
    "checkpoint_provenance",
    "validate_external_eap_checkpoint",
    "validate_external_ssl_checkpoint",
    "validate_external_ttc_checkpoint",
]
