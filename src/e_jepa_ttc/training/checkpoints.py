"""Checkpoint provenance helpers for downstream experiment ledgers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from e_jepa_ttc.data.carla_looming import CARLA_LOOMING_DATASET_ID


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


def validate_external_ssl_checkpoint(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    source_split_path: str | Path,
) -> dict[str, Any]:
    """Validate an approved external SSL encoder before EvTTC fine-tuning."""

    path = Path(checkpoint_path)
    if checkpoint.get("external_pretraining") is not True:
        raise ValueError("External SSL checkpoint must declare external_pretraining=true.")
    if checkpoint.get("pretraining_dataset_id") != CARLA_LOOMING_DATASET_ID:
        raise ValueError(
            "External SSL checkpoint dataset is not approved by the closed protocol."
        )
    if checkpoint.get("model_name") != "event-tubelet-transformer":
        raise ValueError("External SSL checkpoint must use event-tubelet-transformer.")
    if int(checkpoint.get("in_channels", -1)) != 21 or int(checkpoint.get("bins", -1)) != 5:
        raise ValueError("External SSL checkpoint is incompatible with 21-channel EvTTC BASE.")
    state = checkpoint.get("encoder_state_dict")
    if not isinstance(state, Mapping) or "event_embed.weight" not in state:
        raise ValueError("External SSL checkpoint has no compatible encoder_state_dict.")
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
    source_split = Path(source_split_path)
    if not source_split.is_file():
        raise FileNotFoundError(f"External SSL source split is missing: {source_split}.")
    if checkpoint.get("split_manifest_sha256") != _file_sha256(source_split):
        raise ValueError("External SSL checkpoint source split hash does not match.")
    provenance = checkpoint_provenance(path, checkpoint)
    if provenance["recommended_for_downstream"] is not True:
        raise ValueError("External SSL checkpoint must be the validation-selected best encoder.")
    return {
        **provenance,
        "pretraining_dataset_id": checkpoint["pretraining_dataset_id"],
        "source_split": source_split.as_posix(),
        "source_split_sha256": checkpoint["split_manifest_sha256"],
        "uses_evttc_pretraining_events": False,
        "benchmark10_opened": False,
    }


__all__ = ["checkpoint_provenance", "validate_external_ssl_checkpoint"]
