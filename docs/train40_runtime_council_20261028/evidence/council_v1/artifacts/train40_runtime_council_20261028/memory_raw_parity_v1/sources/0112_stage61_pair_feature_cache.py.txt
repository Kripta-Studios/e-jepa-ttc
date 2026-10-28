"""Typed, signed feature caches for Scientific Recovery V9 Stage 61/62.

The cache deliberately separates model inputs from identity and supervision.
Neither ``PairFeatureBatch`` nor ``LocalTemporalFieldBatch`` can carry a TTC
target, sequence identifier, fold identifier, or other routing shortcut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash

PAIR_FEATURE_DIM = 133
LOCAL_PATCH_COUNT = 16
LOCAL_FEATURE_DIM = 34


@dataclass(frozen=True)
class PairFeatureBatch:
    """Inference-only PAIR features; shape ``[B,133]``."""

    features: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != PAIR_FEATURE_DIM:
            raise ValueError("PAIR features must have shape [B,133]")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("PAIR features must be finite")


@dataclass(frozen=True)
class PairSupervisionBatch:
    """Training-only PAIR target phase and sample weight."""

    target_phase: torch.Tensor
    sample_weight: torch.Tensor

    def __post_init__(self) -> None:
        if self.target_phase.ndim != 1 or self.target_phase.shape != self.sample_weight.shape:
            raise ValueError("PAIR supervision vectors must be matching one-dimensional tensors")
        if not bool(torch.isfinite(self.target_phase).all()):
            raise ValueError("PAIR target phase must be finite")
        if not bool(torch.isfinite(self.sample_weight).all()) or bool(
            (self.sample_weight < 0).any()
        ):
            raise ValueError("PAIR sample weights must be finite and non-negative")


@dataclass(frozen=True)
class LocalTemporalFieldBatch:
    """Inference-only local field inputs for X2."""

    patch_features: torch.Tensor
    patch_valid: torch.Tensor
    a5_phase: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.patch_features.shape[0] if self.patch_features.ndim == 3 else -1
        if self.patch_features.shape != (batch, LOCAL_PATCH_COUNT, LOCAL_FEATURE_DIM):
            raise ValueError("local patch features must have shape [B,16,34]")
        if self.patch_valid.shape != (batch, LOCAL_PATCH_COUNT):
            raise ValueError("local patch validity must have shape [B,16]")
        if self.a5_phase.shape != (batch,):
            raise ValueError("A5 phase must have shape [B]")
        if not bool(torch.isfinite(self.patch_features).all()) or not bool(
            torch.isfinite(self.a5_phase).all()
        ):
            raise ValueError("local-field model inputs must be finite")


def save_feature_cache(
    path: Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Write an NPZ plus metadata CSV and return its signed manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    forbidden = {"target", "target_phase", "target_ttc", "sample_weight", "label", "bucket"}
    if forbidden & set(arrays):
        raise ValueError("supervision is forbidden inside an inference feature cache")
    if not arrays or metadata.empty or metadata["sample_token"].duplicated().any():
        raise ValueError("feature cache requires non-empty arrays and unique metadata tokens")
    rows = len(metadata)
    for name, value in arrays.items():
        if value.shape[0] != rows or not np.isfinite(value).all():
            raise ValueError(f"cache array {name!r} is row-misaligned or non-finite")
    np.savez_compressed(path, **arrays)
    metadata_path = path.with_suffix(".metadata.csv")
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    manifest = sign_artifact(
        {
            "artifact_type": "stage61_stage62_feature_cache_v1",
            "status": "completed",
            "identity": identity,
            "row_count": rows,
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
            "cache": {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": compute_file_hash(str(path)),
            },
            "metadata": {
                "path": metadata_path.name,
                "bytes": metadata_path.stat().st_size,
                "sha256": compute_file_hash(str(metadata_path)),
            },
            "contains_targets": False,
            "metadata_is_outside_model_forward": True,
            "sealed_evaluation_opened": False,
        }
    )
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_feature_cache(path: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    """Load a cache only after verifying the signed physical manifest."""

    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(manifest):
        raise ValueError("feature cache manifest signature mismatch")
    if compute_file_hash(str(path)) != manifest["cache"]["sha256"]:
        raise ValueError("feature cache SHA-256 mismatch")
    metadata_path = path.with_suffix(".metadata.csv")
    if compute_file_hash(str(metadata_path)) != manifest["metadata"]["sha256"]:
        raise ValueError("feature cache metadata SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != manifest["row_count"] or metadata["sample_token"].duplicated().any():
        raise ValueError("feature cache metadata identity mismatch")
    for name, values in arrays.items():
        if list(values.shape) != manifest["array_shapes"][name] or not np.isfinite(values).all():
            raise ValueError(f"feature cache array mismatch: {name}")
    return arrays, metadata, manifest


__all__ = [
    "LOCAL_FEATURE_DIM",
    "LOCAL_PATCH_COUNT",
    "PAIR_FEATURE_DIM",
    "LocalTemporalFieldBatch",
    "PairFeatureBatch",
    "PairSupervisionBatch",
    "load_feature_cache",
    "save_feature_cache",
]
