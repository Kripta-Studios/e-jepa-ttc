"""Read-only export of tensors already exposed by a detector forward pass.

The observer never hooks modules, changes the output mapping, or reconstructs
values that the historical producer did not expose.  This keeps it outside the
frozen checkpoint runtime while still making query-level diagnostics auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

SCHEMA = "monodgp_external_diagnostic_sidecar_v1"
UNAVAILABLE_FIELDS = (
    "decoded_box_2d_per_query",
    "geometric_depth_prior_per_query",
    "geometric_depth_residual_per_query",
    "foreground_membership_per_query",
)
QUERY_FIELDS = (
    "pred_logits",
    "pred_boxes",
    "pred_3d_dim",
    "pred_depth",
    "pred_angle",
    "pred_velocity",
)
SPATIAL_FIELDS = ("pred_depth_map_logits", "pred_region_prob")


def sha256_file(path: Path) -> str:
    """Hash an observer or evidence file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_batch(tensor: torch.Tensor, batch_index: int, name: str) -> torch.Tensor:
    """Select one batch from a known historical ``[B,Q,D]`` query output."""
    if tensor.ndim != 3:
        raise ValueError(f"Known query output {name} must have shape [B,Q,D]")
    if not 0 <= batch_index < tensor.shape[0]:
        raise IndexError("Diagnostic batch index is outside the exposed tensor")
    return tensor[batch_index].detach().to(device="cpu").clone()


def capture_exposed_outputs(
    outputs: Mapping[str, Any], *, sample_token: str, batch_index: int = 0
) -> dict[str, Any]:
    """Serialize finite query tensors from an output mapping.

    Non-tensor values are described but not guessed.  Tensor values without a
    query axis are recorded as unavailable rather than ambiguously flattened.
    """
    if not sample_token:
        raise ValueError("sample_token must be nonempty")
    tensors: dict[str, Any] = {}
    non_query_tensors: dict[str, Any] = {}
    unavailable = {name: "not_exposed_by_historical_forward_output" for name in UNAVAILABLE_FIELDS}
    for name in sorted(outputs):
        value = outputs[name]
        if not isinstance(value, torch.Tensor):
            unavailable[f"output:{name}"] = f"unsupported_exposed_type:{type(value).__name__}"
            continue
        if name not in QUERY_FIELDS:
            non_query_tensors[name] = {
                "source_shape": list(value.shape),
                "dtype": str(value.dtype),
                "classification": "known_spatial_output"
                if name in SPATIAL_FIELDS
                else "unknown_non_query_output",
                "values_exported": False,
            }
            unavailable[f"output:{name}"] = "not_a_whitelisted_query_output"
            continue
        selected = _query_batch(value, batch_index, name)
        if not torch.isfinite(selected).all().item():
            raise ValueError(f"Non-finite exposed diagnostic tensor: {name}")
        tensors[name] = {
            "source_shape": list(value.shape),
            "selected_shape": list(selected.shape),
            "dtype": str(selected.dtype),
            "values": selected.tolist(),
        }
    if not tensors:
        raise ValueError("No query tensors were exposed by the detector")
    return {
        "sample_token": sample_token,
        "batch_index": batch_index,
        "selection": "direct batch selection from historical final [B,Q,D] outputs",
        "tensors": tensors,
        "non_query_tensors": non_query_tensors,
        "unavailable": unavailable,
    }


def build_sidecar(
    frames: list[dict[str, Any]],
    *,
    checkpoint_sha256: str,
    prediction_output_sha256: str,
    observer_path: Path,
) -> dict[str, Any]:
    """Bind query evidence to the exact checkpoint, predictions, and observer."""
    if len(checkpoint_sha256) != 64 or len(prediction_output_sha256) != 64:
        raise ValueError("Sidecar inputs require full SHA-256 digests")
    tokens = [frame.get("sample_token") for frame in frames]
    if not frames or any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("Sidecar requires nonempty frame tokens")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Sidecar frame tokens must be unique")
    return {
        "schema": SCHEMA,
        "scope": "external observation of exposed forward outputs; no labels or internal hooks",
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_output_sha256": prediction_output_sha256,
        "observer_sha256": sha256_file(observer_path),
        "sample_count": len(frames),
        "frames": frames,
        "claims": {
            "changes_predictions": False,
            "ground_truth_access": False,
            "internal_unexposed_values_reconstructed": False,
        },
    }


def validate_json_finite(value: object) -> None:
    """Reject NaN/Infinity before writing scientific evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Diagnostic evidence contains a non-finite float")
    if isinstance(value, dict):
        for nested in value.values():
            validate_json_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            validate_json_finite(nested)
    json.dumps(value, allow_nan=False)
