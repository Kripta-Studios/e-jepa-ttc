"""Explicit model-only initialization for a new data campaign, never strict resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def load_model_warm_start(
    model: torch.nn.Module, receipt_path: Path, effective_model_sha256: str
) -> dict[str, Any]:
    """Validate checkpoint identity/architecture and copy only the model parameters."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    path = Path(receipt["checkpoint"])
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != receipt["checkpoint_sha256"]:
        raise ValueError("Warm-start checkpoint SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        payload["runtime_provenance"]["effective_model_sha256"] != effective_model_sha256
        or payload["update"] != receipt["source_update"]
    ):
        raise ValueError("Warm-start effective model or completed source update differs")
    model.load_state_dict(payload["model"], strict=True)
    return {
        "schema": "phase_model_only_warm_start_v1",
        "checkpoint": str(path),
        "checkpoint_sha256": actual,
        "source_update": payload["update"],
        "source_config_sha256": payload["config_sha256"],
        "source_runtime_provenance": payload["runtime_provenance"],
        "optimizer_scheduler_sampler_rng": "new campaign state; not imported from parent",
    }
