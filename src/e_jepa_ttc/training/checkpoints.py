"""Checkpoint provenance helpers for downstream experiment ledgers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


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
