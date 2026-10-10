"""Additive host-pipeline sidecar for the frozen RGB-PORT campaign."""

from pathlib import Path
from typing import Any


def build_pipeline_freeze(
    run: Path,
    cpu_admission: Path,
    gpu_admission: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Build the V2 pipeline freeze without importing its CLI eagerly."""
    from .contracts import build_pipeline_freeze as build

    return build(run, cpu_admission, gpu_admission, output)


def validate_pipeline_freeze(path: Path) -> dict[str, Any]:
    """Validate the V2 pipeline freeze without importing its CLI eagerly."""
    from .contracts import validate_pipeline_freeze as validate

    return validate(path)


__all__ = ["build_pipeline_freeze", "validate_pipeline_freeze"]
