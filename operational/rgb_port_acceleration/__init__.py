"""Auditable runtime-only acceleration sidecar for the frozen RGB-PORT DAG."""

from pathlib import Path
from typing import Any


def build_acceleration_freeze(
    run: Path, admission: Path, output: Path | None = None
) -> dict[str, Any]:
    """Build the supplemental runtime freeze without eager CLI imports."""
    from .contracts import build_acceleration_freeze as build

    return build(run, admission, output)


def validate_acceleration_freeze(path: Path) -> dict[str, Any]:
    """Validate the supplemental runtime freeze without eager CLI imports."""
    from .contracts import validate_acceleration_freeze as validate

    return validate(path)


__all__ = ["build_acceleration_freeze", "validate_acceleration_freeze"]
