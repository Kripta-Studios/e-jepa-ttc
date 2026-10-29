"""Small IO helpers for manifests and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from e_jepa_ttc.artifacts.hashing import sign_artifact


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def read_structured(path: str | Path) -> dict[str, Any]:
    """Read a JSON or YAML mapping."""

    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        msg = f"Expected a mapping in {input_path}, got {type(data).__name__}."
        raise ValueError(msg)
    return data


def write_structured(path: str | Path, data: dict[str, Any]) -> None:
    """Write a JSON or YAML mapping."""

    output_path = ensure_parent(path)
    if output_path.suffix.lower() == ".json":
        if "artifact_type" in data:
            data = sign_artifact(data)
        output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    else:
        output_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
