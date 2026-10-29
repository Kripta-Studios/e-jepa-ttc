#!/usr/bin/env python
"""Atomically sign a locally generated V8 orchestration status payload.

This intentionally has a narrow surface: PowerShell writes an unsigned JSON
payload to a private temporary location and this helper adds the standard V8
artifact digest before replacing the destination.  It never reads data splits
or results, so monitoring cannot accidentally evaluate a sealed split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402


def _payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V8 status payload must be a JSON object")
    if "artifact_sha256" in value:
        raise ValueError("PowerShell status payload must be unsigned before signing")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        _atomic_write(args.output.resolve(), sign_artifact(_payload(args.input.resolve())))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"V8 status signing failed closed: {type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
