"""Passive audit of official GarlTTC release files (checkpoints, manifests, configs).

This utility strictly inspects file paths, sizes, SHA256 hashes, and optional
top-level state dict keys WITHOUT importing model architectures or loading weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_official_release(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    file_size_bytes = checkpoint_path.stat().st_size
    sha256_hash = _sha256_file(checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path.as_posix(),
        "size_bytes": file_size_bytes,
        "sha256": sha256_hash,
        "top_level_keys": None,
        "keys_inspected": False,
        "audit_mode": "passive_file_inventory",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to official GarlTTC release artifact",
    )
    parser.add_argument("--output", type=Path, help="Path to save audit JSON report")
    args = parser.parse_args()

    report = audit_official_release(args.checkpoint)

    formatted = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")
        print(f"Passive audit written to {args.output}")
    else:
        print(formatted)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
