#!/usr/bin/env python
"""Delegate V8 robustness only with an explicit frozen winner checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_winner(manifest_path: Path, checkpoint: Path) -> None:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError("winner manifest must be a signed artifact")
    closed = value.get("closed_evaluation", {})
    if not isinstance(closed, dict) or any(flag is not False for flag in closed.values()):
        raise ValueError("winner manifest must keep every sealed evaluation split closed")
    reference = value.get("checkpoint")
    if not isinstance(reference, dict):
        raise ValueError("winner manifest lacks the exact frozen checkpoint reference")
    if reference.get("sha256") != _sha256(checkpoint):
        raise ValueError("checkpoint hash differs from the signed frozen winner manifest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--winner-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--factory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args()
    if (
        args.checkpoint is None
        or args.winner_manifest is None
        or args.output_dir is None
        or args.factory is None
    ):
        if args.dry_run:
            print(
                "planned: robustness requires --checkpoint --winner-manifest --output-dir --factory"
            )
            return 0
        parser.error("robustness requires --checkpoint --winner-manifest --output-dir --factory")
    try:
        _validate_winner(args.winner_manifest, args.checkpoint)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"V8 robustness failed closed: {type(error).__name__}: {error}\n")
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/evaluate_scientific_recovery_v8_robustness.py",
        "--factory",
        args.factory,
        "--output",
        str(args.output_dir / "robustness.json"),
        "--device",
        args.device,
        *unknown,
    ]
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
