#!/usr/bin/env python3
"""Materialize deterministic v4.6-v4.8 configs with a true per-run seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("train"), dict):
        raise ValueError(f"Config lacks train mapping: {path}")
    return raw


def materialize(*, seed: int, sources: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for name, source in sources.items():
        payload = _load(source)
        original_seed = int(payload["train"].get("seed", -1))
        payload["train"]["seed"] = int(seed)
        destination = output_dir / f"{name}_seed_{seed}.yaml"
        destination.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        reloaded = _load(destination)
        if int(reloaded["train"]["seed"]) != seed:
            raise RuntimeError(f"Failed to persist seed {seed} in {destination}")
        records[name] = {
            "source": source.resolve().as_posix(),
            "source_sha256": _sha256(source),
            "source_seed": original_seed,
            "materialized": destination.resolve().as_posix(),
            "materialized_sha256": _sha256(destination),
            "materialized_seed": seed,
        }
    manifest = {
        "artifact_type": "object_event_v4_10_seed_configs",
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "configs": records,
        "scientific_contract": {
            "only_train_seed_is_overridden": True,
            "architecture_and_loss_are_unchanged": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--v46-config", type=Path, required=True)
    parser.add_argument("--v47-config", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        seed=args.seed,
        sources={
            "v46": args.v46_config.resolve(),
            "v47": args.v47_config.resolve(),
            "v48": args.v48_config.resolve(),
        },
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
