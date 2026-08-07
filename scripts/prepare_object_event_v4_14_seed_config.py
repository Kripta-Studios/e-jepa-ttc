#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diff_paths(a: Any, b: Any, prefix: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        result: list[str] = []
        for key in keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                result.append(child)
            else:
                result.extend(_diff_paths(a[key], b[key], child))
        return result
    return [] if a == b else [prefix]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("train"), dict):
        raise ValueError("v4.12 source config must contain train mapping")
    materialized = copy.deepcopy(source)
    source_seed = int(materialized["train"]["seed"])
    materialized["train"]["seed"] = int(args.seed)
    differences = _diff_paths(source, materialized)
    if differences != ["train.seed"]:
        raise RuntimeError(f"Unexpected seed config differences: {differences}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(materialized, sort_keys=False), encoding="utf-8")
    manifest = {
        "artifact_type": "object_event_v4_14_true_seed_probe_config",
        "created_at": datetime.now(UTC).isoformat(),
        "source": args.source.resolve().as_posix(),
        "source_sha256": _sha256(args.source),
        "source_seed": source_seed,
        "materialized": args.output.resolve().as_posix(),
        "materialized_sha256": _sha256(args.output),
        "materialized_seed": int(args.seed),
        "changed_paths": differences,
        "scientific_contract": {
            "only_train_seed_is_overridden": True,
            "architecture_loss_and_gates_are_unchanged": True,
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
