"""Atomic, recoverable cache artifacts and explicit migration of verified head states."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from operational.evttc_transfer_audit.common import canonical_sha, file_sha


def atomic_receipt(path: Path, value: object) -> None:
    """Commit a complete immutable JSON receipt with atomic replacement of a temp file."""
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to replace different evidence: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_npz(path: Path, values: dict[str, Any]) -> None:
    """Recover an orphan only if fresh computation reproduces every stored array exactly."""
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(values) or any(
                stored[k].dtype != v.dtype
                or not np.array_equal(stored[k], v, equal_nan=np.issubdtype(v.dtype, np.inexact))
                for k, v in values.items()
            ):
                raise ValueError(f"Orphan contains different feature arrays: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resume_binding(old: dict, current: dict) -> dict:
    """Keep original keys when only unrelated sources have been added to the checkout."""
    if {k: v for k, v in old.items() if k != "sources"} != {
        k: v for k, v in current.items() if k != "sources"
    }:
        raise ValueError("Resume candidate, data or policy changed")
    if any(current["sources"].get(name) != digest for name, digest in old["sources"].items()):
        raise ValueError("Executed source changed; use a new output and explicit migration")
    return old


def migrate_shards(old: Path, new: Path, root: Path) -> dict:
    """Reuse v1 shards after orchestration-only changes, recording both source bindings."""

    def read(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    before, after = read(old / "BINDING.json"), read(new / "BINDING.json")
    for name in ("candidate", "manifest_sha256", "role"):
        if before[name] != after[name]:
            raise ValueError(f"Migration {name} differs")
    # Only the extraction orchestration changes. All model, sensor, ROI, target and
    # representation source bytes used in the old run must still match.
    for name, digest in before["sources"].items():
        if name != "scripts/prepare_phase_adapt.py" and file_sha(root / name) != digest:
            raise ValueError(f"Migration scientific source changed: {name}")
    population = read(old / "POPULATION.json")
    if population != read(new / "POPULATION.json"):
        raise ValueError("Migration population changed")
    migrated = []
    for row in population["rows"]:
        old_key = canonical_sha({"row": row, "binding": before})
        old_receipt = old / "shards" / f"{old_key}.json"
        if not old_receipt.exists():
            continue
        receipt = read(old_receipt)
        if receipt["key"] != old_key or receipt["query_id"] != row["query_id"]:
            raise ValueError("Old shard identity differs")
        new_key = canonical_sha({"row": row, "binding": after})
        if receipt["feature_sha256"] is not None:
            source = old_receipt.with_suffix(".npz")
            if file_sha(source) != receipt["feature_sha256"]:
                raise ValueError("Old shard hash differs")
            destination = new / "shards" / f"{new_key}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if file_sha(destination) != file_sha(source):
                    raise ValueError("Migration destination differs")
            else:
                shutil.copyfile(source, destination)
        changed = dict(
            receipt,
            key=new_key,
            migrated_from={
                "binding_sha256": file_sha(old / "BINDING.json"),
                "receipt_sha256": file_sha(old_receipt),
                "old_key": old_key,
            },
        )
        atomic_receipt(new / "shards" / f"{new_key}.json", changed)
        migrated.append(
            {"old_key": old_key, "new_key": new_key, "feature_sha256": receipt["feature_sha256"]}
        )
    report = {
        "old_binding_sha256": file_sha(old / "BINDING.json"),
        "new_binding_sha256": file_sha(new / "BINDING.json"),
        "shards": migrated,
        "reason": "resume fixes and label prefilter; sensor forward unchanged",
    }
    atomic_receipt(new / "MIGRATION.json", report)
    return report
