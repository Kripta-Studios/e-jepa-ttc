"""Verify complete TRAIN40 input/teacher fragments and seal their exact lineage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from operational.efficient_context.common import digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def check_fragment(
    path: Path, receipt: dict, ordinals: np.ndarray, tokens: np.ndarray, freeze_sha: str
) -> None:
    """Verify bytes, recipe and object identity; a count alone cannot admit a shard."""
    if (
        receipt["status"] != "VERIFIED"
        or receipt["rows"] != len(ordinals)
        or receipt["freeze_sha256"] != freeze_sha
        or digest(path) != receipt["sha256"]
    ):
        raise ValueError(f"Invalid TRAIN40 fragment: {path}")
    with np.load(path, allow_pickle=False) as stored:
        if not np.array_equal(stored["ordinals"], ordinals):
            raise ValueError("Fragment ordinals differ from complete population")
        if not np.array_equal(stored["tokens"], tokens):
            raise ValueError("Fragment object identities differ")
        if "relation_targets" in stored.files:
            target, valid = stored["relation_targets"], stored["relation_valid"]
            if target.shape != (len(ordinals), 2, 6, 32, 32):
                raise ValueError("Teacher shape changed")
            if valid.shape != target.shape or valid.dtype != np.bool_:
                raise ValueError("Teacher relation validity changed")
            if target.dtype != np.float16 or not np.isfinite(target).all():
                raise ValueError("Teacher precision or finite-output contract failed")


def run(output: Path) -> None:
    """Hash every shard once and publish manifests only after whole-population admission."""
    for name in ("INPUT_PREPARATION_PROGRESS", "TEACHER_PROGRESS", "MEDIA_VERIFICATION_PROGRESS"):
        if read(output / (name + ".json"))["status"] != "COMPLETE":
            raise ValueError(f"Prerequisite incomplete: {name}")
    audit = read(output / "DATA_AUDIT.json")
    if digest(output / "TRAIN40_INDEX.npz") != audit["index_sha256"]:
        raise ValueError("Audited TRAIN40 index changed")
    if digest(output / "TRAIN40_ROWS.parquet") != audit["rows_sha256"]:
        raise ValueError("Audited TRAIN40 records changed")
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        tokens = stored["tokens"]
    if len(tokens) != 88744:
        raise ValueError("TRAIN40 population changed")
    for kind, freeze_name, directory in (
        ("INPUT", "PREPARE_FREEZE.json", None),
        ("TEACHER", "TEACHER_FREEZE.json", output / "teacher"),
    ):
        freeze_path = output / freeze_name
        freeze = read(freeze_path)
        verify_sources(freeze)
        freeze_sha = digest(freeze_path)
        directory = Path(freeze["cache_root"]) if directory is None else directory
        entries = []
        for start in range(0, len(tokens), 32):
            stop = min(start + 32, len(tokens))
            path = directory / f"shard_{start // 32:05d}.npz"
            receipt = read(path.with_suffix(".json"))
            check_fragment(path, receipt, np.arange(start, stop), tokens[start:stop], freeze_sha)
            entries.append(
                {
                    "name": path.name,
                    "sha256": receipt["sha256"],
                    "receipt_sha256": digest(path.with_suffix(".json")),
                    "start": start,
                    "stop": stop,
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )
        atomic_json(
            output / f"{kind}_MANIFEST.json",
            {
                "schema": "train40_complete_fragment_manifest_v1",
                "status": "COMPLETE_VERIFIED",
                "kind": kind,
                "directory": str(directory),
                "index_sha256": audit["index_sha256"],
                "rows_sha256": audit["rows_sha256"],
                "freeze_sha256": freeze_sha,
                "row_count": len(tokens),
                "sequence_count": 40,
                "ordered_shards": entries,
                "teacher_weights_sha256": freeze.get("ordered_tensor_weights_sha256"),
                "optimizer_updates": 0,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output.resolve())
