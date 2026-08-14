#!/usr/bin/env python
"""Materialize signed per-fold V8-A replay payloads from a train-only V8 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

# ruff: noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v8_cache import (
    ScientificRecoveryV8CacheDataset,  # noqa: E402
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sign_artifact(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not verify_artifact_hash(protocol):
        raise ValueError("autopsy input materialization requires a signed V8 protocol")
    dataset = ScientificRecoveryV8CacheDataset(args.cache_manifest)
    if dataset.manifest.get("raw_materialization") is not True:
        raise ValueError("autopsy inputs refuse fixture-only cache materialization")
    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for index in range(len(dataset)):
        row = dataset[index]
        fold = int(row["outer_fold"])
        if fold not in grouped:
            raise ValueError(f"cache row has invalid outer_fold={fold}")
        grouped[fold].append(row)
    expected = protocol["sample_contract"]["row_count_contract"]["by_outer_fold"]
    for fold, rows in grouped.items():
        if len(rows) != int(expected[str(fold)]):
            raise ValueError(f"cache fold {fold} row count differs from frozen protocol")
        rows.sort(key=lambda row: str(row["sample_token"]))
        events = torch.stack(
            [torch.as_tensor(row["representation"], dtype=torch.float32) for row in rows]
        )
        endpoints = torch.stack(
            [torch.as_tensor(row["endpoint_us"], dtype=torch.int64) for row in rows]
        )
        payload = {
            "events": events,
            "delta_t_s": (endpoints[:, 1:] - endpoints[:, :-1]).to(torch.float32) / 1_000_000.0,
            "target_ttc": torch.tensor([float(row["target_ttc"]) for row in rows]),
            "sample_weight": torch.tensor([float(row["sample_weight"]) for row in rows]),
            "token_id": [str(row["sample_token"]) for row in rows],
            "sequence_id": [str(row["sequence_id"]) for row in rows],
            "track_id": [str(row["track_id"]) for row in rows],
            "outer_fold": [fold] * len(rows),
            "seed": [7] * len(rows),
            "endpoint_us": endpoints,
        }
        output = args.output_dir / f"fold{fold}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        _atomic_json(
            args.output_dir / f"fold{fold}.manifest.json",
            {
                "artifact_type": "scientific_recovery_v8_autopsy_replay_input_v1",
                "protocol_artifact_sha256": protocol["artifact_sha256"],
                "cache_manifest_sha256": _sha256(args.cache_manifest),
                "input_sha256": _sha256(output),
                "outer_fold": fold,
                "rows": len(rows),
                "sealed_splits_opened": False,
            },
        )
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
