#!/usr/bin/env python
"""Inventory A5 8k/16k scaling prerequisites without opening dataset rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _inspect(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path.relative_to(ROOT))}
    result: dict[str, Any] = {
        "exists": True,
        "path": str(path.relative_to(ROOT)),
        "file_sha256": _sha(path),
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            result["artifact_sha256"] = payload.get("artifact_sha256")
            result["split_counts"] = payload.get("split_counts")
            result["scope"] = payload.get("scope")
    except Exception:
        result["json_readable"] = False
    return result


def run(output: Path) -> dict[str, Any]:
    validation = ROOT / "artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json"
    train8 = ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
    dino8 = ROOT / "artifacts/cache/dinov3_convnext_large_relational_a4_train8192_rgb_v1/manifest.json"
    train16_candidates = [
        ROOT / "artifacts/cache/garl_object_event_common_roi_train16384_v1/manifest.json",
        ROOT / "artifacts/cache/garl_object_event_common_roi_train16k_v1/manifest.json",
    ]
    dino16_candidates = [
        ROOT / "artifacts/cache/dinov3_convnext_large_relational_a4_train16384_rgb_v1/manifest.json",
        ROOT / "artifacts/cache/dinov3_convnext_large_relational_a4_train16k_rgb_v1/manifest.json",
    ]
    train16 = next((p for p in train16_candidates if p.is_file()), train16_candidates[0])
    dino16 = next((p for p in dino16_candidates if p.is_file()), dino16_candidates[0])
    inspected = {
        "frozen_validation_2048": _inspect(validation),
        "train_8192": _inspect(train8),
        "dino_8192": _inspect(dino8),
        "train_16384": _inspect(train16),
        "dino_16384": _inspect(dino16),
    }
    payload = {
        "artifact_type": "a5_scale_readiness_v1",
        "manifests": inspected,
        "ready_8192": all(inspected[key]["exists"] for key in ("frozen_validation_2048", "train_8192", "dino_8192")),
        "ready_16384": all(inspected[key]["exists"] for key in ("frozen_validation_2048", "train_16384", "dino_16384")),
        "dataset_rows_opened": False,
        "validation_rows_opened": False,
        "rule": "Do not scale until A5 seed replication supports the transport mechanism; freeze manifest hashes in a new config before training.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.output.resolve())
    print(json.dumps({"ready_8192": payload["ready_8192"], "ready_16384": payload["ready_16384"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
