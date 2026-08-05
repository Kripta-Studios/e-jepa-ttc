#!/usr/bin/env python3
"""Fail-closed preflight for the v4.2 full event-only screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEAD = "e16e6e76c5367dc5a8ad916fc255cd9a1e9ff0a0"
EXPECTED_V41_HASHES = {
    "src/e_jepa_ttc/models/object_event_v4_1.py": "9250191e8515a0b4c2d18712eeeb1da24ae4ef39ac8028462ac5cfefdfcbbb24",
    "src/e_jepa_ttc/training/object_event_v4_1.py": "1e6a96a58f9fe3b80b6ec4213361df3f8a6a97a0d0cc3d2e1b2e59815f53cf63",
    "scripts/diagnose_object_event_v4_1.py": "c9ede4ff732ee9cfaca6be14d9b685563571589e152ce8371ee275d125564a4e",
    "configs/experiment/e_jepa_garl_object_event_overfit_v4_1.yaml": "87a8065530015bc7adc40e4f010de96d95554e298d98c8c9fa30f478948f2715",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def check_cache(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    counts = manifest.get("split_counts", {})
    extension = manifest.get("object_lhr_extension", {})
    precontext = float(manifest.get("event_v4_precontext_valid_fraction", 0.0))
    checks = {
        "train_count": int(counts.get("train", 0)) >= 2048,
        "validation_count": int(counts.get("validation", 0)) >= 2048,
        "precontext": precontext >= 0.80,
        "shape": extension.get("event_v4_common_roi_shape") == [3, 12, 128, 128],
        "common_coordinates": extension.get("event_v4_independent_endpoint_resize") is False,
        "preserved_scale": extension.get("event_v4_preserves_absolute_scale_inside_common_roi") is True,
        "no_boxes_model_input": extension.get("uses_boxes_for_model_input") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v4 cache contract failed: {checks}")
    return {"path": path.resolve().as_posix(), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument(
        "--v41-summary",
        type=Path,
        default=ROOT / "artifacts/debug/object_event_v4_1_overfit/summary.json",
    )
    args = parser.parse_args()
    head = git_head()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"Expected HEAD {EXPECTED_HEAD}, found {head}")
    hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_V41_HASHES.items():
        path = ROOT / relative
        actual = sha256(path)
        hashes[relative] = actual
        if actual != expected:
            raise RuntimeError(f"Unexpected v4.1 hash for {relative}: {actual}")
    summary = json.loads(args.v41_summary.read_text(encoding="utf-8"))
    if not bool(summary.get("screen_passed")):
        raise RuntimeError("v4.1 screen did not pass; v4.2 is not justified")
    validation = summary.get("validation_metrics", {}).get("branches", {}).get("event", {})
    if float(validation.get("pearson", 0.0)) < 0.20:
        raise RuntimeError("v4.1 validation Pearson is below 0.20")
    result = {
        "status": "passed",
        "head": head,
        "expected_head": EXPECTED_HEAD,
        "v4_1_hashes": hashes,
        "v4_1_summary": {
            "path": args.v41_summary.resolve().as_posix(),
            "screen_passed": True,
            "validation_pearson": validation.get("pearson"),
        },
        "cache": check_cache(args.cache_manifest.resolve()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
