#!/usr/bin/env python3
"""Fail-closed preflight for Object Event v4.3 multiseed repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEAD = "1d11b3ee81dc58ac3b07d6a190bbd52baf72daeb"
EXPECTED_HASHES = {
    "scripts/train_e_jepa_object_event_v4_2.py": "c7cd7498cd490f3dacb4b1f4c478133b585787aed5a95ba556fae68838f928df",
    "src/e_jepa_ttc/models/object_event_v4_2.py": "6f802c15d1a9a33d9f5ae10c4321e9ab6ed43a845627b99bb316c6e89d6d35ad",
    "src/e_jepa_ttc/training/object_event_v4_2.py": "7a805e57b83191e883e9a2726607d4b5df20c36725e3e624885e82bae69ad2d0",
    "configs/experiment/e_jepa_garl_object_event_screen_v4_2.yaml": "fd3d7d3cbaca5242840ee6c8bc5e8b004821f23292940512969813de5b16173c",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git rev-parse failed")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--seed7-summary", type=Path)
    args = parser.parse_args()
    head = _head()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"Expected HEAD {EXPECTED_HEAD}, got {head}")
    actual_hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        actual_hashes[relative] = actual
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {relative}: expected {expected}, got {actual}")
    manifest = json.loads(args.cache_manifest.resolve().read_text(encoding="utf-8"))
    counts = manifest.get("split_counts", {})
    extension = manifest.get("object_lhr_extension", {})
    checks = {
        "train_count": int(counts.get("train", 0)) >= 2048,
        "validation_count": int(counts.get("validation", 0)) >= 2048,
        "precontext": float(manifest.get("event_v4_precontext_valid_fraction", 0.0)) >= 0.8,
        "shape": extension.get("event_v4_common_roi_shape") == [3, 12, 128, 128],
        "no_boxes_model_input": extension.get("uses_boxes_for_model_input") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Cache contract failed: {checks}")
    seed7 = None
    if args.seed7_summary and args.seed7_summary.exists():
        seed7 = json.loads(args.seed7_summary.read_text(encoding="utf-8"))
        if not bool(seed7.get("screen_passed", False)):
            raise RuntimeError("Existing seed-7 v4.2 summary did not pass")
    print(json.dumps({
        "status": "passed",
        "head": head,
        "expected_head": EXPECTED_HEAD,
        "v4_2_hashes": actual_hashes,
        "cache_checks": checks,
        "seed7_reusable": seed7 is not None,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
