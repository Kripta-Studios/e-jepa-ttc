#!/usr/bin/env python
"""Fail-closed preflight for the exact v3 patch base and cache contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASE_COMMIT = "e3ecf32b23d526c024eec4d704c97e78d3c88bb8"
EXPECTED_CORE_HASHES = {
    "src/e_jepa_ttc/models/object_expansion.py": (
        "cdb285a7d31076b9675d11ecc3b44bc97381395cef251acc9786a2c466bcd4ed"
    ),
    "src/e_jepa_ttc/training/object_expansion.py": (
        "96723ba4638ca3d97f4c51b4961b0bda66d5c8bea95e6f15848d1e0e2a59b9bc"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "garl_object_lhr_screen_v2" / "manifest.json",
    )
    parser.add_argument("--allow-descendant", action="store_true")
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != EXPECTED_BASE_COMMIT and not args.allow_descendant:
        raise RuntimeError(
            f"Expected exact patch base {EXPECTED_BASE_COMMIT}, got {head}. "
            "Use --allow-descendant only after git apply --check succeeds."
        )
    hashes = {}
    for relative, expected in EXPECTED_CORE_HASHES.items():
        path = ROOT / relative
        actual = _sha256(path)
        hashes[relative] = actual
        if actual != expected:
            raise RuntimeError(f"Core file drift for {relative}: {actual} != {expected}")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    required_inputs = {
        "jepa_event_roi",
        "garl_delta_t_s",
        "observable_motion",
        "jepa_context_motion",
        "jepa_pair_valid",
        "precontext_motion_valid",
    }
    missing = sorted(required_inputs - set(manifest.get("model_input_fields", [])))
    if missing:
        raise RuntimeError(f"Cache lacks v3 observable inputs: {missing}")
    if manifest.get("uses_official_garl_ttc_labels") is not True:
        raise RuntimeError("Cache does not declare official GarlTTC labels")
    report = {
        "status": "passed",
        "head": head,
        "expected_base": EXPECTED_BASE_COMMIT,
        "core_hashes": hashes,
        "manifest": args.manifest.resolve().as_posix(),
        "split_counts": manifest.get("split_counts"),
        "object_lhr_extension": manifest.get("object_lhr_extension"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
