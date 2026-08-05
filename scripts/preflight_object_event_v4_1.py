#!/usr/bin/env python3
"""Fail-closed preflight for the Object Event v4.1 diagnostic patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASE = "e16e6e76c5367dc5a8ad916fc255cd9a1e9ff0a0"
EXPECTED_HASHES = {
    "src/e_jepa_ttc/data/garlttc_lhr_cache.py": "5cd40940bccc8986dd61744882665c60292751587fc34dfb301a12a6e4d2310b",
    "src/e_jepa_ttc/data/object_event_v4.py": "8b20a0ed5b799641ed910f113d4f9b3bde4d1322053b5942e42f7173ad4f84cb",
    "src/e_jepa_ttc/models/object_event_v4.py": "6560673c38deb76f6f7fa26fd022c110a2e88aca6278e11b224835cbe25c5260",
    "scripts/train_e_jepa_object_event_v4.py": "25958e577b2b94070d58c699e3a44bbce4966f59f49d5f4cf05649c4f11048b3",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path)
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != EXPECTED_BASE:
        raise RuntimeError(f"Expected HEAD {EXPECTED_BASE}, found {head}")
    observed = {}
    for relative, expected in EXPECTED_HASHES.items():
        value = sha256(ROOT / relative)
        observed[relative] = value
        if value != expected:
            raise RuntimeError(f"Base file hash mismatch for {relative}: {value} != {expected}")
    manifest_report = None
    if args.cache_manifest:
        manifest_path = args.cache_manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extension = manifest.get("object_lhr_extension", {})
        checks = {
            "train_count": int(manifest.get("split_counts", {}).get("train", 0)) >= 64,
            "validation_count": int(manifest.get("split_counts", {}).get("validation", 0)) >= 256,
            "precontext": float(manifest.get("event_v4_precontext_valid_fraction", 0.0)) >= 0.80,
            "shape": extension.get("event_v4_common_roi_shape", [])[:2] == [3, 12],
            "common_coordinates": extension.get("event_v4_independent_endpoint_resize") is False,
            "preserved_scale": extension.get("event_v4_preserves_absolute_scale_inside_common_roi") is True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Cache contract failed: {checks}")
        manifest_report = {"path": manifest_path.as_posix(), "checks": checks}
    print(
        json.dumps(
            {
                "status": "passed",
                "head": head,
                "expected_base": EXPECTED_BASE,
                "base_hashes": observed,
                "cache": manifest_report,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
