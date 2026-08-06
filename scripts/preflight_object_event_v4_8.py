#!/usr/bin/env python3
"""Fail-closed preflight for Object Event TTC v4.8."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REQUIRED_BASE = "488b433857090e525c447bf2974ac72639f25194"
CRITICAL_HASHES = {
    "src/e_jepa_ttc/models/object_event_v4_7.py": "de91bf32f181af3b1f6d32087f7b31309c222a86ae5a8954222cd50023b57477",
    "src/e_jepa_ttc/training/object_event_v4_7.py": "530e7bca7837455c1c18ef927fe38155c922bf6007527702a3370109079cab59",
    "scripts/train_e_jepa_object_event_v4_7.py": "87d5924b90d43da4fbe7ab7e13398dab5438f558766ae274a138cb9ab1b9fbaf",
    "scripts/run_object_event_v4_7_highres_extent.ps1": "07acfc6fcb6c494b61744a8ff994fd0d525bd646cebd50c3256c211fc0e3eb0f",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v47-summary", type=Path, required=True)
    parser.add_argument("--v47-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_BASE, head],
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RuntimeError(f"HEAD {head} does not descend from required base {REQUIRED_BASE}")
    actual_hashes: dict[str, str] = {}
    for relative, expected in CRITICAL_HASHES.items():
        path = Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        actual_hashes[relative] = actual
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {relative}: expected {expected}, got {actual}")
    for path in (args.cache_manifest, args.v47_summary, args.v47_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(args.v47_summary.read_text(encoding="utf-8"))
    if summary.get("artifact_type") != "object_event_v4_7_highres_foreground_extent":
        raise RuntimeError("Unexpected v4.7 summary artifact type")
    validation = summary.get("validation_metrics", {})
    geometry = validation.get("geometry", {})
    mask_iou = float(geometry.get("foreground_soft_iou", 0.0))
    height_pearson = float(geometry.get("height_log_eta_pearson", 0.0))
    if mask_iou < 0.70:
        raise RuntimeError(f"v4.7 foreground IoU is too low for v4.8: {mask_iou}")
    result = {
        "status": "passed",
        "head": head,
        "required_base_ancestor": REQUIRED_BASE,
        "critical_v47_hashes": actual_hashes,
        "cache_manifest": args.cache_manifest.resolve().as_posix(),
        "v47_summary": args.v47_summary.resolve().as_posix(),
        "v47_checkpoint": args.v47_checkpoint.resolve().as_posix(),
        "v47_foreground_soft_iou": mask_iou,
        "v47_height_log_eta_pearson": height_pearson,
        "scientific_contract": {
            "event_tensor_is_only_inference_input": True,
            "boxes_are_supervision_only": True,
            "visible_heights_are_supervision_only": True,
            "foreground_is_frozen": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
