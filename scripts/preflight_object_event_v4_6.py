#!/usr/bin/env python3
"""Fail-closed preflight for Object Event TTC v4.6."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "488b433857090e525c447bf2974ac72639f25194"
EXPECTED_HASHES = {
    "scripts/train_e_jepa_object_event_v4_2.py": "c7cd7498cd490f3dacb4b1f4c478133b585787aed5a95ba556fae68838f928df",
    "src/e_jepa_ttc/models/object_event_v4_2.py": "6f802c15d1a9a33d9f5ae10c4321e9ab6ed43a845627b99bb316c6e89d6d35ad",
    "src/e_jepa_ttc/training/object_event_v4_2.py": "7a805e57b83191e883e9a2726607d4b5df20c36725e3e624885e82bae69ad2d0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False)


def _head() -> str:
    result = _git("rev-parse", "HEAD")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git rev-parse HEAD failed")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v42-checkpoint", type=Path, required=True)
    parser.add_argument("--v45-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    head = _head()
    ancestor = _git("merge-base", "--is-ancestor", BASE_COMMIT, head)
    if ancestor.returncode != 0:
        raise RuntimeError(f"HEAD {head} does not descend from v4.3 base {BASE_COMMIT}")

    hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {relative}: expected {expected}, got {actual}")
        hashes[relative] = actual

    for relative in (
        "src/e_jepa_ttc/object_event_v4_4.py",
        "src/e_jepa_ttc/training/object_event_v4_5.py",
        "scripts/train_e_jepa_object_event_v4_5.py",
    ):
        if not (ROOT / relative).exists():
            raise FileNotFoundError(ROOT / relative)

    cache_path = args.cache_manifest.resolve()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    model_fields = set(cache.get("model_input_fields", []))
    supervision_fields = set(cache.get("supervision_only_fields", []))
    required_model_fields = {"event_v4_common_roi", "event_v4_boxes_xyxy"}
    required_supervision = {"ttc_s", "garl_visible_heights_px"}
    if missing := sorted(required_model_fields - model_fields):
        raise ValueError(f"Cache lacks v4.6 fields: {missing}")
    if missing := sorted(required_supervision - supervision_fields):
        raise ValueError(f"Cache lacks v4.6 supervision: {missing}")
    extension = cache.get("object_lhr_extension", {})
    if extension.get("event_v4_preserves_absolute_scale_inside_common_roi") is not True:
        raise ValueError("v4.6 requires preserved absolute scale")
    if extension.get("uses_boxes_for_model_input") is not False:
        raise ValueError("Cache does not declare boxes excluded from model input")

    checkpoint_path = args.v42_checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid v4.2 checkpoint: {checkpoint_path}")
    checkpoint_seed = int(payload.get("train_config", {}).get("seed", -1))
    if checkpoint_seed != args.seed:
        raise ValueError(f"Checkpoint seed mismatch: expected {args.seed}, got {checkpoint_seed}")

    v45_path = args.v45_summary.resolve()
    v45 = json.loads(v45_path.read_text(encoding="utf-8"))
    if v45.get("artifact_type") != "object_event_v4_5_paired_reciprocal_mid_multiseed":
        raise ValueError(f"Unexpected v4.5 artifact: {v45.get('artifact_type')}")
    ensemble = v45.get("ensemble_metrics", {})
    weighted_mid = ensemble.get("official_eap", {}).get("weighted_mid")
    pearson = ensemble.get("expansion", {}).get("pearson")
    if weighted_mid is None or not 100.0 < float(weighted_mid) < 400.0:
        raise ValueError(f"Invalid v4.5 weighted MiD: {weighted_mid}")
    if pearson is None or not 0.2 < float(pearson) < 0.9:
        raise ValueError(f"Invalid v4.5 Pearson: {pearson}")

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    result: dict[str, Any] = {
        "status": "passed",
        "head": head,
        "required_base_ancestor": BASE_COMMIT,
        "critical_v4_2_hashes": hashes,
        "cache_manifest": cache_path.as_posix(),
        "cache_manifest_sha256": _sha256(cache_path),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_seed": checkpoint_seed,
        "v4_5_summary": v45_path.as_posix(),
        "v4_5_status": v45.get("status"),
        "v4_5_weighted_mid": float(weighted_mid),
        "v4_5_pearson": float(pearson),
        "scientific_contract": {
            "event_tensor_is_only_inference_input": True,
            "boxes_are_supervision_only": True,
            "visible_heights_are_supervision_only": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
