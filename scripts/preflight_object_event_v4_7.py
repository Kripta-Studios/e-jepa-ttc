#!/usr/bin/env python3
"""Fail-closed preflight for Object Event TTC v4.7."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "488b433857090e525c447bf2974ac72639f25194"
EXPECTED_V46_HASHES = {
    "src/e_jepa_ttc/models/object_event_v4_6.py": "e66010a72ecd58804688e8d509d76b2ca6c468bbc4982232fdb426474a52dac8",
    "src/e_jepa_ttc/training/object_event_v4_6.py": "9c5ad353a1030a8fe30079f0ef2f222aafbc05bbddde13cba7922c57523b09d5",
    "scripts/train_e_jepa_object_event_v4_6.py": "aec5469c0b9c51f6ac7387e801be5e957473846aa16df7ae34dfd9cd4842c1a5",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v46-summary", type=Path, required=True)
    parser.add_argument("--v46-checkpoint", type=Path, required=True)
    parser.add_argument("--v46-validation-predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    head_result = _git("rev-parse", "HEAD")
    if head_result.returncode != 0:
        raise RuntimeError(head_result.stderr.strip() or "git rev-parse HEAD failed")
    head = head_result.stdout.strip()
    ancestor = _git("merge-base", "--is-ancestor", BASE_COMMIT, head)
    if ancestor.returncode != 0:
        raise RuntimeError(f"HEAD {head} does not descend from v4.3 base {BASE_COMMIT}")

    hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_V46_HASHES.items():
        path = ROOT / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        hashes[relative] = actual

    cache_path = args.cache_manifest.resolve()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    model_fields = set(cache.get("model_input_fields", []))
    supervision_fields = set(cache.get("supervision_only_fields", []))
    required_model = {"event_v4_common_roi", "event_v4_boxes_xyxy"}
    required_supervision = {"ttc_s", "garl_visible_heights_px"}
    if missing := sorted(required_model - model_fields):
        raise ValueError(f"Cache lacks v4.7 fields: {missing}")
    if missing := sorted(required_supervision - supervision_fields):
        raise ValueError(f"Cache lacks v4.7 supervision: {missing}")
    extension = cache.get("object_lhr_extension", {})
    if extension.get("uses_boxes_for_model_input") is not False:
        raise ValueError("Cache does not declare boxes excluded from model input")

    summary_path = args.v46_summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("artifact_type") != "object_event_v4_6_learned_foreground_height_ratio":
        raise ValueError(f"Unexpected v4.6 artifact: {summary.get('artifact_type')}")
    validation = summary.get("validation_metrics", {})
    geometry = validation.get("geometry", {})
    gates = summary.get("gates", {})
    mask_iou = float(geometry.get("foreground_soft_iou", 0.0))
    height_pearson = float(geometry.get("height_log_eta_pearson", 0.0))
    if mask_iou < 0.65:
        raise ValueError(f"v4.7 requires learned foreground; v4.6 IoU={mask_iou}")
    if height_pearson >= 0.35:
        raise ValueError(
            "v4.7 is specifically for the learned-mask/failed-height regime; "
            f"v4.6 height Pearson={height_pearson}"
        )
    if gates.get("foreground_learned") is not True:
        raise ValueError("v4.6 foreground gate did not pass")
    if gates.get("height_ratio_learned") is not False:
        raise ValueError("v4.6 height-ratio gate did not fail as expected")

    predictions_path = args.v46_validation_predictions.resolve()
    frame = pd.read_csv(predictions_path)
    required_columns = {"target_expansion", "target_height_log_eta"}
    if missing := sorted(required_columns - set(frame.columns)):
        raise ValueError(f"v4.6 predictions lack columns: {missing}")
    target_expansion = frame["target_expansion"].to_numpy(dtype=np.float64)
    target_log_eta = np.log1p(np.clip(-target_expansion, -0.999999, None))
    height_target = frame["target_height_log_eta"].to_numpy(dtype=np.float64)
    oracle_pearson = _pearson(target_log_eta, height_target)
    oracle_mae = float(np.mean(np.abs(target_log_eta - height_target)))
    if oracle_pearson < 0.98:
        raise ValueError(f"Visible-height target is not aligned with TTC: {oracle_pearson}")

    checkpoint_path = args.v46_checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid v4.6 checkpoint: {checkpoint_path}")
    state = payload["model_state_dict"]
    if not any(str(key).startswith("geometry_encoder.") for key in state):
        raise ValueError("v4.6 checkpoint has no geometry_encoder state")

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    result: dict[str, Any] = {
        "status": "passed",
        "head": head,
        "required_base_ancestor": BASE_COMMIT,
        "critical_v4_6_hashes": hashes,
        "cache_manifest": cache_path.as_posix(),
        "cache_manifest_sha256": _sha256(cache_path),
        "v4_6_summary": summary_path.as_posix(),
        "v4_6_checkpoint": checkpoint_path.as_posix(),
        "v4_6_checkpoint_sha256": _sha256(checkpoint_path),
        "v4_6_foreground_soft_iou": mask_iou,
        "v4_6_height_log_eta_pearson": height_pearson,
        "oracle_height_to_ttc_log_eta_pearson": oracle_pearson,
        "oracle_height_to_ttc_log_eta_mae": oracle_mae,
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
