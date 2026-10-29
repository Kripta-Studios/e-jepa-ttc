#!/usr/bin/env python3
"""Preflight checks for Object Event TTC v4.12."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v410-summary", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.cache_manifest,
        args.v48_checkpoint,
        args.ensemble_train,
        args.ensemble_validation,
        args.v410_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(args.v410_summary.read_text(encoding="utf-8"))
    if summary.get("artifact_type") != "object_event_v4_10_true_seed_fixed_fusion_robustness":
        raise RuntimeError("source summary is not v4.10")
    if summary.get("status") not in {"robust_passed", "robust_failed"}:
        raise RuntimeError("v4.10 source is incomplete")
    contract = summary.get("scientific_contract", {})
    if not contract.get("official_eap_test_not_opened", False):
        raise RuntimeError("v4.10 official-test contract is missing")
    if not contract.get("evttc_not_opened", False):
        raise RuntimeError("v4.10 EvTTC contract is missing")

    payload = torch.load(args.v48_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError("invalid v4.8 checkpoint")

    required = {
        "sequence_id",
        "sample_token",
        "track_id",
        "delta_t_s",
        "target_ttc_s",
        "target_expansion",
        "fused_prediction_expansion",
        "fused_zero_events_expansion",
        "fused_shuffled_mean_expansion",
    }
    rows = {}
    for name, path in (
        ("train", args.ensemble_train),
        ("validation", args.ensemble_validation),
    ):
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{name} ensemble missing columns: {missing}")
        if frame.duplicated(["sequence_id", "sample_token", "track_id"]).any():
            raise RuntimeError(f"{name} ensemble contains duplicate identities")
        rows[name] = len(frame)

    result = {
        "status": "passed",
        "cache_manifest": str(args.cache_manifest.resolve()),
        "cache_manifest_sha256": _sha256(args.cache_manifest),
        "v48_checkpoint": str(args.v48_checkpoint.resolve()),
        "v48_checkpoint_sha256": _sha256(args.v48_checkpoint),
        "v48_checkpoint_epoch": payload.get("epoch"),
        "v410_status": summary.get("status"),
        "rows": rows,
        "scientific_contract": {
            "v410_failed_result_is_used_as_diagnostic_source_not_relabelled": True,
            "v48_backbone_checkpoint_is_complete": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
