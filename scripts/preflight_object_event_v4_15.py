#!/usr/bin/env python3
"""Preflight for Object Event TTC v4.15 shared odd projection."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v414-summary", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    args = parser.parse_args()

    required = [
        args.cache_manifest,
        args.v414_summary,
        args.ensemble_train,
        args.ensemble_validation,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoints: dict[int, Path] = {}
    for item in args.v48_checkpoint:
        seed_text, path_text = item.split("=", 1)
        seed = int(seed_text)
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise RuntimeError(f"Incomplete v4.8 checkpoint: {path}")
        checkpoints[seed] = path
    if sorted(checkpoints) != [7, 13, 23]:
        raise RuntimeError("v4.15 expects exact v4.8 seeds 7, 13 and 23")

    v414 = json.loads(args.v414_summary.read_text(encoding="utf-8"))
    if v414.get("artifact_type") != "object_event_v4_14_locked_dual_head_multiseed":
        raise RuntimeError("Unexpected v4.14 artifact")
    if v414.get("status") not in {"locked_multiseed_passed", "locked_multiseed_failed"}:
        raise RuntimeError("v4.14 aggregate is incomplete")
    if not v414.get("scientific_contract", {}).get("no_validation_retuning"):
        raise RuntimeError("v4.14 scientific contract is incomplete")

    rows: dict[str, int] = {}
    for name, path in {
        "train": args.ensemble_train,
        "validation": args.ensemble_validation,
    }.items():
        frame = pd.read_csv(path)
        required_columns = {
            "sequence_id",
            "sample_token",
            "track_id",
            "target_expansion",
            "fused_prediction_expansion",
            "fused_zero_events_expansion",
            "fused_shuffled_mean_expansion",
        }
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        rows[name] = len(frame)

    result = {
        "status": "passed",
        "v414_status": v414.get("status"),
        "rows": rows,
        "v48_checkpoints": {
            seed: {"path": path.resolve().as_posix(), "sha256": _sha256(path)}
            for seed, path in checkpoints.items()
        },
        "scientific_contract": {
            "v414_is_diagnostic_source_not_relabelled": True,
            "three_true_seed_v48_backbones_required": True,
            "threshold_must_be_selected_from_train_oof": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
