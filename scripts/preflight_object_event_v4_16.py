#!/usr/bin/env python3
"""Preflight for Object Event TTC v4.16 temporal dual head."""
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
    parser.add_argument("--v415-summary", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    args = parser.parse_args()

    for path in (
        args.cache_manifest,
        args.v415_summary,
        args.ensemble_train,
        args.ensemble_validation,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoints: dict[int, Path] = {}
    for value in args.v48_checkpoint:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise RuntimeError(f"Incomplete v4.8 checkpoint: {path}")
        checkpoints[seed] = path
    if sorted(checkpoints) != [7, 13, 23]:
        raise RuntimeError("v4.16 requires true v4.8 seeds 7, 13 and 23")

    v415 = json.loads(args.v415_summary.read_text(encoding="utf-8"))
    if v415.get("artifact_type") != "object_event_v4_15_shared_odd_projection":
        raise RuntimeError("Unexpected v4.15 artifact")
    if v415.get("status") not in {"screen_passed", "screen_failed"}:
        raise RuntimeError("v4.15 screen is incomplete")
    contract = v415.get("scientific_contract", {})
    required_contract = (
        "three_frozen_v48_backbones",
        "one_shared_odd_sign_head",
        "validation_not_used_for_threshold_or_checkpoint_selection",
        "no_new_near_zero_cancellation",
        "official_eap_test_not_opened",
        "evttc_not_opened",
    )
    missing_contract = [key for key in required_contract if not contract.get(key)]
    if missing_contract:
        raise RuntimeError(f"v4.15 scientific contract incomplete: {missing_contract}")

    rows: dict[str, int] = {}
    required_columns = {
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
    for name, path in {
        "train": args.ensemble_train,
        "validation": args.ensemble_validation,
    }.items():
        frame = pd.read_csv(path)
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        if frame.duplicated(["sequence_id", "sample_token", "track_id"]).any():
            raise RuntimeError(f"{path} has duplicate identities")
        rows[name] = len(frame)

    result = {
        "status": "passed",
        "v415_status": v415.get("status"),
        "v415_failed_gates": [
            key for key, value in v415.get("gates", {}).items() if not bool(value)
        ],
        "rows": rows,
        "v48_checkpoints": {
            seed: {"path": path.resolve().as_posix(), "sha256": _sha256(path)}
            for seed, path in checkpoints.items()
        },
        "scientific_contract": {
            "v415_is_diagnostic_source_not_relabelled": True,
            "v410_teacher_is_train_loss_only_not_forward_input": True,
            "track_ids_are_grouping_metadata_only": True,
            "three_true_seed_v48_backbones_required": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
