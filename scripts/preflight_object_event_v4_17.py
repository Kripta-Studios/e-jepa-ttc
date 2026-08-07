#!/usr/bin/env python3
"""Preflight for Object Event TTC v4.17 signed-anchor temporal sign."""
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
    parser.add_argument("--v416-summary", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    args = parser.parse_args()

    for path in (args.cache_manifest, args.v416_summary, args.ensemble_train, args.ensemble_validation):
        if not path.is_file():
            raise FileNotFoundError(path)

    v416 = json.loads(args.v416_summary.read_text(encoding="utf-8"))
    if v416.get("artifact_type") != "object_event_v4_16_temporal_dual_head":
        raise RuntimeError("Unexpected v4.16 artifact")
    if v416.get("status") not in {"screen_passed", "screen_failed"}:
        raise RuntimeError("v4.16 screen is incomplete")
    contract = v416.get("scientific_contract", {})
    required = (
        "three_frozen_v48_backbones",
        "causal_track_windows",
        "one_exact_odd_temporal_sign_head",
        "validation_not_used_for_epoch_or_hyperparameter_selection",
        "official_eap_test_not_opened",
        "evttc_not_opened",
    )
    missing = [key for key in required if not contract.get(key)]
    if missing:
        raise RuntimeError(f"v4.16 scientific contract incomplete: {missing}")

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
        raise RuntimeError("v4.17 requires true v4.8 seeds 7, 13 and 23")

    rows: dict[str, int] = {}
    required_columns = {
        "sequence_id", "sample_token", "track_id", "delta_t_s", "target_ttc_s",
        "target_expansion", "fused_prediction_expansion",
    }
    for name, path in {"train": args.ensemble_train, "validation": args.ensemble_validation}.items():
        frame = pd.read_csv(path)
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise RuntimeError(f"{path} missing columns: {missing_columns}")
        if frame.duplicated(["sequence_id", "sample_token", "track_id"]).any():
            raise RuntimeError(f"{path} has duplicate identities")
        rows[name] = len(frame)

    result = {
        "status": "passed",
        "v416_status": v416.get("status"),
        "v416_failed_gates": [key for key, value in v416.get("gates", {}).items() if not bool(value)],
        "rows": rows,
        "v48_checkpoints": {
            seed: {"path": path.resolve().as_posix(), "sha256": _sha256(path)}
            for seed, path in checkpoints.items()
        },
        "scientific_contract": {
            "v416_is_diagnostic_source_not_relabelled": True,
            "signed_anchor_is_computed_only_from_frozen_v48_events": True,
            "v410_is_not_a_forward_input": True,
            "track_ids_are_grouping_metadata_only": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
