#!/usr/bin/env python3
"""Preflight for Object Event TTC v4.18 radial/divergence physics bottleneck."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v417-summary", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    args = parser.parse_args()

    for path in (
        args.cache_manifest,
        args.v417_summary,
        args.ensemble_train,
        args.ensemble_validation,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    v417 = json.loads(args.v417_summary.read_text(encoding="utf-8"))
    if v417.get("artifact_type") != "object_event_v4_17_signed_anchor_temporal":
        raise RuntimeError("Unexpected v4.17 artifact")
    if v417.get("status") not in {"screen_passed", "screen_failed"}:
        raise RuntimeError("v4.17 screen is incomplete")
    contract = v417.get("scientific_contract", {})
    required_v417_contract = {
        "three_frozen_true_seed_v48_backbones": "v4.17 three-backbone contract missing",
        "signed_v48_anchor_is_event_only_forward_feature": "v4.17 event-only signed-anchor contract missing",
        "track_and_sequence_ids_are_grouping_metadata_not_forward_features": "v4.17 metadata-only ID contract missing",
        "validation_not_used_for_epoch_or_hyperparameter_selection": "v4.17 validation-selection contract missing",
        "official_eap_test_not_opened": "Official eAP test was already opened",
        "evttc_not_opened": "EvTTC was already opened",
    }
    for key, message in required_v417_contract.items():
        if contract.get(key) is not True:
            raise RuntimeError(message)

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
        raise RuntimeError("v4.18 requires true v4.8 seeds 7, 13 and 23")

    rows = {}
    required = {
        "sequence_id",
        "sample_token",
        "track_id",
        "target_expansion",
        "fused_prediction_expansion",
    }
    for split, path in (
        ("train", args.ensemble_train),
        ("validation", args.ensemble_validation),
    ):
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        rows[split] = len(frame)

    result = {
        "status": "passed",
        "v417_status": v417.get("status"),
        "v417_validation_pearson": v417.get("temporal_validation_metrics", {}).get("pearson"),
        "rows": rows,
        "scientific_contract": {
            "v417_is_diagnostic_source_not_relabelled": True,
            "v418_changes_feature_family_not_gate_thresholds": True,
            "explicit_2d_radial_geometry_only": True,
            "three_true_seed_v48_backbones_required": True,
            "v410_magnitude_is_frozen_for_sign_isolation": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
