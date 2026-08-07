#!/usr/bin/env python3
"""Preflight for Object Event TTC v4.23."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _parse(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        result[int(seed_text)] = Path(path_text)
    if sorted(result) != [7, 13, 23]:
        raise RuntimeError("v4.23 requires exact adapted seeds 7,13,23")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v422-summary", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
    args = parser.parse_args()

    if not args.cache_manifest.exists():
        raise FileNotFoundError(args.cache_manifest)
    summary = json.loads(args.v422_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise RuntimeError("v4.22 did not complete")
    if summary.get("decision", {}).get("recommendation") != "partial_unfreeze_supported_integrate_geometry_auxiliary_with_ttc":
        raise RuntimeError("v4.22 did not support joint geometry+TTC integration")
    contract = summary.get("scientific_contract", {})
    required = (
        "three_true_seed_v48_backbones_adapted_independently",
        "only_geometry_encoder_tail_parameters_trainable",
        "boxes_are_train_only_pseudoflow_targets",
        "vertical_height_ratio_is_train_only_geometry_supervision",
        "fixed_epoch_schedule_no_validation_selection",
        "official_eap_test_not_opened",
        "evttc_not_opened",
    )
    missing = [key for key in required if contract.get(key) is not True]
    if missing:
        raise RuntimeError(f"v4.22 scientific contract missing: {missing}")

    checkpoints = _parse(args.adapted_checkpoint)
    records = {int(record["seed"]): record for record in summary.get("seed_records", [])}
    for seed, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("artifact_type") != "object_event_v4_22_adapted_v48" or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"invalid v4.22 adapted checkpoint for seed {seed}: {path}")
        if seed not in records:
            raise RuntimeError(f"v4.22 summary lacks seed record {seed}")

    print(json.dumps({
        "status": "passed",
        "v422_recommendation": summary["decision"]["recommendation"],
        "adapted_seeds": sorted(checkpoints),
        "scientific_contract": {
            "start_from_v422_geometry_adapted_seeds": True,
            "ttc_labels_train_only": True,
            "geometry_auxiliary_retained": True,
            "no_validation_model_selection": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
