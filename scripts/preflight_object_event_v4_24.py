#!/usr/bin/env python3
"""Preflight for the v4.24 train-only schedule orchestrator."""
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
        raise RuntimeError("v4.24 requires exact v4.22 adapted seeds 7,13,23")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v422-summary", type=Path, required=True)
    parser.add_argument("--v423-summary", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
    args = parser.parse_args()

    for path in (args.cache_manifest, args.v422_summary, args.v423_summary):
        if not path.exists():
            raise FileNotFoundError(path)
    v422 = json.loads(args.v422_summary.read_text(encoding="utf-8"))
    v423 = json.loads(args.v423_summary.read_text(encoding="utf-8"))
    if v422.get("status") != "completed" or v423.get("status") != "completed":
        raise RuntimeError("v4.22 and v4.23 must both be completed")
    if v422.get("decision", {}).get("recommendation") != "partial_unfreeze_supported_integrate_geometry_auxiliary_with_ttc":
        raise RuntimeError("v4.22 does not support joint geometry+TTC")
    if v423.get("decision", {}).get("recommendation") != "joint_geometry_ttc_promising_keep_architecture_adjust_train_only_loss_schedule":
        raise RuntimeError("v4.23 does not request train-only schedule search")
    contract = v423.get("scientific_contract", {})
    required = (
        "starts_from_three_independent_v422_adapted_seeds",
        "geometry_tail_and_existing_v48_motion_head_only_trainable",
        "ttc_labels_used_on_train_only",
        "boxes_and_visible_heights_are_train_only_targets_not_forward_features",
        "fixed_epoch_schedule_no_validation_selection",
        "official_eap_test_not_opened",
        "evttc_not_opened",
    )
    missing = [key for key in required if contract.get(key) is not True]
    if missing:
        raise RuntimeError(f"v4.23 scientific contract missing: {missing}")

    checkpoints = _parse(args.adapted_checkpoint)
    for seed, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("artifact_type") != "object_event_v4_22_adapted_v48" or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"invalid v4.22 checkpoint for seed {seed}: {path}")

    print(json.dumps({
        "status": "passed",
        "v423_recommendation": v423["decision"]["recommendation"],
        "seeds": sorted(checkpoints),
        "scientific_contract": {
            "candidate_selection_uses_train_sequences_only": True,
            "development_validation_touched_once_after_champion_selection": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
