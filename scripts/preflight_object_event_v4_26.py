#!/usr/bin/env python3
"""Preflight for leak-free Object Event TTC v4.26 OOF residual stacking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _check_seeded_checkpoints(values: list[str], *, artifact_type: str, arm: str | None = None) -> list[int]:
    seen: list[int] = []
    for item in values:
        seed_text, path_text = item.split("=", 1)
        seed = int(seed_text)
        path = Path(path_text)
        payload = torch.load(path, map_location="cpu")
        if payload.get("artifact_type") != artifact_type:
            raise RuntimeError(f"bad artifact type for {path}: {payload.get('artifact_type')!r}")
        if int(payload.get("seed")) != seed:
            raise RuntimeError(f"seed mismatch for {path}")
        if arm is not None and payload.get("arm") != arm:
            raise RuntimeError(f"arm mismatch for {path}: {payload.get('arm')!r}")
        seen.append(seed)
    if sorted(seen) != [7, 13, 23]:
        raise RuntimeError("exact seeds 7,13,23 required")
    return sorted(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v424-summary", type=Path, required=True)
    parser.add_argument("--v425-summary", type=Path, required=True)
    parser.add_argument("--champion-checkpoint", action="append", required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
    args = parser.parse_args()

    v424 = json.loads(args.v424_summary.read_text(encoding="utf-8"))
    if v424.get("status") != "completed" or v424.get("champion") != "geometry_only_regularized":
        raise RuntimeError("v4.24 geometry_only_regularized champion is required")
    contract = v424.get("scientific_contract", {})
    for key in ("official_eap_test_not_opened", "evttc_not_opened"):
        if contract.get(key) is not True:
            raise RuntimeError(f"v4.24 contract missing {key}")

    v425 = json.loads(args.v425_summary.read_text(encoding="utf-8"))
    if v425.get("status") != "completed" or v425.get("selected_readout") != "baseline_control":
        raise RuntimeError("v4.26 expects the completed v4.25 baseline_control outcome")
    old_contract = v425.get("scientific_contract", {})
    for key in ("official_eap_test_not_opened", "evttc_not_opened"):
        if old_contract.get(key) is not True:
            raise RuntimeError(f"v4.25 contract missing {key}")

    champions = _check_seeded_checkpoints(
        args.champion_checkpoint,
        artifact_type="object_event_v4_24_orchestrated_champion",
        arm="geometry_only_regularized",
    )
    adapted = _check_seeded_checkpoints(
        args.adapted_checkpoint,
        artifact_type="object_event_v4_22_adapted_v48",
    )
    print(json.dumps({
        "status": "passed",
        "v424_champion": "geometry_only_regularized",
        "v425_selected_readout": "baseline_control",
        "champion_seeds": champions,
        "adapted_seeds": adapted,
        "scientific_contract": {
            "oof_anchor_and_geometry_same_model_family": True,
            "v410_in_sample_train_anchor_forbidden_in_selection": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
