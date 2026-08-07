#!/usr/bin/env python3
"""Preflight for Object Event TTC v4.19 dense correspondence/divergence probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v418-summary", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    args = parser.parse_args()

    for path in (args.cache_manifest, args.v418_summary, args.ensemble_train, args.ensemble_validation):
        if not path.is_file():
            raise FileNotFoundError(path)

    v418 = json.loads(args.v418_summary.read_text(encoding="utf-8"))
    if v418.get("artifact_type") != "object_event_v4_18_radial_physics_bottleneck":
        raise RuntimeError("Unexpected v4.18 artifact")
    if v418.get("status") != "completed":
        raise RuntimeError("v4.18 decision record is incomplete")
    recommendation = v418.get("decision", {}).get("recommendation")
    if recommendation != "foreground_geometry_insufficient_use_dense_flow_divergence_supervision":
        raise RuntimeError(f"v4.19 is not the contracted next experiment: {recommendation}")
    contract = v418.get("scientific_contract", {})
    for key in ("three_frozen_true_seed_v48_backbones", "official_eap_test_not_opened", "evttc_not_opened"):
        if contract.get(key) is not True:
            raise RuntimeError(f"v4.18 contract missing: {key}")

    seeds: dict[int, Path] = {}
    for value in args.v48_checkpoint:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise RuntimeError(f"Incomplete v4.8 checkpoint: {path}")
        seeds[seed] = path
    if sorted(seeds) != [7, 13, 23]:
        raise RuntimeError("v4.19 requires true v4.8 seeds 7,13,23")

    required = {"sequence_id", "sample_token", "track_id", "target_expansion", "fused_prediction_expansion"}
    rows: dict[str, int] = {}
    for split, path in (("train", args.ensemble_train), ("validation", args.ensemble_validation)):
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        rows[split] = len(frame)

    print(json.dumps({
        "status": "passed",
        "v418_recommendation": recommendation,
        "rows": rows,
        "scientific_contract": {
            "no_new_trainable_sign_head": True,
            "dense_feature_correspondence_probe": True,
            "three_true_seed_v48_backbones_required": True,
            "v410_magnitude_frozen_for_sign_isolation": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
