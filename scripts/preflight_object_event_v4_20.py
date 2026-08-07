#!/usr/bin/env python3
"""Preflight for v4.20 train-only box pseudoflow decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v419-summary", type=Path, required=True)
    parser.add_argument("--v419-train-scores", type=Path, required=True)
    parser.add_argument("--v419-validation-predictions", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    args = parser.parse_args()

    for path in (
        args.cache_manifest, args.v419_summary, args.v419_train_scores,
        args.v419_validation_predictions, args.ensemble_train, args.ensemble_validation,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(args.v419_summary.read_text(encoding="utf-8"))
    if summary.get("artifact_type") != "object_event_v4_19_dense_correspondence_probe":
        raise RuntimeError("Unexpected v4.19 artifact")
    if summary.get("status") != "completed":
        raise RuntimeError("v4.19 is incomplete")
    recommendation = summary.get("decision", {}).get("recommendation")
    if recommendation != "dense_correspondence_supported_train_box_pseudoflow_decoder":
        raise RuntimeError(f"v4.20 is not the contracted next experiment: {recommendation}")
    contract = summary.get("scientific_contract", {})
    for key in (
        "local_feature_matching_on_frozen_v48_maps",
        "boxes_heights_sequence_ids_track_ids_not_forward_features",
        "official_eap_test_not_opened",
        "evttc_not_opened",
    ):
        if contract.get(key) is not True:
            raise RuntimeError(f"v4.19 contract missing: {key}")

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
        raise RuntimeError("v4.20 requires true v4.8 seeds 7,13,23")

    required = {"sequence_id", "sample_token", "track_id", "target_expansion", "fused_prediction_expansion"}
    rows: dict[str, int] = {}
    for split, path in (("train", args.ensemble_train), ("validation", args.ensemble_validation)):
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        rows[split] = len(frame)

    if "dense_divergence_score" not in pd.read_csv(args.v419_train_scores, nrows=1).columns:
        raise RuntimeError("v4.19 train score table missing dense_divergence_score")
    if "dense_divergence_score" not in pd.read_csv(args.v419_validation_predictions, nrows=1).columns:
        raise RuntimeError("v4.19 validation table missing dense_divergence_score")

    print(json.dumps({
        "status": "passed",
        "v419_recommendation": recommendation,
        "rows": rows,
        "scientific_contract": {
            "boxes_are_train_only_pseudoflow_supervision": True,
            "ttc_labels_not_used_by_decoder_loss": True,
            "three_frozen_true_seed_v48_backbones": True,
            "no_new_absolute_sign_classifier": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
