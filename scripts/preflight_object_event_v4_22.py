#!/usr/bin/env python3
"""Preflight for v4.22 object-centric geometry partial-unfreeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def _object_centric_audit(frame: pd.DataFrame, *, split: str) -> dict[str, Any]:
    required = {"sequence_id", "sample_token", "track_id", "target_expansion", "fused_prediction_expansion"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{split} ensemble missing columns: {missing}")
    if frame.empty:
        raise RuntimeError(f"{split} ensemble is empty")
    for column in ("sequence_id", "sample_token", "track_id"):
        if frame[column].isna().any() or (frame[column].astype(str).str.len() == 0).any():
            raise RuntimeError(f"{split} has empty {column}")
    key = ["sequence_id", "sample_token", "track_id"]
    duplicate_rows = int(frame.duplicated(key).sum())
    if duplicate_rows:
        raise RuntimeError(f"{split} has {duplicate_rows} duplicate object rows for {key}")

    sequence = frame["sequence_id"].astype(str)
    sample = frame["sample_token"].astype(str)
    track = frame["track_id"].astype(str)
    starts_with_track = pd.Series(
        [sample_value.startswith(track_value + "_") for sample_value, track_value in zip(sample, track, strict=True)],
        index=frame.index,
        dtype=bool,
    )
    if not bool(starts_with_track.all()):
        raise RuntimeError(
            f"{split} sample_token/track_id contract broken: "
            f"{int((~starts_with_track).sum())} rows do not encode the object track"
        )

    # Current object cache token contract is <track_id>_<timestamp>.  Grouping by
    # the final timestamp component lets us verify that different objects from
    # the same source frame survive as separate rows instead of sharing one TTC.
    timestamp = sample.str.rsplit("_", n=1).str[-1]
    frame_key = sequence + "::" + timestamp
    objects_per_frame = (
        pd.DataFrame({"frame_key": frame_key, "track_id": track})
        .groupby("frame_key", sort=False)["track_id"]
        .nunique()
    )
    multi_object_frames = int((objects_per_frame > 1).sum())
    if multi_object_frames == 0:
        raise RuntimeError(
            f"{split} contains no multi-object frame keys; refuse encoder adaptation until object-centric cache alignment is verified"
        )
    return {
        "rows": int(len(frame)),
        "sequence_count": int(sequence.nunique()),
        "track_count": int(track.nunique()),
        "frame_key_count": int(len(objects_per_frame)),
        "multi_object_frame_keys": multi_object_frames,
        "maximum_objects_per_frame_key": int(objects_per_frame.max()),
        "mean_objects_per_frame_key": float(objects_per_frame.mean()),
        "duplicate_object_rows": duplicate_rows,
        "sample_token_starts_with_track_id_fraction": float(starts_with_track.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v419-summary", type=Path, required=True)
    parser.add_argument("--v420-summary", type=Path, required=True)
    parser.add_argument("--v421-summary", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    args = parser.parse_args()

    for path in (
        args.cache_manifest,
        args.v419_summary,
        args.v420_summary,
        args.v421_summary,
        args.ensemble_train,
        args.ensemble_validation,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    v419 = json.loads(args.v419_summary.read_text(encoding="utf-8"))
    if v419.get("artifact_type") != "object_event_v4_19_dense_correspondence_probe" or v419.get("status") != "completed":
        raise RuntimeError("v4.19 representation probe is not complete")
    if v419.get("decision", {}).get("recommendation") != "dense_correspondence_supported_train_box_pseudoflow_decoder":
        raise RuntimeError("v4.19 did not support dense correspondence")

    v420 = json.loads(args.v420_summary.read_text(encoding="utf-8"))
    if v420.get("artifact_type") != "object_event_v4_20_box_pseudoflow_decoder" or v420.get("status") != "completed":
        raise RuntimeError("v4.20 pseudoflow decoder is not complete")
    v420_recommendation = v420.get("decision", {}).get("recommendation")
    if v420_recommendation != "frozen_refiner_insufficient_move_pseudoflow_divergence_supervision_into_encoder":
        raise RuntimeError(f"v4.22 is not the contracted next experiment after v4.20: {v420_recommendation}")

    v421 = json.loads(args.v421_summary.read_text(encoding="utf-8"))
    if v421.get("artifact_type") != "object_event_v4_21_box_pseudoflow_target_audit" or v421.get("status") != "completed":
        raise RuntimeError("v4.21 box-pseudoflow target audit is not complete")
    v421_recommendation = v421.get("decision", {}).get("recommendation")
    if v421_recommendation != "box_pseudoflow_supervision_supported_proceed_partial_unfreeze":
        raise RuntimeError(f"v4.21 did not support partial unfreeze: {v421_recommendation}")
    v421_contract = v421.get("scientific_contract", {})
    for key in (
        "no_model_trained",
        "box_pseudoflow_target_audited_before_encoder_unfreeze",
        "orientation_fit_on_train_only",
        "validation_boxes_used_only_for_oracle_diagnostic",
        "boxes_not_forward_features",
        "official_eap_test_not_opened",
        "evttc_not_opened",
    ):
        if v421_contract.get(key) is not True:
            raise RuntimeError(f"v4.21 contract missing: {key}")

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
        raise RuntimeError("v4.22 requires true v4.8 seeds 7,13,23")

    object_audit: dict[str, Any] = {}
    for split, path in (("train", args.ensemble_train), ("validation", args.ensemble_validation)):
        object_audit[split] = _object_centric_audit(pd.read_csv(path), split=split)

    print(json.dumps({
        "status": "passed",
        "v420_recommendation": v420_recommendation,
        "v421_recommendation": v421_recommendation,
        "object_centric_audit": object_audit,
        "scientific_contract": {
            "one_cache_row_is_one_object_track_sample": True,
            "multi_object_source_frames_survive_as_distinct_rows": True,
            "partial_unfreeze_is_now_supported_by_oracle_target_audit": True,
            "boxes_remain_train_only_targets": True,
            "ttc_labels_excluded_from_encoder_loss": True,
            "three_true_seed_v48_checkpoints_required": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
