"""Audit public-TRAIN TTC alignment for detector/PHASE development diagnostics.

Only public train metadata and labels are read.  Private test labels, event
payloads, RGB pixels, model inference and optimizer state are out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

JOIN_KEYS = ("sequence_id", "sample_token", "track_id", "public_track_id", "timestamp_us")


def sha256_file(path: Path) -> str:
    """Hash a metadata/evidence source incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(path: Path) -> dict[str, Any]:
    """Describe one admitted source without copying its content."""
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_sha256(rows: list[list[Any]]) -> str:
    """Hash sorted small identity records."""
    encoded = "\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in sorted(rows)
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    """Return an admission receipt for the currently replayed detector frames."""
    paths = {
        "audit_script": Path(__file__),
        "garl_train_inputs": args.garl_root / "data/train.parquet",
        "garl_train_annotations": args.garl_root / "annotations/train.parquet",
        "eap_train_media": args.eap_root / "data/train.parquet",
        "detector_predictions": args.predictions,
        "phase_replay": args.replay,
        "model_seal": args.candidate_root / "MODEL_SEAL.json",
        "deployment_train40_scope": args.deployment_scope,
        "historical_evttc_transfer": args.evttc_transfer,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions_payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = predictions_payload.get("predictions", predictions_payload.get("results"))
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("Detector artifact needs a nonempty predictions/results mapping")
    detector_tokens = list(predictions)

    eap_rows = pq.read_table(
        paths["eap_train_media"],
        columns=[
            "sample_token",
            "split",
            "sequence_id",
            "rgb_member_path",
            "rgb_exposure_start_timestamp_us",
            "rgb_exposure_end_timestamp_us",
            "labels_path",
        ],
        filters=[("sample_token", "in", detector_tokens)],
        use_threads=False,
    ).to_pylist()
    if len(eap_rows) != len(detector_tokens) or {row["split"] for row in eap_rows} != {"train"}:
        raise ValueError("Every detector token must resolve uniquely to public eAP train metadata")
    eap_by_member = {
        (str(row["sequence_id"]), str(row["rgb_member_path"])): row for row in eap_rows
    }
    if len(eap_by_member) != len(eap_rows):
        raise ValueError("Detector media identities are not unique")
    sequences = sorted({str(row["sequence_id"]) for row in eap_rows})

    input_columns = [*JOIN_KEYS, "frame_timestamps_us", "rgb_member_paths", "boxes_xyxy"]
    garl_inputs = pq.read_table(
        paths["garl_train_inputs"],
        columns=input_columns,
        filters=[("sequence_id", "in", sequences)],
        use_threads=False,
    ).to_pylist()
    references = []
    matched_queries: dict[str, dict[str, Any]] = {}
    for query in garl_inputs:
        members = list(query["rgb_member_paths"])
        for endpoint, member in enumerate(members):
            media = eap_by_member.get((str(query["sequence_id"]), str(member)))
            if media is None:
                continue
            record = {
                "detector_sample_token": str(media["sample_token"]),
                "garl_query_token": str(query["sample_token"]),
                "garl_public_track_id": str(query["public_track_id"]),
                "endpoint_index": endpoint,
                "endpoint_count": len(members),
                "is_ttc_target_endpoint": endpoint == len(members) - 1,
                "rgb_member_path": str(member),
                "garl_timestamp_us": int(query["frame_timestamps_us"][endpoint]),
                "eap_exposure_start_us": int(media["rgb_exposure_start_timestamp_us"]),
                "eap_exposure_end_us": int(media["rgb_exposure_end_timestamp_us"]),
                "clock_offset_garl_minus_eap_start_us": (
                    int(query["frame_timestamps_us"][endpoint])
                    - int(media["rgb_exposure_start_timestamp_us"])
                ),
            }
            references.append(record)
            matched_queries[str(query["sample_token"])] = query

    query_tokens = sorted(matched_queries)
    annotations = pq.read_table(
        paths["garl_train_annotations"],
        columns=[*JOIN_KEYS, "ttc", "frame_ttc"],
        filters=[("sample_token", "in", query_tokens)],
        use_threads=False,
    ).to_pylist()
    input_keys = {tuple(row[key] for key in JOIN_KEYS) for row in matched_queries.values()}
    annotation_keys = {tuple(row[key] for key in JOIN_KEYS) for row in annotations}
    if input_keys != annotation_keys:
        raise ValueError("Matched Garl input and TTC annotation five-key populations differ")
    annotation_by_token = {str(row["sample_token"]): row for row in annotations}
    frame_targets_agree = sum(
        bool(
            row["frame_ttc"]
            and np.isclose(float(row["ttc"]), float(row["frame_ttc"][-1]), rtol=0, atol=0)
        )
        for row in annotations
    )
    if frame_targets_agree != len(annotations):
        raise ValueError("Garl ttc differs from frame_ttc at the final endpoint")
    target_references = [row for row in references if row["is_ttc_target_endpoint"]]
    finite_targets = sum(
        bool(np.isfinite(float(annotation_by_token[row["garl_query_token"]]["ttc"])))
        for row in target_references
    )

    public_label_rows: list[dict[str, Any]] = []
    label_paths = sorted({str(row["labels_path"]) for row in eap_rows})
    for relative in label_paths:
        path = args.eap_root / relative
        paths[f"eap_train_labels:{relative}"] = path
        table = pq.read_table(
            path,
            columns=["sample_token", "split", "sequence_id", "instance_id", "track_id"],
            filters=[("sample_token", "in", detector_tokens)],
            use_threads=False,
        )
        public_label_rows.extend(table.to_pylist())
    ids_by_token: dict[str, set[str]] = {}
    for row in public_label_rows:
        ids_by_token.setdefault(str(row["sample_token"]), set()).update(
            (str(row["instance_id"]), str(row["track_id"]))
        )
    direct_public_id_matches = sum(
        row["garl_public_track_id"] in ids_by_token.get(row["detector_sample_token"], set())
        for row in references
    )

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    replay_detections = replay.get("result", {}).get("detections", [])
    final_token = str(replay["final_sample_token"])
    final_target_references = [
        row for row in target_references if row["detector_sample_token"] == final_token
    ]
    seal = json.loads(paths["model_seal"].read_text(encoding="utf-8"))
    deployment = json.loads(paths["deployment_train40_scope"].read_text(encoding="utf-8"))

    join_rows = [[row[key] for key in JOIN_KEYS] for row in matched_queries.values()]
    return {
        "schema": "ttc_evaluation_admission_audit_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "DEVELOPMENT_DIAGNOSTIC_ADMISSIBLE_INDEPENDENT_GENERALIZATION_BLOCKED",
        "scope": "public_train_metadata_and_labels_only_no_events_pixels_inference_or_updates",
        "private_test_labels_read": False,
        "sources": {name: source(path) for name, path in paths.items()},
        "alignment_contract": {
            "garl_input_annotation_join_keys": list(JOIN_KEYS),
            "garl_to_eap_frame_keys": ["sequence_id", "rgb_member_path"],
            "timestamp_rule": (
                "Do not equate Garl timestamp_us with the eAP sensor clock; bind the exact "
                "frame by RGB member and retain both clocks plus their measured offset."
            ),
            "ttc_target_time": "last endpoint of each Garl query (frame_ttc[-1]/ttc)",
            "matched_query_join_keys_sha256": canonical_sha256(join_rows),
        },
        "counts": {
            "detector_frames": len(detector_tokens),
            "detector_sequences": len(sequences),
            "garl_rows_in_detector_sequences": len(garl_inputs),
            "garl_queries_referencing_detector_frames": len(matched_queries),
            "garl_endpoint_references_to_detector_frames": len(references),
            "target_endpoint_references": len(target_references),
            "target_endpoint_detector_frames": len(
                {row["detector_sample_token"] for row in target_references}
            ),
            "finite_public_train_ttc_targets": finite_targets,
            "matched_queries_ttc_equals_final_frame_ttc": frame_targets_agree,
            "garl_tracks_in_overlap": len({row["garl_public_track_id"] for row in references}),
            "eap_public_label_rows_on_detector_frames": len(public_label_rows),
            "direct_garl_id_to_eap_label_id_matches": direct_public_id_matches,
            "current_replay_detections": len(replay_detections),
            "current_replay_available_ttc": sum(
                row.get("ttc_status") == "available" for row in replay_detections
            ),
            "current_final_frame_target_endpoint_references": len(final_target_references),
        },
        "current_replay": {
            "final_sample_token": final_token,
            "has_exact_public_train_ttc_at_runtime_query": bool(final_target_references),
            "identity_crosswalk_field_present": False,
            "missing_identity_field": (
                "No released field maps generated detector track_id:generation to "
                "GarlTTC public_track_id; eAP instance_id/track_id is not equal to the "
                "overlapping Garl public_track_id."
            ),
        },
        "producer_exposure": {
            "candidate": seal["candidate"],
            "model_seal_blind_validation_claims": sorted(
                {
                    bool(stage.get("blind_validation_claim"))
                    for stage in seal["training_feature_ancestry"].values()
                }
            ),
            "deployment_status": deployment["status"],
            "train40_rows": int(deployment["event_rows"]["row_count"]),
            "train40_sequences": int(deployment["event_rows"]["sequence_count"]),
            "overlap_sequence_in_train40": all(
                sequence in set(deployment["rgb_union"]["group_ids"]) for sequence in sequences
            ),
            "exact_overlapping_queries_are_training_rows": True,
            "evttc_previously_observed": True,
            "evttc_evidence": (
                "The bound historical transfer report explicitly labels EvTTC as previously "
                "observed; it is descriptive transfer, not virgin independent confirmation."
            ),
        },
        "admission": {
            "existing_train40_development_ttc_diagnostic": True,
            "current_final_replay_direct_scoring": False,
            "independent_generalization": False,
            "reason": (
                "Four detector frames have exact public TRAIN TTC target endpoints, but the "
                "current final replay frame does not. The candidate and every deployment "
                "producer were fit on full TRAIN40, including the overlap sequence/queries."
            ),
        },
        "references": references,
        "limitations": [
            (
                "TTC numeric values are read only to verify finiteness and are not copied "
                "to this receipt."
            ),
            (
                "Direct ID equality is absent; geometric/evaluation matching must never "
                "become an inference oracle."
            ),
            (
                "The current eight-frame replay is too short to query PHASE at all four "
                "earlier target endpoints with a 600 ms track-history gate."
            ),
            "No public Test12 TTC labels exist and no private label source was accessed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garl-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--deployment-scope", type=Path, required=True)
    parser.add_argument("--evttc-transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path")
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
