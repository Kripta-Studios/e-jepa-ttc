"""Bounded TRAIN-only 3D-to-2D parity audit, never a production history source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.data.eap import project_box_3d_to_event
from e_jepa_ttc.simplex_t.coordination import verified_ack


def main() -> None:
    """Compare exact identity joins on at most 128 published original-group pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    sequence = sorted(ack["interfaces"]["role_manifest"]["roles"]["original"])[0]
    eap = Path(paths["eap_root"])
    media_columns = [
        "sample_token",
        "sequence_id",
        "rgb_member_path",
        "K_event",
        "T_event_ego",
        "rgb_exposure_start_timestamp_us",
        "rgb_exposure_end_timestamp_us",
    ]
    media = (
        ds.dataset(eap / "data/train.parquet")
        .to_table(columns=media_columns, filter=ds.field("sequence_id") == sequence)
        .to_pylist()
    )
    label_path = eap / "data/train" / sequence / "labels.parquet"
    object_columns = ["sample_token", "track_id", "instance_id", "bbox_3d_ego"]
    objects = ds.dataset(label_path).to_table(columns=object_columns).to_pylist()
    lookup = {(str(row["sample_token"]), str(row["instance_id"])): row for row in objects}
    frames = {row["rgb_member_path"]: row for row in media}
    if len(lookup) != len(objects) or len(frames) != len(media):
        raise ValueError("ambiguous frame/object identity")
    pair_columns = ["sample_token", "sequence_id", "track_id", "rgb_member_paths", "boxes_xyxy"]
    scanner = ds.dataset(paths["garl_annotations_candidate"]).scanner(
        columns=pair_columns, filter=ds.field("sequence_id") == sequence, batch_size=128
    )
    pairs = scanner.head(128).to_pylist()
    comparisons = []
    missing = 0
    for pair in pairs:
        for member, box in zip(pair["rgb_member_paths"], pair["boxes_xyxy"], strict=True):
            frame = frames.get(member)
            obj = lookup.get((str(frame["sample_token"]), str(pair["track_id"]))) if frame else None
            if obj is None or frame is None:
                missing += 1
                continue
            record = {
                "pair": pair["sample_token"],
                "frame": frame["sample_token"],
                "track": pair["track_id"],
                "published_xyxy": box,
            }
            try:
                _, projected, _, _ = project_box_3d_to_event(
                    obj["bbox_3d_ego"], frame["K_event"], frame["T_event_ego"]
                )
                error = float(np.max(np.abs(np.asarray(projected) - box)))
                record.update(projected_xyxy=projected, max_corner_error_px=error)
            except ValueError as error:
                record["projection_error"] = str(error)
            comparisons.append(record)
    errors = [row["max_corner_error_px"] for row in comparisons if "max_corner_error_px" in row]
    write_new_json(
        args.output,
        {
            "scope": "USER_AUTHORIZED_GEOMETRY_FEASIBILITY_ONLY",
            "sequence": sequence,
            "ack_sha256": config["ack_sha256"],
            "object_source_sha256": sha256(label_path),
            "projection_source_sha256": sha256(Path("src/e_jepa_ttc/data/eap.py")),
            "media_count": len(media),
            "object_count": len(objects),
            "pairs_checked": len(pairs),
            "unmatched_frame_objects": missing,
            "compared": len(errors),
            "exact_float_matches": sum(value == 0 for value in errors),
            "maximum_corner_error_px": max(errors) if errors else None,
            "median_corner_error_px": float(np.median(errors)) if errors else None,
            "comparisons": comparisons,
            "columns": {"media": media_columns, "objects": object_columns, "pairs": pair_columns},
            "ttc_or_velocity_read": False,
            "depth_used_for_geometric_projection_only": True,
            "full_timeline_completeness_proven": False,
            "production_history_authorized": False,
            "expert_preprocessing_changed": False,
            "scientific_fits": 0,
        },
    )
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "compared": len(errors),
                "unmatched": missing,
                "max_error_px": max(errors) if errors else None,
            }
        )
    )


if __name__ == "__main__":
    main()
