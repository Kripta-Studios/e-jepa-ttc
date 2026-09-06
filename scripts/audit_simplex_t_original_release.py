"""Audit original input histories against Garl without target-based membership."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.original_release import input_observation, read_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    if args.sequence not in ack["interfaces"]["role_manifest"]["roles"]["original"]:
        raise ValueError("original groups only")
    root = Path("artifacts/simplex_t/original_annotations") / args.sequence
    receipt = json.loads((root / "DOWNLOAD_RECEIPT.json").read_text())
    for entry in receipt["members"]:
        if sha256(root / Path(entry["zip_member"]).name) != entry["sha256"]:
            raise ValueError("downloaded source changed")
    source = read_records(root / "annotations.pkl")
    frames = read_records(root / "frames.pkl")
    observations = [input_observation(row) for row in source]

    def digest(rows: list[dict]) -> str:
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    original_digest = digest(observations)
    for row in source:
        row["ttc"] = "PERTURBED_NONNUMERIC_UNUSED"
        row["velocity"] = "PERTURBED_UNUSED"
        row["bbox_3d"] = "PERTURBED_UNUSED"
    invariant = digest([input_observation(row) for row in source]) == original_digest
    del source
    lookup = {(row["file_name"], row["instance_id"]): row for row in observations}
    by_frame = defaultdict(list)
    for row in observations:
        if row["sequence_id"] != args.sequence:
            raise ValueError("source contains another role/sequence")
        by_frame[row["file_name"]].append(row)
    frame_keys = {row["file_name"] for row in frames}
    columns = ["sample_token", "track_id", "rgb_member_paths", "boxes_xyxy"]
    pairs = (
        ds.dataset(paths["garl_annotations_candidate"])
        .scanner(columns=columns, filter=ds.field("sequence_id") == args.sequence, batch_size=128)
        .head(128)
        .to_pylist()
    )
    counts = Counter()
    diagnostics = []
    for pair in pairs:
        for member, box in zip(pair["rgb_member_paths"], pair["boxes_xyxy"], strict=True):
            direct = lookup.get((member, pair["track_id"]))
            counts["direct_identity_found" if direct else "direct_identity_missing"] += 1
            matches = [
                row["instance_id"]
                for row in by_frame[member]
                if np.array_equal(row["boxes_xyxy"], box)
            ]
            raw_matches = []
            for row in by_frame[member]:
                x, y, right, bottom = row["boxes_xyxy"]
                if np.array_equal([x, y, right - x, bottom - y], box):
                    raw_matches.append(row["instance_id"])
            counts["raw_xyxy_unique_match" if len(raw_matches) == 1 else "raw_xyxy_not_unique"] += 1
            counts[
                "unique_exact_box_match" if len(matches) == 1 else "no_unique_exact_box_match"
            ] += 1
            diagnostics.append(
                {
                    "pair": pair["sample_token"],
                    "member": member,
                    "garl_track": pair["track_id"],
                    "exact_box_candidates": matches,
                    "raw_xyxy_candidates_diagnostic_only": raw_matches,
                    "direct_box_exact": bool(direct and np.array_equal(direct["boxes_xyxy"], box)),
                }
            )
    write_new_json(
        args.output,
        {
            "sequence": args.sequence,
            "observation_count": len(observations),
            "frame_count": len(frames),
            "unique_observation_keys": len(lookup),
            "objects_without_frame": sum(
                row["file_name"] not in frame_keys for row in observations
            ),
            "input_observation_sha256": original_digest,
            "target_depth_velocity_perturbation_invariant": invariant,
            "pair_audit_count": len(pairs),
            "counts": dict(counts),
            "diagnostics": diagnostics,
            "source_receipt_sha256": sha256(root / "DOWNLOAD_RECEIPT.json"),
            "pickle_decoder": "primitive-only opcode scan and globals/persistent references denied",
            "target_values_used": False,
            "production_history_ready": False,
            "limitation": (
                "Exact box matches diagnose correspondence; not adopted as a persistent-ID mapping"
            ),
        },
    )
    print(
        json.dumps(
            {
                "sequence": args.sequence,
                "observations": len(observations),
                "counts": dict(counts),
                "perturbation_invariant": invariant,
            }
        )
    )


if __name__ == "__main__":
    main()
